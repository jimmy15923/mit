import argparse
import logging
import os
import random
import time

import cv2
import MinkowskiEngine as ME
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
from MinkowskiEngine import SparseTensor
from sklearn import metrics
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from util import config
from util.util import (AsymmetricLossOptimized, AverageMeter,
                       get_pseudo_floor_label_by_z_quantile,
                       intersectionAndUnionGPU, poly_learning_rate,
                       save_checkpoint, seg_to_onehot)

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
np.set_printoptions(suppress=True)

best_iou = 0.0


def worker_init_fn(worker_id):
    random.seed(time.time() + worker_id)


def get_parser():
    parser = argparse.ArgumentParser(description='BPNet')
    parser.add_argument('--config', type=str, default='config/scannet/bpnet_5cm.yaml', help='config file')
    parser.add_argument('opts', help='see config/scannet/bpnet_5cm.yaml for all options', default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def main_process():
    return not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    cudnn.benchmark = True
    # for cudnn bug at https://github.com/pytorch/pytorch/issues/4107
    # https://github.com/Microsoft/human-pose-estimation.pytorch/issues/8
    # https://discuss.pytorch.org/t/training-performance-degrades-with-distributeddataparallel/47152/7
    # torch.backends.cudnn.enabled = False

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        # cudnn.benchmark = False
        # cudnn.deterministic = True
    # Log for check version
    print(
        'torch.__version__:%s\ntorch.version.cuda:%s\ntorch.backends.cudnn.version:%s\ntorch.backends.cudnn.enabled:%s' % (
            torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.backends.cudnn.enabled))

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
        args.use_apex = False

    if args.data_name == 'scannet_3d_mink':
        from dataset.scanNet3D import ScanNet3D, collation_fn
        _ = ScanNet3D(dataPathPrefix=args.data_root, voxelSize=args.voxelSize, split='train', aug=args.aug,
                      memCacheInit=True, loop=5)
        if args.evaluate:
            _ = ScanNet3D(dataPathPrefix=args.data_root, voxelSize=args.voxelSize, split='val', aug=args.aug,
                          memCacheInit=True)
    else:
        from dataset.scanNetCross import (ScanNetCross, collation_fn,
                                          collation_fn_eval_all)
        _ = ScanNetCross(dataPathPrefix=args.data_root, n_views=args.viewNum, voxelSize=args.voxelSize, split='train', aug=args.aug,
                         memCacheInit=True, loop=args.loop)
        if args.evaluate:
            _ = ScanNetCross(dataPathPrefix=args.data_root, voxelSize=args.voxelSize, split='val', aug=False,
                             memCacheInit=True, eval_all=True)

    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    global best_iou
    args = argss

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
                                rank=args.rank)

    model = get_model(args)
    if args.sync_bn_2d:
        print("using DDP synced BN for 2D")
        model.net2d.layer0 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.net2d.layer0)
        model.net2d.layer1 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.net2d.layer1)
        model.net2d.layer2 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.net2d.layer2)
        model.net2d.layer3 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.net2d.layer3)
        model.net2d.layer4 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.net2d.layer4)

    if main_process():
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        logger.info(model)

    # ####################### Optimizer ####################### #
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr)
    # scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=args.base_lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20)

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int(args.workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu], find_unused_parameters=False)
    else:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda(gpu)

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight, map_location=lambda storage, loc: storage.cuda())
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            if main_process():
                logger.info("=> no weight found at '{}'".format(args.weight))
    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            # checkpoint = torch.load(args.resume)
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_iou = checkpoint['best_iou']
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # ####################### Data Loader ####################### #
    if args.data_name == 'scannet_3d_mink':
        from dataset.scanNet3D import (ScanNet3D, collation_fn, collation_fn_eval_all)
        train_data = ScanNet3D(dataPathPrefix=args.data_root, voxelSize=args.voxelSize, split='train', aug=args.aug,
                               memCacheInit=True, loop=args.loop)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data) if args.distributed else None
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                                   shuffle=(train_sampler is None),
                                                   num_workers=args.workers, pin_memory=True, sampler=train_sampler,
                                                   drop_last=True, collate_fn=collation_fn,
                                                   worker_init_fn=worker_init_fn)
        if args.evaluate:
            val_data = ScanNet3D(dataPathPrefix=args.data_root, voxelSize=args.voxelSize, split='val', aug=False,
                                 memCacheInit=True, eval_all=True)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_data) if args.distributed else None
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val,
                                                     shuffle=False, num_workers=args.workers, pin_memory=True,
                                                     drop_last=False, collate_fn=collation_fn_eval_all,
                                                     sampler=val_sampler)
    elif args.data_name == 'scannet_cross' or args.data_name == 'scannet_2d':
        from dataset.scanNetCross import (ScanNetCross, collation_fn,
                                          collation_fn_eval_all)
        train_data = ScanNetCross(
            dataPathPrefix=args.data_root, voxelSize=args.voxelSize, n_views=args.viewNum, split='train', aug=args.aug,
            memCacheInit=True, loop=args.loop)

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data) if args.distributed else None
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                                   shuffle=(train_sampler is None),
                                                   num_workers=args.workers, pin_memory=True, sampler=train_sampler,
                                                   drop_last=True, collate_fn=collation_fn,
                                                   worker_init_fn=worker_init_fn)
        if args.evaluate:
            val_data = ScanNetCross(dataPathPrefix=args.data_root, voxelSize=args.voxelSize, split='val', aug=False,
                                    memCacheInit=True, eval_all=True)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_data) if args.distributed else None
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val,
                                                     shuffle=False, num_workers=args.workers, pin_memory=True,
                                                     drop_last=False, collate_fn=collation_fn_eval_all,
                                                     sampler=val_sampler)
    else:
        raise Exception('Dataset not supported yet'.format(args.data_name))

    # ####################### Train ####################### #
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            if args.evaluate:
                val_sampler.set_epoch(epoch)
        if args.data_name == 'scannet_cross':
            loss_train, mIoU_train, mAcc_train, allAcc_train, \
            loss_train_2d, mIoU_train_2d, mAcc_train_2d, allAcc_train_2d \
                = train_cross_weak(train_loader, model, criterion, optimizer, scheduler, epoch)
        elif args.data_name == 'scannet_2d':
            loss_train, mIoU_train, mAcc_train, allAcc_train = train_view_weak(train_loader, model, criterion, optimizer,epoch)            
        else:
            loss_train, mIoU_train, mAcc_train, allAcc_train = train_weak(train_loader, model, criterion, optimizer, epoch)
        epoch_log = epoch + 1
        scheduler.step() 
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
            writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            writer.add_scalar('allAcc_train', allAcc_train, epoch_log)
            if args.data_name == 'scannet_cross':
                writer.add_scalar('loss_train_2d', loss_train_2d, epoch_log)
                writer.add_scalar('mIoU_train_2d', mIoU_train_2d, epoch_log)
                writer.add_scalar('mAcc_train_2d', mAcc_train_2d, epoch_log)
                writer.add_scalar('allAcc_train_2d', allAcc_train_2d, epoch_log)

            is_best = False
            is_best = mIoU_train > best_iou
            best_iou = max(best_iou, mIoU_train)
            logger.info('Best Iou: %.3f' % (best_iou))

        if args.evaluate and (epoch_log % args.eval_freq == 0):
            if args.data_name == 'scannet_cross':
                loss_val, mIoU_val, mAcc_val, allAcc_val, \
                loss_val_2d, mIoU_val_2d, mAcc_val_2d, allAcc_val_2d \
                    = validate_cross(val_loader, model, criterion)
            else:
                loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion)

            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
                if args.data_name == 'scannet_cross':
                    writer.add_scalar('loss_val_2d', loss_val_2d, epoch_log)
                    writer.add_scalar('mIoU_val_2d', mIoU_val_2d, epoch_log)
                    writer.add_scalar('mAcc_val_2d', mAcc_val_2d, epoch_log)
                    writer.add_scalar('allAcc_val_2d', allAcc_val_2d, epoch_log)
                # remember best iou and save checkpoint
                # is_best = mIoU_train > best_iou
                # best_iou = max(best_iou, mIoU_train)

        if (epoch_log % args.save_freq == 0) and main_process():
            save_checkpoint(
                {
                    'epoch': epoch_log,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_iou': best_iou
                }, is_best, os.path.join(args.save_path, 'model')
            )
    if main_process():
        writer.close()
        logger.info('==>Training done!\nBest Iou: %.3f' % (best_iou))


def get_model(cfg):
    if cfg.arch == 'mink_18A':
        from models.unet_3d import MinkUNet18A as Model
        model = Model(in_channels=3, out_channels=cfg.classes, cfg=cfg, D=3, is_cls=cfg.is_cls)
    elif cfg.arch == 'mink_18B':
        from models.unet_3d import MinkUNet18B as Model
        model = Model(in_channels=3, out_channels=cfg.classes, cfg=cfg, D=3, is_cls=cfg.is_cls, is_attn=cfg.use_attn)    
    elif cfg.arch == 'mink_18C':
        from models.unet_3d import MinkUNet18C as Model
        model = Model(in_channels=3, out_channels=cfg.classes, D=3, is_cls=True, is_attn=cfg.use_attn)            
    elif cfg.arch == 'mink_18D':
        from models.unet_3d import MinkUNet18D as Model
        model = Model(in_channels=3, out_channels=cfg.classes, cfg=cfg, D=3, is_cls=cfg.is_cls)
    elif cfg.arch == 'mink_14A':
        from models.unet_3d import MinkUNet14A as Model
        model = Model(in_channels=3, out_channels=cfg.classes, D=3, is_cls=True, is_attn=cfg.use_attn)   
    elif cfg.arch == 'mink_14B':
        from models.unet_3d import MinkUNet14B as Model
        model = Model(in_channels=3, out_channels=cfg.classes, D=3, is_cls=True, is_attn=cfg.use_attn)             
    elif cfg.arch == 'mink_14C':
        from models.unet_3d import MinkUNet14C as Model
        model = Model(in_channels=3, out_channels=cfg.classes, D=3, is_cls=True, is_attn=cfg.use_attn)
    elif cfg.arch == 'mink_14D':
        from models.unet_3d import MinkUNet14D as Model
        model = Model(in_channels=3, out_channels=cfg.classes, D=3, is_cls=True, is_attn=cfg.use_attn)                 
    elif cfg.arch == 'mink_34A':
        from models.unet_3d import MinkUNet34A as Model
        model = Model(in_channels=3, out_channels=cfg.classes, D=3, is_cls=True, is_attn=cfg.use_attn)   
    elif cfg.arch == 'mink_34B':
        from models.unet_3d import MinkUNet34B as Model
        model = Model(in_channels=3, out_channels=cfg.classes, D=3, is_cls=True, is_attn=cfg.use_attn)             
    elif cfg.arch == 'mink_34C':
        from models.unet_3d import MinkUNet34C as Model
        model = Model(in_channels=3, out_channels=cfg.classes, cfg=cfg, D=3, is_cls=cfg.is_cls)
    elif cfg.arch == 'mink_34D':
        from models.unet_3d import MinkUNet34D as Model
        model = Model(in_channels=3, out_channels=cfg.classes, cfg=cfg, D=3, is_cls=cfg.is_cls)
    elif cfg.arch == 'mink_res50':
        from models.resnet_mink import MinkResNet as Model
        model = Model(depth=50, in_channels=3, is_attn=cfg.use_attn)
    elif cfg.arch == 'mink_res34':
        from models.resnet_mink import ResNet34 as Model
        model = Model(in_channels=3, out_channels=cfg.classes, cfg=cfg, D=3, is_cls=True)        
    elif cfg.arch == 'mink_resunet50':
        from models.unet_3d import MinkResNet50 as Model
        model = Model(in_channels=3, out_channels=cfg.classes, cfg=cfg, D=3, is_cls=cfg.is_cls)        
    elif cfg.arch == 'bpnet':
        from models.mit import MIT as Model
        model = Model(cfg=cfg)
    elif cfg.arch == 'resnet2d':
        from models.unet_2d import MctResNet as Model
        model = Model(cfg=cfg, layers=cfg.layers_2d)
    elif cfg.arch == 'mct_2d':
        from models.mct_2d import MCTformerV1 as Model
        model = Model(num_classes=cfg.classes, last_opt='average')
    else:
        raise Exception('architecture not supported yet'.format(cfg.arch))
    return model


def train_cross_weak(train_loader, model, criterion, optimizer, scheduler, epoch):
    # raise NotImplemented
    torch.backends.cudnn.enabled = True
    batch_time = AverageMeter()
    data_time = AverageMeter()

    loss_meter = AverageMeter()
    loss_meter_cam_2d = AverageMeter()
    loss_meter_cam_3d = AverageMeter()
    loss_meter_mct_2d = AverageMeter()
    loss_meter_mct_3d = AverageMeter()
    loss_meter_cross = AverageMeter()
    loss_meter_sim = AverageMeter()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    ap_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    iters = len(train_loader)
    criterion = nn.BCELoss()
    for i, batch_data in enumerate(train_loader):
        data_time.update(time.time() - end)

        (coords, feat, label_3d, supervoxel, color, link, label_2d) = batch_data
        coords[:, 1:] += (torch.rand(3) * 100).type_as(coords)
        sinput = SparseTensor(feat.cuda(non_blocking=True), coords.cuda(non_blocking=True))
        supervoxel = [x.cuda(non_blocking=True) for x in supervoxel]

        color, link = color.cuda(non_blocking=True), link.cuda(non_blocking=True)

        output_dict = model(sinput, supervoxel, color, link)        # pdb.set_trace()
        cross_mct_logits = output_dict['cross_mct_logits']
        mct_logits_3d = output_dict['mct_logits_3d']
        mct_logits_2d = output_dict['mct_logits_2d']
        cls_logits_3d = output_dict['cam_logits_3d']
        cls_logits_2d = output_dict['cam_logits_2d']
        loss_sim = output_dict['token_sim_loss']
        loss_cons = output_dict['consistent_loss']
        batch_pool_cam = output_dict['cams']
        batch_pool_cam_2d = output_dict['view_cams']

        ## 3D global pooling
        global_labels = seg_to_onehot(label_3d)
        loss_cross = F.multilabel_soft_margin_loss(cross_mct_logits, global_labels)
        loss_mct_2d = F.multilabel_soft_margin_loss(mct_logits_2d, global_labels)
        loss_mct_3d = F.multilabel_soft_margin_loss(mct_logits_3d, global_labels)
        loss_cam_2d = F.multilabel_soft_margin_loss(cls_logits_2d, global_labels)
        loss_cam_3d = criterion(cls_logits_3d, global_labels)
        loss = loss_mct_3d + loss_mct_2d + loss_cross  + loss_cam_3d + loss_cam_2d + loss_sim + args.alpha * loss_cons

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(epoch + i / iters)

        for idx, pool_cam in enumerate(batch_pool_cam):
            pool_cam = pool_cam * global_labels[idx]

            score, y_pred = pool_cam.max(1)
            supervoxel_to_seglabel = {}
            for voxel_i in supervoxel[idx].unique():
                if score[voxel_i] < 0.5:
                    supervoxel_to_seglabel[int(voxel_i)] = 255
                else:
                    supervoxel_to_seglabel[int(voxel_i)] = int(y_pred[voxel_i])
                    
            for c in torch.where(global_labels == 1)[1]:
                max_activate_class = (torch.argmax(pool_cam[:, c]))
                supervoxel_to_seglabel[int(max_activate_class)] = int(c)
      
            y_pred_all = supervoxel[idx].cpu().apply_(supervoxel_to_seglabel.get).cuda()

            intersection, union, target = intersectionAndUnionGPU(
                y_pred_all, label_3d[idx].cuda(), args.classes, True, args.ignore_label,
            )

            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)


        for idx, pool_cam in enumerate(batch_pool_cam_2d):
            # ############ 2D ############ #
            view_labels = label_2d[idx]
            for view_idx, y_true in enumerate(view_labels):
                y_pred = np.zeros(args.classes)
                y_pred[torch.where(pool_cam[:, view_idx].sigmoid() > 0.9)[0].cpu().numpy()] = 1
                ap = metrics.precision_score(y_true.cpu().numpy(), y_pred, zero_division=0)
                ap_meter.update(ap)


        loss_meter.update(loss.item(), args.batch_size)
        loss_meter_cam_2d.update(loss_cam_2d.item(), args.batch_size)
        loss_meter_cam_3d.update(loss_cam_3d.item(), args.batch_size)
        loss_meter_mct_2d.update(loss_mct_2d.item(), args.batch_size)
        loss_meter_mct_3d.update(loss_mct_3d.item(), args.batch_size)
        loss_meter_cross.update(loss_cross.item(), args.batch_size)
        loss_meter_sim.update(loss_sim.item(), args.batch_size)
        batch_time.update(time.time() - end)
        end = time.time()

        # # Adjust lr
        current_iter = epoch * len(train_loader) + i + 1

        # calculate remain time
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss 3D {loss_meter_cam_3d.val:.4f} '
                        'Loss 2D {loss_meter_cam_2d.val:.4f} '
                        'Loss mct3D {loss_meter_mct_3d.val:.4f} '
                        'Loss mct2D {loss_meter_mct_2d.val:.4f} '
                        'Loss cross {loss_meter_cross.val:.4f} '
                        'Loss sim {loss_meter_sim.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(
                            epoch + 1, args.epochs, i + 1, len(train_loader),
                            batch_time=batch_time, data_time=data_time,
                            remain_time=remain_time,
                            loss_meter_cam_3d=loss_meter_cam_3d,
                            loss_meter_cam_2d=loss_meter_cam_2d,
                            loss_meter_mct_3d=loss_meter_mct_3d,
                            loss_meter_mct_2d=loss_meter_mct_2d,
                            loss_meter_cross=loss_meter_cross,
                            loss_meter_sim=loss_meter_sim,
                            accuracy=accuracy,
                        ))
        if main_process():
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            writer.add_scalar('loss3d_train_batch', loss_meter_mct_3d.val, current_iter)
            writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    mIoU_2d = ap_meter.avg

    if main_process():
        logger.info(
            'Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch + 1, args.epochs,
                                                                                           mIoU, mIoU_2d, allAcc))    
        print(iou_class)    
    return loss_meter_cam_3d.avg, mIoU, mAcc, allAcc, \
           loss_meter_mct_3d.avg, mIoU, mAcc, allAcc



def train_weak(train_loader, model, criterion, optimizer, epoch):
    # raise NotImplemented
    torch.backends.cudnn.enabled = True
    batch_time = AverageMeter()
    data_time = AverageMeter()

    loss_meter = AverageMeter()
    loss_meter_cam = AverageMeter()
    loss_meter_voxel = AverageMeter()
    loss_meter_mct = AverageMeter()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    for i, batch_data in enumerate(train_loader):
        data_time.update(time.time() - end)

        (coords, feat, label_3d, overseg) = batch_data
        plabel = get_pseudo_floor_label_by_z_quantile(coords).long().cuda()
        coords[:, 1:] += (torch.rand(3) * 100).type_as(coords)
        sinput = SparseTensor(feat.cuda(non_blocking=True), coords.cuda(non_blocking=True))
        overseg = [x.cuda(non_blocking=True) for x in overseg]
        # Feed-forward pass and get the prediction

        output_dict = model(sinput, overseg)
        cls_logits = output_dict['cls_logits']
        mct_logits = output_dict['cls_mct_logits']
        loss_cons = output_dict['smooth_loss']
        batch_pool_cam = output_dict['cams']
        output_3d = output_dict['point_feature']
        
        # pdb.set_trace()

        ## 3D global label
        global_labels = seg_to_onehot(label_3d, num_classes=args.classes, ignore_index=args.ignore_label)

        # plabel
        # loss_seg = criterion(point_logits, label_3d)
        for idx, pool_cam in enumerate(batch_pool_cam):
            pool_cam = pool_cam * global_labels[idx]

            score, y_pred = pool_cam.max(1)
            supervoxel_to_seglabel = {}
            for voxel_i in overseg[idx].unique():
                if score[voxel_i] < 0.5:
                    supervoxel_to_seglabel[int(voxel_i)] = args.ignore_label
                else:
                    supervoxel_to_seglabel[int(voxel_i)] = int(y_pred[voxel_i])
                    
            for c in torch.where(global_labels == 1)[1]:
                max_activate_class = (torch.argmax(pool_cam[:, c]))
                supervoxel_to_seglabel[int(max_activate_class)] = int(c)

            y_pred_all = overseg[idx].cpu().apply_(supervoxel_to_seglabel.get).cuda()

            intersection, union, target = intersectionAndUnionGPU(
                y_pred_all, label_3d[idx].cuda(), args.classes, True, args.ignore_label,
            )

            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)


        if args.criterion == 'asl':
            loss_voxel = criterion(cls_logits, global_labels)
        elif args.criterion == 'weight':
            loss_voxel = criterion(cls_logits, global_labels)
        else:
            loss_voxel = F.multilabel_soft_margin_loss(cls_logits, global_labels)

        if len(mct_logits) > 0:
            loss_mct = F.multilabel_soft_margin_loss(mct_logits, global_labels)
        else:
            loss_mct = torch.zeros(1)[0].cuda()

        loss_3d = criterion(output_3d, torch.cat(label_3d).cuda())

        loss = loss_voxel + loss_mct + loss_cons + loss_3d

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ############ 3D ############ #
        for idx, pool_cam in enumerate(batch_pool_cam):
            pool_cam = pool_cam.softmax(-1)
            pool_cam = pool_cam * global_labels[idx]
            score, y_pred = pool_cam.max(1)

            supervoxel_to_seglabel = {}
            for voxel_i in overseg[idx].unique():
                supervoxel_to_seglabel[int(voxel_i)] = int(y_pred[voxel_i])

            y_pred_all = overseg[idx].cpu().apply_(supervoxel_to_seglabel.get).cuda()

            intersection, union, target = intersectionAndUnionGPU(
                y_pred_all, label_3d[idx].cuda(), args.classes, args.ignore_label,
            )

            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)

        loss_meter.update(loss.item(), args.batch_size)
        loss_meter_voxel.update(loss_voxel.item(), args.batch_size)
        # loss_meter_cam.update(loss_seg.item(), args.batch_size)
        loss_meter_mct.update(loss_mct.item(), args.batch_size)
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        current_iter = epoch * len(train_loader) + i + 1

        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        # 'Loss cam {loss_meter_cam.val:.4f} '
                        'Loss mct {loss_meter_mct.val:.4f} '
                        'Loss voxel {loss_meter_voxel.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch + 1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time, data_time=data_time,
                                                          remain_time=remain_time,
                                                          loss_meter=loss_meter,
                                                        #   loss_meter_cam=loss_meter_cam,
                                                          loss_meter_mct=loss_meter_mct,
                                                          loss_meter_voxel=loss_meter_voxel,
                                                          accuracy=accuracy))
        if main_process():
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            writer.add_scalar('loss3d_train_batch', loss_meter_cam.val, current_iter)
            writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info(
            'Train result at epoch [{}/{}]: mAP/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch + 1, args.epochs,
                                                                                           mIoU, mAcc, allAcc))                                                                                                
        print(iou_class)
        logger.info(f'Best IoU: {best_iou}')
    return loss_meter_cam.avg, mIoU, mAcc, allAcc


def train_view_weak(train_loader, model, criterion, optimizer, epoch):
    class_names = np.loadtxt('./dataset/scannet/scannet_names.txt', dtype=str)
    # raise NotImplemented
    torch.backends.cudnn.enabled = True
    batch_time = AverageMeter()
    data_time = AverageMeter()

    loss_meter = AverageMeter()
    loss_meter_cam = AverageMeter()
    loss_meter_mct = AverageMeter()
    # intersection_meter_3d, intersection_meter_2d = AverageMeter(), AverageMeter()
    # union_meter_3d, union_meter_2d = AverageMeter(), AverageMeter()
    # target_meter_3d, target_meter_2d = AverageMeter(), AverageMeter()
    ap_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    # pos_weight = torch.ones(20).cuda()
    # pos_weight[0] = 0.1
    # pos_weight[1] = 0.1
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    for i, batch_data in enumerate(train_loader):
        data_time.update(time.time() - end)

        _, _, label_3d, _, color, link, label_2d = batch_data

        color, link = color.cuda(non_blocking=True), link.cuda(non_blocking=True)

        output = model(color, link)
        scene_cls_logits = output['scene_logits']
        batch_pool_cam_2d = output['cams']
        # scene_cls_logits = model(color)
        # pdb.set_trace()

        ## scene-level labels
        global_labels = seg_to_onehot(label_3d)
        # global_labels = label_2d.view(-1, 20).cuda()
        loss_cam = F.multilabel_soft_margin_loss(scene_cls_logits, global_labels)
        if args.use_attn:
            mct_cls_logits = output['scene_mct_logits']
            loss_mct = F.multilabel_soft_margin_loss(mct_cls_logits, global_labels)

            loss = loss_cam + loss_mct
        else:
            loss = loss_cam

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ############ 3D ############ #
        for idx, pool_cam in enumerate(batch_pool_cam_2d):
            # ############ 3D ############ #
            view_labels = label_2d[idx]
            for view_idx, y_true in enumerate(view_labels):
                y_pred = np.zeros(20)
                y_pred[torch.where(pool_cam[:, view_idx].sigmoid() > 0.5)[0].cpu().numpy()] = 1
                ap = metrics.precision_score(y_true.cpu().numpy(), y_pred, zero_division=0)
                ap_meter.update(ap)


        loss_meter_cam.update(loss_cam.item(), args.batch_size)
        if args.use_attn:
            loss_meter_mct.update(loss_mct.item(), args.batch_size)
        loss_meter.update(loss.item(), args.batch_size)
        batch_time.update(time.time() - end)
        end = time.time()

        # Adjust lr
        current_iter = epoch * len(train_loader) + i + 1
        current_lr = poly_learning_rate(args.base_lr, current_iter, max_iter, power=args.power)
        # if args.arch == 'cross_p5' or args.arch == 'cross_p2':
        for index in range(0, len(optimizer.param_groups)):
            optimizer.param_groups[index]['lr'] = current_lr
        # for index in range(args.index_split, len(optimizer.param_groups)):
        #     optimizer.param_groups[index]['lr'] = current_lr * 10
        # else:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = current_lr

        # calculate remain time
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(float(remain_time), 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss cam {loss_meter_cam.val:.4f} '
                        'Loss mct {loss_meter_mct.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch + 1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time, data_time=data_time,
                                                          remain_time=remain_time,
                                                          loss_meter_cam=loss_meter_cam,
                                                          loss_meter_mct=loss_meter_mct,
                                                          accuracy=ap_meter.val))
        if main_process():
            # writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            # writer.add_scalar('loss3d_train_batch', loss_meter_3d.val, current_iter)
            writer.add_scalar('loss2d_train_batch', loss_meter_cam.val, current_iter)
            # writer.add_scalar('mIoU3d_train_batch', np.mean(intersection_meter_3d.val / (union_meter_3d.val + 1e-10)),
            #                   current_iter)
            # writer.add_scalar('mAcc3d_train_batch', np.mean(intersection_meter_3d.val / (target_meter_3d.val + 1e-10)),
            #                   current_iter)
            # writer.add_scalar('allAcc3d_train_batch', accuracy_3d, current_iter)

            # writer.add_scalar('mIoU2d_train_batch', np.mean(intersection_meter_2d.val / (union_meter_2d.val + 1e-10)),
                            #   current_iter)
            # writer.add_scalar('mAcc2d_train_batch', np.mean(intersection_meter_2d.val / (target_meter_2d.val + 1e-10)),
                            #   current_iter)
            # writer.add_scalar('allAcc2d_train_batch', accuracy_2d, current_iter)

            writer.add_scalar('learning_rate', current_lr, current_iter)

    # iou_class_3d = intersection_meter_3d.sum / (union_meter_3d.sum + 1e-10)
    # accuracy_class_3d = intersection_meter_3d.sum / (target_meter_3d.sum + 1e-10)
    # mIoU_3d = np.mean(iou_class_3d)
    # mAcc_3d = np.mean(ap_meter)
    allAcc_3d = 0
    # iou_class_2d = intersection_meter_2d.sum / (union_meter_2d.sum + 1e-10)
    # accuracy_class_2d = intersection_meter_2d.sum / (target_meter_2d.sum + 1e-10)
    mIoU_2d = ap_meter.avg
    # mAcc_2d = np.mean(accuracy_class_2d)
    # allAcc_2d = sum(intersection_meter_2d.sum) / (sum(target_meter_2d.sum) + 1e-10)
    # precision = metrics.precision_score(y_trues, y_preds, average=None, zero_division=0)
    # accuracy_3d = np.mean(precision)
    if main_process():
        logger.info(
            'Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch + 1, args.epochs,
                                                                                           mIoU_2d, mIoU_2d, allAcc_3d))    
    return loss_meter_cam.avg, mIoU_2d, mIoU_2d, allAcc_3d


def train_cross(train_loader, model, criterion, optimizer, epoch, is_gap=False):
    # raise NotImplemented
    torch.backends.cudnn.enabled = True
    batch_time = AverageMeter()
    data_time = AverageMeter()

    loss_meter, loss_meter_3d, loss_meter_2d = AverageMeter(), AverageMeter(), AverageMeter()
    intersection_meter_3d, intersection_meter_2d = AverageMeter(), AverageMeter()
    union_meter_3d, union_meter_2d = AverageMeter(), AverageMeter()
    target_meter_3d, target_meter_2d = AverageMeter(), AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    if is_gap:
        mil_criterion = nn.MultiLabelSoftMarginLoss()    
    for i, batch_data in enumerate(train_loader):
        data_time.update(time.time() - end)

        if args.data_name == 'scannet_cross':
            (coords, feat, label_3d, color, label_2d, link) = batch_data
            # For some networks, making the network invariant to even, odd coords is important
            # coords[:, 1:4] += (torch.rand(3) * 100).type_as(coords)

            sinput = SparseTensor(feat.cuda(non_blocking=True), coords)
       
            color, link = color.cuda(non_blocking=True), link.cuda(non_blocking=True)
            label_3d, label_2d = label_3d.cuda(non_blocking=True), label_2d.cuda(non_blocking=True)

            output_3d, output_2d = model(sinput, color, link)
            # pdb.set_trace()
            if is_gap:
                global_features = torch.mean(output_3d, dim=0)
                global_labels = torch.zeros(20).cuda()
                for i in set(label_3d):
                    if i != 255:
                        global_labels[i] = 1
                loss_3d = mil_criterion(global_features.unsqueeze(0), global_labels.unsqueeze(0))
            else:
                loss_3d = criterion(output_3d, label_3d)
            loss_2d = criterion(output_2d, label_2d)
            loss = loss_3d + args.weight_2d * loss_2d
        else:
            raise NotImplemented
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ############ 3D ############ #
        output_3d = output_3d.detach().max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output_3d, label_3d.detach(), args.classes,
                                                              args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter_3d.update(intersection)
        union_meter_3d.update(union)
        target_meter_3d.update(target)
        accuracy_3d = sum(intersection_meter_3d.val) / (sum(target_meter_3d.val) + 1e-10)
        # ############ 2D ############ #
        output_2d = output_2d.detach().max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output_2d, label_2d.detach(), args.classes,
                                                              args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter_2d.update(intersection)
        union_meter_2d.update(union)
        target_meter_2d.update(target)
        accuracy_2d = sum(intersection_meter_2d.val) / (sum(target_meter_2d.val) + 1e-10)

        loss_meter.update(loss.item(), args.batch_size)
        loss_meter_2d.update(loss_2d.item(), args.batch_size)
        loss_meter_3d.update(loss_3d.item(), args.batch_size)
        batch_time.update(time.time() - end)
        end = time.time()

        # Adjust lr
        current_iter = epoch * len(train_loader) + i + 1
        current_lr = poly_learning_rate(args.base_lr, current_iter, max_iter, power=args.power)
        # if args.arch == 'cross_p5' or args.arch == 'cross_p2':
        # for index in range(0, args.index_split):
        #     optimizer.param_groups[index]['lr'] = current_lr
        # for index in range(args.index_split, len(optimizer.param_groups)):
        #     optimizer.param_groups[index]['lr'] = current_lr * 10
        # else:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = current_lr

        # calculate remain time
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch + 1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time, data_time=data_time,
                                                          remain_time=remain_time,
                                                          loss_meter=loss_meter_3d,
                                                          accuracy=accuracy_3d))
        if main_process():
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            writer.add_scalar('loss3d_train_batch', loss_meter_3d.val, current_iter)
            writer.add_scalar('loss2d_train_batch', loss_meter_2d.val, current_iter)
            writer.add_scalar('mIoU3d_train_batch', np.mean(intersection_meter_3d.val / (union_meter_3d.val + 1e-10)),
                              current_iter)
            writer.add_scalar('mAcc3d_train_batch', np.mean(intersection_meter_3d.val / (target_meter_3d.val + 1e-10)),
                              current_iter)
            writer.add_scalar('allAcc3d_train_batch', accuracy_3d, current_iter)

            writer.add_scalar('mIoU2d_train_batch', np.mean(intersection_meter_2d.val / (union_meter_2d.val + 1e-10)),
                              current_iter)
            writer.add_scalar('mAcc2d_train_batch', np.mean(intersection_meter_2d.val / (target_meter_2d.val + 1e-10)),
                              current_iter)
            writer.add_scalar('allAcc2d_train_batch', accuracy_2d, current_iter)

            writer.add_scalar('learning_rate', current_lr, current_iter)

    iou_class_3d = intersection_meter_3d.sum / (union_meter_3d.sum + 1e-10)
    accuracy_class_3d = intersection_meter_3d.sum / (target_meter_3d.sum + 1e-10)
    mIoU_3d = np.mean(iou_class_3d)
    mAcc_3d = np.mean(accuracy_class_3d)
    allAcc_3d = sum(intersection_meter_3d.sum) / (sum(target_meter_3d.sum) + 1e-10)

    iou_class_2d = intersection_meter_2d.sum / (union_meter_2d.sum + 1e-10)
    accuracy_class_2d = intersection_meter_2d.sum / (target_meter_2d.sum + 1e-10)
    mIoU_2d = np.mean(iou_class_2d)
    mAcc_2d = np.mean(accuracy_class_2d)
    allAcc_2d = sum(intersection_meter_2d.sum) / (sum(target_meter_2d.sum) + 1e-10)

    if main_process():
        logger.info(
            'Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch + 1, args.epochs,
                                                                                           mIoU_3d, mAcc_3d, allAcc_3d))
    return loss_meter_3d.avg, mIoU_3d, mAcc_3d, allAcc_3d, \
           loss_meter_2d.avg, mIoU_2d, mAcc_2d, allAcc_2d







def validate_cross(val_loader, model, criterion):
    torch.backends.cudnn.enabled = False  # for cudnn bug at https://github.com/pytorch/pytorch/issues/4107
    loss_meter, loss_meter_3d, loss_meter_2d = AverageMeter(), AverageMeter(), AverageMeter()
    intersection_meter_3d, intersection_meter_2d = AverageMeter(), AverageMeter()
    union_meter_3d, union_meter_2d = AverageMeter(), AverageMeter()
    target_meter_3d, target_meter_2d = AverageMeter(), AverageMeter()

    model.eval()
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):

            if args.data_name == 'scannet_cross':
                (coords, feat, label_3d, color, label_2d, link, inds_reverse) = batch_data
                sinput = SparseTensor(feat.cuda(non_blocking=True), coords)
                color, link = color.cuda(non_blocking=True), link.cuda(non_blocking=True)
                label_3d, label_2d = label_3d.cuda(non_blocking=True), label_2d.cuda(non_blocking=True)

                output_3d, output_2d = model(sinput, color, link)
                output_3d = output_3d[inds_reverse, :]
                # pdb.set_trace()
                loss_3d = criterion(output_3d, label_3d)
                loss_2d = criterion(output_2d, label_2d)
                loss = loss_3d + args.weight_2d * loss_2d
            else:
                raise NotImplemented
            # ############ 3D ############ #
            output_3d = output_3d.detach().max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(output_3d, label_3d.detach(), args.classes,
                                                                  args.ignore_label)
            if args.multiprocessing_distributed:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter_3d.update(intersection)
            union_meter_3d.update(union)
            target_meter_3d.update(target)
            accuracy_3d = sum(intersection_meter_3d.val) / (sum(target_meter_3d.val) + 1e-10)
            # ############ 2D ############ #
            output_2d = output_2d.detach().max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(output_2d, label_2d.detach(), args.classes,
                                                                  args.ignore_label)
            if args.multiprocessing_distributed:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter_2d.update(intersection)
            union_meter_2d.update(union)
            target_meter_2d.update(target)
            accuracy_2d = sum(intersection_meter_2d.val) / (sum(target_meter_2d.val) + 1e-10)

            loss_meter.update(loss.item(), args.batch_size)
            loss_meter_2d.update(loss_2d.item(), args.batch_size)
            loss_meter_3d.update(loss_3d.item(), args.batch_size)

    iou_class_3d = intersection_meter_3d.sum / (union_meter_3d.sum + 1e-10)
    accuracy_class_3d = intersection_meter_3d.sum / (target_meter_3d.sum + 1e-10)
    mIoU_3d = np.mean(iou_class_3d)
    mAcc_3d = np.mean(accuracy_class_3d)
    allAcc_3d = sum(intersection_meter_3d.sum) / (sum(target_meter_3d.sum) + 1e-10)

    iou_class_2d = intersection_meter_2d.sum / (union_meter_2d.sum + 1e-10)
    accuracy_class_2d = intersection_meter_2d.sum / (target_meter_2d.sum + 1e-10)
    mIoU_2d = np.mean(iou_class_2d)
    mAcc_2d = np.mean(accuracy_class_2d)
    allAcc_2d = sum(intersection_meter_2d.sum) / (sum(target_meter_2d.sum) + 1e-10)

    if main_process():
        logger.info(
            'Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU_3d, mAcc_3d, allAcc_3d))
    return loss_meter_3d.avg, mIoU_3d, mAcc_3d, allAcc_3d, \
           loss_meter_2d.avg, mIoU_2d, mAcc_2d, allAcc_2d


def train(train_loader, model, criterion, optimizer, epoch):
    torch.backends.cudnn.enabled = True
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    mil_criterion = nn.MultiLabelSoftMarginLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, batch_data in enumerate(train_loader):
        data_time.update(time.time() - end)
        (coords, feat, label) = batch_data
        # print(feat.cuda(non_blocking=True).shape)
        # logger.info('feat', np.min(feat), np.max(feat))
        # For some networks, making the network invariant to even, odd coords is important
        coords[:, :3] += (torch.rand(3) * 100).type_as(coords)

        sinput = SparseTensor(feat, coordinates=coords, device=device)

        permutations = [inds for inds in sinput.decomposition_permutations 
                            if inds.nelement() != 0]
        cls_label = torch.zeros((train_loader.batch_size, 20)).to(device)
        for b_i, inds in enumerate(permutations):
            per_scene_label = label[inds]
            for cls in torch.unique(per_scene_label):
                if cls != 255: # ignore 255
                    cls_label[b_i, cls] = 1

        label = label.cuda(non_blocking=True)

        # mil_feat, seg_feat, global_feat = model(sinput)
        seg_feat, cls_loss, mil_loss = model(sinput, cls_label)
        # pdb.set_trace()
        seg_loss = criterion(seg_feat, label)

        loss = seg_loss + 0.05*cls_loss + 0.01*mil_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        seg_feat = seg_feat.detach().max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(seg_feat, label.detach(), args.classes, args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), args.batch_size)
        batch_time.update(time.time() - end)
        end = time.time()

        # Adjust lr
        current_iter = epoch * len(train_loader) + i + 1
        current_lr = poly_learning_rate(args.base_lr, current_iter, max_iter, power=args.power)
        if args.arch == 'psp50':
            for index in range(0, args.index_split):
                optimizer.param_groups[index]['lr'] = current_lr
            for index in range(args.index_split, len(optimizer.param_groups)):
                optimizer.param_groups[index]['lr'] = current_lr * 10
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        # calculate remain time
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch + 1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time, data_time=data_time,
                                                          remain_time=remain_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))
        if main_process():
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            writer.add_scalar('allAcc_train_batch', accuracy, current_iter)
            writer.add_scalar('learning_rate', current_lr, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info(
            'Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch + 1, args.epochs, mIoU,
                                                                                           mAcc, allAcc))
    torch.cuda.empty_cache()                                                                                 
    return loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model, criterion):
    torch.backends.cudnn.enabled = False  # for cudnn bug at https://github.com/pytorch/pytorch/issues/4107
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            (coords, feat, label, inds_reverse) = batch_data
            sinput = SparseTensor(feat, coordinates=coords, device=device)
            label = label.cuda(non_blocking=True)
            output, _, _ = model(sinput)
            # pdb.set_trace()
            output = output[inds_reverse, :]
            loss = criterion(output, label)

            output = output.detach().max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(output, label.detach(), args.classes,
                                                                  args.ignore_label)
            if args.multiprocessing_distributed:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

            loss_meter.update(loss.item(), args.batch_size)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info(
            'Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    torch.cuda.empty_cache() 
    return loss_meter.avg, mIoU, mAcc, allAcc



if __name__ == '__main__':
    main()
