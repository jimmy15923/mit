import json
import math
import random
from glob import glob
from os.path import exists, join

import cv2
import imageio
import numpy as np
import PIL
import SharedArray as SA
import tifffile
import torch
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from timm.data.transforms import RandomResizedCropAndInterpolation

from tqdm import tqdm
from dataset.scanNet3D import ScanNet3D
from util.util import seg_to_onehot


def sa_create(name, var):
    x = SA.create(name, var.shape, dtype=var.dtype)
    x[...] = var[...]
    x.flags.writeable = False
    return x


# create camera intrinsics
def make_intrinsic(fx, fy, mx, my):
    intrinsic = np.eye(4)
    intrinsic[0][0] = fx
    intrinsic[1][1] = fy
    intrinsic[0][2] = mx
    intrinsic[1][2] = my
    return intrinsic


# create camera intrinsics
def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
    return intrinsic


class LinkCreator(object):
    def __init__(self, fx=577.870605, fy=577.870605, mx=319.5, my=239.5, image_dim=(320, 240), voxelSize=0.05):
        self.intricsic = make_intrinsic(fx=fx, fy=fy, mx=mx, my=my)
        self.intricsic = adjust_intrinsic(self.intricsic, intrinsic_image_dim=[640, 480], image_dim=image_dim)
        self.imageDim = image_dim
        self.voxel_size = voxelSize

    def computeLinking(self, camera_to_world, coords, depth):
        """
        :param camera_to_world: 4 x 4
        :param coords: N x 3 format
        :param depth: H x W format
        :return: linking, N x 3 format, (H,W,mask)
        """
        link = np.zeros((3, coords.shape[0]), dtype=np.int)
        coordsNew = np.concatenate([coords, np.ones([coords.shape[0], 1])], axis=1).T
        assert coordsNew.shape[0] == 4, "[!] Shape error"

        world_to_camera = np.linalg.inv(camera_to_world)
        p = np.matmul(world_to_camera, coordsNew)
        p[0] = (p[0] * self.intricsic[0][0]) / p[2] + self.intricsic[0][2]
        p[1] = (p[1] * self.intricsic[1][1]) / p[2] + self.intricsic[1][2]
        pi = np.round(p).astype(np.int)
        inside_mask = (pi[0] >= 0) * (pi[1] >= 0) \
                      * (pi[0] < self.imageDim[0]) * (pi[1] < self.imageDim[1])
        occlusion_mask = np.abs(depth[pi[1][inside_mask], pi[0][inside_mask]]
                                - p[2][inside_mask]) <= self.voxel_size
        inside_mask[inside_mask == True] = occlusion_mask
        link[0][inside_mask] = pi[1][inside_mask]
        link[1][inside_mask] = pi[0][inside_mask]
        link[2][inside_mask] = 1

        return link.T


class ScanNetCross(ScanNet3D):
    IMG_DIM = (320, 240)

    def __init__(self, dataPathPrefix='Data', voxelSize=0.05,
                 n_views=64,
                 split='train', aug=False, memCacheInit=False,
                 identifier=1233, loop=1,
                 data_aug_color_trans_ratio=0.1,
                 data_aug_color_jitter_std=0.05, data_aug_hue_max=0.5,
                 data_aug_saturation_max=0.2, eval_all=False,
                 val_benchmark=False
                 ):
        super(ScanNetCross, self).__init__(dataPathPrefix=dataPathPrefix, voxelSize=voxelSize,
                                           split=split, aug=aug, memCacheInit=memCacheInit,
                                           identifier=identifier, loop=loop,
                                           data_aug_color_trans_ratio=data_aug_color_trans_ratio,
                                           data_aug_color_jitter_std=data_aug_color_jitter_std,
                                           data_aug_hue_max=data_aug_hue_max,
                                           data_aug_saturation_max=data_aug_saturation_max,
                                           eval_all=eval_all)
        self.val_benchmark = val_benchmark
        self.n_views = n_views
        self.data_paths = sorted(glob(join(dataPathPrefix, split, '*.pth')))
        if self.val_benchmark:
            self.offset = 0
        # Prepare for 2D
        self.data2D_paths = []
        for x in tqdm(self.data_paths):
            # ps = glob(join(x[:-15].replace(split, '2D'), 'color', '*.jpg'))
            # ps = glob(join(x[:-4].replace(split, '2D'), 'pose', '*.txt'))
            ps = glob(join(x[:-4].replace(split, '2D'), 'coords', '*.tiff'))
            if len(ps) < self.n_views:
                print(f'{x} has only {len(ps)} frames, random appened')
                append_views = np.random.choice(
                    np.arange(len(ps)), self.n_views - len(ps), replace=True,
                )
                append_views = np.array(ps)[append_views].tolist()
                ps.extend(append_views)

            ps.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
            if val_benchmark:
                ps = ps[::5]
            self.data2D_paths.append(ps)

        self.remapper = np.ones(256) * 255
        for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
            self.remapper[x] = i


        transform = create_transform(
            input_size=256,
            is_training=True,
        )
        transform.transforms[0] = RandomResizedCropAndInterpolation(size=224, scale=(0.9, 1.5))

        t = []
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))

        self.transform_2d = transforms.Compose(t)


    def __getitem__(self, index_long):
        index = index_long % len(self.data_paths)
        locs_in = SA.attach("shm://wbhu_scannet_3d_%s_%06d_locs_%08d" % (self.split, self.identifier, index)).copy()
        feats_in = SA.attach("shm://wbhu_scannet_3d_%s_%06d_feats_%08d" % (self.split, self.identifier, index)).copy()
        labels_in = SA.attach("shm://wbhu_scannet_3d_%s_%06d_labels_%08d" % (self.split, self.identifier, index)).copy()
        segs_in = SA.attach("shm://wbhu_scannet_3d_%s_%06d_segs_%08d" % (self.split, self.identifier, index)).copy()

        locs = self.prevoxel_transforms(locs_in) if self.aug else locs_in
        locs, feats, labels, segs, inds_reconstruct = self.voxelizer.voxelize(locs, feats_in, labels_in, segs_in)

        colors, labels_2d, poses = self.get_2d(index)
        if self.eval_all:
            labels = labels_in
        if self.aug:
            locs, feats, labels, segs = self.input_transforms(locs, feats, labels, segs)

        coords = torch.from_numpy(locs).int()
        coords = torch.cat((torch.ones(coords.shape[0], 1, dtype=torch.int), coords), dim=1)
        feats = torch.from_numpy(feats).float() / 127.5 - 1.
        labels = torch.from_numpy(labels).long()
        segs = torch.from_numpy(segs).long()
    
        if self.eval_all:
            return coords, feats, labels, segs, colors, poses, torch.from_numpy(inds_reconstruct).long()
        return coords, feats, labels, segs, colors, poses, labels_2d


    def get_2d(self, room_id, coords=None):
        """
        :param      room_id:
        :param      coords: Nx3
        :return:    imgs:   CxHxWxV Tensor
                    labels: HxWxV Tensor
                    links: Nx4xV(1,H,W,mask) Tensor
        """
        frames_path = self.data2D_paths[room_id]
        partial = int(len(frames_path) / self.n_views)
        imgs, labels, world_coords = [], [], []

        for v in range(self.n_views):
            if not self.val_benchmark:
                f = random.sample(frames_path[v * partial:v * partial + partial], k=1)[0]
            else:
                select_id = (v * partial+self.offset) % len(frames_path)
                f = frames_path[select_id]

            # In ICCV version, we do not use 2D poses information, you can skip the loading of coords by modifing the dataloader and collate_fn
            img = PIL.Image.open(f.replace('coords', 'color').replace('.tiff', '.jpg')).convert("RGB")
            coords = tifffile.imread(f)
            label = imageio.imread(f.replace('coords', 'label').replace('.tiff', '.png'))
            label = self.remapper[label]
            onehot = np.zeros(20)
            for x in np.unique(label):
                if x != 255:
                    onehot[int(x)] = 1
            label_2d = torch.from_numpy(onehot).long()

            img = self.transform_2d(img)
            imgs.append(img)
            world_coords.append(np.mean(coords, axis=(0, 1)))
            labels.append(label_2d)

        imgs = torch.stack(imgs, dim=-1)
        labels = torch.stack(labels, dim=0)
        world_coords = torch.from_numpy(np.array(world_coords))

        return imgs, labels, world_coords



def collation_fn(batch):
    """
    :param batch:
    :return:    coords: N x 4 (batch,x,y,z)
                feats:  N x 3
                labels: N
                colors: B x C x H x W x V
                labels_2d:  B x H x W x V
                links:  N x 4 x V (B,H,W,mask)
    """
    coords, feats, labels, segs, colors, links, labels_2d = list(zip(*batch))
    # pdb.set_trace()

    for i in range(len(coords)):
        coords[i][:, 0] *= i

    return torch.cat(coords), torch.cat(feats), labels, segs, \
           torch.stack(colors), torch.stack(links), torch.stack(labels_2d)



def collation_fn_eval_all(batch):
    """
    :param batch:
    :return:    coords: N x 4 (x,y,z,batch)
                feats:  N x 3
                labels: N
                colors: B x C x H x W x V
                labels_2d:  B x H x W x V
                links:  N x 4 x V (B,H,W,mask)
                inds_recons:ON
    """
    coords, feats, labels, segs, colors, links, inds_recons = list(zip(*batch))
    inds_recons = list(inds_recons)
    # pdb.set_trace()

    accmulate_points_num = 0
    for i in range(len(coords)):
        coords[i][:, 0] *= i
        inds_recons[i] = accmulate_points_num + inds_recons[i]
        accmulate_points_num += coords[i].shape[0]

    return torch.cat(coords), torch.cat(feats), labels, segs, \
           torch.stack(colors), torch.stack(links), torch.cat(inds_recons)




if __name__ == '__main__':
    import time

    from tensorboardX import SummaryWriter

    data_root = '/research/dept6/wbhu/Dataset/ScanNet'
    train_data = ScanNetCross(dataPathPrefix=data_root, aug=True, split='train', memCacheInit=True, voxelSize=0.05)
    val_data = ScanNetCross(dataPathPrefix=data_root, aug=False, split='val', memCacheInit=True, voxelSize=0.05,
                            eval_all=True)
    coords, feats, labels, colors, labels_2d, links = train_data.__getitem__(0)
    print(coords.shape, feats.shape, labels.shape, colors.shape, labels_2d.shape, links.shape)
    coords, feats, labels, colors, labels_2d, links, inds_recons = val_data.__getitem__(0)
    print(coords.shape, feats.shape, labels.shape, colors.shape, labels_2d.shape, links.shape, inds_recons.shape)
    exit(0)

    manual_seed = 123


    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)


    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True, num_workers=2, pin_memory=True,
                                               worker_init_fn=worker_init_fn, collate_fn=collation_fn)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=8, shuffle=False, num_workers=2, pin_memory=True,
                                             worker_init_fn=worker_init_fn, collate_fn=collation_fn_eval_all)
    # _ = iter(train_loader).__next__()
    trainLog = SummaryWriter('Exp/scannet/statistic_cross/train')
    valLog = SummaryWriter('Exp/scannet/statistic_cross/val')

    for idx in range(1):
        end = time.time()
        for step, (coords, feats, labels, colors, labels_2d, links) in enumerate(train_loader):
            print(
                'time: {}/{}--{}'.format(step + 1, len(train_loader), time.time() - end))
            trainLog.add_histogram('voxel_coord_x', coords[:, 0], global_step=step)
            trainLog.add_histogram('voxel_coord_y', coords[:, 1], global_step=step)
            trainLog.add_histogram('voxel_coord_z', coords[:, 2], global_step=step)
            trainLog.add_histogram('color', feats, global_step=step)
            trainLog.add_histogram('2D_image', colors, global_step=step)
            # time.sleep(0.3)
            end = time.time()

        for step, (coords, feats, labels, colors, labels_2d, links, inds_reverse) in enumerate(val_loader):
            print(
                'time: {}/{}--{}'.format(step + 1, len(val_loader), time.time() - end))
            valLog.add_histogram('voxel_coord_x', coords[:, 0], global_step=step)
            valLog.add_histogram('voxel_coord_y', coords[:, 1], global_step=step)
            valLog.add_histogram('voxel_coord_z', coords[:, 2], global_step=step)
            valLog.add_histogram('color', feats, global_step=step)
            valLog.add_histogram('2D_image', colors, global_step=step)
            # time.sleep(0.3)
            end = time.time()

    trainLog.close()
    valLog.close()