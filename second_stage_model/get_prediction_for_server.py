import cv2
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
import MinkowskiEngine as ME
from MinkowskiEngine import CoordsManager, SparseTensor
from dataset.scanNet3D import ScanNet3D, collation_fn_eval_all
from sklearn.metrics import accuracy_score

from pathlib import Path
val_data = ScanNet3D(dataPathPrefix='/home/jimmy15923/mnt/data/scannet/scannet_cross/', voxelSize=0.02, split='test', aug=False,
                     memCacheInit=True, eval_all=True, identifier=6797)
val_sampler = None
val_loader = torch.utils.data.DataLoader(val_data, batch_size=1,
                                         shuffle=False, num_workers=1, pin_memory=True,
                                         drop_last=False, collate_fn=collation_fn_eval_all,
                                         sampler=val_sampler)

from models.unet_3d import MinkUNet18A as Model
model = Model(in_channels=3, out_channels=20, D=3)

model = torch.nn.parallel.DataParallel(model)
model.cuda()

checkpoint = torch.load('./Exp/scannet/ours_2cm_34c_trainval/model/model_best.pth.tar', 
                        map_location=lambda storage, loc: storage.cuda())
model.load_state_dict(checkpoint['state_dict'], strict=True)

index_mapping = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])

from tqdm import tqdm
for i, batch_data in enumerate(tqdm(val_loader)):
    if i < 68:
        continue
    scene = Path(val_loader.dataset.data_paths[i]).stem
    scene = scene.replace('_vh_clean_2', '')
    score = 0
    for _ in range(7):
        (coords, feat, label, inds_reverse) = batch_data
        coords[:,1:] = coords[:,1:] + np.random.rand(3) * 100
        sinput = SparseTensor(feat, coords, device='cuda')
        predictions = model(sinput)
        predictions_enlarge = predictions[inds_reverse, :]
        score = score + predictions_enlarge.detach().cpu()
    seg = torch.argmax(score, 1).numpy()
    pred40 = index_mapping[seg]
    pred40 = pred40.astype(np.int64)
    np.savetxt((f'./Exp/scannet/ours_2cm_34c_trainval/result/best/{scene}.txt'), pred40, '%s', encoding='utf-8')
    torch.cuda.empty_cache()