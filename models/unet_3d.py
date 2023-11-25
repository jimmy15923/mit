# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import MinkowskiEngine as ME
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
from timm.models.layers import trunc_normal_, Mlp
from torch import nn
from torch.autograd import Variable

from models.avggroup import avg_grouping
from models.resnet_mink import ResNetBase
# from models.transformer import TransformerEncoder, TransformerEncoderLayer
from models.sem_transformer import TransformerEncoder
from models.utils import MLP, min_max_norm


def soft_crossentropy(input, target):
    logprobs = torch.nn.functional.log_softmax(input, dim=1)
    return  -(target * logprobs).sum() / input.shape[0]


def c2_xavier_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.
    Args:
        module (torch.nn.Module): module to initialize.
    """
    # Caffe2 implementation of XavierFill in fact
    # corresponds to kaiming_uniform_ in PyTorch
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        # pyre-fixme[6]: Expected `Tensor` for 1st param but got `Union[nn.Module,
        #  torch.Tensor]`.
        nn.init.constant_(module.bias, 0)


class MinkUNetBase(ResNetBase):
    BLOCK = None
    PLANES = None
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, in_channels, out_channels, cfg, D, is_cls, *args, **kwargs):
        ResNetBase.__init__(self, in_channels, out_channels, cfg, D, is_cls)
        self.is_extract_feature = is_cls
        self.is_cls = is_cls
        self.cfg = cfg

    def network_initialization(self, in_channels, out_channels, D=3):
        # Output of the first conv concated to conv6
        self.num_classes = self.cfg.classes
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, dimension=D)

        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)
        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0],
                                       self.LAYERS[0])

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1],
                                       self.LAYERS[1])

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)

        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2],
                                       self.LAYERS[2])

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3],
                                       self.LAYERS[3])

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D)
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4],
                                       self.LAYERS[4])
        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D)
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5],
                                       self.LAYERS[5])
        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D)
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6],
                                       self.LAYERS[6])
        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D)
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7],
                                       self.LAYERS[7])

        # self.final = ME.MinkowskiConvolution(
        #     self.PLANES[7],
        #     self.embed_dim,
        #     kernel_size=1,
        #     bias=True,
        #     dimension=D)

        # if self.is_cls:
            # if self.is_attn:
        if not self.is_cls:
            if self.cfg.pos_emb:
                self.coord_emb = Mlp(3, self.PLANES[7], self.PLANES[7])
                self.cls_pos_emb =  nn.Parameter(torch.zeros(self.num_classes, self.PLANES[7]))
                trunc_normal_(self.cls_pos_emb, std=0.02)

            if self.cfg.encoder:
                self.cls_token = nn.Parameter(torch.zeros(self.num_classes, self.PLANES[7]))
                trunc_normal_(self.cls_token, std=0.02)

                self.encoder = TransformerEncoder(
                    embed_dim=self.PLANES[7],
                    depth=self.cfg.depth,
                    num_heads=self.cfg.num_heads,
                    mlp_ratio=self.cfg.mlp_ratio,
                    drop_rate=self.cfg.drop_rate,
                    attn_drop_rate=self.cfg.attn_drop_rate,
                    drop_path_rate=self.cfg.drop_path_rate,
                )
            self.head = nn.Conv1d(self.PLANES[7], self.num_classes, kernel_size=1)
        self.relu = ME.MinkowskiReLU(inplace=True)
        # self.pool = ME.MinkowskiAvgPooling(kernel_size=3, stride=3, dimension=D)

    def forward(self, x, overseg=None, debug=False):
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)

        # tensor_stride=8
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)

        out = ME.cat(out, out_b3p8)
        out = self.block5(out)

        # tensor_stride=4
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        out = ME.cat(out, out_b2p4)
        out = self.block6(out)

        # tensor_stride=2
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        out = ME.cat(out, out_b1p2)
        out = self.block7(out)

        # tensor_stride=1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)
        out = ME.cat(out, out_p1)
        point_tensor = self.block8(out)
        # out = self.final(out)

        if self.is_cls:
            return point_tensor

        cls_logits = []
        cls_mct_logits = []
        cams = []
        consistent_loss = []
        cls_attn_maps = []
        for batch_idx in range(len(point_tensor.C[:, 0].unique())):
            mask_i = (point_tensor.C[:, 0] == batch_idx)
            point_feat = point_tensor.F[mask_i]
            point_coord = point_tensor.C[mask_i][:, 1:].float()

            if self.cfg.overseg_pool:
                # overseg_idx = rearange_consecutive_label(overseg[batch_idx])
                overseg_idx = overseg[batch_idx]
                if self.cfg.overseg_maxpool:
                    pooled_feat = max_grouping(point_feat.contiguous(), overseg_idx)
                    pooled_coord = max_grouping(point_coord.contiguous(), overseg_idx) 
                else:
                    # pooled_feat = avg_grouping(point_feat.contiguous(), overseg_idx)
                    # pooled_coord = avg_grouping(point_coord.contiguous(), overseg_idx)
                    pooled_feat = avg_grouping(point_feat.contiguous(), overseg_idx)
                    pooled_coord = avg_grouping(point_coord.contiguous(), overseg_idx)
                    pooled_feat = torch.where(torch.isnan(pooled_feat), torch.zeros_like(pooled_feat), pooled_feat)
                    pooled_coord = torch.where(torch.isnan(pooled_coord), torch.zeros_like(pooled_coord), pooled_coord)

                if self.cfg.feat_smooth:
                    mean_target = pooled_feat[overseg_idx]
                    consistent_loss.append(F.mse_loss(point_feat, mean_target))

                last_feature = pooled_feat.unsqueeze(0).permute(0, 2, 1)
            else:
                last_feature = point_feat.unsqueeze(0).permute(0, 2, 1)

            cam = self.head(last_feature)  # (1, C, N)
            cam = cam.softmax(1)

            if self.cfg.pool == 'avg':
                cls_logits.append(torch.mean(cam, dim=2))
            else:
                cls_logits.append(F.adaptive_max_pool1d(cam, output_size=(1)).squeeze(-1))  

            if self.cfg.encoder:
                voxel_token = torch.cat((self.cls_token, pooled_feat))

            if self.cfg.pos_emb:
                pos = self.coord_emb(pooled_coord)  # NxD
                pos = torch.cat((self.cls_pos_emb, pos))
                voxel_token = voxel_token + pos
            
            if self.cfg.encoder:        
                # voxel_tokens.append(voxel_token)
                voxel_token, attn_weights = self.encoder(voxel_token.unsqueeze(0))
                cls_attn_maps.append(attn_weights)

                cls_tokens = voxel_token[0, :self.num_classes] # 20 x D
                cls_mct_logits.append(cls_tokens.mean(-1).unsqueeze(0))
                # if self.cfg.cls_on_attn:
                #     cam = self.head(voxel_token[0, 20:].unsqueeze(0).permute(0, 2, 1))
                #     if self.cfg.pool == 'avg':
                #         cls_logits.append(torch.mean(cam, dim=2))
                #     else:
                #         cls_logits.append(F.adaptive_max_pool1d(cam, output_size=(1)).squeeze(-1))  

            # if self.cfg.encoder:
            #     attn_weights = torch.stack(attn_weights)  # 12 * B * H * N * N
            #     attn_weights = torch.mean(attn_weights, dim=2)
            #     mtatt = attn_weights[-3:].sum(0)[:, 0:20, 20:]  # (1, C, N)
            #     mct_cam = mtatt * F.relu(cam)

            if self.cfg.overseg_pool and self.cfg.encoder:
                cams.append(cam[0].permute(1, 0))
            elif self.cfg.overseg_pool:
                cams.append(cam[0].permute(1, 0))
            else:
                cams.append(cam[0].permute(1, 0))

        if len(cls_mct_logits):
            cls_mct_logits = torch.cat(cls_mct_logits)
        
        if len(consistent_loss):
            consistent_loss = torch.mean(torch.stack(consistent_loss)) 
        else:
            consistent_loss = torch.zeros(1)[0].cuda()

        output = {
            'cls_logits': torch.cat(cls_logits),
            'cls_mct_logits': cls_mct_logits,
            'smooth_loss': consistent_loss,
            'attn_maps': cls_attn_maps,
            'cams': cams,
        }
        return output


class MinkUNet14(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)


class MinkUNet18(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)


class MinkUNet34(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class MinkUNet50(MinkUNetBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class MinkUNet101(MinkUNetBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 23, 2, 2, 2, 2)


class MinkResNet18(MinkUNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class MinkResNet34(MinkUNet34):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class MinkResNet50(MinkUNet50):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class MinkUNet14A(MinkUNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class MinkUNet14B(MinkUNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class MinkUNet14C(MinkUNet14):
    PLANES = (32, 64, 128, 256, 192, 192, 128, 128)


class MinkUNet14D(MinkUNet14):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class MinkUNet18A(MinkUNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class MinkUNet18B(MinkUNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class MinkUNet18C(MinkUNet18):
    PLANES = (32, 64, 128, 256, 256, 256, 256, 256)


class MinkUNet18D(MinkUNet18):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class MinkUNet34A(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 64)


class MinkUNet34B(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 32)


class MinkUNet34C(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)


class MinkUNet34D(MinkUNet34):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


def mink_unet(in_channels=3, out_channels=20, cfg=None, D=3, is_cls=False, arch='MinkUNet18A'):
    if arch == 'MinkUNet18A':
        return MinkUNet18A(in_channels, out_channels, cfg, D, is_cls, arch)
    elif arch == 'MinkUNet18B':
        return MinkUNet18B(in_channels, out_channels, D, is_cls)
    elif arch == 'MinkUNet50':
        return MinkUNet50(in_channels, out_channels, D, is_cls)
    elif arch == 'MinkUNet18D':
        return MinkUNet18D(in_channels, out_channels, D, is_cls)
    elif arch == 'MinkUNet34A':
        return MinkUNet34A(in_channels, out_channels, D, is_cls)
    elif arch == 'MinkUNet34B':
        return MinkUNet34B(in_channels, out_channels, D, is_cls)
    elif arch == 'MinkUNet34C':
        return MinkUNet34C(in_channels, out_channels, cfg, D, is_cls, arch)
    elif arch == 'MinkUNet34D':
        return MinkUNet34D(in_channels, out_channels, D, is_cls)        
    elif arch == 'MinkUNet14A':
        return MinkUNet14A(in_channels, out_channels, D, is_cls)
    elif arch == 'MinkUNet14B':
        return MinkUNet14B(in_channels, out_channels, D, is_cls)
    elif arch == 'MinkUNet14C':
        return MinkUNet14C(in_channels, out_channels, D, is_cls)
    elif arch == 'MinkUNet14D':
        return MinkUNet14D(in_channels, out_channels, D, is_cls)
    else:
        raise Exception('architecture not supported yet'.format(arch))
