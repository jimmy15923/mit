#!/usr/bin/env python
from collections import OrderedDict

import MinkowskiEngine as ME
import numpy as np
import torch
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from torch import nn

from models.avggroup import avg_grouping
from models.sem_transformer import (
    InterlanceDecoder,
    TransformerEncoder,
)

from models.unet_2d import ResNet as model2D
from models.unet_3d import mink_unet as model3D
from models.utils import MLP, min_max_norm


def state_dict_remove_moudle(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:]  # remove 'module.' of dataparallel
        name = k.replace("module.", "")
        new_state_dict[name] = v
    return new_state_dict


def constructor3d(*args, **kwargs):
    model = model3D(*args, **kwargs)
    # model = model.cuda()
    return model


def constructor2d(**kwargs):
    model = model2D(**kwargs)
    # model = model.cuda()
    return model


class MIT(nn.Module):
    def __init__(self, cfg=None):
        super(MIT, self).__init__()
        self.cfg = cfg
        self.viewNum = cfg.viewNum
        self.voxel_size = cfg.voxelSize
        # 2D
        self.net2d = constructor2d(cfg=cfg, layers=cfg.layers_2d, classes=cfg.classes)

        # 3D
        self.net3d = constructor3d(
            in_channels=3,
            out_channels=cfg.classes,
            cfg=cfg,
            D=3,
            is_cls=True,
            arch=cfg.arch_3d,
        )
        self.embed_dim = self.net3d.PLANES[7]
        self.num_classes = 20

        self.voxel_encoder = TransformerEncoder(
            embed_dim=cfg.embed_dim,
            depth=self.cfg.depth,
            num_heads=self.cfg.num_heads,
            mlp_ratio=self.cfg.mlp_ratio,
            drop_rate=self.cfg.drop_rate,
            attn_drop_rate=self.cfg.attn_drop_rate,
            drop_path_rate=self.cfg.drop_path_rate,
        )

        self.view_encoder = TransformerEncoder(
            embed_dim=cfg.embed_dim,
            depth=self.cfg.depth,
            num_heads=self.cfg.num_heads,
            mlp_ratio=self.cfg.mlp_ratio,
            drop_rate=self.cfg.drop_rate,
            attn_drop_rate=self.cfg.attn_drop_rate,
            drop_path_rate=self.cfg.drop_path_rate,
        )

        self.interlance_decoder = InterlanceDecoder(
            embed_dim=cfg.embed_dim,
            depth=4,
            num_classes=cfg.classes,
            num_heads=4,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
        )

        self.pos_emb = MLP(3, self.embed_dim, self.embed_dim, num_layers=3)
        self.cls_token = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.cls_pos_emb = nn.Parameter(
            torch.zeros(1, self.num_classes, self.embed_dim)
        )
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.cls_pos_emb, std=0.02)
        self.cls_head_2d = nn.Conv1d(self.embed_dim, cfg.classes, kernel_size=1)
        self.cls_head_3d = nn.Conv1d(self.embed_dim, cfg.classes, kernel_size=1)
        #         if cfg.learn_view_emb:
        self.view_pos_emb = nn.Parameter(torch.zeros(1, self.viewNum, self.embed_dim))


    def forward(self, sinput, supervoxel, images, poses):
        """
        images:BCHWV
        """
        # 2D feature extract
        batch, channel, height, width, n_view = images.size()
        poses = poses.permute(0, 2, 1).contiguous()  # -> BVC
        poses = poses.view(batch * n_view, 3) / self.voxel_size

        data_2d = images.permute(0, 4, 1, 2, 3).contiguous()  # -> BVCHW
        data_2d = data_2d.view(batch * n_view, *data_2d.shape[2:])

        # Extract features and concat with multi-class tokens
        image_tokens = self.net2d(data_2d)
        image_view_tokens = image_tokens.view(batch, n_view, -1)  # (B, N ,C)
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        image_tokens = torch.cat((cls_tokens, image_view_tokens), dim=1)

        # Get coordinates embedding
        image_pos_emb = self.view_pos_emb.expand(batch, -1, -1)
        cls_pos_emb = self.cls_pos_emb.expand(batch, -1, -1)
        image_pos_emb = torch.cat((cls_pos_emb, image_pos_emb), dim=1)

        # Pass views token into encoder
        attn_view_token, view_attn_map = self.view_encoder(
            image_tokens + image_pos_emb
        )  # (B, N ,C)
        view_cls_token = attn_view_token[:, :self.num_classes]  # (B, 20, F)
        mct_logits_2d = view_cls_token.mean(-1)

        # 3D feature extract
        point_tensor = self.net3d(sinput)

        cam_logits_2d = []
        cam_logits_3d = []
        mct_logits_3d = []
        cross_mct_logits = []
        cams = []
        cls_attn_maps = []
        consistent_loss = []
        token_sim_loss = []
        for batch_idx in range(len(point_tensor.C[:, 0].unique())):
            mask_i = point_tensor.C[:, 0] == batch_idx
            point_feat = point_tensor.F[mask_i]
            point_coord = point_tensor.C[mask_i][:, 1:].float()

            # supervoxel pooling
            pooled_feat = avg_grouping(point_feat.contiguous(), supervoxel[batch_idx])
            pooled_coord = avg_grouping(point_coord.contiguous(), supervoxel[batch_idx])
            pooled_coord = torch.where(
                torch.isnan(pooled_coord), torch.zeros_like(pooled_coord), pooled_coord
            )
            pooled_feat = torch.where(
                torch.isnan(pooled_feat), torch.zeros_like(pooled_feat), pooled_feat
            )

            # smooth regularization
            mean_target = pooled_feat[supervoxel[batch_idx]]
            consistent_loss.append(F.mse_loss(point_feat, mean_target))

            # 3D coordinate embedding
            pos = self.pos_emb(pooled_coord)  # NxD
            pos = torch.cat((self.cls_pos_emb[0], pos))

            # Concat with multi-class tokens
            voxel_token = torch.cat((self.cls_token[0], pooled_feat))
            voxel_token = (voxel_token + pos).unsqueeze(0)

            # Pass to voxel encoder
            voxel_token, voxel_attn = self.voxel_encoder(voxel_token)
            # cls_attn_maps.append(voxel_attn)

            # Get the self attention maps
            voxel_attn = torch.stack(voxel_attn)  # 12 * B * H * N * N
            voxel_attn = torch.mean(voxel_attn, dim=2)
            mtatt = voxel_attn[-3:].sum(0)[:, 0:self.num_classes, self.num_classes:]

            # Get the attned multi-class token for classification loss
            cls_tokens = voxel_token[0, :self.num_classes]  # 20 x D
            mct_logits_3d.append(cls_tokens.mean(-1).unsqueeze(0))

            # Pass to interlance decoder
            view_attn_token, voxel_attn_token, cross_attn = self.interlance_decoder(
                attn_view_token[batch_idx].unsqueeze(0),
                voxel_token,
            )  # 1xVxC
            cls_attn_maps.append(cross_attn)

            # Get the affinity map across layers, and calculate the N-pair loss
            if self.cfg.is_pair_loss:
                for attention_map in cross_attn:
                    attention_amp = torch.mean(attention_map, dim=1)  # Mean over head
                    sim_matrix = attention_amp[
                        0, :self.num_classes, :self.num_classes
                    ]  # Get the cls token affinity
                    labels = torch.arange(self.num_classes, device="cuda")
                    view_loss = F.cross_entropy(sim_matrix, labels)
                    voxel_loss = F.cross_entropy(sim_matrix.T, labels)
                    token_sim_loss.append((view_loss + voxel_loss) / 2.0)

            # Get the cross-attend multi-class token for classification loss
            cross_cls_token1 = voxel_attn_token[0, :self.num_classes]  # 20 x D
            cross_cls_token2 = view_attn_token[0, :self.num_classes]
            cross_cls_token = (cross_cls_token1 + cross_cls_token2) / 2
            cross_mct_logits.append(cross_cls_token.mean(-1).unsqueeze(0))

            if self.cfg.cam_on_attn:
                voxel_cam = self.cls_head_3d(
                    F.relu(voxel_token.permute(0, 2, 1)[0, :, self.num_classes:][None])
                )
            else:
                voxel_cam = self.cls_head_3d(pooled_feat.unsqueeze(0).permute(0, 2, 1))
                voxel_cam = voxel_cam.softmax(1)

            view_cam = self.cls_head_2d(
                image_view_tokens[batch_idx].unsqueeze(0).permute(0, 2, 1)
            )
            if self.cfg.pool == "avg":
                cam_logits_3d.append(torch.mean(voxel_cam, dim=2))
                cam_logits_2d.append(torch.mean(view_cam, dim=2))
            elif self.cfg.pool == "mix":
                x1 = F.adaptive_max_pool1d(voxel_cam, 1).squeeze(-1)
                x2 = F.adaptive_avg_pool1d(voxel_cam, 1).squeeze(-1)
                cam_logits_3d.append(torch.maximum(x1, x2))

                x1 = F.adaptive_max_pool1d(view_cam, 1).squeeze(-1)
                x2 = F.adaptive_avg_pool1d(view_cam, 1).squeeze(-1)
                cam_logits_2d.append(torch.maximum(x1, x2))
            else:
                cam_logits_3d.append(
                    F.adaptive_max_pool1d(voxel_cam, output_size=(1)).squeeze(-1)
                )
                cam_logits_2d.append(
                    F.adaptive_max_pool1d(view_cam, output_size=(1)).squeeze(-1)
                )

            # mct_cam = mtatt * F.relu(voxel_cam)
            # cams.append(torch.maximum(mct_cam, voxel_cam))

            # # sometimes result from 3D encoder is better
            cams.append(voxel_cam[0].permute(1, 0))

        output = {
            "cross_mct_logits": torch.cat(cross_mct_logits),
            "mct_logits_2d": mct_logits_2d,
            "mct_logits_3d": torch.cat(mct_logits_3d),
            "cam_logits_2d": torch.cat(cam_logits_2d),
            "cam_logits_3d": torch.cat(cam_logits_3d),
            "consistent_loss": torch.mean(torch.stack(consistent_loss)),
            "token_sim_loss": torch.mean(torch.stack(token_sim_loss)),
            "attn_maps": cls_attn_maps,
            "view_cams": view_cam,
            "cams": cams,
        }

        return output
