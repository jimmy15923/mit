import torch
import torch.nn.functional as F
from torch import nn

from torchvision import models

from models.sem_transformer import TransformerEncoder
from models.utils import MLP
import timm


class ResNet(nn.Module):
    def __init__(
        self,
        cfg,
        layers=18,
        classes=2,
        BatchNorm=nn.BatchNorm2d,
    ):
        super(ResNet, self).__init__()

        if layers == 18:
            self.resnet = models.resnet18(pretrained=cfg.pretrained)
        elif layers == 34:
            self.resnet = models.resnet34(pretrained=cfg.pretrained)
        elif layers == 50:
            self.resnet = timm.create_model(
                "resnetv2_50", pretrained=cfg.pretrained, num_classes=0
            )
        elif layers == 101:
            self.resnet = timm.create_model(
                "resnetv2_101", pretrained=cfg.pretrained, num_classes=0
            )

        # Parameters of newly constructed modules have requires_grad=True by default
        self.fc = nn.Linear(2048, cfg.embed_dim)

    def forward(self, x):
        return self.fc(self.resnet(x))


class MctResNet(ResNet):
    def __init__(self, cfg, layers):
        ResNet.__init__(self, cfg, layers)
        self.is_attn = cfg.use_attn
        if cfg.use_attn:
            self.encoder = TransformerEncoder(
                embed_dim=cfg.embed_dim,
                depth=3,
                num_heads=4,
                mlp_ratio=1,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.1,
            )
            self.pos_emb_3d = MLP(3, cfg.embed_dim, cfg.embed_dim, num_layers=2)
            self.cls_token = nn.Parameter(torch.zeros(1, 20, cfg.embed_dim))
            self.cls_pos_emb = nn.Parameter(torch.zeros(1, 20, cfg.embed_dim))

        self.head = nn.Conv1d(cfg.embed_dim, 20, kernel_size=1)


    def forward(self, images, poses):
        batch, channel, height, width, n_view = images.size()

        data_2d = images.permute(0, 4, 1, 2, 3).contiguous()  # -> BVCHW
        data_2d = data_2d.view(batch * n_view, *data_2d.shape[2:])

        image_tokens = self.fc(self.resnet(data_2d))
        raw_image_tokens = image_tokens.view(batch, n_view, -1)  # (B, N ,C)
        cls_attn_maps = []
        if self.is_attn:
            cls_tokens = self.cls_token.expand(batch, -1, -1)
            poses = poses.permute(0, 2, 1).contiguous()  # -> BVC
            poses = poses.view(batch * n_view, 3)
            pos_emb = self.pos_emb_3d(poses.float())

            pos_emb = pos_emb.view(batch, n_view, -1)

            cls_pos_emb = self.cls_pos_emb.expand(batch, -1, -1)
            pos_emb = torch.cat((cls_pos_emb, pos_emb), dim=1)

            image_tokens = torch.cat(
                (cls_tokens, raw_image_tokens), dim=1
            )  # (B, N+20 ,C)
            image_tokens, attn_weights = self.encoder(image_tokens + pos_emb)
            cls_tokens = image_tokens[:, :20, :]
            scene_mct_logits = cls_tokens.mean(-1)
            cls_attn_maps.append(attn_weights)

            cams = self.head(image_tokens.permute(0, 2, 1)[:, :, 20:])
        else:
            scene_mct_logits = None
            cams = self.head(raw_image_tokens.permute(0, 2, 1))

        scene_logits = torch.mean(cams, 2)

        output = {
            "scene_logits": scene_logits,
            "scene_mct_logits": scene_mct_logits,
            "cams": cams,
        }

        return output
