import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from .unet import MYASPPHead
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class OccConvDecoder3D(BaseModule):
    def __init__(self, 
                 embed_dims,
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg_3d=None, 
                 act_cfg_3d=None):
        super(OccConvDecoder3D, self).__init__()
        self.embed_dims = embed_dims
        use_bias_3d = norm_cfg_3d is None
        self.middle_dims = 32  # decrease memory cost

        self.conv1 = ConvModule(
            self.embed_dims,
            self.middle_dims,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=use_bias_3d,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg_3d,
            act_cfg=act_cfg_3d
        )

        self.aspp = MYASPPHead(
            in_channels=self.middle_dims,
            in_index=3,
            channels=self.middle_dims,
            dilations=(1, 3, 6, 9),
            dropout_ratio=0.1,
            num_classes=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg_3d,
            align_corners=False,
            # loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        )

        self.conv2 = ConvModule(
            self.middle_dims,
            embed_dims,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=use_bias_3d,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg_3d,
            act_cfg=act_cfg_3d
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.aspp(x)
        x = self.conv2(x)
        return x
