import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from .unet import MYASPPHead
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class OccConvDecoder(BaseModule):
    def __init__(self, 
                 embed_dims=256, 
                 conv_num=3, 
                 pillar_h=16, 
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN',),
                 act_cfg=dict(type='ReLU',inplace=True),):
        super(OccConvDecoder, self).__init__()
        self.embed_dims = embed_dims
        self.conv_num = conv_num
        self.pillar_h = pillar_h
        self.use_bias = norm_cfg is None

        # Define convolutional layers
        self.conv_layers = nn.ModuleList()
        for _ in range(conv_num):
            conv_layer = ConvModule(
                self.embed_dims,
                self.embed_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=self.use_bias,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
            self.conv_layers.append(conv_layer)

        # ASPP module
        self.aspp_head = MYASPPHead(
            is_volume=False,
            in_channels=self.embed_dims,
            in_index=3,
            channels=self.embed_dims,
            dilations=(1, 3, 6, 9),
            dropout_ratio=0.1,
            num_classes=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            align_corners=False,
            # loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        )

        # Deconvolution to original shape
        _out_dim = self.embed_dims * self.pillar_h
        self.deconv = ConvModule(
            self.embed_dims,
            _out_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=self.use_bias,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

    def forward(self, x):
        # Forward pass through convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x) # 256 -> 256

        # Forward pass through ASPP module
        x = self.aspp_head(x) # 256 -> 256

        # Forward pass through deconvolutional layer
        x = self.deconv(x) # 256 -> 256 * pillar_h

        return x
