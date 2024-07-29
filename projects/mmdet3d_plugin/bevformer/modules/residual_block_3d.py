import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

class ResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 conv_cfg=dict(type='Conv3d'), 
                 norm_cfg=dict(type='BN3d'), 
                 act_cfg=dict(type='ReLU',inplace=True)):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvModule(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1,
            conv_cfg=conv_cfg, 
            norm_cfg=norm_cfg, 
            act_cfg=act_cfg,
        )
        self.conv2 = ConvModule(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1,
            conv_cfg=conv_cfg, 
            norm_cfg=norm_cfg, 
            act_cfg=None,
        )
        self.downsample = ConvModule(
            in_channels, 
            out_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0,
            conv_cfg=conv_cfg, 
            norm_cfg=norm_cfg, 
            act_cfg=None,
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.downsample(x)
        out = F.relu(out)
        return out
    
