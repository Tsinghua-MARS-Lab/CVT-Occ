import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

class TemporalNet(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 conv_cfg=dict(type='Conv3d'), 
                 norm_cfg=dict(type='BN3d'), 
                 act_cfg=dict(type='ReLU',inplace=True)):
        super(TemporalNet, self).__init__()
        self.conv1 = ConvModule(
            in_channels, 
            out_channels, 
            kernel_size=1,
            stride=1, 
            padding=0,
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
            act_cfg=act_cfg,
        )
        self.conv3 = ConvModule(
            out_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
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
        out = self.conv3(out)
        out += self.downsample(x)
        out = F.relu(out)
        return out

class ConcatConvModule(BaseModule):
    def __init__(self, 
                 bev_h=200, 
                 bev_w=200, 
                 total_z=16, # not bev_z
                 channels=16, 
                 pc_range=[-40, -40, -1.0, 40, 40, 5.4],
                 voxel_size=[0.4, 0.4, 0.4],
                 sampled_queue_length=7,
                 ):
        super(ConcatConvModule, self).__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.total_z = total_z
        self.channels = channels
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.sampled_queue_length = sampled_queue_length
        self.concatChannels=self.channels * self.sampled_queue_length
        self.resnet = TemporalNet(in_channels=self.concatChannels,
                             out_channels=self.channels,
                             conv_cfg=dict(type='Conv3d'),
                             norm_cfg=dict(type='BN3d', ),
                             act_cfg=dict(type='ReLU',inplace=True),
                            )
        bias_x = (pc_range[3] + pc_range[0]) / 2 # 0
        bias_y = (pc_range[4] + pc_range[1]) / 2 # 0
        bias_z = (pc_range[5] + pc_range[2]) / 2 # 2.2
        half_x = (pc_range[3] - pc_range[0]) / 2 # 40
        half_y = (pc_range[4] - pc_range[1]) / 2 # 40
        half_z = (pc_range[5] - pc_range[2]) / 2 # 3.2
        self.bias = torch.tensor([bias_x, bias_y, bias_z], device='cuda') # [0, 0, 2.2]
        self.norm_factors = torch.tensor([half_x, half_y, half_z], device='cuda') # [40, 40, 3.2]

        self.x_idx_begins = self.pc_range[0] + self.voxel_size[0]/2
        self.y_idx_begins = self.pc_range[1] + self.voxel_size[1]/2
        self.z_idx_begins = self.pc_range[2] + self.voxel_size[2]/2

        self.x_idx_ends = self.pc_range[3] - self.voxel_size[0]/2
        self.y_idx_ends = self.pc_range[4] - self.voxel_size[1]/2
        self.z_idx_ends = self.pc_range[5] - self.voxel_size[2]/2

    def get_bev_ref(self, W, H, Z):
        """
        Get reference point for reshaped BEV volume. 
        Args: 
            W (int): width of the BEV volume.
            H (int): height of the BEV volume.
            Z (int): depth of the BEV volume.
        Returns: 
            ref_3d (torch.Tensor): shape (w, h, z, 4, 1)
        """

        x_idx = torch.linspace(self.x_idx_begins, self.x_idx_ends, W, device='cuda', dtype=torch.float64) 
        y_idx = torch.linspace(self.y_idx_begins, self.y_idx_ends, H, device='cuda', dtype=torch.float64) 
        z_idx = torch.linspace(self.z_idx_begins, self.z_idx_ends, Z, device='cuda', dtype=torch.float64) 

        grid_x_idx, grid_y_idx, grid_z_idx = torch.meshgrid(x_idx, y_idx, z_idx, indexing='ij')
        grid_ones = torch.ones_like(grid_x_idx) 
        ref_3d = torch.stack((grid_x_idx, grid_y_idx, grid_z_idx, grid_ones), -1)
        ref_3d = ref_3d.unsqueeze(4)
        return ref_3d
    
    def gather_feature(self, features, locations, device='cuda', dtype=torch.float32):
        """
        Args: 
            features (torch.Tensor): (len-1, c, z, h, w)
            locations (torch.Tensor): (len-1, w, h, z, 3)
        returns:
            grid_sampled_features (torch.Tensor): (len-1, c, z, h, w) no mask
        """
        features = features.to(dtype)
        locations = locations.to(dtype)

        # norm the location and reshape locations to (len-1, z, h, w, 3)
        locations = (locations - self.bias) / self.norm_factors # norm the location to [-1, 1]
        locations = locations.permute(0, 3, 2, 1, 4) # (len-1, z, h, w, 3)

        grid_sampled_features = F.grid_sample(features, locations, align_corners=False) # (len-1, c, z, h, w)
        # default to be bilinear interpolation and no align corners

        return grid_sampled_features # (len-1, c, z, h, w)

    def forward(self, bev_list, img_metas_list):
        """
        Args: 
            bev_list (list[torch.Tensor]): each has shape (bs, h*w, embed_dims).
            img_metas_list (list[dict]): Include current img meta info dict.
        return:
            updated_bev (torch.Tensor): fused BEV feature with shape (bs, h*w, embed_dims)
        """

        # Step 1: prepare the input parameters
        bs, bev_HW, embed_dims = bev_list[0].shape
        bev_h = self.bev_h
        bev_w = self.bev_w
        total_z = self.total_z
        channels = self.channels
        pc_range = self.pc_range

        # get ego2global from img_metas
        ego2global_list = []
        for i in range(len(bev_list)):
            img_metas = img_metas_list[i]
            ego2global = img_metas['ego2global'] # (4, 4) numpy array
            ego2global_list.append(torch.tensor(ego2global, device='cuda', dtype=torch.float64))
        cat_prev_ego2global = torch.stack(ego2global_list[:-1], dim=0) # (len-1, 4, 4)
        assert bs == 1, "Only support batch size 1"
        cat_prev_ego2global = cat_prev_ego2global.unsqueeze(0).repeat(bs, 1, 1, 1) # (bs, len-1, 4, 4)
        cur_ego2global = ego2global_list[-1] # (4, 4)
        cur_ego2global = cur_ego2global.unsqueeze(0).repeat(bs, 1, 1) # (bs, 4, 4)

        # Reshape the input
        cat_bev = torch.cat(bev_list, dim=0)
        cat_bev = cat_bev.permute(0, 2, 1).reshape(bs*len(bev_list), channels, total_z, bev_h, bev_w)

        # Step 2: prepare reference point for BEV volume
        # sample reference point for current ego coordinate
        bev_ref = self.get_bev_ref(H=bev_h, W=bev_w, Z=total_z)
        bev_ref = bev_ref.unsqueeze(3).repeat(1, 1, 1, len(bev_list)-1, 1, 1) # (w, h, z, len-1, 4, 1)

        # Step 3: use `ego2global` to wrap the reference point to the previous BEV volume
        prev_bev_ref = torch.inverse(cat_prev_ego2global) @ cur_ego2global @ bev_ref
        prev_bev_ref = prev_bev_ref.squeeze(-1)
        prev_bev_ref = prev_bev_ref[..., :3]
        prev_bev_ref = prev_bev_ref.permute(3, 0, 1, 2, 4) # (len-1, w, h, z, 3)

        # Step 4: get mask
        mask = (
            (prev_bev_ref[..., 0] > pc_range[0]) | (prev_bev_ref[..., 0] < pc_range[3]) |
            (prev_bev_ref[..., 1] > pc_range[1]) | (prev_bev_ref[..., 1] < pc_range[4]) |
            (prev_bev_ref[..., 2] > pc_range[2]) | (prev_bev_ref[..., 2] < pc_range[5])
        ) # out-of-range points will be 0. with shape (len-1, w, h, z)

        # Step 5: gather_feature
        prev_bev = self.gather_feature(cat_bev[:-1], prev_bev_ref) # (len-1, c, z, h, w)
        prev_bev = prev_bev.permute(0, 4, 3, 2, 1) # (len-1, w, h, z, c)
        prev_bev[~mask] = 0
        prev_bev = prev_bev.permute(0, 4, 3, 2, 1) # (len-1, c, z, h, w)
        prev_bev = torch.flatten(prev_bev, start_dim=0, end_dim=1) # (len-1 * c, z, h, w)
        
        # Step 6: padding
        cur_bev = cat_bev[-1] # c, z, h, w
        cat_bev = torch.cat((prev_bev, cur_bev), dim=0) # (len * c, z, h, w)
        if len(bev_list) < self.sampled_queue_length:
            pad_len = self.sampled_queue_length - len(bev_list)
            pad_bev = torch.zeros((pad_len*channels, total_z, bev_h, bev_w), device='cuda', dtype=cat_bev.dtype)
            cat_bev = torch.cat((pad_bev, cat_bev), dim=0) # (queue_length * c, z, h, w)

        # Step 7: use resnet to fuse the temporal information into a updated BEV feature
        cat_bev = cat_bev.unsqueeze(0) # (bs, c * queue_length, z, h, w)
        update_bev = self.resnet(cat_bev) # (bs, c, z, h, w)
        update_bev = update_bev.reshape(bs, embed_dims, bev_HW).permute(0, 2, 1) # (bs, h*w, embed_dims)

        return update_bev