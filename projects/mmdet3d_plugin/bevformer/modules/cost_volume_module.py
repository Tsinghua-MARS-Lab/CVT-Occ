import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from .residual_block_3d import ResidualBlock

class TemporalNet(nn.Module):
    def __init__(self, in_channels=240,
                 out_channels=[16, 128, 64, 32],
                 conv_cfg=dict(type='Conv3d'), 
                 norm_cfg=dict(type='BN3d'), 
                 act_cfg=dict(type='ReLU',inplace=True),
                 ):
        super(TemporalNet, self).__init__()
        self.conv_head = ConvModule(in_channels, 
                                    out_channels[0],
                                    kernel_size=3, 
                                    stride=1, 
                                    padding=1, 
                                    conv_cfg=conv_cfg,
                                    norm_cfg=norm_cfg, 
                                    act_cfg=act_cfg)

        self.layer1 = self.make_layer(out_channels[0], out_channels[1], num_blocks=2, 
                                      conv_cfg=conv_cfg, 
                                      norm_cfg=norm_cfg, 
                                      act_cfg=act_cfg)

        self.layer2 = self.make_layer(out_channels[1], out_channels[2], num_blocks=2, 
                                      conv_cfg=conv_cfg, 
                                      norm_cfg=norm_cfg, 
                                      act_cfg=act_cfg)

        self.layer3 = self.make_layer(out_channels[2], out_channels[3], num_blocks=2, 
                                      conv_cfg=conv_cfg, 
                                      norm_cfg=norm_cfg, 
                                      act_cfg=act_cfg)

        self.conv_back = ConvModule(out_channels[3], 
                                    2,
                                    kernel_size=1, 
                                    stride=1, 
                                    padding=0, 
                                    conv_cfg=conv_cfg, 
                                    norm_cfg=norm_cfg, 
                                    act_cfg=None)

    def make_layer(self, in_channels, 
                        out_channels, 
                        num_blocks=2, 
                        conv_cfg=dict(type='Conv3d'), 
                        norm_cfg=dict(type='BN3d'), 
                        act_cfg=dict(type='ReLU',inplace=True)):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(in_channels, 
                                        out_channels, 
                                        conv_cfg=conv_cfg, 
                                        norm_cfg=norm_cfg, 
                                        act_cfg=act_cfg))
            in_channels = out_channels # after one round, the inchannel will become outchannel
        return nn.Sequential(*layers)

    def forward(self, bev_3d):
        bev_3d = self.conv_head(bev_3d)
        bev_3d = self.layer1(bev_3d)
        bev_3d = self.layer2(bev_3d)
        bev_3d = self.layer3(bev_3d)
        bev_3d = self.conv_back(bev_3d)
        return bev_3d

class CostVolumeModule(BaseModule):
    def __init__(self, 
                 bev_h=200, 
                 bev_w=200, 
                 total_z=16, # not bev_z
                 channels=16, 
                 pc_range=[-40, -40, -1.0, 40, 40, 5.4],
                 voxel_size=[0.4, 0.4, 0.4],
                 sampled_queue_length=7,
                 scales=[0.8, 0.9, 1.0, 1.1, 1.2],
                 ):
        super(CostVolumeModule, self).__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.total_z = total_z
        self.channels = channels
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.sampled_queue_length = sampled_queue_length
        self.scales = scales
        self.scales_len = len(scales)
        self.scalesChannels=self.channels * self.sampled_queue_length * self.scales_len
        self.out_channels=[16, 128, 64, 32]

        self.resnet = TemporalNet(in_channels=self.scalesChannels,
                             out_channels=self.out_channels,
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

        grid_x_idx, grid_y_idx, grid_z_idx = torch.meshgrid(x_idx, y_idx, z_idx, indexing='ij') # all with shape (w, h, z)
        grid_ones = torch.ones_like(grid_x_idx) 
        ref_3d = torch.stack((grid_x_idx, grid_y_idx, grid_z_idx, grid_ones), -1) # (x, h, z, 4)
        ref_3d = ref_3d.unsqueeze(4)

        return ref_3d
    
    def gather_feature_scales(self, features, locations, dtype=torch.float32):
        """
        Use grid sample to gather feature from previous BEV volume.
        Args: 
            features (torch.Tensor): shape (bs, len, w, h, z, c).
            locations (torch.Tensor): shape (bs, len*len(scales), w, h, z, 3).
        Returns:
            grid_sampled_features (torch.Tensor): shape (bs, len, len(scales), w, h, z, c).
        """
        
        features = features.to(dtype)
        locations = locations.to(dtype)

        # norm the location and reshape locations shape
        locations = (locations - self.bias) / self.norm_factors # [-1, 1]
        locations = locations.permute(0, 1, 4, 3, 2, 5) # (bs, len*len(scales), z, h, w, 3)
        locations = locations.reshape(-1, *locations.shape[2:]) # (bs*len*len(scales), z, h, w, 3)

        # reshape features shape
        queue_len = features.shape[1]
        features = features.permute(0, 1, 5, 4, 3, 2) # (bs, len, c, z, h, w)
        features = features.unsqueeze(2) # (bs, len, 1, c, z, h, w)
        features = features.repeat(1, 1, self.scales_len, 1, 1, 1, 1) # (bs, len, len(scales), c, z, h, w)
        features = features.reshape(-1, *features.shape[3:]) # (bs*len*len(scales), c, z, h, w)

        grid_sampled_features = F.grid_sample(features, locations, align_corners=False) # (bs*len*len(scales), c, z, h, w)
        # default to be bilinear interpolation and no align corners

        grid_sampled_features = grid_sampled_features.reshape(-1, queue_len, self.scales_len, *grid_sampled_features.shape[1:]) # (bs, len, len(scales), c, z, h, w)
        grid_sampled_features = grid_sampled_features.permute(0, 1, 2, 6, 5, 4, 3) # (bs, len, len(scales), w, h, z, c)

        return grid_sampled_features

    def forward(self, bev_list, img_metas_list):
        """
        Forward pass of the `CostVolume` temporal fusion method. 
        Args: 
            bev_list (list[torch.Tensor]): each has shape (bs, h*w, embed_dims).
            img_metas_list (list[dict]): include current img meta info dict.
        return:
            updated_bev (torch.Tensor): fused BEV feature with shape (bs, h*w, embed_dims)
            refine_feat_w (torch.Tensor): shape (w, h, z, c). used to calculate refine_feat loss
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
        len_time = len(bev_list)
        cat_bev = torch.stack(bev_list, dim=1) # (bs, len_time, h*w, embed_dims)
        cat_bev = cat_bev.permute(0, 1, 3, 2) # (bs, len_time, embed_dims, h*w)
        cat_bev = cat_bev.reshape(bs, len_time, channels, total_z, bev_h, bev_w)
        cat_bev = cat_bev.permute(0, 1, 5, 4, 3, 2) # (bs, len_time, w, h, z, c)

        # Step 2: prepare the reference point for BEV volume
        # sample reference point for current ego coordinate
        bev_ref = self.get_bev_ref(H=bev_h, W=bev_w, Z=total_z) # (w, h, z, 4, 1)
        cur_center_bev_ref = bev_ref.squeeze(-1) # (w, h, z, 4)
        cur_center_bev_ref = cur_center_bev_ref[..., :3] # (w, h, z, 3)

        # sample cur point to a line according to a series of scales(strides)
        scales = self.scales
        scales = torch.tensor(scales, device='cuda', dtype=torch.float32) # (scales_len, )
        scales = scales.unsqueeze(1).repeat(1, 3) # shape (scales_len, 3)
        # Because we need to scale the x, y, and z components by the same multiple, we set this component to 3 here

        repeat_cur_center_bev_ref = cur_center_bev_ref.unsqueeze(3).repeat(1, 1, 1, self.scales_len, 1) # (w, h, z, scales_len, 3)
        cur_line_bev_ref = repeat_cur_center_bev_ref * scales # (w, h, z, scales_len, 3)

        # reshape cur_line prepared for wrapping
        repeat_cur_line_bev_ref = cur_line_bev_ref.unsqueeze(3).repeat(1, 1, 1, len_time-1, 1, 1) # (w, h, z, len_time-1, scales_len, 3)
        repeat_cur_line_bev_ref = repeat_cur_line_bev_ref.permute(4, 0, 1, 2, 3, 5) # (scales_len, w, h, z, len-1, 3)
        ones_tensor = torch.ones_like(repeat_cur_line_bev_ref[..., 0:1]) # (scales_len, w, h, z, len-1, 1)
        repeat_cur_line_bev_ref = torch.cat((repeat_cur_line_bev_ref, ones_tensor), dim=-1) # (scales_len, w, h, z, len-1, 4)
        repeat_cur_line_bev_ref = repeat_cur_line_bev_ref.unsqueeze(6) # (scales_len, w, h, z, len-1, 4, 1)

        # Step 3: use `ego2global` to wrap the reference point to the previous BEV volume        
        # process the data add batch size
        repeat_cur_line_bev_ref = repeat_cur_line_bev_ref.unsqueeze(4) # (scales_len, w, h, z, 1, len-1, 4, 1)
        repeat_cur_line_bev_ref = repeat_cur_line_bev_ref.repeat(1, 1, 1, 1, bs, 1, 1, 1) # (scales_len, w, h, z, bs, len-1, 4, 1)
        cur_ego2global = cur_ego2global.unsqueeze(1).repeat(1, len_time-1, 1, 1) # (bs, len-1, 4, 4)

        # use ego2global to transform the reference point to prev_bev coordinate, estimate egomotion
        prev_line_bev_ref = torch.inverse(cat_prev_ego2global) @ cur_ego2global @ repeat_cur_line_bev_ref # (scales_len, w, h, z, bs, len-1, 4, 1) 
        repeat_cur_line_bev_ref = repeat_cur_line_bev_ref.permute(4, 5, 0, 1, 2, 3, 6, 7) # (bs, len_time-1, scales_len, w, h, z, 4, 1)
        cur_line_bev_ref = repeat_cur_line_bev_ref[:, 0:1, ...] # (bs, 1, scales_len, w, h, z, 4, 1)
        prev_line_bev_ref = prev_line_bev_ref.permute(4, 5, 0, 1, 2, 3, 6, 7) # (bs, len_time-1, scales_len, w, h, z, 4, 1)
        line_bev_ref = torch.cat((prev_line_bev_ref, cur_line_bev_ref), dim=1) # (bs, len_time, scales_len, w, h, z, 4, 1)

        line_bev_ref = line_bev_ref.squeeze(-1) # (bs, len, scales_len, w, h, z, 4)
        line_bev_ref = line_bev_ref[..., :3] # (bs, len, scales_len, w, h, z, 3)

        # Step 4: get mask
        mask_for_all = (
                (line_bev_ref[..., 0] > pc_range[0]) & (line_bev_ref[..., 0] < pc_range[3]) & 
                (line_bev_ref[..., 1] > pc_range[1]) & (line_bev_ref[..., 1] < pc_range[4]) & 
                (line_bev_ref[..., 2] > pc_range[2]) & (line_bev_ref[..., 2] < pc_range[5])
            ) # out-of-range points will be 0. with shape (len, scales_len, w, h, z) 

        # Step 5: gather feature for all reference
        line_bev_ref = line_bev_ref.reshape(bs, len_time*self.scales_len, bev_w, bev_h, total_z, 3) # (bs, len*scales_len, w, h, z, 3)
        line_bev = self.gather_feature_scales(cat_bev, line_bev_ref, dtype=cat_bev.dtype) # (bs, len, scales_len, w, h, z, c)
        line_bev[~mask_for_all] = 0

        extra = {}
        # Step 6: padding the line_bev by zero tensor. (because convolution need same length)
        true_queue_length = self.sampled_queue_length
        if len_time < true_queue_length:
            zero_bev = torch.zeros_like(line_bev[:, 0:1]) # (bs, 1, scales_len, w, h, z, c) 
            zero_bev = zero_bev.repeat(1, true_queue_length-len_time, 1, 1, 1, 1, 1) # (bs, pad_len, scales_len, w, h, z, c)
            line_bev = torch.cat((zero_bev, line_bev), dim=1) # (bs, true_q_len, scales_len, w, h, z, c)

        # Step 7: use resnet to fuse the temporal information into a weight
        line_bev = line_bev.reshape(bs, -1, *line_bev.shape[3:]) # (bs, true_q_len*scales_len, w, h, z, c)
        line_bev = line_bev.permute(0, 1, 5, 4, 3, 2) # (bs, true_q_len*scales_len, c, z, h, w)
        line_bev = line_bev.reshape(bs, -1, *line_bev.shape[3:]) # (bs, true_q_len*scales_len*c, z, h, w)

        refine_feat_w = self.resnet(line_bev) # (bs, 2, z, h, w) # 2 channels setting is used for focal loss
        refine_feat_w = refine_feat_w.permute(0, 4, 3, 2, 1) # (bs, w, h, z, 2)
        extra['refine_feat_w'] = refine_feat_w

        # Step 8: update the BEV volume
        update_bev = cat_bev[:, -1] * refine_feat_w[..., 1:].sigmoid() # (bs, w, h, z, c) * (bs, w, h, z, 1) = (bs, w, h, z, c)
        update_bev = update_bev.permute(0, 4, 3, 2, 1) # (bs, c, z, h, w)
        update_bev = update_bev.reshape(-1, embed_dims, bev_HW).permute(0, 2, 1) # (1, h*w, embed_dims)

        return update_bev, extra