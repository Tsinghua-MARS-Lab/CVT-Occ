# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import normal_
from torchvision.transforms.functional import rotate

from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule
from mmcv.runner.base_module import BaseModule
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import PLUGIN_LAYERS, Conv2d,Conv3d, ConvModule
from mmdet.models.utils.builder import TRANSFORMER

from projects.mmdet3d_plugin.bevformer.modules.temporal_self_attention import TemporalSelfAttention
from projects.mmdet3d_plugin.bevformer.modules.spatial_cross_attention import MSDeformableAttention3D
from projects.mmdet3d_plugin.bevformer.modules.decoder import CustomMSDeformableAttention
from projects.mmdet3d_plugin.bevformer.modules.cost_volume_module import CostVolumeModule

@TRANSFORMER.register_module()
class CVTOccTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 bev_h=200,
                 bev_w=200,
                 channels=16,
                 pc_range=None,
                 voxel_size=None,
                 rotate_prev_bev=False,
                 use_shift=False,
                 use_can_bus=False,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 use_free_mask=False,
                 use_3d=False,
                 use_conv=False,
                 rotate_center=[100, 100],
                 num_classes=18,
                 out_dim=32,
                 pillar_h=16,
                 queue_length=None,
                 use_padding=False,
                 use_temporal=None,
                 scales=None,
                 act_cfg=dict(type='ReLU',inplace=True),
                 norm_cfg=dict(type='BN', ),
                 norm_cfg_3d=dict(type='BN3d', ),
                 **kwargs):
        super(CVTOccTransformer, self).__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.use_free_mask = use_free_mask
        self.embed_dims = embed_dims
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.channels = channels
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.grid_length = ((pc_range[4] - pc_range[1]) / bev_h, 
                            (pc_range[3] - pc_range[0]) / bev_w)
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False
        if use_free_mask:
            num_classes = num_classes - 1
        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds
        self.use_3d = use_3d
        self.use_conv = use_conv
        self.pillar_h = pillar_h
        self.queue_length = queue_length
        self.use_padding = use_padding
        self.use_temporal = use_temporal
        self.scales = scales
        self.out_dim = out_dim
        if not use_3d:
            if use_conv:
                use_bias = norm_cfg is None
                self.decoder  = nn.Sequential(
                    ConvModule(
                        self.embed_dims,
                        self.embed_dims,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=use_bias,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg),
                    ConvModule(
                        self.embed_dims,
                        self.embed_dims*2,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=use_bias,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg),)

            else:
                self.decoder = nn.Sequential(
                    nn.Linear(self.embed_dims, self.embed_dims * 2),
                    nn.Softplus(),
                    nn.Linear(self.embed_dims * 2, self.embed_dims * 2),
                )
        else:
            use_bias_3d = norm_cfg_3d is None

            self.middle_dims=self.embed_dims//pillar_h
            self.decoder = nn.Sequential(
                ConvModule(
                    self.middle_dims,
                    self.out_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias_3d,
                    conv_cfg=dict(type='Conv3d'),
                    norm_cfg=norm_cfg_3d,
                    act_cfg=act_cfg),
                ConvModule(
                    self.out_dim,
                    self.out_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias_3d,
                    conv_cfg=dict(type='Conv3d'),
                    norm_cfg=norm_cfg_3d,
                    act_cfg=act_cfg),
            )
        self.predicter = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim*2),
            nn.Softplus(),
            nn.Linear(self.out_dim*2,num_classes),
        )
        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center
        if self.use_temporal == "costvolume":
            self.costvolume = CostVolumeModule(bev_h=self.bev_h, 
                                                bev_w=self.bev_w,
                                                total_z=self.pillar_h,
                                                channels=self.channels,
                                                pc_range=self.pc_range,
                                                voxel_size=self.voxel_size,
                                                sampled_queue_length=self.queue_length, 
                                                scales=self.scales,)

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        # self.reference_points = nn.Linear(self.embed_dims, 3)
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        # xavier_init(self.reference_points, distribution='uniform', bias=0.)
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)

    @auto_fp16(apply_to=('multi_level_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def get_bev_features(self, multi_level_feats, 
                         bev_queries, 
                         bev_pos=None,
                         cur_img_metas=None, 
                         prev_bev=None, 
                         **kwargs):
        """
        This method is used to obtain Bird's Eye View (BEV) features from multi-level features, BEV queries, and other related parameters.
        use bev queries to find feature in multi-view camera img
        Args:
            multi_level_feats (list[torch.Tensor]): Current multi level img features from the upstream network.
                                                    Each is a 5D-tensor img_feats with shape (bs, num_cams, embed_dims, h, w).
            cur_img_metas (list[dict]): Meta information of each sample. The list has length of batch size.
            bev_queries (torch.Tensor): (bev_h*bev_w, c). used in decoder
            bev_pos (torch.Tensor): (bs, embed_dims, bev_h, bev_w). used in decoder
            prev_bev (torch.Tensor): BEV features of the previous sample.
        Returns: 
            results (dict): with keys "bev_embed, feat_flatten, spatial_shapes, level_start_index, shift". 
            bev_embed (torch.Tensor): BEV feature for current frame.
            feat_flatten (torch.Tensor): Each level img feature, flattens the height and width dimensions and combine together. 
                                         shape (num_cam, bs, h*w, c). h*w are sum of all fallten h*w of all levels = 12750.
            spatial_shapes (torch.Tensor): Record the shape of each level img feature. 
                                           tensor([[ 80, 120],[ 40,  60],[ 20,  30],[ 10,  15]]).
            level_start_index (torch.Tensor): Record the start index of each level img feature in feat_flatten.
                                              tensor([0, 9600, 12000, 12600]).
            shift (torch.Tensor): shift of ego car in x and y axis. shape (1, 2). 
        """

        # Step 1: obtain parameters
        bs = multi_level_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)
        grid_length_y = self.grid_length[0]
        grid_length_x = self.grid_length[1]

        # Step 2: obtain rotation angle and shift with ego motion
        if self.use_can_bus:
            delta_x = np.array([each['can_bus'][0] for each in cur_img_metas])
            delta_y = np.array([each['can_bus'][1] for each in cur_img_metas])
            ego_angle = np.array([each['can_bus'][-2] / np.pi * 180 for each in cur_img_metas])
            translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
            translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
            bev_angle = ego_angle - translation_angle
            shift_y = translation_length * \
                np.cos(bev_angle / 180 * np.pi) / grid_length_y / self.bev_h
            shift_x = translation_length * \
                np.sin(bev_angle / 180 * np.pi) / grid_length_x / self.bev_w
            shift_y = shift_y * self.use_shift
            shift_x = shift_x * self.use_shift
            shift = bev_queries.new_tensor(np.array([shift_x, shift_y])).permute(1, 0)  # xy, bs -> bs, xy
        else:
            shift = bev_queries.new_zeros(bs, 2)

        # Step 3: apply rotation to previous BEV features
        if prev_bev is not None:
            if prev_bev.shape[1] == self.bev_h * self.bev_w:
                prev_bev = prev_bev.permute(1, 0, 2) # (bev_h*bev_w, bs, embed_dims)

            # elif len(prev_bev.shape) == 4: # (bs, embed_dims, bev_h, bev_w)
            #     prev_bev = prev_bev.view(bs, -1, self.bev_h * self.bev_w).permute(2, 0, 1) # (bev_h*bev_w, bs, embed_dims)
            
            if self.rotate_prev_bev:
                for i in range(bs):
                    rotation_angle = cur_img_metas[i]['can_bus'][-1]
                    tmp_prev_bev = prev_bev[:, i].reshape(
                        self.bev_h, self.bev_w, -1).permute(2, 0, 1)
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle, center=self.rotate_center)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(self.bev_h * self.bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        # Step 4: apply ego motion shift to BEV queries
        if self.use_can_bus:
            can_bus = bev_queries.new_tensor(np.array([each['can_bus'] for each in cur_img_metas]))
            can_bus = self.can_bus_mlp(can_bus)[None, :, :]
            bev_queries = bev_queries + can_bus

        # Step 5: flatten the multi level image features
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(multi_level_feats):
            # For each level feature, flattens the height and width dimensions (last two dimensions) and permutes the dimensions to make the shape compatible with concatenation.
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None, None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        # Step 6: Use the encoder the get the BEV features
        bev_embed = self.encoder(bev_query=bev_queries,
                                 key=feat_flatten,
                                 value=feat_flatten,
                                 bev_pos=bev_pos,
                                 spatial_shapes=spatial_shapes,
                                 level_start_index=level_start_index,
                                 prev_bev=prev_bev,
                                 img_metas=cur_img_metas,
                                 shift=shift,
                                 **kwargs)

        results = {
            "bev_embed": bev_embed,
            "feat_flatten": feat_flatten,
            "spatial_shapes": spatial_shapes,
            "level_start_index": level_start_index,
            "shift": shift,
        }

        return results
    
    def temporal_fusion(self, cur_bev, cur_img_metas, prev_bev_list, prev_img_metas):
        """
        Do Temporal Fusion.
        Args:
        Returns:
            fusion_results (dict): with keys "bev_embed, extra".
            bev_embed (torch.Tensor): Updated BEV features after some temporal fusion method.
            extra (dict): with keys "refine_feat_w", which is a tensor with shape (w, h, z, 2).
                          or maybe empty dict. 
        """

        # Step 1: prepare parameters
        bev_list = list(prev_bev_list)
        bev_list.append(cur_bev)
        img_metas = list(prev_img_metas)
        for batch_idx in range(len(cur_img_metas)): # for i in batch_size
            each_batch_img_metas = img_metas[batch_idx] # dict[dict]
            each_batch_img_metas[self.queue_length - 1] = cur_img_metas[batch_idx]

        # Step 2: padding(controlled by flag `use_padding`)
        if len(bev_list) < self.queue_length and self.use_padding:
            bev_list = [bev_list[0].clone() for _ in range(self.queue_length - len(bev_list))] + bev_list
            queue_begin = list(img_metas[0].keys())[0] # the min index
            for batch_idx in range(len(img_metas)):
                for queue_idx in range(0, queue_begin):
                    img_metas[batch_idx][queue_idx] = img_metas[batch_idx][queue_begin].copy()

        # Step 3: reshape the img_metas to a list
        keys_list = list(img_metas[0].keys())
        keys_list.sort() # HERE!
        img_metas_list = []
        for key in keys_list:
            for batch_idx in range(len(img_metas)):
                img_metas_list.append(img_metas[batch_idx][key]) # list[dict]
            
        # Step 4: do temporal fusion
        if self.use_temporal == 'costvolume' and len(bev_list) > 1:
            update_bev, extra = self.costvolume(bev_list, img_metas_list)
            fusion_results = {'bev_embed': update_bev, 'extra': extra}
        
        else:
            # no fusion
            fusion_results = {'bev_embed': cur_bev, 'extra': {}}
        
        return fusion_results
    
    @auto_fp16(apply_to=('multi_level_feats', 'bev_queries', 'prev_bev_list', 'bev_pos'))
    def forward(self, multi_level_feats,
                bev_queries,
                bev_pos=None,
                cur_img_metas=None,
                prev_bev_list=[],
                prev_img_metas=[],
                **kwargs):
        """
        Forward function for `Detr3DTransformer`. 
        Args:
            multi_level_feats (list(torch.Tensor)): Current multi level img features from the upstream network.
                                                    Each element has shape (bs, num_cams, embed_dims, h, w).
            bev_queries (torch.Tensor): bev embedding with shape (hwz, embed_dims). 
            bev_pos (torch.Tensor): bev position embedding with shape (bs, embed_dims, 1, h, w). 
            img_metas (list[dict]): current img meta info. The list has length of batch size. 
            prev_bev_list (list(torch.Tensor)): BEV features of previous frames. Each has shape (bs, bev_h*bev_w, embed_dims). 
            prev_img_metas (list[dict[dict]]): Meta information of each sample.
                                               The list has length of batch size.
                                               The dict has keys len_queue-1-prev_bev_list_len, ..., len_queue-2. 
                                               The element of each key is a dict.
                                               So each dict has length of prev_bev_list_len. 
        Returns:
            bev_for_history (torch.Tensor): directly from self.get_bev_features with shape (bs, h*w, embed_dims) only used in inference. 
            outputs (torch.Tensor): bev_embed after fusion, decoder and predictor. shape (bs, w, h, z, c).
            extra (dict): with keys "refine_feat_w", which is a tensor with shape (w, h, z, 2).
                          or maybe empty dict. 
        """

        # Step 1: prepare parameters
        bev_h = self.bev_h
        bev_w = self.bev_w
        if len(prev_bev_list) > 0:
            prev_bev = prev_bev_list[-1] # (bs, h*w*z, c)
        else: 
            prev_bev = None

        # Step 2: get BEV features
        get_bev_features_outputs = self.get_bev_features(multi_level_feats,
                                                         bev_queries,
                                                         bev_pos,
                                                         cur_img_metas,
                                                         prev_bev,
                                                         **kwargs)
        bev_embed = get_bev_features_outputs['bev_embed']
        bev_for_history = bev_embed.clone() # (bs, h*w, embed)

        # Step 3: do temporal fusion
        outputs = self.temporal_fusion(bev_embed, cur_img_metas, prev_bev_list, prev_img_metas)
        bev_embed = outputs['bev_embed']
        extra = outputs['extra']

        # Step 4: Decoder and predictor
        bs = multi_level_feats[0].size(0)
        # bev_embed = bev_embed.permute(0, 2, 1).view(bs, -1, bev_h, bev_w) # (bs, embed_dims, h, w)
        if self.use_3d:
            assert NotImplementedError

        elif self.use_conv:
            assert NotImplementedError

        else:
            bev_embed = bev_embed.permute(0, 2, 1).view(bs, -1, bev_h, bev_w) # (bs, embed_dims, h, w)
            # outputs = self.decoder(bev_embed.permute(0,2,3,1))
            outputs = self.decoder(bev_embed.permute(0, 3, 2, 1)) # bs, w, h, embed_dims, 
            outputs = outputs.view(bs, bev_w, bev_h, self.pillar_h, self.out_dim)

            # outputs = self.decoder(bev_embed) # (bs, bev_h*bev_w, embed_dims * 2)
            # outputs = outputs.permute(0, 2, 1).view(bs, self.out_dim, self.pillar_h, bev_h, bev_w) # (bs, out_dim, pillar_h, h, w)
            # outputs = outputs.permute(0, 4, 3, 2, 1) # (bs, w, h, pillar_h, out_dim)
            outputs = self.predicter(outputs) # (bs, w, h, pillar_h, num_classes)

        return bev_for_history, outputs, extra
