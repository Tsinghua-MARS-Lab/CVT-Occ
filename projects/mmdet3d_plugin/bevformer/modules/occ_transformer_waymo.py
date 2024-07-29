# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_
from torchvision.transforms.functional import rotate

from mmcv.cnn import xavier_init
from mmcv.runner import force_fp32, auto_fp16
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule
from mmdet.models.utils.builder import TRANSFORMER

from projects.mmdet3d_plugin.bevformer.modules.temporal_self_attention import TemporalSelfAttention
from projects.mmdet3d_plugin.bevformer.modules.spatial_cross_attention import MSDeformableAttention3D
from projects.mmdet3d_plugin.bevformer.modules.decoder import CustomMSDeformableAttention
from projects.mmdet3d_plugin.bevformer.modules.cost_volume_module import CostVolumeModule
from projects.mmdet3d_plugin.bevformer.modules.concat_conv_module import ConcatConvModule

@TRANSFORMER.register_module()
class CVTOccTransformerWaymo(BaseModule):
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
                 volume_flag=True,
                 num_feature_levels=4,
                 num_cams=6,
                 queue_length=3,
                 sampled_queue_length=1,
                 two_stage_num_proposals=300,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 pc_range=None,
                 voxel_size=None,
                 occ_voxel_size=None,
                 use_larger=True,
                 use_temporal=None,
                 rotate_prev_bev=False,
                 use_shift=False,
                 use_can_bus=False,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 use_3d_decoder=False,
                 use_conv_decoder=False,
                 rotate_center=[100, 100],
                 scales=[0.8, 0.9, 1.0, 1.1, 1.2],
                 num_classes=18,
                 out_dim=32,
                 pillar_h=16,
                 bev_z=1,
                 bev_h=200,
                 bev_w=200,
                 total_z=16,
                 iter_encoders=None,
                 use_padding=False,
                 topK_method='foreground',
                 **kwargs):
        super(CVTOccTransformerWaymo, self).__init__(**kwargs)
        self.volume_flag = volume_flag
        self.encoder = build_transformer_layer_sequence(encoder)
        if iter_encoders is not None: # default is None
            self.iter_encoders = torch.nn.ModuleList([build_transformer_layer_sequence(encoder) for encoder in iter_encoders])
        self.decoder = build_transformer_layer_sequence(decoder)
        self.topK_method = topK_method
        self.queue_length = queue_length
        self.sampled_queue_length = sampled_queue_length
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.occ_voxel_size = occ_voxel_size
        self.use_larger = use_larger
        self.use_temporal = use_temporal
        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds
        self.use_3d_decoder=use_3d_decoder
        self.use_conv_decoder = use_conv_decoder
        self.pillar_h = pillar_h
        self.out_dim = out_dim
        self.bev_z = bev_z
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.total_z = total_z
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.grid_length = (self.real_h / self.bev_h, self.real_w / self.bev_w)
        self.channels=self.embed_dims//self.total_z # 256//16=16
        self.scales=scales
        self.use_padding = use_padding
        if self.use_temporal == 'costvolume':
            self.costvolume = CostVolumeModule(bev_h=self.bev_h, 
                                                bev_w=self.bev_w,
                                                total_z=self.total_z,
                                                channels=self.channels,
                                                pc_range=self.pc_range,
                                                voxel_size=self.voxel_size,
                                                sampled_queue_length=self.sampled_queue_length, 
                                                scales=self.scales,)
        if self.use_temporal == 'concat_conv':
            self.concatconv = ConcatConvModule(bev_h=self.bev_h,
                                             bev_w=self.bev_w,
                                             total_z=self.total_z,
                                             channels=self.channels,
                                             pc_range=self.pc_range,
                                             voxel_size=self.voxel_size,
                                             sampled_queue_length=self.sampled_queue_length,)
        
        # choose predictor
        if not self.use_3d_decoder:
            if self.use_conv_decoder:
                _out_dim = self.embed_dims*self.pillar_h
                # because not use 3d, so total_z=1
                self.predicter = nn.Sequential(
                    nn.Linear(_out_dim//total_z, self.embed_dims//2),
                    nn.Softplus(),
                    nn.Linear(self.embed_dims//2,num_classes),
                )
            else:raise NotImplementedError
        else:
            # use_3d_decoder enter here
            _out_dim = self.embed_dims
            self.predicter = nn.Sequential(
                nn.Linear(_out_dim, self.embed_dims//2),
                nn.Softplus(),
                nn.Linear(self.embed_dims//2, 2), # binary classify
            ) 

        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center

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
                         img_metas=None,
                         prev_bev=None,
                         **kwargs):
        """
        This method is used to obtain Bird's Eye View (BEV) features from multi-level features, BEV queries, and other related parameters.
        use bev queries to find feature in multi-view camera img
        Args:
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
        bev_z = self.bev_z
        bev_h = self.bev_h
        bev_w = self.bev_w
        grid_length = self.grid_length
        bs = multi_level_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1) # TODO why reshape here?
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        # Step 2: obtain rotation angle and shift with ego motion
        grid_length_y = grid_length[0] # 0.4
        grid_length_x = grid_length[1] # 0.4
        if self.use_can_bus: # Default value is False
            delta_x = np.array([each['can_bus'][0]for each in img_metas])
            delta_y = np.array([each['can_bus'][1]for each in img_metas])
            ego_angle = np.array([each['can_bus'][-2] / np.pi * 180 for each in img_metas])
            translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
            translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
            bev_angle = ego_angle - translation_angle
            shift_y = translation_length * \
                np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
            shift_x = translation_length * \
                np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
            shift_y = shift_y * self.use_shift
            shift_x = shift_x * self.use_shift
            shift = bev_queries.new_tensor([shift_x, shift_y]).permute(1, 0) # (2, 1) -> (1, 2)
        else:
            shift = bev_queries.new_zeros((1, 2))

        # Step 3: apply rotation to previous BEV features
        if prev_bev is not None:
            if self.volume_flag:
                if prev_bev.shape[1] == bev_h * bev_w * bev_z:
                    prev_bev = prev_bev.permute(1, 0, 2)
                elif len(prev_bev.shape) == 4:
                    prev_bev = prev_bev.view(bs,-1,bev_h * bev_w).permute(2, 0, 1)
                elif len(prev_bev.shape) == 5:
                    prev_bev = prev_bev.view(bs, -1,bev_z* bev_h * bev_w).permute(2, 0, 1)
            else:
                # HERE
                if prev_bev.shape[1] == bev_h * bev_w:
                    prev_bev = prev_bev.permute(1, 0, 2) # (bs, bev_h*bev_w, embed_dims) -> (bev_h*bev_w, bs, embed_dims)
                elif len(prev_bev.shape) == 4: # nuscene
                    prev_bev = prev_bev.view(bs, -1, bev_h * bev_w).permute(2, 0, 1) # (bs, embed_dims, h, w) -> (bev_h*bev_w, bs, embed_dims)
                
            if self.rotate_prev_bev: # Default value is False
                for i in range(bs):
                    rotation_angle = img_metas[i]['can_bus'][-1]
                    if self.volume_flag:
                        tmp_prev_bev = prev_bev[:, i].reshape(
                            bev_z, bev_h, bev_w, -1).permute(3, 0, 1, 2)
                        tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                            center=self.rotate_center)
                        tmp_prev_bev = tmp_prev_bev.permute(1, 2,3, 0).reshape(
                            bev_z * bev_h * bev_w, 1, -1)
                    else:
                        tmp_prev_bev = prev_bev[:, i].reshape(
                            bev_h, bev_w, -1).permute(2, 0, 1)
                        tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                            center=self.rotate_center)
                        tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                            bev_h * bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]
        
        # Step 4: apply ego motion shift to BEV queries
        if self.use_can_bus: # Default value is False
            can_bus = bev_queries.new_tensor([each['can_bus'] for each in img_metas])
            can_bus = self.can_bus_mlp(can_bus)[None, :, :]
            bev_queries = bev_queries + can_bus

        # Step 5: flatten the multi level image features
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(multi_level_feats):
            # For each level feature, flattens the height and width dimensions (last two dimensions) and permutes the dimensions to make the shape compatible with concatenation.
            bs, _, _, h, w = feat.shape # bs, n_views, c, h, w
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape) # list[tuple]
            feat_flatten.append(feat)
        feat_flatten = torch.cat(feat_flatten, 2) # (num_cam, bs, h*w, c). h*w are sum of all fallten h*w of all levels = 12750

        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=bev_queries.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        feat_flatten = feat_flatten.permute(0, 2, 1, 3) # (num_cam, h*w, bs, embed_dims)

        # Step 6: Use the encoder the get the BEV features
        bev_embed = self.encoder(bev_query=bev_queries,
                                 key=feat_flatten, 
                                 value=feat_flatten,
                                 bev_pos=bev_pos,
                                 spatial_shapes=spatial_shapes,
                                 level_start_index=level_start_index,
                                 prev_bev=prev_bev,
                                 img_metas=img_metas,
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
    
    def temporal_fusion(self, cur_bev, prev_bev_list, prev_img_metas, cur_img_metas):
        """
        Do Temporal Fusion.
        Args:
        Returns:
            fusion_results (dict): with keys "_bev_embed, extra".
            _bev_embed (torch.Tensor): Updated BEV features after some temporal fusion method.
            extra (dict): with keys "refine_feat_w", which is a tensor with shape (w, h, z, 2).
                          or maybe empty dict. 
        """

        # Step 1: prepare parameters
        bev_list = list(prev_bev_list)
        bev_list.append(cur_bev)
        img_metas = list(prev_img_metas)
        for batch_idx in range(len(cur_img_metas)): # for i in batch_size
            each_batch_img_metas = img_metas[batch_idx] # dict[dict]
            each_batch_img_metas[self.sampled_queue_length - 1] = cur_img_metas[batch_idx]

        # Step 2: padding(controlled by flag `use_padding`)
        if len(bev_list) < self.sampled_queue_length and self.use_padding:
            bev_list = [bev_list[0].clone() for _ in range(self.sampled_queue_length - len(bev_list))] + bev_list
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
            fusion_results = {'_bev_embed': update_bev, 'extra': extra}
        
        elif self.use_temporal == 'concat_conv' and len(bev_list) > 1:
            update_bev = self.concatconv(bev_list, img_metas_list)
            fusion_results = {'_bev_embed': update_bev, 'extra': {}}
        
        else:
            # no fusion
            fusion_results = {'_bev_embed': cur_bev, 'extra': {}}
        
        return fusion_results

    @auto_fp16(apply_to=('multi_level_feats', 'bev_queries', 'object_query_embed', 'prev_bev_list', 'bev_pos'))
    def forward(self, multi_level_feats,
                bev_queries,
                bev_pos=None,
                img_metas=None,
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
        bev_z = self.bev_z
        bev_h = self.bev_h
        bev_w = self.bev_w
        if not self.volume_flag: assert bev_z == 1
        if len(prev_bev_list) > 0:
            prev_bev = prev_bev_list[-1] # (bs, h*w*z, c)
        else: 
            prev_bev = None

        # Step 2: get BEV features
        get_bev_features_outputs = self.get_bev_features(multi_level_feats,
                                                        bev_queries,
                                                        bev_pos=bev_pos,
                                                        img_metas=img_metas,
                                                        prev_bev=prev_bev,
                                                        **kwargs)
        _bev_embed = get_bev_features_outputs['bev_embed']
        bev_for_history = _bev_embed
        
        # Step 3: do temporal fusion
        outputs = self.temporal_fusion(_bev_embed, prev_bev_list, prev_img_metas, img_metas)
        _bev_embed = outputs['_bev_embed']
        extra = outputs['extra'] # a empty dict or containing refine_feat_w
        _bev_embed = _bev_embed.to(bev_for_history.dtype)

        # Step 4: Decoder and Predictor
        # assert bev_embed in [bs, DHW, C] order
        bev_embed_bs_DHW_C  = _bev_embed # [bs, 40000, 256]
        feat_flatten = get_bev_features_outputs['feat_flatten'] # [num_cams, 12750, bs, 256]
        spatial_shapes = get_bev_features_outputs['spatial_shapes'] # [80, 120]=>[10,15]
        level_start_index = get_bev_features_outputs['level_start_index']
        shift = get_bev_features_outputs['shift']

        bs = multi_level_feats[0].size(0)
        if self.use_3d_decoder:
            zz = bev_z if self.volume_flag else self.pillar_h
            bev_embed_bs_C_D_H_W = bev_embed_bs_DHW_C.permute(0, 2, 1).view(bs, -1, zz, bev_h, bev_w)
            res_bs_C_D_H_W = self.decoder(bev_embed_bs_C_D_H_W)
            bev_embed_bs_C_D_H_W = bev_embed_bs_C_D_H_W + res_bs_C_D_H_W
            bev_embed_bs_W_H_D_C = bev_embed_bs_C_D_H_W.permute(0,4,3,2,1)
            outputs_bs_W_H_D_C = self.predicter(bev_embed_bs_W_H_D_C)

            bev_embed_list = [bev_embed_bs_W_H_D_C]
            outputs_list = [outputs_bs_W_H_D_C]
            topk_dim = 1 # 1 for foreground
            for iter_i, iter_encoder in enumerate(self.iter_encoders):
                # topk voxel
                topk_ratio = iter_encoder.topk_ratio
                if self.topK_method == 'foreground' or self.topK_method == 'no_cross_atten' or self.topK_method == 'no_conv':
                    outputs_onedim_bs_W_H_D = outputs_bs_W_H_D_C[:, :, :, :, topk_dim]
                    outputs_squeeze_bsWHD = outputs_onedim_bs_W_H_D.reshape(-1)
                    topk_mask_bs_W_H_D = torch.zeros_like(outputs_onedim_bs_W_H_D, dtype=torch.bool)
                    topk_mask_squeeze_bsWHD = topk_mask_bs_W_H_D.reshape(-1)
                    topk = int(outputs_squeeze_bsWHD.shape[0] * topk_ratio)
                    indices = torch.topk(outputs_squeeze_bsWHD, topk).indices
                    topk_mask_squeeze_bsWHD[indices] = True
                elif self.topK_method == 'ambiguous':
                    scores_bs_W_H_D = outputs_bs_W_H_D_C.softmax(dim=-1)[:, :, :, :, topk_dim]
                    ambiguous_bs_W_H_D = 1 - torch.abs(0.5 - scores_bs_W_H_D)
                    ambiguous_squeeze_bsWHD = ambiguous_bs_W_H_D.reshape(-1)
                    topk_mask_bs_W_H_D = torch.zeros_like(ambiguous_bs_W_H_D, dtype=torch.bool)
                    topk_mask_squeeze_bsWHD = topk_mask_bs_W_H_D.reshape(-1)
                    topk = int(ambiguous_squeeze_bsWHD.shape[0] * topk_ratio)
                    indices = torch.topk(ambiguous_squeeze_bsWHD, topk).indices
                    topk_mask_squeeze_bsWHD[indices] = True                    
                elif self.topK_method == 'mixed':
                    scores_bs_W_H_D = outputs_bs_W_H_D_C.softmax(dim=-1)[:, :, :, :, topk_dim]
                    ambiguous_bs_W_H_D = 1 - torch.abs(0.5 - scores_bs_W_H_D)
                    ambiguous_squeeze_bsWHD = ambiguous_bs_W_H_D.reshape(-1)
                    topk_mask_bs_W_H_D = torch.zeros_like(ambiguous_bs_W_H_D, dtype=torch.bool)
                    topk_mask_squeeze_bsWHD = topk_mask_bs_W_H_D.reshape(-1)
                    topk = int(ambiguous_squeeze_bsWHD.shape[0] * topk_ratio * 0.5)
                    indices = torch.topk(ambiguous_squeeze_bsWHD, topk).indices
                    topk_mask_squeeze_bsWHD[indices] = True

                    outputs_onedim_bs_W_H_D = outputs_bs_W_H_D_C[:, :, :, :, topk_dim]
                    outputs_squeeze_bsWHD = outputs_onedim_bs_W_H_D.reshape(-1)
                    topk = int(outputs_squeeze_bsWHD.shape[0] * topk_ratio * 0.5)
                    indices = torch.topk(outputs_squeeze_bsWHD, topk).indices
                    topk_mask_squeeze_bsWHD[indices] = True
                elif self.topK_method == 'random':
                    outputs_onedim_bs_W_H_D = outputs_bs_W_H_D_C[:, :, :, :, topk_dim]
                    outputs_squeeze_bsWHD = outputs_onedim_bs_W_H_D.reshape(-1)
                    topk_mask_bs_W_H_D = torch.zeros_like(outputs_onedim_bs_W_H_D, dtype=torch.bool)
                    topk_mask_squeeze_bsWHD = topk_mask_bs_W_H_D.reshape(-1)
                    topk = int(outputs_squeeze_bsWHD.shape[0] * topk_ratio)
                    # indices = torch.topk(outputs_squeeze_bsWHD, topk).indices
                    indices = torch.randint(low=0, high=outputs_squeeze_bsWHD.shape[0], size=(topk,)).to(topk_mask_squeeze_bsWHD.device)
                    topk_mask_squeeze_bsWHD[indices] = True 
                else:
                    raise NotImplementedError

                # upsample
                bs, C, D, H, W = bev_embed_bs_C_D_H_W.shape
                tg_D, tg_H, tg_W = iter_encoder.DHW
                topk_mask_bs_D_H_W = topk_mask_bs_W_H_D.permute(0, 3, 2, 1)
                topk_mask_bs_C_D_H_W = topk_mask_bs_D_H_W.unsqueeze(dim=1) # => bs,1,D,H,W 
                update_bev_embed_bs_C_D_H_W  = F.interpolate(bev_embed_bs_C_D_H_W, size=(tg_D, tg_H, tg_W), mode='trilinear', align_corners=True)
                update_topk_bs_C_D_H_W  = F.interpolate(topk_mask_bs_C_D_H_W.float(), size=(tg_D, tg_H, tg_W), mode='trilinear', align_corners=True)
                update_topk_bs_C_D_H_W = update_topk_bs_C_D_H_W > 0
                update_topk_bs_D_H_W = update_topk_bs_C_D_H_W.squeeze(dim=1)
                update_bev_embed_bs_C_DHW = update_bev_embed_bs_C_D_H_W.reshape(bs, C, tg_D*tg_H*tg_W)
                update_bev_embed_DHW_bs_C = update_bev_embed_bs_C_DHW.permute(2, 0, 1) # => (DHW, bs, C)
                update_topk_bs_DHW = update_topk_bs_D_H_W.reshape(bs, tg_D*tg_H*tg_W)
                bev_embed_bs_DHW_C = iter_encoder(
                    update_bev_embed_DHW_bs_C,
                    feat_flatten,
                    feat_flatten,
                    bev_z=tg_D,
                    bev_h=tg_H,
                    bev_w=tg_W,
                    bev_pos=None,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    prev_bev=None,
                    shift=shift,
                    topk_mask=update_topk_bs_DHW,
                    **kwargs
                )
                update_bev_embed_bs_DHW_C = update_bev_embed_bs_C_DHW.permute(0, 2, 1)
                if self.topK_method != 'no_cross_atten':
                    bev_embed_bs_DHW_C = bev_embed_bs_DHW_C + update_bev_embed_bs_DHW_C
                else:
                    bev_embed_bs_DHW_C = update_bev_embed_bs_DHW_C
                bev_embed_bs_C_D_H_W = bev_embed_bs_DHW_C.permute(0, 2, 1).view(bs, -1, tg_D, tg_H, tg_W)
                if self.topK_method != 'no_conv':
                    res_bs_C_D_H_W = self.iter_decoders[iter_i](bev_embed_bs_C_D_H_W)
                    bev_embed_bs_C_D_H_W = bev_embed_bs_C_D_H_W + res_bs_C_D_H_W
                bev_embed_bs_W_H_D_C = bev_embed_bs_C_D_H_W.permute(0,4,3,2,1)
                outputs_bs_W_H_D_C = self.iter_predicters[iter_i](bev_embed_bs_W_H_D_C)
                outputs = outputs_bs_W_H_D_C
                # previous binary seg, last semantic seg
                if iter_i != len(self.iter_encoders)-1:
                    bev_embed_list.append(bev_embed_bs_W_H_D_C)
                    outputs_list.append(outputs_bs_W_H_D_C)

            extra['bev_embed_list'] = bev_embed_list
            extra['outputs_list'] = outputs_list
        
        elif self.use_conv_decoder:
            """
            If the `use_conv_decoder` flag is set to True, the BEV features are processed using the conventional convolutional decoder.
            The BEV features are reshaped and passed through the decoder and predicter. 
            """

            bev_embed = bev_embed_bs_DHW_C # [1, 40000, 256]
            total_z = self.total_z
            bev_embed = bev_embed.permute(0, 2, 1).view(bs, -1, bev_h, bev_w) # [1, 40000, 256] -> [1, 256, 200, 200]
            outputs = self.decoder(bev_embed) # [1, 256, 200, 200]
            outputs = outputs.view(bs, -1, self.total_z, bev_h, bev_w).permute(0,4,3,2,1).contiguous() # [bs, c, z, h, w] -> [bs, w, h, z, c]
            outputs = outputs.reshape(bs * bev_w * bev_h * total_z, -1) # [640000, 16]
            outputs = self.predicter(outputs) # [640000, 16]
            outputs = outputs.view(bs, bev_w, bev_h, total_z, -1) # [1, 200, 200, 16, 16]

        else:
            assert NotImplementedError

        return bev_for_history, outputs, extra
