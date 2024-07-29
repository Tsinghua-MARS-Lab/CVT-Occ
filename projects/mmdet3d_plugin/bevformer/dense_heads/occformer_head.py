# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import BaseModule, force_fp32
from mmdet.models.utils import build_transformer
from mmdet.models.builder import build_loss
from mmdet.models import HEADS
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet.models import LOSSES as mmdet_LOSSES

@HEADS.register_module()
class CVTOccHead(BaseModule):
    def __init__(self,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 bev_h=200,
                 bev_w=200,
                 num_classes=18,
                 occ_thr=0.3,
                 loss_occ=None,
                 use_camera_mask=False,
                 use_lidar_mask=False,
                 use_free_mask=False,
                 use_focal_loss=False,
                 positional_encoding=None,
                 use_refine_feat_loss=False,
                 refine_feat_loss_weight=None,
                 **kwargs):
        super(CVTOccHead, self).__init__()
        self.use_camera_mask = use_camera_mask
        self.use_lidar_mask = use_lidar_mask
        self.use_free_mask = use_free_mask
        self.use_focal_loss = use_focal_loss
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.occ_thr = occ_thr
        self.fp16_enabled = False
        self.num_classes = num_classes
        if use_free_mask:
            self.num_classes = self.num_classes - 1
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_occ = build_loss(loss_occ)
        self.use_refine_feat_loss = use_refine_feat_loss
        _loss=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0) # refine_feat_w not sigmoid, so here use_sigmoid=True
        self.refine_feat_loss_func = mmdet_LOSSES.build(_loss)
        self.refine_feat_loss_weight = refine_feat_loss_weight

        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims
        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()

    @auto_fp16(apply_to=('multi_level_feats'))
    def forward(self,
                multi_level_feats, 
                cur_img_metas, 
                prev_bev_list=[], 
                prev_img_metas=[],
                only_bev=False,
                **kwargs):
        """
        Forward function.
        Args:
            multi_level_feats (list[torch.Tensor]): Current multi level img features from the upstream network.
                                                    Each is a 5D-tensor img_feats with shape (bs, num_cams, embed_dims, h, w).
            cur_img_metas (list[dict]): Meta information of each sample. The list has length of batch size.
            prev_bev_list (list[torch.Tensor]): BEV features of previous frames. Each has shape (bs, bev_h*bev_w, embed_dims). 
            prev_img_metas (list[dict[dict]]): Meta information of each sample.
                                               The list has length of batch size.
                                               The dict has keys len_queue-1-prev_bev_list_len, ..., len_queue-2. 
                                               The element of each key is a dict.
                                               So each dict has length of prev_bev_list_len. 
            only_bev: If this flag is true. The head only computes BEV features with encoder.
        Returns:
            If only_bev:
            _bev_embed (torch.Tensor): BEV features of the current frame with shape (bs, bev_h*bev_w, embed_dims). 
            else: 
            outs (dict): with keys "bev_embed, occ, extra".
            - bev_embed (torch.Tensor): BEV features of the current frame with shape (bs, bev_h*bev_w, embed_dims).
            - occ (torch.Tensor): Predicted occupancy features with shape (bs, w, h, total_z, c).
            - extra (dict): extra information. if 'costvolume' in self.transformer, it will have 'refine_feat_w' key.
        """

        # Step 1: initialize BEV queries and mask
        bs = multi_level_feats[0].shape[0]
        dtype = multi_level_feats[0].dtype
        bev_queries = self.bev_embedding.weight.to(dtype)
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w), device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        # Step 2: get BEV features
        if only_bev:
            if len(prev_bev_list) == 0:
                prev_bev = None
            else:
                prev_bev = prev_bev_list[-1]

            outputs = self.transformer.get_bev_features(multi_level_feats,
                                                        bev_queries,
                                                        bev_pos,
                                                        cur_img_metas,
                                                        prev_bev,
                                                        **kwargs)
            _bev_embed = outputs['bev_embed']
            
            return _bev_embed

        else:
            outputs = self.transformer(multi_level_feats,
                                       bev_queries,
                                       bev_pos,
                                       cur_img_metas,
                                       prev_bev_list,
                                       prev_img_metas,
                                       **kwargs)
            
            bev_for_history, occ_outs, extra = outputs
            outs = {'bev_embed': bev_for_history, 'occ':occ_outs, 'extra':extra}

            return outs

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, voxel_semantics,
             preds_dicts,
             mask_camera=None,
             mask_lidar=None,
             **kwargs):
        '''
        Loss function. 
        Args:
            voxel_semantics (torch.Tensor): Shape (bs, w, h, total_z)
            valid_mask (torch.Tensor): 1 represent valid voxel, 0 represent invalid voxel. 
                                       Directly get from the data loader. shape (bs, w, h, total_z)
            preds_dicts (dict): result from head with keys "bev_embed, occ, extra".
            - occ (torch.Tensor): Predicted occupancy features with shape (bs, w, h, total_z, c). 
        Returns:
            loss_dict (dict): Losses of different branch. 
                              Default cvtocc model has refine_feat_loss loss and loss_occ_coheam loss. 
        '''

        loss_dict = dict()
        occ = preds_dicts['occ']
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= self.num_classes-1, "semantic gt out of range"
        losses = self.loss_single(voxel_semantics, mask_camera, occ)
        loss_dict['loss_occ'] = losses

        extra = preds_dicts['extra']
        if self.use_refine_feat_loss:
            if 'refine_feat_w' in extra: # has the key means it will not be None
                refine_feat_w = extra['refine_feat_w']
                loss_dict['refine_feat_loss'] = self.get_refine_feat_loss(voxel_semantics, refine_feat_w, mask_camera, mask_lidar)
            else:
                loss_dict['refine_feat_loss'] = occ.reshape(-1).sum() * 0

        return loss_dict
    
    def get_refine_feat_loss(self, voxel_semantics, refine_feat_w, mask_camera=None, mask_lidar=None):
        """
        Calculate refine_feat_loss from refine_feat_w
        Args:
            refine_feat_w (torch.Tensor): The weight without sigmoid. shape (bev_w, bev_h, total_z, 2).
        Returns:
            refine_feat_loss (float): 
        """
        
        # Step 1: reshape refine_feat_w 
        refine_feat_w = refine_feat_w.reshape(-1, refine_feat_w.shape[-1]) # (w*h*total_z, 2)

        # Step 2: get the ground truth for refine feat weight from the occupancy ground truth. 
        refine_feat_gt = (voxel_semantics != self.num_classes-1)
        refine_feat_gt = refine_feat_gt.reshape(-1).long()

        # Step 3: use `mask_camera` and `mask_lidar` to filter out the invalid points
        mask = torch.ones_like(voxel_semantics)
        if self.use_lidar_mask:
            mask = torch.logical_and(mask, mask_lidar)
        if self.use_camera_mask:
            mask = torch.logical_and(mask, mask_camera)

        mask = mask.reshape(-1)
        refine_feat_w_masked = refine_feat_w[mask]
        refine_feat_gt_masked = refine_feat_gt[mask]

        # Step 4: calculate the loss
        refine_feat_loss = self.refine_feat_loss_func(refine_feat_w_masked, refine_feat_gt_masked)
        refine_feat_loss = self.refine_feat_loss_weight * refine_feat_loss

        return refine_feat_loss

    def loss_single(self, voxel_semantics, mask_camera, preds_dicts):
        if self.use_camera_mask:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds_dicts = preds_dicts.reshape(-1,self.num_classes)
            mask_camera = mask_camera.reshape(-1)
            num_total_samples = mask_camera.sum()
            loss_occ = self.loss_occ(preds_dicts,
                                     voxel_semantics,
                                     mask_camera, 
                                     avg_factor=num_total_samples)
            
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds_dicts = preds_dicts.reshape(-1, self.num_classes)
            if self.use_free_mask:
                free_mask = voxel_semantics < self.num_classes
                voxel_semantics = voxel_semantics[free_mask]
                preds_dicts = preds_dicts[free_mask]
                pos_num = voxel_semantics.shape[0]

            else:
                pos_num = voxel_semantics.shape[0]

            loss_occ = self.loss_occ(preds_dicts, voxel_semantics.long(), avg_factor=pos_num)

        return loss_occ
    
    @force_fp32(apply_to=('preds_dicts'))
    def get_occ(self, preds_dicts):
        """
        Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
        Returns:
            occ_out (torch.Tensor): Predicted occupancy map with shape (bs, h, w, z).
        """

        occ_out = preds_dicts['occ']
        if self.use_focal_loss:
            # Default value of this flag is False
            occ_out = occ_out.sigmoid()

        if self.use_free_mask:
            # Default value of this flag is False
            bs, h, w, z, channels = occ_out.shape
            occ_out = occ_out.reshape(bs, -1, self.num_classes)
            occ_out = torch.cat((occ_out, torch.ones_like(occ_out)[:,:, :1] * self.occ_thr), dim=-1)
            occ_out = occ_out.reshape(bs,h,w,z,-1)
        else:
            occ_out = occ_out.softmax(-1)

        occ_out = occ_out.argmax(-1)

        return occ_out

    def compute_count_matrix(self, gtocc, predocc):
        """
        Calculate count matrix.
        Args:
            voxel_semantics (torch.Tensor): semantic occpuancy ground truth.
            voxel_semantics_preds (torch.Tensor): predicted semantic occpuancy.
            both input are masked
        Returns:
            count_matrix (numpy.ndarray): count_matrix[i][j] counts the number of voxel with gt type i and pred type j. shape (num_classes, num_classes)
        """

        n_cl = self.num_classes
        count_matrix = torch.zeros((n_cl, n_cl), device='cuda')
        correct_idx = (gtocc >= 0) & (gtocc < n_cl)
        count_matrix = torch.bincount(n_cl * gtocc[correct_idx].to(torch.int) + predocc[correct_idx].to(torch.int), 
                                        weights=None, minlength=n_cl ** 2).reshape(n_cl, n_cl)
        
        return count_matrix

    def eval_metrics(self, voxel_semantics, voxel_semantics_preds, camera_mask):
        """
        Evaluation.
        Args:
            voxel_semantics (torch.Tensor): semantic occpuancy ground truth.
            voxel_semantics_preds (torch.Tensor): predicted semantic occpuancy.
            camera_mask (torch.Tensor): camera mask.
            all of them have shape (bs, w, h, total_z)
        Returns: 
            results (dict): with key "count_matrix".
            - count_matrix (numpy.ndarray): count_matrix[i][j] counts the number of voxel with gt type i and pred type j. shape (num_classes, num_classes)
        """

        masked_semantics_gt = voxel_semantics[camera_mask]
        masked_semantics_pred = voxel_semantics_preds[camera_mask]
        count_matrix = self.compute_count_matrix(gtocc=masked_semantics_gt, predocc=masked_semantics_pred)

        results = {"count_matrix": count_matrix.cpu().numpy()}

        return results