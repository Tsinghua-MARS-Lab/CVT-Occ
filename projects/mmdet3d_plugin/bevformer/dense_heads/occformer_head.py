# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.utils import TORCH_VERSION, digit_version

from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import force_fp32, auto_fp16
from projects.mmdet3d_plugin.models.utils.bricks import run_time
import numpy as np
import mmcv
import cv2 as cv
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet.models.utils import build_transformer
from mmdet.models.builder import build_loss
from mmcv.runner import BaseModule, force_fp32

@HEADS.register_module()
class OccFormerHead(BaseModule):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 loss_occ=None,
                 use_mask=False,
                 positional_encoding=None,
                 **kwargs):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.num_classes=kwargs['num_classes']
        self.use_mask=use_mask

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]



        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1
        super(OccFormerHead, self).__init__()

        self.loss_occ = build_loss(loss_occ)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims

        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)
    # def _init_layers(self):
    #     """Initialize classification branch and regression branch of head."""
    #     cls_branch = []
    #     for _ in range(self.num_reg_fcs):
    #         cls_branch.append(Linear(self.embed_dims, self.embed_dims))
    #         cls_branch.append(nn.LayerNorm(self.embed_dims))
    #         cls_branch.append(nn.ReLU(inplace=True))
    #     cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
    #     fc_cls = nn.Sequential(*cls_branch)
    #
    #     reg_branch = []
    #     for _ in range(self.num_reg_fcs):
    #         reg_branch.append(Linear(self.embed_dims, self.embed_dims))
    #         reg_branch.append(nn.ReLU())
    #     reg_branch.append(Linear(self.embed_dims, self.code_size))
    #     reg_branch = nn.Sequential(*reg_branch)
    #
    #     def _get_clones(module, N):
    #         return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    #
    #     # last reg_branch is used to generate proposal from
    #     # encode feature map when as_two_stage is True.
    #     num_pred = (self.transformer.decoder.num_layers + 1) if \
    #         self.as_two_stage else self.transformer.decoder.num_layers
    #
    #     if self.with_box_refine:
    #         self.cls_branches = _get_clones(fc_cls, num_pred)
    #         self.reg_branches = _get_clones(reg_branch, num_pred)
    #     else:
    #         self.cls_branches = nn.ModuleList(
    #             [fc_cls for _ in range(num_pred)])
    #         self.reg_branches = nn.ModuleList(
    #             [reg_branch for _ in range(num_pred)])
    #
    #     if not self.as_two_stage:
    #         self.bev_embedding = nn.Embedding(
    #             self.bev_h * self.bev_w, self.embed_dims)
    #         self.query_embedding = nn.Embedding(self.num_query,
    #                                             self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        # if self.loss_cls.use_sigmoid:
        #     bias_init = bias_init_with_prob(0.01)
        #     for m in self.cls_branches:
        #         nn.init.constant_(m[-1].bias, bias_init)

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas, prev_bev=None, only_bev=False, test=False):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder.
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = None
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        else:
            outputs = self.transformer(
                mlvl_feats,
                bev_queries,
                object_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=None,  # noqa:E501
                cls_branches=None,
                img_metas=img_metas,
                prev_bev=prev_bev
            )
        bev_embed, occ_outs = outputs
        # bev_embed, hs, init_reference, inter_references = outputs
        #
        #
        # outs = {
        #     'bev_embed': bev_embed,
        #     'all_cls_scores': outputs_classes,
        #     'all_bbox_preds': outputs_coords,
        #     'enc_cls_scores': None,
        #     'enc_bbox_preds': None,
        # }

        # if test:
        #     return bev_embed, occ_outs
        # else:
        #     return occ_outs
        outs = {
            'bev_embed': bev_embed,
            'occ':occ_outs,
        }

        return outs

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             # gt_bboxes_list,
             # gt_labels_list,
             voxel_semantics_list,
             mask_camera_list,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):

        loss_dict=dict()
        occ=preds_dicts['occ']
        assert voxel_semantics_list.min()>=0 and voxel_semantics_list.max()<=17
        losses = self.loss_single(voxel_semantics_list,mask_camera_list,occ)
        loss_dict['loss_occ']=losses
        return loss_dict

    def loss_single(self,voxel_semantics,mask_camera,preds_dicts):
        if self.use_mask:
            voxel_semantics=voxel_semantics.reshape(-1)
            preds_dicts=preds_dicts.reshape(-1,self.num_classes)
            mask_camera=mask_camera.reshape(-1)
            num_total_samples=mask_camera.sum()
            loss_occ=self.loss_occ(preds_dicts,voxel_semantics,mask_camera, avg_factor=num_total_samples)
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds_dicts = preds_dicts.reshape(-1, self.num_classes)
            loss_occ = self.loss_occ(preds_dicts, voxel_semantics,)
        return loss_occ

    @force_fp32(apply_to=('preds_dicts'))
    def get_occ(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        # return self.transformer.get_occ(
        #     preds_dicts, img_metas, rescale=rescale)
        # print(img_metas[0].keys())
        occ_out=preds_dicts['occ']
        occ_score=occ_out.softmax(-1)
        occ_score=occ_score.argmax(-1)


        return occ_score
