# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import torch
import torch.nn as nn
from mmdet.models import HEADS
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models.builder import build_loss
from mmcv.runner import BaseModule, force_fp32
from mmcv.cnn import PLUGIN_LAYERS, Conv2d, Conv3d, ConvModule, caffe2_xavier_init
from mmseg.models import LOSSES as LOSSES_SEG
from ..modules.unet import MYASPPHead

@HEADS.register_module()
class SOLOOccHeadWaymo(BaseModule):
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
                 FREE_LABEL=None,
                 embed_dims=256,
                 bev_z=1,
                 bev_h=200,
                 bev_w=200,
                 total_z=16,
                 num_classes=16,
                 use_infov_mask=True,
                 use_lidar_mask=False,
                 use_camera_mask=True, 
                 act_cfg=dict(type='ReLU',inplace=True),
                 norm_cfg=dict(type='BN',),
                 loss_occ=None,
                 **kwargs):
        self.FREE_LABEL = FREE_LABEL
        self.embed_dims=embed_dims
        self.bev_z = bev_z
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.total_z = total_z
        self.fp16_enabled = False
        self.num_classes = num_classes
        self.use_infov_mask = use_infov_mask
        self.use_lidar_mask = use_lidar_mask
        self.use_camera_mask = use_camera_mask

        super(SOLOOccHeadWaymo, self).__init__()
        self.loss_occ_fun = dict()
        for name, loss_dict in loss_occ.items():
            if LOSSES_SEG.get(loss_dict['type']) is not None:
                self.loss_occ_fun['loss_occ_' + name] = LOSSES_SEG.build(loss_dict)
            else:
                _type = loss_dict['type']
                raise KeyError(f'{_type} not in LOSSES_SEG registry')

        use_bias = norm_cfg is None
        self.decoder = []
        conv_cfg = dict(type='Conv2d')
        conv_num = 3
        # conv module
        decoder_layers = []
        for _ in range(conv_num):
            decoder_layers.append(
                ConvModule(
                    self.embed_dims, # 256
                    self.embed_dims, # 256
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
            )   # 256 -> 256
        # aspp
        decoder_layers.append(
            MYASPPHead(
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
            )   # 256 -> 256
        )
        # deconv to origin shape
        _out_dim = self.embed_dims*4
        decoder_layers.append(
            ConvModule(
                self.embed_dims,
                _out_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            ) # 256 -> 256 * 4
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.predicter = nn.Sequential(
            nn.Linear(_out_dim//self.total_z, self.num_classes*2), # 256 * 4 // 16 -> 32 
            nn.Softplus(),
            nn.Linear(self.num_classes*2, self.num_classes),
        )
        self.embed_dims = self.embed_dims

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, bev_feats, **kwargs):
        """
        Forward function of occupancy head of solofusion.
        Args:
            bev_feats (torch.Tensor): shape (bs, embed_dims=256, bev_h, bev_w)
        Returns:
            outs (dict): Output results.
            - occ (torch.Tensor): shape (bs, bev_w, bev_h, total_z, num_classes)
        """
        bs, _, _, _ = bev_feats.shape
        occ_out = self.decoder(bev_feats) # (bs, embed_dim * 4, h, w)
        occ_out = occ_out.permute(0, 3, 2, 1) # (bs, w, h, embed_dim * 4)
        occ_out = occ_out.reshape(bs, self.bev_w, self.bev_h, self.total_z, -1) # (bs, w, h, z, channels * 4)
        occ_out = occ_out.reshape(bs * self.bev_w * self.bev_h * self.total_z, -1) # (bs * w * h * z, channels * 4)
        occ_out = self.predicter(occ_out) # (bs * w * h * z, channels * 4)
        occ_out = occ_out.reshape(bs, self.bev_w, self.bev_h, self.total_z, self.num_classes) # (bs, w, h, z, num_classes)
        outs = {'occ': occ_out}

        return outs

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             voxel_semantics,
             valid_mask,
             preds_dicts,
             **kwargs,
             ):
        '''
        Loss function of occupancy head of solofusion.
        Args:
            voxel_semantics (torch.Tensor): 3D occupancy ground truth. shape (B, H, W, Z)
            valid_mask (torch.Tensor): mask of valid area. shape (B, H, W, Z)
            preds_dicts (dict): output of forward function.
            - occ (torch.Tensor): shape (B, W, H, Z, num_classes)
        Returns:
            - loss_dict (dict): loss of occupancy head.
        '''
        loss_dict = dict()
        occ = preds_dicts['occ']
        loss_dict = self.loss_single(voxel_semantics, valid_mask, occ)

        return loss_dict
    

    def get_loss(self,loss_occ_fun, cls_score, labels, weight=None):
        assert labels.max() <= (self.num_classes - 1) and labels.min() >= 0, f"score out of range: {labels.max()} vs {labels.min()}"
        assert cls_score.shape[0] == labels.shape[0], f"shape mismatch: {cls_score.shape} vs {labels.shape}"

        loss_occ = dict()
        for loss_name in sorted(list(loss_occ_fun.keys())):
            if 'lovasz' in loss_name:
                cls_score = cls_score.reshape(*cls_score.shape, 1, 1)
                labels = labels.reshape(*labels.shape, 1, 1)
            _loss = loss_occ_fun[loss_name](cls_score, labels, weight)
            loss_occ[loss_name] = _loss

        return loss_occ

    def loss_single(self, 
                    voxel_semantics, 
                    valid_mask, 
                    occ_preds, 
                    **kwargs
                    ):
        valid_mask = valid_mask.reshape(-1) # (bs, w, h, z) -> (bs*w*h*z, )
        valid_mask = valid_mask.bool()
        occ_preds = occ_preds.reshape(-1, self.num_classes) # (bs*w*h*z, num_classes)
        voxel_semantics = voxel_semantics.reshape(-1) # (bs*w*h*z, )
        loss_ce = self.loss_ce = self.get_loss(self.loss_occ_fun, occ_preds[valid_mask], voxel_semantics[valid_mask])

        return loss_ce #,loss_lovasz

    @force_fp32(apply_to=('preds_dicts'))
    def get_occ(self, preds_dicts):
        """
        Generate Occupancy semantics prediction.
        Args:
            preds_dicts (dict): with keys "bev_embed, occ, extra"
            occ (torch.Tensor): Predicted occupancy features with shape (bs, w, h, total_z, c). 
        Returns:
            occ_label (torch.Tensor): Occupancy semantics prediction with shape (bs, w, h, total_z).
        """

        occ_out = preds_dicts['occ']
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

    def eval_metrics(self, voxel_semantics, voxel_semantics_preds, valid_mask=None):
        """
        Evaluation.
        Args:
            voxel_semantics (torch.Tensor): semantic occpuancy ground truth.
            voxel_semantics_preds (torch.Tensor): predicted semantic occpuancy.
            valid_mask (torch.Tensor): 1 represent valid voxel, 0 represent invalid voxel. Directly get from the data loader. 
            all of them have shape (bs, w, h, total_z)
        Returns: 
            count_matrix (numpy.ndarray): count_matrix[i][j] counts the number of voxel with gt type i and pred type j. 
                                          shape (num_classes, num_classes)
        """

        masked_semantics_gt = voxel_semantics[valid_mask]
        masked_semantics_pred = voxel_semantics_preds[valid_mask]
        count_matrix = self.compute_count_matrix(gtocc=masked_semantics_gt, predocc=masked_semantics_pred)

        # use count matrix is the same
        # gt_count = torch.sum(count_matrix, dim=1)
        # pred_count = torch.sum(count_matrix, dim=0)

        occ_results = {"count_matrix": count_matrix.cpu().numpy(),}

        return occ_results