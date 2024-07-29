# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import BaseModule, force_fp32
from mmdet.models import HEADS
from mmdet.models.utils import build_transformer
from mmdet.models.builder import build_loss
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmdet.models import LOSSES as mmdet_LOSSES
from mmseg.models import LOSSES as LOSSES_SEG
from sklearn.neighbors import NearestNeighbors

@HEADS.register_module()
class CVTOccHeadWaymo(BaseModule):
    def __init__(self,
                 volume_flag=False,
                 with_box_refine=False,
                 as_two_stage=False,
                 voxel_size=None,
                 occ_voxel_size=None,
                 use_larger=True,
                 transformer=None,
                 bev_h=200,
                 bev_w=200,
                 bev_z=1,
                 num_classes=None,
                 loss_occ=None,
                 loss_binary_occ=None,
                 use_CDist=False,
                 CLASS_NAMES=None,
                 positional_encoding=None,
                 use_refine_feat_loss=False,
                 refine_feat_loss_weight=None,
                 **kwargs):
        super(CVTOccHeadWaymo, self).__init__()
        if not volume_flag: assert bev_z == 1
        self.volume_flag = volume_flag
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z
        self.fp16_enabled = False
        self.num_classes = num_classes
        self.use_CDist = use_CDist
        self.CLASS_NAMES = CLASS_NAMES
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        
        self.voxel_size = voxel_size
        self.occ_voxel_size = occ_voxel_size
        self.use_larger = use_larger
        self.loss_occ_fun = dict()

        for name, loss_dict in loss_occ.items():
            if LOSSES_SEG.get(loss_dict['type']) is not None:
                self.loss_occ_fun['loss_occ_' + name] = LOSSES_SEG.build(loss_dict)
            else:
                _type = loss_dict['type']
                raise KeyError(f'{_type} not in LOSSES_SEG registry')
            
        if loss_binary_occ is not None:
            self.loss_binary_occ_func = dict()
            for name, loss_dict in loss_binary_occ.items():
                if LOSSES_SEG.get(loss_dict['type']) is not None:
                    self.loss_binary_occ_func['loss_occ_' + name] = LOSSES_SEG.build(loss_dict)
                else:
                    _type = loss_dict['type']
                    raise KeyError(f'{_type} not in LOSSES_SEG registry')
        
        self.use_refine_feat_loss = use_refine_feat_loss
        _loss=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0) # refine_feat_w not sigmoid, so here use_sigmoid=True
        self.refine_feat_loss_func = mmdet_LOSSES.build(_loss)
        self.refine_feat_loss_weight = refine_feat_loss_weight

        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims

        if not self.as_two_stage:
            # self.as_two_stage default value is False
            self.bev_embedding = nn.Embedding(self.bev_z * self.bev_h * self.bev_w, self.embed_dims)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()

    @auto_fp16(apply_to=('multi_level_feats'))
    def forward(self, 
                multi_level_feats, 
                img_metas, 
                prev_bev_list=[], 
                prev_img_metas=[],
                only_bev=False, 
                **kwargs,):
        """
        Forward function.
        Args:
            multi_level_feats (list[torch.Tensor]): Current multi level img features from the upstream network.
                                                    Each is a 5D-tensor img_feats with shape (bs, num_cams, embed_dims, h, w).
            img_metas (list[dict]): Meta information of each sample. The list has length of batch size.
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
        """

        # Step 1: initialize BEV queries and mask
        bs = multi_level_feats[0].shape[0]
        dtype = multi_level_feats[0].dtype
        bev_queries = self.bev_embedding.weight.to(dtype)
        if self.volume_flag:
            bev_mask = torch.zeros((bs, self.bev_z, self.bev_h, self.bev_w),
                                   device=bev_queries.device).to(dtype)
        else:
            bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                                   device=bev_queries.device).to(dtype)
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
                                                        img_metas,
                                                        prev_bev,
                                                        **kwargs,)
            _bev_embed = outputs['bev_embed']
            
            return _bev_embed

        else:
            outputs = self.transformer(multi_level_feats,
                                       bev_queries,
                                       bev_pos,
                                       img_metas,
                                       prev_bev_list,
                                       prev_img_metas,
                                       **kwargs,)
            
            bev_for_history, occ_outs, extra = outputs
            outs = {'bev_embed': bev_for_history, 'occ':occ_outs, 'extra': extra}

            return outs

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, voxel_semantics, 
             valid_mask, 
             preds_dicts,
             **kwargs):
        """
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
        """
        loss_dict = dict()
        occ_outs = preds_dicts['occ']
        loss_dict = self.loss_single(voxel_semantics, valid_mask, occ_outs, binary_loss=False) 
        extra = preds_dicts['extra']
        # if 'outputs_list' in extra:
        #     assert False, "not implemented"
        #     pred_list = extra['outputs_list']
        #     for iter_i, preds in enumerate(pred_list):
        #         losses = self.loss_single(voxel_semantics, mask_infov, mask_lidar, mask_camera, preds, binary_loss=True)
        #         for k,v in losses.items():
        #             loss_dict['loss_occ_iter{}_{}'.format(iter_i, k)] = v

        if self.use_refine_feat_loss:
            if 'refine_feat_w' in extra: # has the key means it will not be None
                refine_feat_w = extra['refine_feat_w']
                loss_dict['refine_feat_loss'] = self.get_refine_feat_loss(voxel_semantics, refine_feat_w, valid_mask)
            else:
                loss_dict['refine_feat_loss'] = occ_outs.reshape(-1).sum() * 0

        return loss_dict

    def get_refine_feat_loss(self, voxel_semantics, refine_feat_w, valid_mask):
        """
        Calculate refine_feat_loss from refine_feat_w
        Args:
            refine_feat_w (torch.Tensor): The weight without sigmoid. shape (bev_w, bev_h, total_z, 2).
        Returns:
            refine_feat_loss (float): 
        """
        
        # Step 1: reshape refine_feat_w 
        refine_feat_w = refine_feat_w.unsqueeze(0)
        channels = refine_feat_w.shape[-1]
        refine_feat_w = refine_feat_w.reshape(-1, channels)

        # Step 2: get the ground truth for refine feat weight from the occupancy ground truth. 
        refine_feat_gt = (voxel_semantics != self.num_classes-1)
        refine_feat_gt = refine_feat_gt.reshape(-1).long()

        # Step 3: use valid_mask to filter out the invalid points
        valid_mask = valid_mask.reshape(-1)
        refine_feat_w_masked = refine_feat_w[valid_mask]
        refine_feat_gt_masked = refine_feat_gt[valid_mask]

        # Step 4: calculate the loss
        refine_feat_loss = self.refine_feat_loss_func(refine_feat_w_masked, refine_feat_gt_masked)
        refine_feat_loss = self.refine_feat_loss_weight * refine_feat_loss

        return refine_feat_loss 
    
    def get_loss(self, loss_occ_fun, cls_score, labels, valid_mask, weight=None):
        """
        Calculate multiple losses using different loss functions.
        Args:
            loss_occ_fun (dict): A dictionary containing different loss functions.
                                The keys are loss function names, and the values are the corresponding loss functions.
                                Each loss function should accept the following arguments:
                                - cls_score: The predicted scores or logits from the model.
                                - labels: The ground-truth labels for the data.
                                - weight: Optional, a weighting tensor to apply to the loss. Default is None.
                                - avg_factor: Optional, a scalar factor to normalize the loss. Default is None.
            cls_score (torch.Tensor): The predicted scores or logits from the model.
                                    It should have a shape of (N, C, *), where N is the batch size,
                                    C is the number of classes, and * denotes additional dimensions.
                                    cls_score is pred[mask] (N, )
            labels (torch.Tensor): The ground-truth labels for the data.
                                    It should have a shape of (N, *), where N is the batch size,
                                    and * denotes additional dimensions. The values should be integers
                                    representing class indices.
                                    labels is voxel_semantics[mask](N, )
            weight (torch.Tensor, optional): Optional, a weighting tensor to apply to the loss.
                                            It should have a shape that is broadcastable to the shape of `cls_score`.
                                            Default is None, which means no weighting is applied.
            avg_factor (int or None, optional): Optional, a scalar factor to normalize the loss.
                                                If 'focal' is in the loss function names, this factor
                                                represents the number of positive samples (mask.sum()).
                                                For other loss functions, the default is None, which means
                                                no normalization is performed.

        Returns:
            dict: A dictionary containing the computed loss values for each loss function.
                The keys are the loss function names, and the values are the corresponding loss values.
        """

        loss_occ = dict()
        for loss_name in sorted(list(loss_occ_fun.keys())):
            if 'focal' in loss_name:
                avg_factor = valid_mask.sum()
            else:
                avg_factor = None
            if 'lovasz' in loss_name:
                cls_score = cls_score.reshape(*cls_score.shape, 1, 1)
                labels = labels.reshape(*labels.shape, 1, 1)
            
            _loss = loss_occ_fun[loss_name](cls_score, labels, weight, avg_factor=avg_factor)
            loss_occ[loss_name] = _loss
            
        return loss_occ # dict, key is loss_name, value is loss
    
    def loss_single(self, voxel_semantics, valid_mask, occ_outs, binary_loss=False):
        """
        Args:
            voxel_semantics (torch.Tensor): 
            occ_outs (torch.Tensor): Predicted occupancy features with shape (bs, w, h, total_z, c)
        Returns:
            loss_occ (dict): A dictionary containing the computed loss values for each loss function.
                             The keys are the loss function names, and the values are the corresponding loss values.
                             Default loss_occ_coheam loss
        """

        if binary_loss:
            assert occ_outs.shape[-1] == 2
            binary_gt = voxel_semantics != self.num_classes-1
            bs, W, H, D = voxel_semantics.shape
            _bs, _W, _H, _D, _ = occ_outs.shape
            assert W % _W == 0 and H % _H == 0 and D % _D == 0
            scale_W, scale_H, scale_D = W//_W, H//_H, D//_D
            
            _scale = 1
            while _scale != scale_W:
                binary_gt = binary_gt.reshape(bs, -1, 2, H,  D)
                binary_gt = torch.logical_or(binary_gt[:, :, 0, :, :], binary_gt[:, :, 1, :, :, :])
                _scale *= 2
            _scale = 1
            while _scale != scale_H:
                binary_gt = binary_gt.reshape(bs, _W,  -1,  2, D)
                binary_gt = torch.logical_or(binary_gt[:, :, :, 0, :], binary_gt[:, :, :, 1, :])
                _scale *= 2
            _scale = 1
            while _scale != scale_D:
                binary_gt = binary_gt.reshape(bs, _W,  _H,  -1, 2)
                binary_gt = torch.logical_or(binary_gt[:, :, :, :, 0], binary_gt[:, :, :, :, 1])
                _scale *= 2
            binary_gt = binary_gt.long()
            binary_gt=binary_gt.reshape(-1)
            occ_outs=occ_outs.reshape(-1, 2)
            mask=torch.ones_like(binary_gt, dtype=torch.bool)
            loss_occ = self.get_loss(self.loss_binary_occ_func, occ_outs[mask], binary_gt[mask], mask)
        else:
            voxel_semantics=voxel_semantics.reshape(-1)
            occ_outs = occ_outs.reshape(-1, self.num_classes)
            valid_mask = valid_mask.reshape(-1)
            loss_occ = self.get_loss(self.loss_occ_fun, 
                                     cls_score=occ_outs[valid_mask], 
                                     labels=voxel_semantics[valid_mask], 
                                     valid_mask=valid_mask)

        return loss_occ

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
        occ_out=preds_dicts['occ']
        occ_score=occ_out.softmax(-1)
        occ_label=occ_score.argmax(-1)

        return occ_label
    
    def compute_CDist(self, gtocc, predocc, mask):
        alpha = 1/3  # Hyperparameter
        
        # Squeeze dimensions
        gtocc = gtocc.squeeze(0)
        predocc = predocc.squeeze(0)
        mask = mask.squeeze(0)
        
        # Use mask to change unobserved into 16 (out of range)
        gtocc = torch.where(mask, gtocc, torch.ones_like(gtocc) * self.num_classes)
        predocc = torch.where(mask, predocc, torch.ones_like(predocc) * self.num_classes)
        
        # Get all unique class labels
        labels_tensor = torch.unique(torch.cat((gtocc, predocc), dim=0))
        labels_list = labels_tensor.tolist()
        labels_list = [x for x in labels_list if x < (self.num_classes-1)] # skip free type
        
        CDist_tensor = torch.zeros((self.num_classes-1), device='cuda')
        for label in labels_list:
            
            # Extract points for the current class
            labeled_gtocc = torch.nonzero(gtocc == label).float()  # (N_1, 3)
            labeled_predocc = torch.nonzero(predocc == label).float() # (N_2, 3)
            
            if labeled_gtocc.shape[0] == 0 or labeled_predocc.shape[0] == 0:
                # CDist_tensor[label] = 2
                CDist_tensor[label] = labeled_gtocc.shape[0] + labeled_predocc.shape[0]
                continue

            # convert tensor to numpy
            labeled_gtocc_np = labeled_gtocc.cpu().numpy()
            labeled_predocc_np = labeled_predocc.cpu().numpy()

            # Use sklearn's NearestNeighbors to find nearest neighbors
            reference_gt = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(labeled_gtocc_np)
            reference_pred = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(labeled_predocc_np)

            dist_pred_to_gt, _ = reference_gt.kneighbors(labeled_predocc_np)
            dist_gt_to_pred, _ = reference_pred.kneighbors(labeled_gtocc_np)

            dist_pred_to_gt = torch.from_numpy(dist_pred_to_gt).squeeze().to('cuda')
            dist_gt_to_pred = torch.from_numpy(dist_gt_to_pred).squeeze().to('cuda')
            
            exp_dist1 = 1 - torch.exp(-dist_pred_to_gt * alpha)
            exp_dist2 = 1 - torch.exp(-dist_gt_to_pred * alpha)
            chamfer_distance = torch.sum(exp_dist1) + torch.sum(exp_dist2)
            
            CDist_tensor[label] = chamfer_distance.item()
            
        return CDist_tensor

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

    def eval_metrics(self, voxel_semantics, voxel_semantics_preds, valid_mask):
        """
        Evaluation.
        Args:
            voxel_semantics (torch.Tensor): semantic occpuancy ground truth.
            voxel_semantics_preds (torch.Tensor): predicted semantic occpuancy.
            valid_mask (torch.Tensor): 1 represent valid voxel, 0 represent invalid voxel. Directly get from the data loader. 
            all of them have shape (bs, w, h, total_z)
        Returns: 
            count_matrix (numpy.ndarray): count_matrix[i][j] counts the number of voxel with gt type i and pred type j. shape (num_classes, num_classes)
            CDist_tensor (numpy.ndarray): CDist_tensor[i] is the chamfer distance for class i. (without free type) 
        """
        
        # Step 1: compute chamfer distance (controlled by `self.use_CDist`)
        if self.use_CDist:
            CDist_tensor = self.compute_CDist(gtocc=voxel_semantics, predocc=voxel_semantics_preds, mask=valid_mask)
        else:
            CDist_tensor = torch.zeros((self.num_classes-1), device='cuda')
        
        # Step 2: compute mIoU
        masked_semantics_gt = voxel_semantics[valid_mask]
        masked_semantics_pred = voxel_semantics_preds[valid_mask]
        count_matrix = self.compute_count_matrix(gtocc=masked_semantics_gt, predocc=masked_semantics_pred)

        # Step 3: count dict
        # use count matrix is the same
        # gt_count = torch.sum(count_matrix, dim=1)
        # pred_count = torch.sum(count_matrix, dim=0)

        occ_results = {"CDist_tensor": CDist_tensor.cpu().numpy(),
                       "count_matrix": count_matrix.cpu().numpy(),}

        return occ_results
