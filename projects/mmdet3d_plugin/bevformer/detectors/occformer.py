# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import copy
import numpy as np
import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask

@DETECTORS.register_module()
class CVTOcc(MVXTwoStageDetector):
    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False,
                 queue_length=None,
                 save_results=False,
                 **kwargs):
        super(CVTOcc,
              self).__init__(pts_voxel_layer, 
                             pts_voxel_encoder,
                             pts_middle_encoder, 
                             pts_fusion_layer,
                             img_backbone, 
                             pts_backbone, 
                             img_neck, pts_neck,
                             pts_bbox_head, 
                             img_roi_head, 
                             img_rpn_head,
                             train_cfg, 
                             test_cfg, 
                             pretrained)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.save_results = save_results

        # temporal
        self.queue_length = queue_length
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev_list': [],
            'prev_img_metas_list': [],
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

    def extract_img_feat(self, img, len_queue=None):
        """Extract features of images.
        Args: 
            img (torch.Tensor): Image tensor with shape (bs, n_views, C, H, W).
                                But for previous img, its shape will be (bs*len_queue, n_views, C, H, W).
            len_queue (int): The length of the queue. It is less or equal to self.queue_length.
                             It is used when extracting features of previous images.
        Returns:
            list[torch.Tensor]: Image features. Each with shape (bs, n_views, C, H, W).
                                But different scales (from FPN) will have different shapes.
                                For previous img, its shape will be (bs, len_queue, n_views, C, H, W).
        """

        bs_length, num_views, C, H, W = img.size()
        bs_length_num_views = bs_length * num_views
        img = img.reshape(bs_length_num_views, C, H, W)

        if self.use_grid_mask:
            img = self.grid_mask(img)
        img_feats = self.img_backbone(img)
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            bs_length_num_views, C, H, W = img_feat.size()
            if len_queue is not None: # for prev imgs
                bs = int(bs_length / len_queue)
                img_feats_reshaped.append(img_feat.view(bs, len_queue, num_views, C, H, W))
            else: # for current imgs
                img_feats_reshaped.append(img_feat.view(bs_length, num_views, C, H, W))

        return img_feats_reshaped

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, len_queue=None):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, len_queue=len_queue)
        return img_feats


    def forward_pts_train(self, multi_level_feats,
                          voxel_semantics,
                          mask_camera,
                          mask_lidar,
                          cur_img_metas,
                          prev_bev_list=[],
                          prev_img_metas=[],
                          **kwargs):
        """
        Forward function
        Args:
            multi_level_feats (list[torch.Tensor]): Multi level img_feats.
            voxel_semantics (torch.Tensor): Occupancy ground truth.
            mask_camera (torch.Tensor): Camera mask.
            mask_lidar (torch.Tensor): Lidar mask.
            cur_img_metas (list[dict]): Meta information of samples. It has length of batch_size. 
            prev_bev_list (list[torch.Tensor]): BEV features of previous frames.
                                                Each has shape (bs, bev_h*bev_w=40000, embed_dims=256). 
            prev_img_metas (list[dict[dict]]): Meta information of previous samples. 
        Returns:
            losses (dict): Losses of each branch.
        """

        # use the occupancy head to get the occupancy output
        outs = self.pts_bbox_head(multi_level_feats, 
                                  cur_img_metas, 
                                  prev_bev_list, 
                                  prev_img_metas,
                                  only_bev=False,
                                  **kwargs)
        
        # calculate the loss
        losses = self.pts_bbox_head.loss(voxel_semantics,
                                         outs,
                                         mask_camera,
                                         mask_lidar,
                                         **kwargs)

        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """
        Calls either forward_train or forward_test depending on whether return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested 
        (i.e. torch.Tensor and list[dict]), 
        and when `resturn_loss=False`, img and img_metas should be 
        double nested (i.e.  list[torch.Tensor], list[list[dict]]), 
        with the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def obtain_history_bev(self, 
                           prev_img_feats_list=[], 
                           prev_img_metas=[], 
                           prev_len_queue=0):
        """
        Obtain history BEV features iteratively. 
        To save GPU memory, gradients are not calculated.
        Args: 
            prev_img_feats_list (list[torch.Tensor]): The list has length eqauls to the scales. 
                                                      Each tensor has shape: bs, prev_len_queue, n_views, C, H, W.
            prev_img_metas (list[dict[dict]]): Meta information of each sample.
                                               The list has length of batch size.
                                               The dict has keys 0, 1, 2, ..., len-2. The element of each key is a dict.
            prev_len_queue (int): The length of the queue - 1.
        Returns:
            prev_bev_list (list[torch.Tensor]): Each has shape ([bs, bev_h*bev_w=40000, embed_dims=256]).
        """

        self.eval()
        prev_bev_list = []
        with torch.no_grad():
            for i in range(prev_len_queue):
                img_feats = [each_scale[:, i, ...] for each_scale in prev_img_feats_list]
                img_metas = [each_batch[i] for each_batch in prev_img_metas] # list[dict] of length equals to batch_size
                if img_metas[0]['prev_bev_exists'] is not True: # HERE assume batch size = 1
                    prev_bev_list = []
                
                prev_bev = self.pts_bbox_head(multi_level_feats=img_feats, 
                                              cur_img_metas=img_metas, 
                                              prev_bev_list=prev_bev_list, 
                                              prev_img_metas=None, # useless
                                              only_bev=True)
                prev_bev_list.append(prev_bev)

            self.train()

            return prev_bev_list

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self, img=None,
                      voxel_semantics=None,
                      mask_lidar=None,
                      mask_camera=None,
                      img_metas=None,
                      **kwargs):
        """
        Forward training function.
        Args:
            img_metas (list[dict[dict]]): Meta information of each sample.
                                          The list has length of batch size.
                                          The dict has keys 0, 1, 2, ..., len-1. The element of each key is a dict.
            img (torch.Tensor): Images of each sample with shape (bs, len, n_views, C, H, W).
            voxel_semantics (torch.Tensor): Occupancy ground truth 
                                            with shape (bs, bev_h, bev_w, total_z).
            mask_camera (torch.Tensor): Camera mask with shape (bs, bev_w, bev_h, total_z).
            mask_lidar (torch.Tensor): Lidar mask with shape (bs, bev_w, bev_h, total_z).
        Returns:
            losses (dict): Losses of different branches.
        """

        # Step 1: prepare cur_img_feats and cur_img_metas
        batch_size, len_queue, _, _, _, _ = img.shape
        cur_img = img[:, -1, ...]
        cur_img_feats = self.extract_feat(img=cur_img) # list[tensor], each tensor is of shape (B, N, C, H, W). H and W are different across scales. 
        img_metas_deepcopy = copy.deepcopy(img_metas)
        cur_img_metas = [each_batch[len_queue-1] for each_batch in img_metas] # list[dict] of length equals to batch_size

        # Step 2: prepare prev_bev_list, prev_img_metas
        if cur_img_metas[0]['prev_bev_exists']:
            prev_img = img[:, :-1, ...]
            bs, prev_len_queue, num_cams, C, H, W = prev_img.shape
            prev_img = prev_img.reshape(bs * prev_len_queue, num_cams, C, H, W)
            with torch.no_grad():
                prev_img_feats = self.extract_feat(img=prev_img, len_queue=prev_len_queue)

            prev_img_metas = []
            for each_batch in img_metas_deepcopy:
                each_batch.pop(len_queue - 1)
                prev_img_metas.append(each_batch) # list[dict[dict]]
                
            prev_bev_list = self.obtain_history_bev(prev_img_feats, prev_img_metas, prev_len_queue)

            # Step 3: adjust the length of these two to be consistent
            prev_bev_list_len = len(prev_bev_list)
            for each_batch in prev_img_metas:
                if len(each_batch) > prev_bev_list_len:
                    for i in range(0, len(each_batch) - prev_bev_list_len): # len(each_batch) = len_queue - 1
                        each_batch.pop(i)
        else:
            prev_bev_list = []
            prev_img_metas = [{} for _ in range(batch_size)]

        # Step 4: forward in head to get losses
        losses = dict()
        losses_pts = self.forward_pts_train(multi_level_feats=cur_img_feats,
                                            voxel_semantics=voxel_semantics,
                                            mask_camera=mask_camera,
                                            mask_lidar=mask_lidar,
                                            cur_img_metas=cur_img_metas,
                                            prev_bev_list=prev_bev_list,
                                            prev_img_metas=prev_img_metas,
                                            **kwargs)

        losses.update(losses_pts)
        return losses

    def forward_test(self,  
                     img_metas,
                     img=None,
                     voxel_semantics=None,
                     mask_camera=None,
                     mask_lidar=None,
                     **kwargs):
        '''
        Forward inference function.
        Args: 
            (all arg are be wrapped one more list. after we take it out, the type of each parameter are below)
            img (torch.Tensor): Images of each sample with shape (bs, n_views, C, H, W).
            img_metas (list[dict]): len is bs. 
            voxel_semantics (torch.Tensor): Occupancy ground truth with shape (bs, bev_h, bev_w, total_z).
            mask_camera (torch.Tensor): Camera mask with shape (bs, bev_w, bev_h, total_z).
            mask_lidar (torch.Tensor): Lidar mask with shape (bs, bev_w, bev_h, total_z).
        Returns: 
            If self.save_result is true. We will save the occ_result for visualization. 
            voxel_semantics (numpy.ndarray): 
            voxel_semantics_preds (numpy.ndarray): Occupancy semantics prediction. The same shape. 
            valid_mask (numpy.ndarray): unified boolean mask for visible voxel. The same shape. 
            sample_idx (int): The index of the sample.
        '''

        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(name, type(var)))
            
        # Step 1: prepare the input
        # all arg are be wrapped one more list, so we need to take the first element
        if img is not None: img = img[0]
        if voxel_semantics is not None: voxel_semantics = voxel_semantics[0]
        if mask_camera is not None: mask_camera = mask_camera[0]
        if mask_lidar is not None: mask_lidar = mask_lidar[0]
        if img_metas is not None: img_metas = img_metas[0]

        # If the input frame is in a new scene, the prev_frame_info need to be reset.
        if img_metas[0]['scene_token'] != self.prev_frame_info['scene_token']:
            self.prev_frame_info['prev_bev_list'] = []
            self.prev_frame_info['prev_img_metas_list'] = []
            # update idx
            self.prev_frame_info['scene_token'] = img_metas[0]['scene_token']

        if not self.video_test_mode:
            # defalut value of self.video_test_mode is True
            self.prev_frame_info['prev_bev_list'] = []
            self.prev_frame_info['prev_img_metas_list'] = []

        # Step 2: Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0]['can_bus'][-1])
        if len(self.prev_frame_info['prev_bev_list']) > 0:
            img_metas[0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0]['can_bus'][-1] = 0
            img_metas[0]['can_bus'][:3] = 0

        # Step 3: prepare prev_bev_list, prev_img_metas_list
        if len(self.prev_frame_info['prev_bev_list']) > 0:
            prev_bev_list = self.prev_frame_info['prev_bev_list']
            prev_img_metas_list = self.prev_frame_info['prev_img_metas_list']
        else:
            prev_bev = torch.zeros([1, 40000, 256], device=img.device, dtype=img.dtype)
            prev_bev_list = [prev_bev]
            prev_img_metas_list = [img_metas[0].copy()]

        # convert the list to dict TODO
        prev_img_metas_list_len = len(prev_img_metas_list)
        prev_img_metas_dict = {}
        for i in range(prev_img_metas_list_len):
            prev_img_metas_dict[self.queue_length - 1 - prev_img_metas_list_len + i] = prev_img_metas_list[i]
            # from 0 to self.queue_length - 2

        # Step 4: forward in head to get occ_results
        outs, occ_results = self.simple_test(img_metas,
                                             img,
                                             prev_bev_list=prev_bev_list,
                                             prev_img_metas=[prev_img_metas_dict],
                                             **kwargs)
        
        # Step 5: During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        new_prev_bev = outs['bev_embed']

        if self.queue_length > 1:
            if len (prev_bev_list) > (self.queue_length - 2):
                del prev_bev_list[0]
            prev_bev_list.append(new_prev_bev)
            self.prev_frame_info['prev_bev_list'] = prev_bev_list

            if len(prev_img_metas_list) > (self.queue_length - 2):
                del prev_img_metas_list[0]
            prev_img_metas_list.append(img_metas[0])
            self.prev_frame_info['prev_img_metas_list'] = prev_img_metas_list

        if self.save_results:
            results = {
                        'occ_pred': occ_results.cpu().numpy().astype(np.uint8),
                        'occ_gt': voxel_semantics,
                        'mask_camera': mask_camera,
                    }
        else:
            results = self.pts_bbox_head.eval_metrics(occ_results, voxel_semantics, mask_camera)
            scene_idx = img_metas[0]['scene_idx']
            frame_idx = img_metas[0]['frame_idx']
            results['scene_id'] = scene_idx
            results['frame_id'] = frame_idx

            
        return results

    def simple_test(self, 
                    img_metas, 
                    img=None, 
                    prev_bev_list=[],
                    prev_img_metas=[],
                    **kwargs):
        """
        Test function without augmentaiton.
        Args:
            img_metas (list[dict]): Meta information of each sample.
            img (torch.Tensor): Images of each sample with shape (bs, n_views, C, H, W).
            prev_bev_list (list[torch.Tensor]): BEV features of previous frames. 
                                                Each has shape (bs, bev_h*bev_w, embed_dims).
            prev_img_metas (list[dict[dict]]): Meta information of previous samples.
        Returns:
            new_prev_bev (torch.Tensor): BEV features of the current frame with shape (bs, bev_h*bev_w, embed_dims). 
            occ (torch.Tensor): Predicted occupancy with shape (bs, bev_h, bev_w, total_z).
        """
        multi_level_feats = self.extract_feat(img=img)
        outs = self.pts_bbox_head(multi_level_feats,
                                  img_metas, 
                                  prev_bev_list,
                                  prev_img_metas,
                                  only_bev=False,
                                  **kwargs) 
        occ = self.pts_bbox_head.get_occ(outs)

        return outs, occ
