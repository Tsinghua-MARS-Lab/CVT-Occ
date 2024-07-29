# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import copy
import torch
from mmcv.runner import auto_fp16, force_fp32
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask

@DETECTORS.register_module()
class CVTOccWaymo(MVXTwoStageDetector):
    def __init__(self, 
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 clip_backbone=None,
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
                 queue_length=1,
                 sampled_queue_length=1,
                 sample_num=None,
                 save_results=False,
                 use_temporal=None, 
                 sample_policy_test=None, 
                 **kwargs):
        super(CVTOccWaymo, self).__init__(pts_voxel_layer, 
                                             pts_voxel_encoder,
                                             pts_middle_encoder, 
                                             pts_fusion_layer,
                                             img_backbone, 
                                             pts_backbone, 
                                             img_neck, 
                                             pts_neck,
                                             pts_bbox_head, 
                                             img_roi_head, 
                                             img_rpn_head,
                                             train_cfg, 
                                             test_cfg, 
                                             pretrained)

        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.queue_length = queue_length
        self.sample_num = sample_num # used in test
        self.sampled_queue_length = sampled_queue_length
        assert self.sampled_queue_length == len(self.sample_num), "sampled_queue_length should equal to len(sample_num)"
        self.video_test_mode = video_test_mode
        self.save_results = save_results
        self.use_temporal = use_temporal
        self.prev_frame_info = {
            'prev_bev_list': [],
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
            'prev_img_metas_list': [],
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
                          valid_mask, 
                          cur_img_metas,
                          prev_bev_list=[],
                          prev_img_metas=[],
                          **kwargs):
        """Forward training function.
        Args:
            multi_level_feats (list[torch.Tensor]): Multi level img_feats.
            voxel_semantics (torch.Tensor): Occupancy ground truth.
            cur_img_metas (list[dict]): Meta information of samples. It has length of batch_size. 
            prev_bev_list (list[torch.Tensor]): BEV features of previous frames.
                                                Each has shape (bs, bev_h*bev_w=40000, embed_dims=256). 
            prev_img_metas (list[dict[dict]]): Meta information of previous samples. 
        Returns:
            losses (dict): Losses of each branch.
        """

        outs = self.pts_bbox_head(multi_level_feats, 
                                  cur_img_metas, 
                                  prev_bev_list, 
                                  prev_img_metas, 
                                  only_bev=False, 
                                  **kwargs)
        losses = self.pts_bbox_head.loss(voxel_semantics, valid_mask, preds_dicts=outs)

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
                
                prev_bev = self.pts_bbox_head(img_feats, 
                                              img_metas, 
                                              prev_bev_list, 
                                              prev_img_metas=None, # useless
                                              only_bev=True)
                prev_bev_list.append(prev_bev)

            self.train()

            return prev_bev_list

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self, img=None,
                      voxel_semantics=None,
                      valid_mask=None,
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
            valid_mask (torch.Tensor): unified boolean mask for visible voxel
                                       with shape (bs, bev_w, bev_h, total_z).
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
            prev_img = prev_img.reshape(bs*prev_len_queue, num_cams, C, H, W)
            with torch.no_grad():
                prev_img_feats = self.extract_feat(img=prev_img, len_queue=prev_len_queue)

            prev_img_metas = []
            for each_batch in img_metas_deepcopy:
                each_batch.pop(len_queue-1)
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
        losses_pts = self.forward_pts_train(cur_img_feats, 
                                            voxel_semantics, 
                                            valid_mask,
                                            cur_img_metas, 
                                            prev_bev_list, 
                                            prev_img_metas,
                                            **kwargs)
        losses.update(losses_pts)

        return losses

    def forward_test(self, img=None,
                    img_metas=None, 
                    voxel_semantics=None,
                    valid_mask=None,
                    **kwargs):
        '''
        Forward inference function.
        Args: 
            (all arg are be wrapped one more list. after we take it out, the type of each parameter are below)
            img (torch.Tensor): Images of each sample with shape (bs, n_views, C, H, W).
            img_metas (list[dict]): len is bs. 
            voxel_semantics (torch.Tensor): Occupancy ground truth with shape (bs, bev_h, bev_w, total_z).
            valid_mask (torch.Tensor): unified boolean mask for visible voxel with shape (bs, bev_w, bev_h, total_z).
        Returns: 
            If self.save_result is true. We will save the occ_result for visualization. 
            voxel_semantics (numpy.ndarray): 
            voxel_semantics_preds (numpy.ndarray): Occupancy semantics prediction. The same shape. 
            valid_mask (numpy.ndarray): unified boolean mask for visible voxel. The same shape. 
            sample_idx (int): The index of the sample.
        '''

        # Step 1: prepare the input
        # all arg are be wrapped one more list, so we need to take the first element
        if img is not None: img = img[0]
        if voxel_semantics is not None: voxel_semantics = voxel_semantics[0]
        if valid_mask is not None: valid_mask = valid_mask[0]
        if img_metas is not None: img_metas = img_metas[0] # list[dict] of length 1
        
        # If the input frame is in a new scene, the prev_frame_info need to be reset.
        scene_token = img_metas[0]['sample_idx'] // 1000
        if scene_token != self.prev_frame_info['scene_token']:
            self.prev_frame_info['prev_bev_list'] = []
            self.prev_frame_info['prev_img_metas_list'] = []
            # update idx
            self.prev_frame_info['scene_token'] = scene_token

        if not self.video_test_mode:
            # defalut value of self.video_test_mode is True
            self.prev_frame_info['prev_bev_list'] = []
            self.prev_frame_info['prev_img_metas_list'] = []

        # Step 2: prepare prev_bev_list, prev_img_metas_list
        prev_bev_list = self.prev_frame_info['prev_bev_list']
        len_queue = len(prev_bev_list)
        prev_img_metas_list = self.prev_frame_info['prev_img_metas_list']
        assert len(prev_bev_list) == len(prev_img_metas_list), "len(prev_bev_list) should equal to len(prev_bev_list)"
        
        # Step 3: Get the delta of ego position and angle between two timestamps.
        # tmp_pos = copy.deepcopy(img_metas[0]['can_bus'][:3])
        # tmp_angle = copy.deepcopy(img_metas[0]['can_bus'][-1])

        # if len_queue > 0:
        #     img_metas[0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
        #     img_metas[0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        # else:
        #     img_metas[0]['can_bus'][-1] = 0
        #     img_metas[0]['can_bus'][:3] = 0

        # Step 4: sample the previous BEV features and img_metas
        index_list = []
        for i in self.sample_num[:-1]:
            if len_queue - i < 0: continue 
            # If the index is out of range, then it will be skipped. 
            # Therefore the total length of index_list will be sometimes shorted. 
            index_list.append(len_queue - i)
                
        # prepare sampled_prev_bev_list, sampled_prev_img_metas_list
        sampled_prev_bev_list = []
        sampled_prev_img_metas_list = []
        if len_queue > 0:
            for index in index_list:
                sampled_prev_bev_list.append(prev_bev_list[index])
                sampled_prev_img_metas_list.append(prev_img_metas_list[index])
        
        if len(sampled_prev_img_metas_list) == 0 and self.use_temporal is not None:
            sampled_prev_img_metas_list.append(img_metas[0].copy())
            sampled_prev_bev_list.append(torch.zeros([1, 40000, 256], device='cuda', dtype=torch.float32))

        # change sampled_prev_img_metas_list into a list[dict[dict]]
        sampled_prev_img_metas_DICT = {}
        sampled_prev_img_metas_list_len = len(sampled_prev_img_metas_list)
        for i in range(sampled_prev_img_metas_list_len):
            sampled_prev_img_metas_DICT[self.sampled_queue_length - 1 - sampled_prev_img_metas_list_len + i] = sampled_prev_img_metas_list[i]
            # if sampled_queue_length - 1 = sampled_prev_img_metas_list_len, then the key will be 0, 1, 2, ..., len_queue-2
            # if sampled_queue_length - 1 > sampled_prev_img_metas_list_len, then the key will be len_queue - 1 - sampled_prev_bev_list_len, ..., len_queue-2
        # if `sampled_prev_img_metas_list` is empty, then `sampled_prev_img_metas_dict` will be an empty dict. 
 
        # Step 5: forward
        outs, voxel_semantics_preds = self.simple_test(img, img_metas, 
                                                       sampled_prev_bev_list, 
                                                       prev_img_metas=[sampled_prev_img_metas_DICT],
                                                       **kwargs)
        
        new_prev_bev = outs['bev_embed']
        extra = outs['extra']
        
        # Step 6: update the self.prev_frame_info
        # During inference, we save the BEV features and ego motion of each timestamp.
        # self.prev_frame_info['prev_pos'] = tmp_pos
        # self.prev_frame_info['prev_angle'] = tmp_angle
        if self.queue_length > 1:
            if len (prev_bev_list) > (self.queue_length - 2):
                del prev_bev_list[0]
            prev_bev_list.append(new_prev_bev)
            self.prev_frame_info['prev_bev_list'] = prev_bev_list

            if len(prev_img_metas_list) > (self.queue_length - 2):
                del prev_img_metas_list[0]
            prev_img_metas_list.append(img_metas[0])
            self.prev_frame_info['prev_img_metas_list'] = prev_img_metas_list
        
        # Step 7: save the results (controlled by `self.save_results`)
        # If you want to do visualization, you can set `self.save_results` to True.
        # If you want to do evaluation, you can set `self.save_results` to False.
        if self.save_results:
            occ_results = {
                "voxel_semantics": voxel_semantics.to(torch.uint8).cpu().numpy(),
                "voxel_semantics_preds": voxel_semantics_preds.to(torch.uint8).cpu().numpy(),
                "mask": valid_mask.to(torch.uint8).cpu().numpy(),
                "sample_idx": img_metas[0]['sample_idx'],
            }

        else:
            occ_results = self.pts_bbox_head.eval_metrics(voxel_semantics, voxel_semantics_preds, valid_mask)
            sample_idx = img_metas[0]['sample_idx']
            scene_id = sample_idx % 1000000 // 1000
            occ_results['scene_id'] = scene_id
            frame_id = sample_idx % 1000
            occ_results['frame_id'] = frame_id
            
        return occ_results
    
    def simple_test(self, img, 
                    img_metas, 
                    prev_bev_list=[], 
                    prev_img_metas=[], 
                    **kwargs):
        """
        Test function without augmentaiton.
        Args:
            img (torch.Tensor): Images of each sample with shape (bs, n_views, C, H, W).
            img_metas (list[dict]): Meta information of each sample.
            prev_bev_list (list[torch.Tensor]): BEV features of previous frames.
                                                Each has shape (bs, bev_h*bev_w=40000, embed_dims=256). 
            prev_img_metas (list[dict[dict]]): Meta information of previous samples.
        Returns:
            outs (dict): with keys "bev_embed, occ, extra"
            occ (torch.Tensor): Occupancy semantics prediction.
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