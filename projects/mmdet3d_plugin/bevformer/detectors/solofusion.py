import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.runner import force_fp32, auto_fp16
# from mmcv.ops.nms import batched_nms
from mmdet.models import DETECTORS
from mmdet3d.models import builder
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from san import tools as san_tools
from detectron2.engine import DefaultTrainer
from detectron2.checkpoint import DetectionCheckpointer
from .bevdet_solo import BEVDet_solo

def generate_forward_transformation_matrix(img_meta_dict):
    # res = torch.eye(3)

    # if 'transformation_3d_flow' in img_meta_dict:
    #     for transform_type in img_meta_dict['transformation_3d_flow']:
    #         if transform_type == "R":
    #             if "pcd_rotation" in img_meta_dict:
    #                 res = img_meta_dict['pcd_rotation'].T @ res # .T since L158 of lidar_box3d has points @ rot
    #         elif transform_type == "S":
    #             if "pcd_scale_factor" in img_meta_dict:
    #                 res = res * img_meta_dict['pcd_scale_factor']
    #         elif transform_type == "T":
    #             if "pcd_trans" in img_meta_dict:
    #                 assert torch.tensor(img_meta_dict['pcd_trans']).abs().sum() == 0, \
    #                     "I'm not supporting translation rn; need to convert to hom coords which is annoying"
    #         elif transform_type == "HF": # Horizontal is Y apparently
    #             if "pcd_horizontal_flip" in img_meta_dict:
    #                 tmp = torch.eye(3)
    #                 tmp[1, 1] = -1
    #                 res = tmp @ res
    #         elif transform_type == "VF":
    #             if "pcd_vertical_flip" in img_meta_dict:
    #                 tmp = torch.eye(3)
    #                 tmp[0, 0] = -1
    #                 res = tmp @ res
    #         else:
    #             raise Exception(str(img_meta_dict))

    # For now, there is no data augmentation
    hom_res = torch.eye(4)
    return hom_res

@DETECTORS.register_module()
class SOLOFusion(BEVDet_solo):
    def __init__(self, 
                 pre_process=None, 
                 pre_process_neck=None, 
                 input_sample_policy=None,
                 do_history=True,
                 interpolation_mode='bilinear',
                 history_cat_num=1, # Number of history frames to cat
                 history_queue_length=1,
                 history_cat_conv_out_channels=None,
                 FREE_LABEL=23, 
                 num_classes=16,
                 do_history_stereo_fusion=False,
                 stereo_neck=None,
                 history_stereo_prev_step=1,
                 **kwargs):
        super(SOLOFusion, self).__init__(**kwargs)
        
        self.FREE_LABEL=FREE_LABEL
        self.num_classes=num_classes
        self.input_sample_policy=input_sample_policy
        #### Prior to history fusion, do some per-sample pre-processing.
        self.single_bev_num_channels = self.img_view_transformer.numC_Trans

        # Lightweight MLP
        self.embed = nn.Sequential(
            nn.Conv2d(self.single_bev_num_channels, self.single_bev_num_channels, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(self.single_bev_num_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.single_bev_num_channels, self.single_bev_num_channels, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(self.single_bev_num_channels),
            nn.ReLU(inplace=True))

        # Preprocessing like BEVDet4D
        self.pre_process = pre_process is not None
        if self.pre_process:
            self.pre_process_net = builder.build_backbone(pre_process)

        #### Deal with history
        self.do_history = do_history
        # if self.do_history:
        self.interpolation_mode = interpolation_mode
        self.history_queue_length = history_queue_length # 30
        self.queue_length = self.history_queue_length + 1
        self.history_cat_num = history_cat_num
        self.sample_interval = self.history_queue_length // self.history_cat_num # 30 / 6 = 5
        self.sample_index = [(i+1) * self.sample_interval - 1 for i in range(self.history_cat_num)]
            # [4, 9, 14, 19, 24, 29]

        self.history_cam_sweep_freq = 0.5 # seconds between each frame
        history_cat_conv_out_channels = (history_cat_conv_out_channels 
                                            if history_cat_conv_out_channels is not None 
                                            else self.single_bev_num_channels)
            # Embed each sample with its relative temporal offset with current timestep
        self.history_keyframe_time_conv = nn.Sequential(
                nn.Conv2d(self.single_bev_num_channels + 1,
                        self.single_bev_num_channels,
                        kernel_size=1,
                        padding=0,
                        stride=1),
                nn.BatchNorm2d(self.single_bev_num_channels),
                nn.ReLU(inplace=True))

            # Then concatenate and send them through an MLP.
        self.history_keyframe_cat_conv = nn.Sequential(
                nn.Conv2d(self.single_bev_num_channels * (self.history_cat_num + 1),
                        history_cat_conv_out_channels,
                        kernel_size=1,
                        padding=0,
                        stride=1),
                nn.BatchNorm2d(history_cat_conv_out_channels),
                nn.ReLU(inplace=True))

        self.history_sweep_time = None

        self.history_bev = None
        self.history_seq_ids = None
        self.history_forward_augs = None
        self.history_global_to_lidar = None
            
        self.prev_frame_info = {
            'prev_bev_list': [],
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
            'prev_img_metas_list': [],
            'prev_stereo_feats_list': [], 
            'prev_sweep_time_list': [],}

        #### Stereo depth fusion
        self.do_history_stereo_fusion = do_history_stereo_fusion
        if self.do_history_stereo_fusion:
            self.stereo_neck = stereo_neck
            if self.stereo_neck is not None:
                self.stereo_neck = builder.build_neck(self.stereo_neck)
            self.history_stereo_prev_step = history_stereo_prev_step

        self.prev_stereo_img_feats = None # B x N x C x H x W
        self.prev_stereo_global_to_img = None # B x N x 4 x 4
        self.prev_stereo_img_forward_augs = None

        self.fp16_enabled = False

    @auto_fp16()
    def image_encoder(self, img):
        '''
        Encoder for image features.
        Args:
            img (torch.Tensor): Image tensor. shape (B, N, C=3, H, W)
        Returns:
            neck_feats (torch.Tensor): Image features. shape (B, N, output_dim, ouput_H, output_W)
            stereo_feats (torch.Tensor): Stereo features. shape (B, N, C, H, W)
        '''

        # Step 1: use image backbone to extract image features
        B, N, C, imH, imW = img.shape
        img = img.view(B * N, C, imH, imW)
        backbone_feats = self.img_backbone(img)
        
        # Step 2: use image neck
        neck_feats = self.img_neck(backbone_feats)
        if isinstance(neck_feats, list):
            assert len(neck_feats) == 1 # SECONDFPN returns a length-one list
            neck_feats = neck_feats[0]
            
        _, output_dim, ouput_H, output_W = neck_feats.shape
        neck_feats = neck_feats.view(B, N, output_dim, ouput_H, output_W)

        # Step 3: use stereo_necks to extract stereo features
        if self.do_history_stereo_fusion:
            backbone_feats_detached = [tmp.detach() for tmp in backbone_feats]
            stereo_feats = self.stereo_neck(backbone_feats_detached)
            if isinstance(stereo_feats, list):
                assert len(stereo_feats) == 1 # SECONDFPN returns a trivial list
                stereo_feats = stereo_feats[0]
            stereo_feats = F.normalize(stereo_feats, dim=1, eps=self.img_view_transformer.stereo_eps)
            return neck_feats, stereo_feats.view(B, N, *stereo_feats.shape[1:])

        else:
            return neck_feats, None

    @force_fp32()
    def get_depth_loss(self, depth_gt, depth):
        """
        This was updated to be more similar to BEVDepth's original depth loss function.
        """
        B, N, H, W = depth_gt.shape
        fg_mask = (depth_gt != 0).view(-1) 
        depth_gt = (depth_gt - self.img_view_transformer.grid_config['dbound'][0])\
                /self.img_view_transformer.grid_config['dbound'][2]
        depth_gt = torch.clip(torch.floor(depth_gt), 0,
                            self.img_view_transformer.D).to(torch.long)
        assert depth_gt.max() < self.img_view_transformer.D

        depth_gt_logit = F.one_hot(depth_gt.reshape(-1),
                                   num_classes=self.img_view_transformer.D)
        depth_gt_logit = depth_gt_logit.reshape(B, N, H, W,
                                                self.img_view_transformer.D).permute(
            0, 1, 4, 2, 3).to(torch.float32) # B x N x D x H x W
        depth = depth.view(B, N, self.img_view_transformer.D, H, W).softmax(dim=2)
        
        depth_gt_logit = depth_gt_logit.permute(0, 1, 3, 4, 2).view(-1, self.img_view_transformer.D)
        depth = depth.permute(0, 1, 3, 4, 2).contiguous().view(-1, self.img_view_transformer.D)

        loss_depth = (F.binary_cross_entropy(
                depth[fg_mask],
                depth_gt_logit[fg_mask],
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum()))
        loss_depth = self.img_view_transformer.loss_depth_weight * loss_depth
        return loss_depth

    @force_fp32(apply_to=('rots', 'trans', 'intrins', 'post_rots', 'post_trans'))
    def process_stereo_before_fusion(self, stereo_feats, img_metas, rots, trans, intrins, post_rots, post_trans):
        '''
        Process stereo features before fusion.
        Args:
            stereo_feats (torch.Tensor): Stereo features. shape (B, N, C, H, W)
            img_metas (List[dict]): Meta information of each sample.
            rots (torch.Tensor): Rotation matrix. shape (B, N, 3, 3)
            trans (torch.Tensor): Translation matrix. shape (B, N, 3)
            intrins (torch.Tensor): Intrinsic matrix. shape (B, N, 3, 3)
            post_rots (torch.Tensor): Post rotation matrix. shape (B, N, 3, 3)
            post_trans (torch.Tensor): Post translation matrix. shape (B, N, 3)
        Returns: 
            self.prev_stereo_img_feats (torch.Tensor): Previous stereo image features. shape (B, self.history_stereo_prev_step, N, C, H, W)
            self.prev_stereo_global_to_img (torch.Tensor): Previous stereo global to image transformation matrix. shape (B, self.history_stereo_prev_step, N, 4, 4)
            self.prev_stereo_img_forward_augs (torch.Tensor): Previous stereo image forward augmentation matrix. shape (B, self.history_stereo_prev_step, N, 4, 4)
            global_to_img (torch.Tensor): Global to image transformation matrix. shape (B, N, 4, 4)
            img_forward_augs(cam_to_cam_aug) (torch.Tensor): Image forward augmentation matrix. shape (B, N, 4, 4)
            curr_unaug_cam_to_prev_unaug_cam (torch.Tensor): Current unaugmented camera to previous unaugmented camera transformation matrix. shape (B, N, N, 4, 4)
        '''

        # Step 1: get `start_of_sequence`, `global_to_curr_lidar_rt`, `lidar_forward_augs`
        B, N, C, H, W = stereo_feats.shape
        device = stereo_feats.device
        start_of_sequence_list = []
        for img_meta in img_metas:
            start_of_sequence_list.append(img_meta['start_of_sequence'])
        start_of_sequence = torch.BoolTensor(start_of_sequence_list).to(device)

        global_to_curr_lidar_rt_list = []
        for img_meta in img_metas:
            global_to_curr_lidar_rt_list.append(torch.tensor(img_meta['global_to_curr_lidar_rt'], device=device, dtype=torch.float32))
        global_to_curr_lidar_rt = torch.stack(global_to_curr_lidar_rt_list, dim=0)  # B x 4 x 4

        lidar_forward_augs_list = []
        for img_meta in img_metas:
            lidar_forward_augs_list.append(generate_forward_transformation_matrix(img_meta))
        lidar_forward_augs = torch.stack(lidar_forward_augs_list, dim=0).to(rots.device)

        # Step 2: get `img_forward_augs`, `intrins4x4`, `cam_to_lidar_aug`
        cam_to_cam_aug = rots.new_zeros((B, N, 4, 4))
        cam_to_cam_aug[:, :, 3, 3] = 1
        cam_to_cam_aug[:, :, :3, :3] = post_rots
        cam_to_cam_aug[:, :, :3, 3] = post_trans
        img_forward_augs = cam_to_cam_aug

        intrins4x4 = rots.new_zeros((B, N, 4, 4))
        intrins4x4[:, :, 3, 3] = 1
        intrins4x4[:, :, :3, :3] = intrins

        cam_to_lidar_aug = rots.new_zeros((B, N, 4, 4))
        cam_to_lidar_aug[:, :, 3, 3] = 1
        cam_to_lidar_aug[:, :, :3, :3] = rots
        cam_to_lidar_aug[:, :, :3, 3] = trans # indeed `sensor2ego` in waymo

        # Step 3: get `global_to_img`
        # Global -> Lidar unaug -> lidar aug -> cam space unaug -> cam xyd unaug
        global_to_img = (intrins4x4 @ torch.inverse(cam_to_lidar_aug) 
                         @ lidar_forward_augs.unsqueeze(1) @ global_to_curr_lidar_rt.unsqueeze(1)) # B x N x 4 x 4

        # Step 3: padding the first frame
        # Then, let's check if stereo saved values are none or we're the first in the sequence.
        if self.prev_stereo_img_feats is None:
            # Detach and clone the current stereo features to create a history buffer
            prev_stereo_img_feats = stereo_feats.detach().clone()
            # Reshape and repeat the stereo features to create a history buffer
            prev_stereo_img_feats = prev_stereo_img_feats.unsqueeze(1).repeat(1, self.history_stereo_prev_step, 1, 1, 1, 1)
            # B x history_stereo_prev_step x N x C x H x W
            self.prev_stereo_img_feats = prev_stereo_img_feats

            # For `global_to_img` and `img_forward_augs`, they are the same
            prev_stereo_global_to_img = global_to_img.clone()
            prev_stereo_global_to_img = prev_stereo_global_to_img.unsqueeze(1).repeat(1, self.history_stereo_prev_step, 1, 1, 1)
            # B x history_stereo_prev_step x N x 4 x 4
            self.prev_stereo_global_to_img = prev_stereo_global_to_img

            prev_stereo_img_forward_augs = img_forward_augs.clone()
            prev_stereo_img_forward_augs = prev_stereo_img_forward_augs.unsqueeze(1).repeat(1, self.history_stereo_prev_step, 1, 1, 1)
            # B x history_stereo_prev_step x N x 4 x 4
            self.prev_stereo_img_forward_augs = prev_stereo_img_forward_augs

            # self.prev_stereo_frame_idx = stereo_feats.new_zeros((B))[:, None].repeat(
            #     1, self.history_stereo_prev_step
            # ) # B x history_stereo_prev_step
        else:
            # self.prev_stereo_img_feats shape # B x history_stereo_prev_step x N x C x H x W
            # stereo_feats # B x N x C x H x W
            self.prev_stereo_img_feats[start_of_sequence] = stereo_feats[start_of_sequence].unsqueeze(1).detach().clone()
            self.prev_stereo_global_to_img[start_of_sequence] = global_to_img[start_of_sequence].unsqueeze(1).clone()
            self.prev_stereo_img_forward_augs[start_of_sequence] = img_forward_augs[start_of_sequence].unsqueeze(1).clone()

        # These are both B x N x 4 x 4. Want the result to be B x prev_N x curr_N x 4 x 4
        curr_unaug_cam_to_prev_unaug_cam = self.prev_stereo_global_to_img[:, 0][:, :, None, :, :] @ torch.inverse(global_to_img)[:, None, :, :, :]

        return (self.prev_stereo_img_feats[:, self.history_stereo_prev_step - 1], 
                self.prev_stereo_global_to_img[:, self.history_stereo_prev_step - 1], 
                self.prev_stereo_img_forward_augs[:, self.history_stereo_prev_step - 1],
                global_to_img,
                img_forward_augs,
                curr_unaug_cam_to_prev_unaug_cam)

    def process_stereo_for_next_timestep(self, stereo_feats, global_to_img, img_forward_augs):
        self.prev_stereo_img_feats[:, 1:] = self.prev_stereo_img_feats[:, :-1].clone()
        self.prev_stereo_img_feats[:, 0] = stereo_feats.detach().clone()
        self.prev_stereo_global_to_img[:, 1:] = self.prev_stereo_global_to_img[:, :-1].clone()
        self.prev_stereo_global_to_img[:, 0] = global_to_img.clone()
        self.prev_stereo_img_forward_augs[:, 1:] = self.prev_stereo_img_forward_augs[:, :-1].clone()
        self.prev_stereo_img_forward_augs[:, 0] = img_forward_augs.detach().clone()

    @force_fp32()
    def fuse_history(self, curr_bev, img_metas):
        '''
        Fuse long term history into current BEV.
        Args: 
            curr_bev (torch.Tensor): Current BEV. shape (B, base_bev_channels, H, W)
            img_metas (List[dict]): Meta information of each sample.
        Returns:
            feats_to_return (torch.Tensor): Fused BEV features. shape (B, history_cat_conv_out_channels, H, W)
        '''

        # Step 1: get `seq_ids`, `start_of_sequence`, `forward_augs`, `global_to_curr_lidar_rt`
        seq_ids = torch.LongTensor([img_meta['sequence_group_idx'] for img_meta in img_metas])
        seq_ids = seq_ids.to(curr_bev.device)

        start_of_sequence = torch.BoolTensor([img_meta['start_of_sequence'] for img_meta in img_metas])
        start_of_sequence = start_of_sequence.to(curr_bev.device)
        
        forward_augs_list = []
        for img_meta in img_metas:
            forward_augs_list.append(generate_forward_transformation_matrix(img_meta))
        forward_augs = torch.stack(forward_augs_list, dim=0).to(curr_bev.device)

        global_to_curr_lidar_rt_list = []
        for img_meta in img_metas:
            global_to_curr_lidar_rt_list.append(torch.tensor(img_meta['global_to_curr_lidar_rt'], device=curr_bev.device, dtype=torch.float32))
        global_to_curr_lidar_rt = torch.stack(global_to_curr_lidar_rt_list, dim=0)  # B x 4 x 4

        # Step 2: initialize history
        if self.history_bev is None:
            self.history_seq_ids = seq_ids.clone()
            self.history_forward_augs = forward_augs.clone()
            self.history_global_to_lidar = global_to_curr_lidar_rt.unsqueeze(1).repeat(1, self.history_queue_length, 1, 1) # B x T x 4 x 4
            # Repeat the first frame feature to be history
            self.history_bev = curr_bev.unsqueeze(1).repeat(1, self.history_queue_length, 1, 1, 1) # B x T x 80 x H x W
            # All 0s, representing current timestep.
            self.history_sweep_time = curr_bev.new_zeros(curr_bev.shape[0], self.history_queue_length) # B x T

        # Step 3: detach history
        self.history_bev = self.history_bev.detach()
        self.history_forward_augs = self.history_forward_augs.detach()
        self.history_sweep_time = self.history_sweep_time.detach()
        self.history_global_to_lidar = self.history_global_to_lidar.detach()

        assert self.history_bev.dtype == torch.float32

        # Step 4: Deal with the new sequences
        # First, sanity check. For every non-start of sequence, history id and seq id should be same.
        assert (self.history_seq_ids != seq_ids)[~start_of_sequence].sum() == 0, \
            "{}, {}, {}".format(self.history_seq_ids, seq_ids, start_of_sequence)
        
        # Step 5: Replace all the new sequences' positions in history with the curr_bev information
        self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].unsqueeze(1) # B x T x 80 x H x W
        self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
        self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]
        self.history_global_to_lidar[start_of_sequence] = global_to_curr_lidar_rt[start_of_sequence].unsqueeze(1)
        
        # Step 6: new timestep, everything in history gets pushed back one. 
        self.history_sweep_time = self.history_sweep_time + 1
        self.history_sweep_time[start_of_sequence] = 0 # zero the new sequence timestep starts

        # Step 7: Get grid idxs & grid2bev
        B, c, h, w = curr_bev.shape
        assert c == self.single_bev_num_channels, "channel dim is wrong in curr_bev"
        dtype = curr_bev.dtype
        device = curr_bev.device
        xs = torch.linspace(0, w - 1, w, dtype=dtype, device=device).view(1, w).expand(h, w)
        ys = torch.linspace(0, h - 1, h, dtype=dtype, device=device).view(h, 1).expand(h, w)
        grid = torch.stack((xs, ys, torch.ones_like(xs), torch.ones_like(xs)), -1) # H x W x 4
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1) # B x H x W x 4
        grid = grid.unsqueeze(-1) # B x H x W x 4 x 1

        # This converts BEV indices to meters
        # IMPORTANT: the feat2bev[0, 3] is changed from feat2bev[0, 2] because previous was 2D rotation
        # which has 2-th index as the hom index. Now, with 3D hom, 3-th is hom
        feat2bev = torch.zeros((4,4),dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.img_view_transformer.dx[0]
        feat2bev[1, 1] = self.img_view_transformer.dx[1]
        feat2bev[0, 3] = self.img_view_transformer.bx[0] - self.img_view_transformer.dx[0] / 2.
        feat2bev[1, 3] = self.img_view_transformer.bx[1] - self.img_view_transformer.dx[1] / 2.
        feat2bev[2, 2] = 1
        feat2bev[3, 3] = 1
        feat2bev = feat2bev.view(1,4,4)

        # Step 8: Get flow for grid sampling.
        # The flow is as follows. Starting from grid locations in curr bev, transform to BEV XY11,
        # backward of current augmentations, curr lidar to prev lidar, forward of previous augmentations,
        # transform to previous grid locations.
        
        ## global to prev lidar
        sample_history_global_to_lidar = [self.history_global_to_lidar[:, sample_i, :, :] for sample_i in self.sample_index]
        sample_history_global_to_lidar = torch.stack(sample_history_global_to_lidar, dim=1).to(curr_bev).detach() # B x cat_num x 4 x 4

        # global to curr lidar
        repeat_global_to_curr_lidar_rt = global_to_curr_lidar_rt.unsqueeze(1).repeat(1, self.history_cat_num, 1, 1) # B x cat_num x 4 x 4

        # bev to grid(lidar)
        repeat_feat2bev = feat2bev.unsqueeze(1).repeat(B, self.history_cat_num, 1, 1) # B x cat_num x 4 x 4

        # transformation between unaug lidar and aug lidar
        repeat_forward_augs = forward_augs.unsqueeze(1).repeat(1, self.history_cat_num, 1, 1) # B x cat_num x 4 x 4
        repeat_history_forward_augs = self.history_forward_augs.unsqueeze(1).repeat(1, self.history_cat_num, 1, 1) # B x cat_num x 4 x 4
        
        # curr bev -> curr grid(lidar) -> global -> prev grid(lidar) -> prev bev
        curr_feat2prev_feat = (torch.inverse(repeat_feat2bev) @ repeat_history_forward_augs @ sample_history_global_to_lidar 
                               @ torch.inverse(repeat_global_to_curr_lidar_rt) @ torch.inverse(repeat_forward_augs) @ repeat_feat2bev)
        # feat curbev global historybev history feat B x cat_num x 4 x 4
        
        # repeat for h*w and batch size
        repeat_curr_feat2prev_feat = curr_feat2prev_feat.view(B, self.history_cat_num, 1, 1, 4, 4).repeat(1, 1, h, w, 1, 1) # B x cat_num x h x w x 4 x 4
        repeat_curr_grid = grid.unsqueeze(1).repeat(1, self.history_cat_num, 1, 1, 1, 1) # B x cat_num x h x w x 4 x 1

        # apply transformation matrix, grid -> grid
        prev_grid = repeat_curr_feat2prev_feat @ repeat_curr_grid # B x cat_num x h x w x 4 x 1
        prev_grid = prev_grid.squeeze(-1) # B x cat_num x h x w x 4

        # Step 9: use the wrapped grid to gridsample the prev bev
        grid = prev_grid.reshape(B * self.history_cat_num, h, w, 4) # (B*cat_num) x h x w x 4
        normalize_factor = torch.tensor([w - 1.0, h - 1.0], dtype=dtype, device=device)
        grid = grid[:,:,:,:2] / normalize_factor.view(1, 1, 1, 2) * 2.0 - 1.0 # (B*cat_num) x h x w x 2
        sample_history_bev = [self.history_bev[:, sample_i, :, :, :] for sample_i in self.sample_index]
        sample_history_bev = torch.stack(sample_history_bev, dim=1).to(curr_bev).detach() # B x cat_num x 80 x h x w
        sample_history_bev = sample_history_bev.reshape(B * self.history_cat_num, self.single_bev_num_channels, h, w) # B*cat_num x 80 x h x w 
        sample_history_bev = F.grid_sample(sample_history_bev, grid.to(dtype), align_corners=True, mode=self.interpolation_mode) # B*cat_num x 80 x h x w
        sample_history_bev = sample_history_bev.reshape(B, self.history_cat_num, self.single_bev_num_channels, h, w) # B x cat_num x 80 x h x w
        feats_to_return = torch.cat([curr_bev.unsqueeze(1), sample_history_bev], dim=1) # B x (1 + cat_num) x 80 x H x W

        # Step 10: Update history
        # Reshape and concatenate features and timestep
        sample_sweep_time = [self.history_sweep_time[:, sample_i] for sample_i in self.sample_index]
        sample_sweep_time = torch.stack(sample_sweep_time, dim=1).detach() # B x cat_num
        sample_sweep_time_add_curr = torch.cat([self.history_sweep_time.new_zeros(self.history_sweep_time.shape[0], 1), sample_sweep_time], dim=1) # B x (cat_num+1)

        feats_to_return = torch.cat(
            [feats_to_return, sample_sweep_time_add_curr[:, :, None, None, None].repeat(
                1, 1, 1, feats_to_return.shape[3], feats_to_return.shape[4]) * self.history_cam_sweep_freq
            ], dim=2) # B x (1 + cat_num) x (80 + 1) x H x W
        
        # Step 11: Time conv
        feats_to_return = self.history_keyframe_time_conv(
            feats_to_return.reshape(-1, *feats_to_return.shape[2:])).reshape(
                feats_to_return.shape[0], feats_to_return.shape[1], self.single_bev_num_channels, *feats_to_return.shape[3:]) # B x (1 + cat_num) x 80 x H x W

        # Step 12: Cat keyframes & conv
        feats_to_return = self.history_keyframe_cat_conv(
            feats_to_return.reshape(
                feats_to_return.shape[0], -1, feats_to_return.shape[3], feats_to_return.shape[4])) # B x [(1 + cat_num)*80] x H x W -> B x 160 x H x W
        
        # Stwp 13: Update history by moving everything down one group of single_bev_num_channels channels
        # and adding in curr_bev.
        # Clone is necessary since we're doing in-place operations on self.history_bev
        self.history_bev[:, :-1] = torch.roll(self.history_bev[:, :-1], shifts=1, dims=1)
        self.history_bev[:, 0] = curr_bev.detach().clone() # B x 80 x h x w -> B x T x 80 x H x W

        self.history_forward_augs = forward_augs.clone() # B x 4 x 4

        self.history_sweep_time = torch.roll(self.history_sweep_time, shifts=1, dims=1)
        self.history_sweep_time[:, 0] = 0

        self.history_global_to_lidar = torch.roll(self.history_global_to_lidar, shifts=1, dims=1)
        self.history_global_to_lidar[:, 0] = global_to_curr_lidar_rt.detach().clone()

        return feats_to_return

    def extract_img_feat(self, img_inputs, img_metas):
        '''
        Extract image featrues from the input image and some image metas.
        Args:
            img_inputs (List[torch.Tensor]): With input image and some meta information. 
            - img (torch.Tensor): Image tensor. shape (B, N, C=3, H, W)
            - rots (torch.Tensor): Rotation matrix. shape (B, N, 3, 3)
            - trans (torch.Tensor): Translation matrix. shape (B, N, 3)
            - intrins (torch.Tensor): Intrinsic matrix. shape (B, N, 3, 3)
            - post_rots (torch.Tensor): Post rotation matrix. shape (B, N, 3, 3)
            - post_trans (torch.Tensor): Post translation matrix. shape (B, N, 3)
            img_metas (List[dict]): Meta information of each sample. length = batch_size.
        Returns:
            bev_encoder_out_list:
            - bev_feat (torch.Tensor): BEV features. shape (B, C=256, H=200, W=200)
            depth_digit (torch.Tensor): Depth digit. shape (B, N, D, H, W)
        '''

        # Step 1: Extract image features and stereo features by `image_encoder`
        img = img_inputs[0]
        rots, trans, intrins, post_rots, post_trans = img_inputs[1:6]
        curr_img_encoder_feats, curr_stereo_feats = self.image_encoder(img)

        # Step 2: Stereo
        if not self.do_history_stereo_fusion:
            bev_feat, depth_digit = self.img_view_transformer(curr_img_encoder_feats, rots, trans, intrins, post_rots, post_trans)
        else:
            # pre_process
            prev_stereo_feats, prev_global2img, prev_img_forward_aug, curr_global2img, curr_img_forward_aug, curr_unaug_cam_to_prev_unaug_cam = \
                self.process_stereo_before_fusion(stereo_feats=curr_stereo_feats, 
                                                  img_metas=img_metas,
                                                    rots=rots,
                                                    trans=trans,
                                                    intrins=intrins,
                                                    post_rots=post_rots,
                                                    post_trans=post_trans)
            
            # short term history fusion
            bev_feat, depth_digit = self.img_view_transformer(
                curr_img_encoder_feats, 
                rots, trans, intrins, post_rots, post_trans,
                curr_stereo_feats, 
                prev_stereo_feats, 
                prev_global2img, 
                prev_img_forward_aug, 
                curr_global2img, 
                curr_img_forward_aug, 
                curr_unaug_cam_to_prev_unaug_cam    
            )

            # post_process for next timestep
            self.process_stereo_for_next_timestep(stereo_feats=curr_stereo_feats, 
                                                  global_to_img=curr_global2img, 
                                                  img_forward_augs=curr_img_forward_aug)

        # Step 3: Pre-process BEV features
        bev_feat = self.pre_process_net(bev_feat)[0] # singleton list
        
        # Step 4: add bev embedding
        bev_feat = self.embed(bev_feat) # B x base_bev_channels x H x W

        # Step 5: Fuse History long term
        if self.do_history:
            bev_feat = self.fuse_history(bev_feat, img_metas)

        # Step 6: BEV encoder
        bev_encoder_out_list = self.bev_encoder(bev_feat) # see definition of `bev_encoder` in class `BEVDet_solo`

        return bev_encoder_out_list, depth_digit
    
    def get_transform_matrix(self, img_metas_list):
        '''
        Args: 
            img_metas (List[dict]): Meta information of each sample. length = batch_size. 
                                    With keys 'rots', 'trans', 'intrins', 'post_rots', 'post_trans' etc.
        Returns:
            outs (list): list of torch.Tensor. Inside are: 
            - rots (torch.Tensor): rotation part of `sensor2ego`. shape (B, N, 3, 3)
            - trans (torch.Tensor): translation part of `sensor2ego`. shape (B, N, 3)
            - intrins (torch.Tensor): intrinsic of camera. shape (B, N, 3, 3)
            - post_rots (torch.Tensor): Because of data augmentation, the `rots` pose matrix will need to be recalibrated. 
                                        `post_rots` is the recalibrated part. If no data augmentation, it is np.eye(3). 
                                        shape (B, N, 3, 3)
            - post_trans (torch.Tensor): Because of data augmentation, the `trans` pose matrix will need to be recalibrated. 
                                        `post_trans` is the recalibrated part. If no data augmentation, it is np.zeros(3). 
                                        shape (B, N, 3)
        '''
        rots_list = []
        trans_list = []
        intrins_list = []
        post_rots_list = []
        post_trans_list = []
        for img_metas in img_metas_list:
            rots_list.append(torch.tensor(img_metas['rots'], device='cuda', dtype=torch.float32))
            trans_list.append(torch.tensor(img_metas['trans'], device='cuda', dtype=torch.float32))
            intrins_list.append(torch.tensor(img_metas['intrins'], device='cuda', dtype=torch.float32))
            post_rots_list.append(torch.tensor(img_metas['post_rots'], device='cuda', dtype=torch.float32))
            post_trans_list.append(torch.tensor(img_metas['post_trans'], device='cuda', dtype=torch.float32))
        rots = torch.stack(rots_list, dim=0) # B x N x 3 x 3
        trans = torch.stack(trans_list, dim=0)
        intrins = torch.stack(intrins_list, dim=0)
        post_rots = torch.stack(post_rots_list, dim=0)
        post_trans = torch.stack(post_trans_list, dim=0)
        outs = [rots, trans, intrins, post_rots, post_trans]
        return outs

    def forward_train(self, img=None,
                      img_metas=None,
                      voxel_semantics=None,
                      valid_mask=None,
                      **kwargs): 
        '''
        Args:
            img (torch.Tensor): multi view image input of current frame. shape (B, N, C, H, W)
            voxel_semantics (torch.Tensor): 3D occupancy ground truth. shape (B, H, W, Z)
            valid_mask (torch.Tensor): unified boolean mask for visible voxel. shape (B, H, W, Z)
            img_metas (List[dict]): Meta information of each sample. length = batch_size. 
        Returns: 
            losses (dict): dict of loss.
        '''
        losses = dict()
        # Step 1: extract some meta information from `img_metas`
        transform_tensor_list = self.get_transform_matrix(img_metas)

        # Step 2: extract image feature
        img_inputs = [img] + transform_tensor_list
        bev_encoder_out_list, depth = self.extract_img_feat(img_inputs, img_metas)

        # # If we're training depth...
        # Step 3: get depth loss
        # depth_gt = img_inputs[-1] 
        # loss_depth = self.get_depth_loss(depth_gt, depth)
        # losses['loss_depth'] = loss_depth
        
        # Step 4: get occ loss, this can refer to bevdet_occ code
        occ_outs = self.pts_bbox_head(bev_encoder_out_list[0], **kwargs)
        loss_inputs = [voxel_semantics, valid_mask, occ_outs] 
        losses_pts = self.pts_bbox_head.loss(*loss_inputs, **kwargs)# the loss will be changed again
        losses.update(losses_pts)

        return losses

    def fuse_history_test(self, all_bev_feats_list, all_img_metas_list, sweep_time_list):
        # padding
        if len(all_bev_feats_list) != (self.history_cat_num + 1):
            assert False
            all_bev_feats_list = [all_bev_feats_list[0].clone() for _ in range((self.history_cat_num + 1) - len(all_bev_feats_list))] + all_bev_feats_list
            all_img_metas_list = [all_img_metas_list[0].copy() for _ in range((self.history_cat_num + 1) - len(all_img_metas_list))] + all_img_metas_list
            sweep_time_list = [sweep_time_list[0] for _ in range((self.history_cat_num + 1) - len(sweep_time_list))] + sweep_time_list
            assert len(all_bev_feats_list) == (self.history_cat_num + 1), "warning! padding is wrong!"
        
        # get the current grid
        curr_bev = all_bev_feats_list[0]
        n, c, h, w = curr_bev.shape
        xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w).expand(h, w)
        ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1).expand(h, w)
        grid = torch.stack((xs, ys, torch.ones_like(xs), torch.ones_like(xs)), -1).view(1, h, w, 4).expand(n, h, w, 4).view(n,h,w,4,1)
        # bs, h, w, 4, 1
        
        # get feat2bev
        feat2bev = torch.zeros((4,4),dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.img_view_transformer.dx[0]
        feat2bev[1, 1] = self.img_view_transformer.dx[1]
        feat2bev[0, 3] = self.img_view_transformer.bx[0] - self.img_view_transformer.dx[0] / 2.
        feat2bev[1, 3] = self.img_view_transformer.bx[1] - self.img_view_transformer.dx[1] / 2.
        feat2bev[2, 2] = 1
        feat2bev[3, 3] = 1
        feat2bev = feat2bev.view(1, 4, 4) # (1, 4, 4)
        feat2bev = feat2bev.repeat(self.history_cat_num, 1, 1).double() # (cat_num, 4, 4)

        # get ego2global 
        ego2global_list = []
        for img_meta in all_img_metas_list:
            ego2global_tr = torch.tensor(img_meta['ego2global'], device='cuda', dtype=torch.float64) # (4, 4)
            ego2global_list.append(ego2global_tr)
        
        cur_ego2global = ego2global_list[0] # (4, 4)
        repeat_cur_ego2global = cur_ego2global.unsqueeze(0).repeat(self.history_cat_num, 1, 1) # (cat_num, 4, 4)
        cat_prev_ego2global = torch.stack(ego2global_list[1:], dim=0) # (cat_num, 4, 4)
        rt_flow = torch.inverse(feat2bev)  @ torch.inverse(cat_prev_ego2global) @ repeat_cur_ego2global @ feat2bev # (cat_num, 4, 4)

        # reshape grid and wrap
        grid = grid.repeat(self.history_cat_num, 1, 1, 1, 1).double() # (cat_num, h, w, 4, 1)
        repeat_rt_flow = rt_flow.unsqueeze(1).unsqueeze(2).repeat(1, h, w, 1, 1) # (cat_num, h, w, 4, 4)
        grid = repeat_rt_flow @ grid # (cat_num, h, w, 4, 1)

        # normalize
        normalize_factor = torch.tensor([w - 1.0, h - 1.0], dtype=curr_bev.dtype, device=curr_bev.device)
        grid = grid[:,:,:,:2,0] / normalize_factor.view(1, 1, 1, 2) * 2.0 - 1.0 # (cat_num, h, w, 2)

        # grid sample
        prev_bev = torch.cat(all_bev_feats_list[1:], dim=0) # (cat_num, channels, H, W) len=7
        sampled_history_bev = F.grid_sample(prev_bev, grid.to(curr_bev.dtype), align_corners=True, mode=self.interpolation_mode) # cat_num, c, h, w
        
        ## cat history and reshape
        feats_cat = torch.cat([curr_bev, sampled_history_bev], dim=0) # (1 + cat_num, 80, H, W)
        feats_to_return = feats_cat.reshape(
            1, self.history_cat_num + 1, self.single_bev_num_channels, feats_cat.shape[2], feats_cat.shape[3]) # B x (1 + cat_num) x 80 x H x W
        
        # concatenate features and timestep
        sweep_time = torch.tensor([sweep_time_list], device='cuda', dtype=curr_bev.dtype) # (1, 1+catnum)
        feats_to_return = torch.cat(
            [feats_to_return, sweep_time[:, :, None, None, None].repeat(
                1, 1, 1, feats_to_return.shape[3], feats_to_return.shape[4]) * self.history_cam_sweep_freq
            ], dim=2) # B x (1 + T) x (80 + 1) x H x W
        
        # Time conv
        feats_to_return = self.history_keyframe_time_conv(
            feats_to_return.reshape(-1, *feats_to_return.shape[2:])).reshape(
                feats_to_return.shape[0], feats_to_return.shape[1], -1, *feats_to_return.shape[3:]) # B x (1 + T) x 80 x H x W

        # Cat keyframes & conv
        feats_to_return = self.history_keyframe_cat_conv(
            feats_to_return.reshape(
                feats_to_return.shape[0], -1, feats_to_return.shape[3], feats_to_return.shape[4])) # B x C x H x W
        
        return feats_to_return
        

    def simple_test(self, img_metas, img=None, **kwargs):
        '''
        Test Function without augmentaiton.
        Args:
            img (torch.Tensor): multi view image input of current frame. shape (B, N, C, H, W)
            img_metas (List[dict]): Meta information of each sample. length = batch_size.
        Returns:
            occ_outs (torch.Tensor): Occupancy prediction. shape (B, H, W, Z)
        '''
        
        # Step 1: prepare the img_inputs
        bs, num_views, C, H, W = img.shape
        curr_img_encoder_feats, curr_stereo_feats = self.image_encoder(img)
        cur_rots = torch.tensor(img_metas[0]['rots'], device='cuda', dtype=torch.float32).unsqueeze(0) # 1 x N x 3 x 3
        cur_trans = torch.tensor(img_metas[0]['trans'], device='cuda', dtype=torch.float32).unsqueeze(0) # 1 x N x 3 x 3
        cur_intrins = torch.tensor(img_metas[0]['intrins'], device='cuda', dtype=torch.float32).unsqueeze(0) # 1 x N x 3 x 3
        cur_post_rots = torch.tensor(img_metas[0]['post_rots'], device='cuda', dtype=torch.float32).unsqueeze(0) # 1 x N x 3 x 3
        cur_post_trans = torch.tensor(img_metas[0]['post_trans'], device='cuda', dtype=torch.float32).unsqueeze(0) # 1 x N x 3 x 3

        # Step 2: prepare `prev_stereo_feats` and `prev_img_metas_list`
        scene_token = img_metas[0]['sample_idx'] // 1000
        if scene_token != self.prev_frame_info['scene_token']:
            self.prev_frame_info['prev_img_metas_list'] = [img_metas[0].copy() for _ in range(self.history_queue_length)] # length = 30
            self.prev_frame_info['prev_stereo_feats_list'] = [curr_stereo_feats.clone() for _ in range(self.history_stereo_prev_step)] # length = 5
        prev_stereo_feats_list = self.prev_frame_info['prev_stereo_feats_list']
        history_last_stereo_feats = prev_stereo_feats_list[self.history_stereo_prev_step - 1]
        prev_img_metas_list = self.prev_frame_info['prev_img_metas_list']

        # Step 3: do short term fusion
        if self.do_history_stereo_fusion:
            curr_ego2global_np = np.asarray(img_metas[0]['ego2global'])
            curr_ego2global = torch.tensor(curr_ego2global_np, device='cuda', dtype=torch.float64) # (4, 4)
            curr_lidar2img_np = np.asarray(img_metas[0]['lidar2img'])
            curr_lidar2img = torch.tensor(curr_lidar2img_np, device='cuda', dtype=torch.float64)
            curr_global2img = curr_lidar2img @ torch.inverse(curr_ego2global)
            curr_global2img = curr_global2img.unsqueeze(0) # 1 x N x 4 x 4

            prev_ego2global_np = np.asarray(prev_img_metas_list[self.history_stereo_prev_step - 1]['ego2global'])
            prev_ego2global = torch.tensor(prev_ego2global_np, device='cuda', dtype=torch.float64) # (4, 4)
            prev_lidar2img_np = np.asarray(prev_img_metas_list[self.history_stereo_prev_step - 1]['lidar2img'])
            prev_lidar2img = torch.tensor(prev_lidar2img_np, device='cuda', dtype=torch.float64) # (5, 4, 4)
            prev_global2img = prev_lidar2img @ torch.inverse(prev_ego2global) # (5, 4, 4)
            prev_global2img = prev_global2img.unsqueeze(0) # 1 x N x 4 x 4
            
            curr_unaug_cam_to_prev_unaug_cam = prev_global2img[:, :, None, :, :] @ torch.inverse(curr_global2img)[:, None, :, :, :] # 1 x N x N x 4 x 4

            prev_img_forward_aug = torch.eye(4).to('cuda').repeat(bs, num_views, 1, 1).double() # B x N x 4 x 4
            curr_img_forward_aug = torch.eye(4).to('cuda').repeat(bs, num_views, 1, 1).double() # B x N x 4 x 4

            raw_bev_feat, _ = self.img_view_transformer(curr_img_encoder_feats, cur_rots, cur_trans, 
                                                        cur_intrins, cur_post_rots, cur_post_trans, # each B x N x 3 (x 3)
                                                        curr_stereo_feats, history_last_stereo_feats, 
                                                        prev_global2img, prev_img_forward_aug, 
                                                        curr_global2img, curr_img_forward_aug, 
                                                        curr_unaug_cam_to_prev_unaug_cam)

        else:
            # no short term fusion
            raw_bev_feat, _ = self.img_view_transformer(curr_img_encoder_feats, cur_rots, cur_trans, cur_intrins, cur_post_rots, cur_post_trans)

        bev_feat = self.pre_process_net(raw_bev_feat)[0]
        bev_feat = self.embed(bev_feat)
        new_bev_for_history = bev_feat.clone()

        # Step 4: long term fusion
        if self.do_history:
            # prepare the history
            if scene_token != self.prev_frame_info['scene_token']:
                self.prev_frame_info['prev_bev_list'] = [bev_feat.clone() for _ in range(self.history_queue_length)] # length = 30
                self.prev_frame_info['prev_sweep_time_list'] = [0 for _ in range(self.history_queue_length)] # length = 30
                prev_sweep_time_list = self.prev_frame_info['prev_sweep_time_list']
            else:
                # new timestep, everything in history gets pushed back one. 
                prev_sweep_time_list = [i+1 for i in self.prev_frame_info['prev_sweep_time_list']]
            prev_bev_list = self.prev_frame_info['prev_bev_list']

            # sample
            sampled_prev_bev_list = []
            sampled_prev_img_metas_list = []
            sampled_sweep_time_list = []
            for index in self.sample_index:
                sampled_prev_bev_list.append(prev_bev_list[index])
                sampled_prev_img_metas_list.append(prev_img_metas_list[index])
                sampled_sweep_time_list.append(prev_sweep_time_list[index])

            all_bev_feats_list = [bev_feat] + list(sampled_prev_bev_list)
            all_img_metas_list = img_metas + list(sampled_prev_img_metas_list)
            sweep_time_list = [0] + sampled_sweep_time_list

            # fuse
            bev_feat = self.fuse_history_test(all_bev_feats_list, all_img_metas_list, sweep_time_list)

            # update the `prev_sweep_time_list` and `prev_bev_list`
            del prev_sweep_time_list[-1]
            prev_sweep_time_list = [0] + prev_sweep_time_list
            self.prev_frame_info['prev_sweep_time_list'] = prev_sweep_time_list

            del prev_bev_list[-1]
            prev_bev_list.insert(0, new_bev_for_history)
            self.prev_frame_info['prev_bev_list'] = prev_bev_list

        # update
        self.prev_frame_info['scene_token'] = scene_token
        
        del prev_img_metas_list[-1]
        prev_img_metas_list.insert(0, img_metas[0])
        self.prev_frame_info['prev_img_metas_list'] = prev_img_metas_list
        
        del prev_stereo_feats_list[-1]
        prev_stereo_feats_list.insert(0, curr_stereo_feats)
        self.prev_frame_info['prev_stereo_feats_list'] = prev_stereo_feats_list

        bev_encoder_out_list = self.bev_encoder(bev_feat)
        
        occ_outs = self.pts_bbox_head(bev_encoder_out_list[0]) # bs, h, w, z, c
        occ_outs = self.pts_bbox_head.get_occ(occ_outs) # bs, h, w, z

        return occ_outs

    def forward_test(self, img_metas, 
                     img=None, 
                     voxel_semantics=None,
                     valid_mask=None,
                     **kwargs):
        '''
        Test function for the model.
        Args: 
            (all arg are be wrapped one more list. after we take it out, the type of each parameter are below)
            img_metas (List[dict]): Meta information of each sample. length = batch_size = 1
            img (torch.Tensor): multi view image input of current frame. shape (B, N, C, H, W)
            voxel_semantics (torch.Tensor): 3D occupancy ground truth. shape (B, H, W, Z)
            valid_mask (torch.Tensor): unified boolean mask for visible voxel. shape (B, H, W, Z)
        Returns:
            occ_results (dict): dict of evaluation results.
        '''

        # Step 1: prepare the input
        if voxel_semantics is not None: voxel_semantics = voxel_semantics[0]
        if valid_mask is not None: valid_mask = valid_mask[0]
        if img is not None: img = img[0]
        if img_metas is not None: img_metas = img_metas[0]

        voxel_semantics_pred = self.simple_test(img_metas=img_metas, img=img, **kwargs)
        
        occ_results = self.pts_bbox_head.eval_metrics(voxel_semantics, voxel_semantics_pred, valid_mask=valid_mask)
        sample_idx = img_metas[0]['sample_idx']
        scene_id = sample_idx % 1000000 // 1000
        occ_results['scene_id'] = scene_id
        frame_id = sample_idx % 1000
        occ_results['frame_id'] = frame_id

        return occ_results
