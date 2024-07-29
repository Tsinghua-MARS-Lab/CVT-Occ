###############################################################################
# Training Details

_base_ = ['../_base_/default_runtime.py']
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

work_dir = None
resume_optimizer = False

# By default, 3D detection datasets randomly choose another sample if there is
# no GT object in the current sample. This does not make sense when doing
# sequential sampling of frames, so we disable it.
filter_empty_gt = False

# Intermediate Checkpointing to save GPU memory.
with_cp = False

###############################################################################
# High-level Model & Training Details

base_bev_channels = 80

# Long-Term Fusion Parameters
input_sample_policy = {
    "type": "normal",
}

do_history = True
history_cat_conv_out_channels = 160
history_cat_num = 6
history_queue_length = 30
queue_length = history_queue_length + 1
if do_history:
    bev_encoder_in_channels = history_cat_conv_out_channels
else:
    bev_encoder_in_channels = base_bev_channels

# Short-Term Fusion Parameters
do_history_stereo_fusion = True
stereo_out_feats = 64
history_stereo_prev_step = 5
stereo_sampling_num = 7

# Loss Weights
depth_loss_weight = 3.0
velocity_code_weight = 0.2

###############################################################################
# General Dataset & Augmentation Details.

point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
voxel_size = [0.4, 0.4, 0.4]
grid_config = {
    'xbound': [-40, 40, 0.4],
    'ybound': [-40, 40, 0.4],
    'zbound': [-10.0, 10.0, 20.0],
    'dbound': [2.0, 58.0, 0.5],
}

num_classes = 16
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
# waymo
CLASS_NAMES = [
    'GO',
    'TYPE_VEHICLE',
    'TYPE_PEDESTRIAN',
    'TYPE_SIGN',
    'TYPE_BICYCLIST',
    'TYPE_TRAFFIC_LIGHT',
    'TYPE_POLE',
    'TYPE_CONSTRUCTION_CONE',
    'TYPE_BICYCLE',
    'TYPE_MOTORCYCLE',
    'TYPE_BUILDING',
    'TYPE_VEGETATION',
    'TYPE_TREE_TRUNK',
    'TYPE_ROAD',
    'TYPE_WALKABLE',
    'TYPE_FREE',
]

data_config = {
    'Ncams': 5,
    'src_size': (1280, 1920),
    'scales': [0.5], 
    'input_size': (640, 960),
}

use_infov_mask = True
use_lidar_mask = False
use_camera_mask = True
FREE_LABEL = 23
class_weight_multiclass = [
    21.996729830048952,
    7.504469780801267, 
    10.597629961083673, 
    12.18107968968811, 
    15.143940258446506, 
    13.035521328502758,
    9.861234292376812, 
    13.64431851057796, 
    15.121236434460473, 
    21.996729830048952, 
    6.201671013759701,
    5.7420517938838325, 
    9.768712859518626, 
    3.4607400626606317, 
    4.152268220983671, 
    1.000000000000000,
]
bev_h_ = 200
bev_w_ = 200
bev_z_ = 1
###############################################################################
# Set-up the model.

model = dict(
    type='SOLOFusion',
    input_sample_policy=input_sample_policy,
    # Long-Term Fusion
    do_history=do_history,
    history_cat_num=history_cat_num,
    history_queue_length=history_queue_length, 
    history_cat_conv_out_channels=history_cat_conv_out_channels,
    # Short-Term Fusion
    do_history_stereo_fusion=do_history_stereo_fusion,
    history_stereo_prev_step=history_stereo_prev_step,
    FREE_LABEL=FREE_LABEL,
    num_classes=num_classes,
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='SECONDFPN_solo',
        in_channels=[256, 512, 1024, 2048],
        upsample_strides=[0.25, 0.5, 1, 2],
        out_channels=[128, 128, 128, 128]),
    # A separate, smaller neck for generating stereo features. Format is
    # similar to MVS works.
    stereo_neck=dict(
        type='SECONDFPN_solo',
        in_channels=[256, 512, 1024, 2048],
        upsample_strides=[1, 2, 4, 8],
        out_channels=[stereo_out_feats, 
                      stereo_out_feats, 
                      stereo_out_feats, 
                      stereo_out_feats],
        final_conv_feature_dim=stereo_out_feats),
    # 2D -> BEV Image View Transformer.
    img_view_transformer=dict(type='ViewTransformerSOLOFusion',
                              do_history_stereo_fusion=do_history_stereo_fusion,
                              stereo_sampling_num=stereo_sampling_num,
                              loss_depth_weight=depth_loss_weight,
                              grid_config=grid_config,
                              data_config=data_config,
                              numC_Trans=base_bev_channels,
                              use_bev_pool=False,
                              extra_depth_net=dict(type='ResNetForBEVDet_solo',
                                                   numC_input=256,
                                                   num_layer=[3,],
                                                   num_channels=[256,],
                                                   stride=[1,])
                                ),
    # Pre-processing of BEV features before using Long-Term Fusion
    pre_process = dict(type='ResNetForBEVDet_solo',numC_input=base_bev_channels,
                       num_layer=[2,], num_channels=[base_bev_channels,],
                       stride=[1,], backbone_output_ids=[0,]),
    # After using long-term fusion, process BEV for detection head.
    img_bev_encoder_backbone = dict(type='ResNetForBEVDet_solo', 
                                    numC_input=bev_encoder_in_channels,
                                    num_channels=[base_bev_channels * 2, 
                                                  base_bev_channels * 4, 
                                                  base_bev_channels * 8],
                                    backbone_output_ids=[-1, 0, 1, 2]),
    img_bev_encoder_neck = dict(type='SECONDFPN_solo',
                                in_channels=[bev_encoder_in_channels, 
                                             160, 320, 640],
                                upsample_strides=[1, 2, 4, 8],
                                out_channels=[64, 64, 64, 64]),
    # occ head
    pts_bbox_head=dict(
        type='SOLOOccHeadWaymo',
        FREE_LABEL=FREE_LABEL,
        embed_dims=256,
        bev_z=bev_z_,
        bev_w=bev_w_,
        bev_h=bev_h_,
        total_z=16,
        num_classes=16,
        use_infov_mask=use_infov_mask,
        use_lidar_mask=use_lidar_mask,
        use_camera_mask=use_camera_mask,
        loss_occ=dict(
            ceohem=dict(
                type='CrossEntropyOHEMLoss',
                class_weight=class_weight_multiclass,
                use_sigmoid=False,
                use_mask=False,
                loss_weight=1.0,
                top_ratio=0.2,
                top_weight=4.0),
        )
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[1024, 1024, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
                          velocity_code_weight, velocity_code_weight])),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            # nms_type='circle',
            pre_max_size=1000,
            post_max_size=83,
            # nms_thr=0.2,
            # Scale-NMS
            nms_type=['rotate', 'rotate', 'rotate', 'circle', 'rotate', 
                      'rotate'],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], 
                                [4.5, 9.0]]
        )))

###############################################################################
# Set-up the dataset

dataset_type = 'CustomWaymoDataset_T'
file_client_args = dict(backend='disk')

PUJIANG = False
use_larger = True
if PUJIANG:
    # pujiang 0.4/0.1
    data_root = '/mnt/petrelfs/zhaohang.p/mmdetection/data/waymo/kitti_format/'
    occ_data_root = '/mnt/petrelfs/zhaohang.p/dataset/waymo_occV2/'
    
else:
    # MARS 0.4/0.1
    data_root = '/public/MARS/datasets/waymo_v1.3.1_untar/kitti_format/' # replace with your won waymo image path
    occ_data_root = '/public/MARS/datasets/waymo_occV2/' # replace with your won occ gt path

ann_file = occ_data_root + 'waymo_infos_train.pkl'
val_ann_file = occ_data_root + 'waymo_infos_val.pkl'
pose_file = occ_data_root + 'cam_infos.pkl'
val_pose_file = occ_data_root + 'cam_infos_vali.pkl'
if use_larger:  # use 0.4 voxel size
    occ_gt_data_root = occ_data_root + 'voxel04/training/'
    occ_val_gt_data_root = occ_data_root + 'voxel04/validation/'
else:
    occ_gt_data_root = occ_data_root + 'voxel01/training/'
    occ_val_gt_data_root = occ_data_root + 'voxel01/validation/'

train_pipeline = [
    dict(type='MyLoadMultiViewImageFromFiles', to_float32=True, img_scale=data_config['src_size']),
    dict(type='LoadOccGTFromFileWaymo',
         data_root=occ_gt_data_root,
         use_larger=use_larger, 
         crop_x=False,
         use_infov_mask=use_infov_mask,
         use_camera_mask=use_camera_mask,
         use_lidar_mask=use_lidar_mask,
         FREE_LABEL=FREE_LABEL,
         num_classes=num_classes,
    ),
    dict(type='RandomScaleImageMultiViewImage', scales=data_config['scales']),
    dict(type='PadMultiViewImage', size_divisor=32),
    # dict(type='PointToMultiViewDepth', grid_config=grid_config), # For fair comparison, we remove depth supervision in SOLOFusion
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', 
         keys=['img','voxel_semantics', 'valid_mask'],
         meta_keys=('filename', 'pts_filename',
                    'sample_idx', 'prev_idx', 'next_idx', 'scene_token',
                    'pad_shape', 'ori_shape', 'img_shape', 
                    'start_of_sequence', 
                    'sequence_group_idx', 
                    'lidar2img','ego2lidar', 'depth2img', 'cam2img', 'cam_intrinsic','lidar2cam',
                    'ego2global', 'can_bus', 
                    'rots', 'trans', 'intrins', 'post_trans', 'post_rots', 
                    'global_to_curr_lidar_rt', 
                    # Below are useless now, but may be used if add data augmentation
                    'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip', 
                    'pcd_trans', 'pcd_scale_factor', 'pcd_rotation', 
                    'transformation_3d_flow',  
                ),
        ),
]

test_pipeline = [
    dict(type='MyLoadMultiViewImageFromFiles', to_float32=True, img_scale=data_config['src_size']), 
    dict(type='LoadOccGTFromFileWaymo',
         data_root=occ_val_gt_data_root, 
         use_larger=use_larger,
         crop_x=False,
         use_infov_mask=use_infov_mask,
         use_camera_mask=use_camera_mask,
         use_lidar_mask=use_lidar_mask,
         FREE_LABEL=FREE_LABEL,
         num_classes=num_classes,
        ),
    dict(type='RandomScaleImageMultiViewImage', scales=data_config['scales']),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1, 1),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='DefaultFormatBundle3D', class_names=class_names),
            dict(type='CustomCollect3D', 
                 keys=['img', 'voxel_semantics', 'valid_mask'],
                 meta_keys=('filename', 'pts_filename',
                    'sample_idx', 'prev_idx', 'next_idx', 'scene_token',
                    'pad_shape', 'ori_shape', 'img_shape', 
                    'start_of_sequence', 
                    'sequence_group_idx', 
                    'lidar2img','ego2lidar', 'depth2img', 'cam2img', 'cam_intrinsic','lidar2cam',
                    'ego2global', 'can_bus', 
                    'rots', 'trans', 'intrins', 'post_trans', 'post_rots', 
                    'global_to_curr_lidar_rt', 
                    # Below are useless now, but may be used if add data augmentation
                    'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip', 
                    'pcd_trans', 'pcd_scale_factor', 'pcd_rotation', 
                    'transformation_3d_flow',  
                 ),
            )
        ]
    )
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False
)

load_interval = 1
test_interval = 1

use_CDist=False
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        load_interval=load_interval, 
        num_views=data_config['Ncams'],
        split='training',
        ann_file=ann_file,
        pose_file=pose_file,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        history_len=queue_length,
        input_sample_policy=input_sample_policy,
        withimage=True, # For streaming input. Do not use `union2one` function in class `CustomWaymoDataset_T`
        box_type_3d='LiDAR',
        use_streaming=True,
        # speed_mode=None,
        # max_interval=None,
        # min_interval=None,
        # prev_only=None,
        # fix_direction=None,
        # img_info_prototype='bevdet',
        # use_sequence_group_flag=True,
        # sequences_split_num=1,
        # filter_empty_gt=filter_empty_gt,
        ),
    val=dict(type=dataset_type,
             data_root=data_root,
             pipeline=test_pipeline,
             load_interval=test_interval, 
             split='training',
             ann_file=val_ann_file,
             pose_file=val_pose_file,
             num_views=data_config['Ncams'],
             test_mode=True,
             classes=class_names,
             modality=input_modality,
             samples_per_gpu=1,
             # below are evaluation parameters
             use_CDist=use_CDist,
             voxel_size=voxel_size,
             point_cloud_range=point_cloud_range,
             CLASS_NAMES=CLASS_NAMES,
            #  img_info_prototype='bevdet',
            #  use_sequence_group_flag=True,
            #  sequences_split_num=1
             ),
    test=dict(type=dataset_type,
              data_root=data_root,
              load_interval=test_interval,
              split='training',
              num_views=data_config['Ncams'],
              ann_file=val_ann_file,
              pose_file=val_pose_file,
              pipeline=test_pipeline, 
              classes=class_names,
              modality=input_modality,
              test_mode=False,
              history_len=queue_length,
              box_type_3d='LiDAR',
              # below are evaluation parameters
              use_CDist=use_CDist,
              voxel_size=voxel_size,
              point_cloud_range=point_cloud_range,
              CLASS_NAMES=CLASS_NAMES,
            #   img_info_prototype='bevdet',
            #   use_sequence_group_flag=True,
            #   sequences_split_num=1
              ),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

# ###############################################################################
# # Optimizer & Training

optimizer = dict(
    type='AdamW',
    lr=4e-4,
    paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=0.1),}),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

full_waymo_dataset_sample_number = 158081
num_gpus = 8 # need to be changed manually. only about train
batch_size = 1
DATASET_LENGTH = full_waymo_dataset_sample_number // load_interval
num_iters_per_epoch = DATASET_LENGTH // (num_gpus * batch_size) 
total_epochs = 8
total_num_of_iters = total_epochs * num_iters_per_epoch
evaluation = dict(interval=total_num_of_iters, pipeline=test_pipeline)
runner = dict(type='IterBasedRunner', max_iters=total_num_of_iters)
load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'

log_config = dict(interval=50,hooks=[dict(type='TextLoggerHook'),dict(type='TensorboardLoggerHook')])
checkpoint_config = dict(interval=num_iters_per_epoch)
find_unused_parameters=True
