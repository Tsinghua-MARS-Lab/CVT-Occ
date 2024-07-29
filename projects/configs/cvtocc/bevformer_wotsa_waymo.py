_base_ = ['../_base_/default_runtime.py']
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
voxel_size = [0.4, 0.4, 0.4]
occ_voxel_size = None # useless
use_larger = True  # means use 0.4 voxel size
num_classes = 16
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
class_names = [
    'car',
    'truck',
    'construction_vehicle',
    'bus',
    'trailer',
    'barrier',
    'motorcycle',
    'bicycle',
    'pedestrian',
    'traffic_cone',
]
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

PUJIANG = False
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

input_modality = dict(
    use_lidar=False, 
    use_camera=True, 
    use_radar=False, 
    use_map=False, 
    use_external=True
)

# mask
use_infov_mask = True
use_lidar_mask = False
use_camera_mask = True
use_CDist = False

_dim_ = 256
num_feats = [_dim_ // 3, _dim_ // 3, _dim_ - _dim_ // 3 - _dim_ // 3]
_ffn_dim_ = _dim_ * 2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
total_z = 16
# for bev
pillar_h = 4
num_points_in_pillar = 4
# for volume
volume_flag = False
bev_z_ = 1
if not volume_flag:
    bev_z_ = 1
# for decoder
use_3d_decoder = False
use_conv_decoder = True

num_views = 5
FREE_LABEL = 23
# for data
load_interval = 1
test_interval = 1
total_epochs = 8

# for cost volume
use_refine_feat_loss = False
use_temporal = None
use_temporal_self_attention = False
use_padding = False
# important parameter
refine_feat_loss_weight = None
scales = None
# for interval
queue_length = 1
input_sample_policy = {
    "type": "normal"
} # only for training

sampled_queue_length = 1 # only for costvolume
sample_num = [0] # only for test

if use_temporal_self_attention:
    attn_cfgs = [
        dict(
            type='TemporalSelfAttention', embed_dims=_dim_, num_points=4, num_levels=1
        ),
        dict(
            type='SpatialCrossAttention',
            num_cams=num_views,
            pc_range=point_cloud_range,
            deformable_attention=dict(
                type='MSDeformableAttention3D',
                embed_dims=_dim_,
                num_points=4,
                num_levels=_num_levels_,
            ),
            embed_dims=_dim_,
        ),
    ]
    operation_order = ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
else:
    attn_cfgs = [
        dict(
            type='SpatialCrossAttention',
            num_cams=num_views,
            pc_range=point_cloud_range,
            deformable_attention=dict(
                type='MSDeformableAttention3D',
                embed_dims=_dim_,
                num_points=4,
                num_levels=_num_levels_,
            ),
            embed_dims=_dim_,
        )
    ]
    operation_order = ('cross_attn', 'norm', 'ffn', 'norm')

class_weight_binary = [5.314075572339673, 1]
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

model = dict(
    type='CVTOccWaymo',
    use_grid_mask=False,
    video_test_mode=True,
    queue_length=queue_length,
    sampled_queue_length=sampled_queue_length,
    sample_num=sample_num, # only for test
    save_results=False, # for visualization
    use_temporal=use_temporal,
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
    ),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True,
    ),
    pts_bbox_head=dict(
        type='CVTOccHeadWaymo',
        volume_flag=volume_flag,
        bev_z=bev_z_,
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_classes=num_classes,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        voxel_size=voxel_size,
        occ_voxel_size=occ_voxel_size,
        use_larger=use_larger,
        # loss_occ=dict(
        #     type='FocalLoss',
        #     use_sigmoid=False,
        #     gamma=2.0,
        #     alpha=0.25,
        #     loss_weight=10.0),
        use_CDist=use_CDist,
        CLASS_NAMES=CLASS_NAMES,
        use_refine_feat_loss=use_refine_feat_loss,
        refine_feat_loss_weight=refine_feat_loss_weight,
        # loss_occ= dict(
        #     type='CrossEntropyLoss',
        #     use_sigmoid=False,
        #     loss_weight=1.0),
        loss_occ=dict(
            ceohem=dict(
                type='CrossEntropyOHEMLoss',
                # Online hard example mining cross-entropy loss
                class_weight=class_weight_multiclass,
                use_sigmoid=False,
                use_mask=False,
                loss_weight=1.0,
                top_ratio=0.2,
                top_weight=4.0,
            ),
            # lovasz=dict(
            #     type='LovaszLoss',
            #     class_weight=class_weight_multiclass,
            #     loss_type='multi_class',
            #     classes='present',
            #     per_image=False,
            #     reduction='none',
            #     loss_weight=1.0)
        ),
        transformer=dict(
            type='CVTOccTransformerWaymo',
            num_cams=num_views,
            queue_length=queue_length,
            sampled_queue_length=sampled_queue_length,
            volume_flag=volume_flag,
            pillar_h=pillar_h,
            bev_z=bev_z_,
            bev_h=bev_h_,
            bev_w=bev_w_,
            total_z=total_z,
            scales=scales,
            num_classes=num_classes,
            use_3d_decoder=use_3d_decoder,
            use_conv_decoder=use_conv_decoder,
            rotate_prev_bev=False,
            # use_shift=True, # use_can_bus is False, so use_shift will not be used
            use_can_bus=False,
            embed_dims=_dim_,
            pc_range=point_cloud_range,
            voxel_size=voxel_size,
            occ_voxel_size=occ_voxel_size,
            use_larger=use_larger,
            use_temporal=use_temporal,
            use_padding=use_padding,
            encoder=dict(
                type='BEVFormerEncoderWaymo',
                num_layers=4,
                volume_flag=volume_flag,
                pc_range=point_cloud_range,
                num_points_in_pillar=num_points_in_pillar,
                return_intermediate=False,
                bev_z=bev_z_,
                bev_h=bev_h_,
                bev_w=bev_w_,
                total_z=total_z,
                transformerlayers=dict(
                    type='BEVFormerLayerWaymo',
                    volume_flag=volume_flag,
                    attn_cfgs=attn_cfgs,
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    bev_z=bev_z_,
                    bev_h=bev_h_,
                    bev_w=bev_w_,
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=_dim_,
                        feedforward_channels=_dim_ * 4,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),
                    operation_order=operation_order,
                ),
            ),
            decoder=dict(
                type='OccConvDecoder',
                embed_dims=_dim_,
                conv_num=3,
                pillar_h=pillar_h,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(
                    type='BN',
                ),
                act_cfg=dict(type='ReLU', inplace=True),
            ),
        ),
        positional_encoding=dict(
            type='LearnedPositionalEncoding3D',
            num_feats=num_feats,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            height_num_embed=9999,
        ),
        loss_cls=dict(
            type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,  # it seems no use
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(
                    type='IoUCost', weight=0.0
                ),  # Fake cost. This is just to make it compatible with DETR head.
                pc_range=point_cloud_range,
            ),
        )
    ),
)

dataset_type = 'CustomWaymoDataset_T'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='MyLoadMultiViewImageFromFiles', to_float32=True, img_scale=(1280, 1920)),
    dict(
        type='LoadOccGTFromFileWaymo',
        data_root=occ_gt_data_root,
        use_larger=use_larger,
        crop_x=False,
        use_infov_mask=use_infov_mask,
        use_camera_mask=use_camera_mask,
        use_lidar_mask=use_lidar_mask,
        FREE_LABEL=FREE_LABEL,
        num_classes=num_classes,
    ),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
    ),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='CustomCollect3D',
        keys=['img', 'voxel_semantics', 'valid_mask'],
        meta_keys=[
            'filename',
            'pts_filename',
            'sample_idx',
            'scene_token',
            'ori_shape',
            'img_shape',
            'pad_shape',
            'lidar2img',
            'sensor2ego',
            'cam_intrinsic',
            'ego2global',
        ],
    ),
]

test_pipeline = [
    dict(type='MyLoadMultiViewImageFromFiles', to_float32=True, img_scale=(1280, 1920)),
    dict(
        type='LoadOccGTFromFileWaymo',
        data_root=occ_val_gt_data_root,
        use_larger=use_larger,
        crop_x=False,
        use_infov_mask=use_infov_mask,
        use_camera_mask=use_camera_mask,
        use_lidar_mask=use_lidar_mask,
        FREE_LABEL=FREE_LABEL,
        num_classes=num_classes,
    ),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1, 1),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D', class_names=class_names, with_label=False
            ),
            dict(
                type='CustomCollect3D',
                keys=['img', 'voxel_semantics', 'valid_mask'],
                meta_keys=[
                    'filename',
                    'pts_filename',
                    'sample_idx',
                    'scene_token',
                    'ori_shape',
                    'img_shape',
                    'pad_shape',
                    'lidar2img',
                    'sensor2ego',
                    'cam_intrinsic',
                    'ego2global',
                ],
            ),
        ],
    ),
]

# class CustomWaymoDataset_T
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        load_interval=load_interval,
        num_views=num_views,
        split='training',
        ann_file=ann_file,
        pose_file=pose_file,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        history_len=queue_length,
        input_sample_policy=input_sample_policy,
        box_type_3d='LiDAR',
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        load_interval=test_interval,
        split='training',
        ann_file=val_ann_file,
        pose_file=val_pose_file,
        num_views=num_views,
        pipeline=test_pipeline,
        test_mode=True,
        classes=class_names,
        modality=input_modality,
        samples_per_gpu=1,
        # below are evaluation parameters
        use_CDist=use_CDist,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        CLASS_NAMES=CLASS_NAMES,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        load_interval=test_interval,
        split='training',
        num_views=num_views,
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
    ),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

optimizer = dict(
    type='AdamW',
    lr=4e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }
    ),
    weight_decay=0.01,
)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

evaluation = dict(interval=total_epochs, pipeline=test_pipeline)
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'

log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')]
)
checkpoint_config = dict(interval=1)
find_unused_parameters = True
