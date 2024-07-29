_base_ = ['../datasets/custom_nus-3d.py', '../_base_/default_runtime.py']
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
voxel_size = [0.4, 0.4, 0.4]
num_classes = 18
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
    'others',
    'barrier',
    'bicycle',
    'bus',
    'car',
    'construction_vehicle',
    'motorcycle',
    'pedestrian',
    'traffic_cone',
    'trailer',
    'truck',
    'driveable_surface',
    'other_flat',
    'sidewalk',
    'terrain',
    'manmade',
    'vegetation',
    'free',
]

class_weight_multiclass = [
    1.552648813025149,
    1.477680635715412,
    1.789915946148316,
    1.454376653104962,
    1.283242744137921,
    1.583160056748120,
    1.758171915228669,
    1.468604241657418,
    1.651769160217543,
    1.454675968105020,
    1.369895420004945,
    1.125140370991227,
    1.399044660772846,
    1.203105344914611,
    1.191157881795851,
    1.155987296237377,
    1.150134564832974,
    1.000000000000000,
]

input_modality = dict(
    use_lidar=False,
    use_camera=True, 
    use_radar=False, 
    use_map=False, 
    use_external=True
)

_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
pillar_h = 16
channels = 16
queue_length = 1
use_padding = False
use_temporal = None
scales = None
use_camera_mask = True
use_lidar_mask = False
use_refine_feat_loss = False
refine_feat_loss_weight = 10

use_temporal_self_attention = False
if use_temporal_self_attention:
    attn_cfgs = [
        dict(type='TemporalSelfAttention', embed_dims=_dim_, num_levels=1),
        dict(
            type='SpatialCrossAttention',
            pc_range=point_cloud_range,
            deformable_attention=dict(
                type='MSDeformableAttention3D',
                embed_dims=_dim_,
                num_points=8,
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
            pc_range=point_cloud_range,
            deformable_attention=dict(
                type='MSDeformableAttention3D',
                embed_dims=_dim_,
                num_points=8,
                num_levels=_num_levels_,
            ),
            embed_dims=_dim_,
        )
    ]
    operation_order = ('cross_attn', 'norm', 'ffn', 'norm')

model = dict(
    type='CVTOcc',
    use_grid_mask=True,
    video_test_mode=True,
    queue_length=queue_length,
    save_results=False,
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(
            type='DCNv2', deform_groups=1, fallback_on_stride=False
        ),  # original DCNv2 will print log when perform load_state_dict
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
        type='CVTOccHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_classes=num_classes,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        use_camera_mask=use_camera_mask,
        use_lidar_mask=use_lidar_mask,
        use_free_mask=False,
        use_focal_loss=False,
        use_refine_feat_loss=use_refine_feat_loss,
        refine_feat_loss_weight=refine_feat_loss_weight,
        loss_occ=dict(
            type='CrossEntropyLoss',
            # class_weight=class_weight_multiclass,
            use_sigmoid=False,
            loss_weight=1.0,
        ),
        transformer=dict(
            type='CVTOccTransformer',
            pillar_h=pillar_h,
            num_classes=num_classes,
            bev_h=bev_h_,
            bev_w=bev_w_,
            channels=channels,
            pc_range=point_cloud_range,
            voxel_size=voxel_size,
            norm_cfg=dict(type='BN',),
            norm_cfg_3d=dict(type='BN2d',),
            use_3d=False,
            use_conv=False,
            rotate_prev_bev=False,
            use_shift=False,
            use_can_bus=False,
            embed_dims=_dim_,
            queue_length=queue_length,
            use_padding=use_padding,
            use_temporal=use_temporal,
            scales=scales,
            encoder=dict(
                type='BEVFormerEncoder',            
                bev_h=bev_h_,
                bev_w=bev_w_,
                num_layers=6,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    bev_h=bev_h_,
                    bev_w=bev_w_,
                    attn_cfgs=attn_cfgs,
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=operation_order,
                ),
            ),
        ),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10,
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
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=point_cloud_range,
            ),
        )
    ),
)

data_root = './data/occ3d-nus/'
img_scales = [1.0]
dataset_type = 'NuSceneOcc'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadOccGTFromFileNuScenes', data_root=data_root),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='RandomScaleImageMultiViewImage', scales=img_scales),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
    ),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='CustomCollect3D',
        keys=[
            'img',
            'voxel_semantics',
            'mask_lidar',
            'mask_camera',
        ],
        meta_keys=(
            'filename',
            'pts_filename',
            'occ_gt_path',
            'scene_token',
            'frame_idx',
            'scene_idx',
            'sample_idx',
            'ori_shape',
            'img_shape',
            'pad_shape',
            'lidar2img',
            'ego2lidar',
            'ego2global',
            'cam_intrinsic',
            'lidar2cam',
            'cam2img',
            'can_bus',
        ),
    ),
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadOccGTFromFileNuScenes', data_root=data_root),
    dict(type='RandomScaleImageMultiViewImage', scales=img_scales),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D', class_names=class_names, with_label=False
            ),
            dict(
                type='CustomCollect3D',
                keys=[
                    'img',
                    'voxel_semantics',
                    'mask_lidar',
                    'mask_camera',
                ],
                meta_keys=(
                    'filename',
                    'pts_filename',
                    'occ_gt_path',
                    'scene_token',
                    'frame_idx',
                    'scene_idx',
                    'sample_idx',
                    'ori_shape',
                    'img_shape',
                    'pad_shape',
                    'lidar2img',
                    'ego2lidar',
                    'ego2global',
                    'cam_intrinsic',
                    'lidar2cam',
                    'cam2img',
                    'can_bus',
                ),
            ),
        ],
    ),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'occ_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        box_type_3d='LiDAR',
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'occ_infos_temporal_val.pkl',
        pipeline=test_pipeline,
        bev_size=(bev_h_, bev_w_),
        classes=class_names,
        modality=input_modality,
        samples_per_gpu=1,
        # below are evaluation settings
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        CLASS_NAMES=CLASS_NAMES,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'occ_infos_temporal_val.pkl',
        pipeline=test_pipeline,
        bev_size=(bev_h_, bev_w_),
        classes=class_names,
        modality=input_modality,
        # below are evaluation settings
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        CLASS_NAMES=CLASS_NAMES,
    ),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
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
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
total_epochs = 24
evaluation = dict(interval=total_epochs, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')]
)

checkpoint_config = dict(interval=3)
find_unused_parameters = True
