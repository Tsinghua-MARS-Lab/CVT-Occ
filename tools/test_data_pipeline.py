"""
run the dataloader pipeline to get the data or exmaine the pipeline without run the model
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from mmcv import Config, DictAction
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets import nuscenes_occ

config = "projects/configs/cvtocc/bevformer_nuscene.py"
cfg = Config.fromfile(config)

dataset = build_dataset(cfg.data.train)
result = dataset.__getitem__(200)

# waymo
# imgs = result["img"].data[-1]
# lidar2img = result["img_metas"].data[2]["lidar2img"]
# voxel_label = result["voxel_semantics"]

breakpoint()
print(result['img_metas'].data.keys()) # dict_keys([0, 1, 2])
print(result['img_metas'].data[2].keys())
# dict_keys(['filename', 'pts_filename', 'occ_gt_path', 'scene_token', 'frame_idx', 'scene_idx', 'sample_idx', 'ori_shape', 'img_shape', 'pad_shape', 'lidar2img', 'ego2lidar', 'cam_intrinsic', 'lidar2cam', 'can_bus', 'prev_bev_exists'])
'''
(Pdb) print(result['img_metas'].data[2]['pts_filename'])
./data/occ3d-nus/samples/LIDAR_TOP/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915375547671.pcd.bin
(Pdb) print(result['img_metas'].data[2]['frame_idx'])
1
(Pdb) print(result['img_metas'].data[2]['scene_idx'])
166
(Pdb) print(result['img_metas'].data[2]['sample_idx'])
e981a119b19040159fe112adca805119
(Pdb) print(result['img_metas'].data[2]['scene_token'])
15e1fa06e30e438a98430cc1fd0e8a69
(Pdb) print(result['img_metas'].data[2]['occ_gt_path'])
gts/scene-0166/e981a119b19040159fe112adca805119/labels.npz
'''