# Copyright 2022 tao.jiang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script processes LiDAR data from the NuScenes or Waymo dataset and visualizes the voxel labels.

Usage:
    python script_name.py

Configuration:
    IS_WAYMO: Boolean flag to toggle between Waymo and NuScenes dataset configurations.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from mmcv import Config, DictAction
from mmdet3d.datasets import build_dataset

from projects.mmdet3d_plugin.datasets import nuscenes_occ
from projects.mmdet3d_plugin.datasets import waymo_temporal_zlt
from .utils import *

config = "projects/configs/bevformer/bevformer_waymo.py"
voxel_size = [0.4, 0.4, 0.4]
IS_WAYMO = True
cfg = Config.fromfile(config)
point_cloud_range = cfg.point_cloud_range

IS_WAYMO = True
if IS_WAYMO:
    NOT_OBSERVED = -1
    FREE = 0
    OCCUPIED = 1
    FREE_LABEL = 23
    MAX_POINT_NUM = 10
    ROAD_LABEL_START_BEFORE_REMAP = 24
    ROAD_LABEL_STOP_BEFORE_REMAP = 27
    ROAD_LABEL_START = 13
    ROAD_LABEL_STOP = 14
    BINARY_OBSERVED = 1
    BINARY_NOT_OBSERVED = 0
    STUFF_START = 9  # 0-10 thing 11-17 stuff
    # DO NOT CHANGE
    FLT_MAX = 1e9
    RAY_STOP_DISTANCE_VOXEL = 1
    DISTANCE_THESHOLD_IGNORE = 1.0
    RAY_ROAD_IGNORE_DISTANCE = 1.0

    num_cams = 5
else:
    NOT_OBSERVED = -1
    FREE = 0
    OCCUPIED = 1
    FREE_LABEL = 17
    MAX_POINT_NUM = 10
    ROAD_LABEL_START_BEFORE_REMAP = 24
    ROAD_LABEL_STOP_BEFORE_REMAP = 27
    ROAD_LABEL_START = 11
    ROAD_LABEL_STOP = 14
    BINARY_OBSERVED = 1
    BINARY_NOT_OBSERVED = 0
    STUFF_START = 10  # 0-10 thing 11-17 stuff
    num_cams = 6
mean = [103.530, 116.280, 123.675]


dataset = build_dataset(cfg.data.train)
result = dataset.__getitem__(1000)
print(result["img_metas"].data.keys())
imgs = result["img"].data[-1]
lidar2img = result["img_metas"].data[2]["lidar2img"]  # DEBUG_TMP
print(result["img_metas"].data[2]["filename"])
voxel_label = result["voxel_semantics"]
voxel_locs = volume2points(voxel_label, voxel_size, point_cloud_range)
points = voxel_locs.reshape(-1, 3)
points_label = voxel_label.reshape(-1)
points_colors = np.zeros((points_label.shape[0], 3))
if points_label is not None:
    for idx in np.unique(points_label):
        if idx == FREE_LABEL:
            continue
        points_colors[points_label == idx] = get_cv_color(idx, begin=1)
points_colors = points_colors / 255.0
mask = points_label != (FREE_LABEL+1) # Here we filter out the free label
points = points[mask]
points_label = points_label[mask]
points_colors = points_colors[mask]

points = np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)
points = points.reshape(-1, 4, 1)


imgs[:, 0] += mean[0]
imgs[:, 1] += mean[1]
imgs[:, 2] += mean[2]
imgs = torch.clip(imgs, min=0, max=255)
for img_idx in range(len(imgs)):
    img = imgs[img_idx]
    lidar2img_ = lidar2img[img_idx]
    lidar2img_ = torch.Tensor(lidar2img_)
    H, W = img.shape[1], img.shape[2]

    n = points.shape[0]
    lidar2img_ = lidar2img_.view(1, 4, 4).repeat(n, 1, 1)
    points_ = torch.Tensor(points)
    reference_points_cam = torch.matmul(
        lidar2img_.to(torch.float32), points_.to(torch.float32)
    ).squeeze(-1)
    eps = 1e-5
    bev_mask = reference_points_cam[..., 2:3] > eps
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3],
        torch.ones_like(reference_points_cam[..., 2:3]) * eps,
    )
    reference_points_cam[..., 0] /= W
    reference_points_cam[..., 1] /= H
    bev_mask = (
        bev_mask
        & (reference_points_cam[..., 1:2] > 0.0)
        & (reference_points_cam[..., 1:2] < 1)
        & (reference_points_cam[..., 0:1] < 1)
        & (reference_points_cam[..., 0:1] > 0.0)
    )
    bev_mask = bev_mask.reshape(-1)

    reference_points_cam_ = reference_points_cam.cpu().numpy()
    bev_mask = bev_mask.cpu().numpy()

    im = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    uu = reference_points_cam[bev_mask, 0] * W
    vv = reference_points_cam[bev_mask, 1] * H

    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    fig.tight_layout()
    # plt.subplots_adjust(wspace=0, hspace=0)
    ax[0].imshow(im)
    ax[0].axis("off")
    ax[1].imshow(im)
    ax[1].scatter(uu, vv, c=points_colors[bev_mask], s=1)
    ax[1].axis("off")
    # plt save too slow, use PIL
    # If we haven't already shown or saved the plot, then we need to draw the figure first...
    fig.canvas.draw()
    img = PIL.Image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )
    img.save(os.path.join("work_dirs/", "{}_{}.jpg".format("ref", img_idx)))
    plt.close("all")
