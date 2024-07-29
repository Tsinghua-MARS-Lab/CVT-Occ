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

'''
The script is used to project the point cloud data onto the image and generate an image file with the result of the point cloud visualization.
'''
import os
import pickle
from pathlib import Path

import numpy as np
import PIL
import torch
from matplotlib import pyplot as plt

from .utils import *

pcd_path = Path(
    "/public/MARS/datasets/waymo_v1.3.1_untar/kitti_format/training/velodyne/0001050.bin"
)
img_path = Path("/public/MARS/datasets/waymo_v1.4.0/gt_points/001/050_cam.pkl")
voxel_path = Path("/public/MARS/datasets/waymo_v1.4.0/voxel/001/050.npz")
point_cloud_range = [-80, -80, -5.0, 80, 80, 7.8]
voxel_size = [0.1, 0.1, 0.2]

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

if __name__ == "__main__":
    with open(img_path, "rb") as f:
        img_infos = pickle.load(f)

    pcd = np.fromfile(pcd_path, dtype=np.float32).reshape((-1, 6))[:, :3]  # ego frame
    labels = np.load(voxel_path)
    voxel_label = labels["voxel_label"]
    origin_voxel_state = labels["origin_voxel_state"]
    final_voxel_state = labels["final_voxel_state"]
    # ego2global = labels['ego2global']
    points = volume2points(voxel_label, voxel_size, point_cloud_range)
    points = points.reshape(-1, 3)
    points_label = voxel_label.reshape(-1)
    # points = pcd
    points_colors = np.zeros((points_label.shape[0], 3))
    if points_label is not None:
        for idx in np.unique(points_label):
            if idx == FREE_LABEL:
                continue
            points_colors[points_label == idx] = get_cv_color(idx, begin=1)
    points_colors = points_colors / 255.0
    mask = points_label != FREE_LABEL
    points = points[mask]
    points_label = points_label[mask]
    points_colors = points_colors[mask]

    for cam_idx in range(5):
        img_info = img_infos[cam_idx]
        vehicle2image = img_info["intrinsics"] @ np.linalg.inv(img_info["sensor2ego"])
        img = np.array(img_info["img"])
        pts, mask = display_laser_on_image(img, points, vehicle2image)
        pts = pts[mask]
        pts_colors = points_colors[mask]
        fig, ax = plt.subplots(1, 2, figsize=(18, 8))
        ax[0].imshow(img)
        ax[1].imshow(img)
        ax[1].scatter(pts[:, 0], pts[:, 1], c=pts_colors, s=1)
        ax[0].axis("off")
        ax[1].axis("off")
        # plt.show()
        # plt save too slow, use PIL
        # If we haven't already shown or saved the plot, then we need to draw the figure first...
        fig.canvas.draw()
        img = PIL.Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
        img.save(os.path.join("work_dirs/", "{}_{}.jpg".format("src", cam_idx)))
        plt.close("all")
