#!/usr/bin/env python3
"""
This script processes data from Waymo or NuScenes datasets and generates annotated images.

Usage:
    python script_name.py

Configuration:
    IS_WAYMO: Boolean flag to toggle between Waymo and NuScenes dataset configurations.
    ref_file: reference file containing 3D reference points and camera reference points.
    metas_file: metadata file containing img_metas. 
"""

import os
import pickle as pkl
import cv2
import numpy as np
import torch

# Configuration for Waymo and NuScenes datasets
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
    STUFF_START = 9
    FLT_MAX = 1e9
    RAY_STOP_DISTANCE_VOXEL = 1
    DISTANCE_THESHOLD_IGNORE = 1.0
    RAY_ROAD_IGNORE_DISTANCE = 1.0
    num_cams = 5
    ref_file = "work_dirs/ref_waymo.pkl"
    metas_file = "work_dirs/metas_waymo.pkl"
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
    STUFF_START = 10
    num_cams = 6
    ref_file = "work_dirs/ref_nuscene.pkl"
    metas_file = "work_dirs/metas_nuscene.pkl"

# Load reference and metadata files
with open(ref_file, "rb") as f:
    data = pkl.load(f)
    ref_3d = data["ref_3d"]
    reference_points_cam = data["reference_points_cam"]
    bev_mask = data["bev_mask"]
with open(metas_file, "rb") as f:
    data = pkl.load(f)
    semantics = data["voxel_semantics"][0]
    imgs = data["imgs"][0]

# Configuration parameters
embed_dims = 1
bs = 1
ref_num_voxel = 2
bev_z, bev_h, bev_w = 16, 200, 200
num_query = bev_z * bev_h * bev_w
_device = torch.device("cuda:0")
mean = [103.530, 116.280, 123.675]

# Reshape semantics data
semantics = semantics.reshape(bev_w, bev_h, bev_z, ref_num_voxel)
query_labels = (
    semantics.permute(2, 1, 0, 3)
    .contiguous()
    .reshape(bev_w * bev_h * bev_z, ref_num_voxel)
    .long()
)

# Process images
D = reference_points_cam.size(3)
indexes = []
imgs[:, 0] += mean[0]
imgs[:, 1] += mean[1]
imgs[:, 2] += mean[2]
imgs = torch.clip(imgs, min=0, max=255)

for cam_id in range(num_cams):
    img_cur = imgs[cam_id]
    W, H = img_cur.shape[2], img_cur.shape[1]
    img_cur_squeeze = img_cur.reshape(-1)
    mask_per_img = bev_mask[cam_id]
    ref_num = mask_per_img.shape[2]

    for ref_idx in range(ref_num):
        print(ref_idx)
        index_query_per_img = mask_per_img[0, :, ref_idx].nonzero().squeeze(-1)
        query_labels_cur = query_labels[index_query_per_img, ref_idx]
        reference_points_cam_cur = reference_points_cam[
            cam_id, 0, index_query_per_img, ref_idx
        ]

        uu, vv = reference_points_cam_cur[:, 0] * W, reference_points_cam_cur[:, 1] * H
        uu, vv = uu.long(), vv.long()
        scalar = (uu + vv * W).long()

        scalar_ = scalar[query_labels_cur < ROAD_LABEL_START]
        img_cur_squeeze[0 * H * W + scalar_] = 0
        img_cur_squeeze[1 * H * W + scalar_] = 0
        img_cur_squeeze[2 * H * W + scalar_] = 255

        scalar_ = scalar[
            torch.logical_and(
                query_labels_cur >= ROAD_LABEL_START,
                query_labels_cur <= ROAD_LABEL_STOP,
            )
        ]
        img_cur_squeeze[0 * H * W + scalar_] = 0
        img_cur_squeeze[1 * H * W + scalar_] = 255
        img_cur_squeeze[2 * H * W + scalar_] = 0

        scalar_ = scalar[
            torch.logical_and(
                query_labels_cur > ROAD_LABEL_STOP, query_labels_cur != FREE_LABEL
            )
        ]
        img_cur_squeeze[0 * H * W + scalar_] = 255
        img_cur_squeeze[1 * H * W + scalar_] = 0
        img_cur_squeeze[2 * H * W + scalar_] = 0

    img_cur = img_cur_squeeze.reshape(3, H, W).permute(1, 2, 0)
    img_show = img_cur.cpu().numpy().astype(np.uint8)
    cv2.imwrite("work_dirs/waymo{}_debug_{}.jpg".format(IS_WAYMO, cam_id), img_show)
    # cv2.imshow("image", img_show)
    # cv2.waitKey()
