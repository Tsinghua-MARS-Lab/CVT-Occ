"""
distinguish the ego car fast scene and ego car slow scene. 
"""

import pickle as pkl
import numpy as np

pose_file = "/public/MARS/datasets/waymo_occV2/cam_infos_vali.pkl"
cam_idx = 0

poses_all = pkl.load(open(pose_file, "rb"))

speed = {}
for scene_idx, poses in poses_all.items():
    _len = len(poses)
    move = poses[_len - 1][cam_idx]["ego2global"] - poses[0][cam_idx]["ego2global"]
    dist = np.linalg.norm(move[:, 3]) / _len
    speed[scene_idx] = dist

_list = sorted([v for k, v in speed.items()])
valid_scene = []
theshold = 1
for scene_idx, dist in speed.items():
    if dist > theshold:
        valid_scene.append(scene_idx)
print(valid_scene)
