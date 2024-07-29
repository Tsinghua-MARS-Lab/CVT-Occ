import os
import copy
import random
import numpy as np
import torch

from tqdm import tqdm
import mmcv
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset

from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
from projects.mmdet3d_plugin.datasets.occ_metrics import Metric_mIoU, Metric_FScore

@DATASETS.register_module()
class NuSceneOcc(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, *args,
                 queue_length=1,
                 load_interval=1,
                 bev_size=(200, 200),
                 overlap_test=False,
                 eval_fscore=False,
                 point_cloud_range=None,
                 voxel_size=None,
                 CLASS_NAMES=None,
                 **kwargs):
        self.eval_fscore = eval_fscore
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size
        self.my_load_interval = load_interval
        self.data_infos_full = self.load_annotations(self.ann_file)
        self.data_infos = self.data_infos_full[::self.my_load_interval]
        self.image_data_root = os.path.join(self.data_root, 'samples')
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.CLASS_NAMES = CLASS_NAMES
        if hasattr(self, 'flag'):
            self.flag = self.flag[::load_interval]
        super().__init__(*args, **kwargs)

    # def __len__(self):
    #     return len(self.data_infos)

    def load_annotations(self, ann_file):
        """
        Load annotations from ann_file.
        Args:
            ann_file (str): Path of the annotation file.
        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """

        data = mmcv.load(ann_file)
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos
    
    def __getitem__(self, idx):
        """
        Get item from infos according to the given index.
        Args:
            idx (int): Index for accessing the target data.
        Returns:
            dict: Data dictionary of the corresponding index.
        """

        if self.test_mode:
            return self.prepare_test_data(idx)
        
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
    
    def prepare_test_data(self, index):
        """
        Prepare data for testing.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Testing data dict of the corresponding index.
        """

        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example
        
    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        
        # Step 1: get the index list of the history data
        index *= self.my_load_interval
        if self.queue_length <= 1:
            index_list = [index]
        else:
            index_list = list(range(index-self.queue_length, index))
            random.shuffle(index_list)
            index_list = sorted(index_list[1:])
            index_list.append(index)

        # Step 2: get the data according to the index list
        data_queue = []
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            
            # Step 3: prepare the data by dataloader pipeline
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            data_queue.append(example)

        # Step 4: union the data_queue into one single sample
        return self.union2one(data_queue)

    def union2one(self, queue):
        """
        convert sample queue into one single sample.
        Args: 
            queue (List[Dict]): the sample queue
        Returns:
            queue (Dict): the single sample
        """

        # Step 1: 1. union the `img` tensor into a single tensor. 
        # 2. union the `img_metas` dict into a dict[dict]
        # 3. add prev_bev_exists and scene_token
        imgs_list = [each['img'].data for each in queue]
        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if metas_map[i]['scene_token'] != prev_scene_token:
                metas_map[i]['prev_bev_exists'] = False
                prev_scene_token = metas_map[i]['scene_token']
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev_exists'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)

        # Step 2: pack them together
        queue[-1]['img'] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]

        return queue

    def get_data_info(self, index):
        """
        Get data info according to the given index.
        Args:
            index (int): Index of the sample data to get.
        Returns:
            dict: Data information that will be passed to the data preprocessing pipelines.
        """

        # Step 1: get the data info
        info = self.data_infos_full[index]

        # Step 2: add some basic info without preprocessing
        input_dict = dict(
            occ_gt_path=info['occ_gt_path'],
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            frame_idx=info['frame_idx'],
        )

        # Step 3: add the `ego2lidar` transformation matrix
        lidar2ego_rotation = info['lidar2ego_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        ego2lidar = transform_matrix(translation=lidar2ego_translation, rotation=Quaternion(lidar2ego_rotation), inverse=True)
        input_dict['ego2lidar'] = ego2lidar
        
        # Step 4: get the image paths, lidar2cam, intrinsics, lidar2img for each image
        if self.modality['use_camera']:
            img_filename = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info['cams'].items():
                data_path = cam_info['data_path'] 
                basename = os.path.basename(data_path)
                img_filename.append(os.path.join(self.image_data_root, cam_type, basename))

                # obtain `lidar2cam` transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t

                # obtain `lidar2img` and `intrinsic` transformation matrix
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)

                lidar2img_rts.append(lidar2img_rt)
                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=img_filename,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                )
            )

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        # Step 5: get the `ego2global` transformation matrix
        rotation = Quaternion(input_dict['ego2global_rotation'])
        translation = input_dict['ego2global_translation']
        ego2global = transform_matrix(translation=translation, rotation=rotation, inverse=False)

        # Step 6: update the `can_bus` info
        can_bus = input_dict['can_bus']
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        input_dict.update(
            dict(
                ego2global=ego2global,
                can_bus=can_bus,
            )
        )
        
        return input_dict

    def evaluate(self, occ_results, runner=None, **eval_kwargs):
        self.occ_eval_metrics = Metric_mIoU(
            point_cloud_range=self.point_cloud_range,
            voxel_size=self.voxel_size,
            CLASS_NAMES=self.CLASS_NAMES,
        )

        if self.eval_fscore: # False
            self.fscore_eval_metrics=Metric_FScore(
                leaf_size=10,
                threshold_acc=0.4,
                threshold_complete=0.4,
                voxel_size=[0.4, 0.4, 0.4],
                range=[-40, -40, -1, 40, 40, 5.4],
                void=[17, 255],
                use_lidar_mask=False,
                use_image_mask=True,
            )
            
        print('\nStarting Evaluation...')
        for index, results in enumerate(tqdm(occ_results)):
            count_matrix = results['count_matrix']
            scene_id = results['scene_id']
            frame_id = results['frame_id']
            self.occ_eval_metrics.add_batch(count_matrix=count_matrix, scene_id=scene_id, frame_id=frame_id)

        self.occ_eval_metrics.print(runner=runner)
