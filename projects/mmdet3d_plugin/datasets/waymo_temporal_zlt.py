import os
import copy
import random
import pickle
from functools import reduce

from tqdm import tqdm
import numpy as np
import torch
import mmcv
from mmcv.parallel import DataContainer as DC
from mmcv.utils import print_log
from mmdet3d.datasets import DATASETS
from mmdet3d.core.bbox import Box3DMode, points_cam2img
from mmdet3d.datasets.kitti_dataset import KittiDataset
from mmdet3d.core.bbox import get_box_type
from mmdet3d.core.bbox import (Box3DMode, CameraInstance3DBoxes, Coord3DMode, LiDARInstance3DBoxes, points_cam2img)
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion

from projects.mmdet3d_plugin.datasets.zltwaymo import CustomWaymoDataset
from projects.mmdet3d_plugin.datasets.occ_metrics import Metric_FScore, Metric_mIoU

@DATASETS.register_module()
class CustomWaymoDataset_T(CustomWaymoDataset):

    CLASSES = ('Car', 'Pedestrian', 'Sign', 'Cyclist')

    def __init__(self,
                 *args,
                 load_interval=1,
                 history_len=1, 
                 input_sample_policy=None,
                 skip_len=0,
                 withimage=True,
                 pose_file=None,
                 offset=0,
                 use_streaming=False,
                 **kwargs):
        with open(pose_file, 'rb') as f:
            pose_all = pickle.load(f)
            self.pose_all = pose_all
        self.length_waymo = sum([len(scene) for k, scene in pose_all.items()])
        self.history_len = history_len
        self.input_sample_policy = input_sample_policy
        self.skip_len = skip_len
        self.withimage = withimage
        self.load_interval_waymo = load_interval
        self.length = self.length_waymo
        self.offset = offset
        self.evaluation_kwargs = kwargs
        self.use_streaming = use_streaming
        super().__init__(*args, **kwargs)

    def __len__(self):
        return self.length_waymo // self.load_interval_waymo

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        if self.use_streaming:
            return self.prepare_streaming_train_data(idx)
        
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
        
    def prepare_streaming_train_data(self, index):
        index = int(index * self.load_interval_waymo)
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def prepare_test_data(self, index):
        """
        Prepare data for testing.
        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """

        index += self.offset
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example
    
    def get_input_idx(self, idx_list):
        '''
        sample the input index list
        Args:
            idx_list (List[int]): the index list from `index - self.history_len` to `index`. 
                                  It contains current frame index, but it dropped another random frame index to add randomness. 
                                  So the length is `self.history_len`.
        Returns:
            sampled_idx_list (List[int]): the index list after sampling
        '''

        if self.input_sample_policy['type'] == 'normal':
            return idx_list
        
        elif self.input_sample_policy['type'] == 'large interval':
            sampled_idx_list = []
            for i in range(0, self.input_sample_policy['number']):
                sampled_idx = max(0, self.history_len - 1 - i * self.input_sample_policy['interval'])
                sampled_idx_list.append(idx_list[sampled_idx])
            return sorted(sampled_idx_list)
        
        elif self.input_sample_policy['type'] == 'random interval':
            fix_interval = self.input_sample_policy['fix interval']
            slow_interval = random.randint(0, fix_interval-1)
            random_interval = random.choice([fix_interval, slow_interval])

            sampled_idx_list = []
            for i in range(0, self.input_sample_policy['number']):
                sampled_idx = max(self.history_len - 1 - i * random_interval, 0)
                sampled_idx_list.append(idx_list[sampled_idx])
                
            return sorted(sampled_idx_list)
        
        else:
            raise NotImplementedError('not implemented input_sample_policy type')

    def prepare_train_data(self, index):
        '''
        prepare data for training
        Args:
            index (Int): the index of the data
        Returns:
            data (Dict): the data dict for training
        '''

        # Step 1: get the index list of the history data
        index *= self.load_interval_waymo
        if self.history_len == 1:
            idx_list = [index]
        else:
            queue_start_index = index - self.history_len
            idx_list = list(range(queue_start_index, index))
            random.shuffle(idx_list)
            idx_list = sorted(idx_list[1:]) # drop one frame to add some randomness
            idx_list.append(index)
            
        # Step 2: sample the index list
        i_list = self.get_input_idx(idx_list)

        # Step 3: get the data info according to the index list
        data_queue = []
        for i in i_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None: 
                return None

            # Step 4: prepare the data by dataloader pipeline
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            data_queue.append(example)

        # Step 5: union the data_queue into one single sample
        if self.filter_empty_gt and (data_queue[0] is None):
            return None
        if self.withimage:
            return self.union2one(data_queue)
        else:
            return data_queue[-1]

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
        prev_scene_token=None
        imgs_list = [each['img'].data for each in queue]
        metas_map = {}
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if metas_map[i]['sample_idx']//1000 != prev_scene_token:
                metas_map[i]['prev_bev_exists'] = False
                prev_scene_token = metas_map[i]['sample_idx'] // 1000
                metas_map[i]['scene_token']= prev_scene_token

            else:
                metas_map[i]['scene_token'] = prev_scene_token
                metas_map[i]['prev_bev_exists'] = True

        # Step 2: pack them together
        queue[-1]['img'] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]

        return queue


    def get_data_info(self, index):
        '''
        get the data info according to the index. Most of them are image meta data. 
        Args: 
            index (Int): the index of the data.
        Returns:
            input dict (Dict): the data info dict.
        '''

        # Step 1: get the data info
        info = self.data_infos_full[index]
        
        # Step 2: get the image file name and idx
        sample_idx = info['image']['image_idx']
        scene_idx = sample_idx % 1000000 // 1000
        frame_idx = sample_idx % 1000000 % 1000
        img_filename = os.path.join(self.data_root, info['image']['image_path'])

        # # Step 3: get the `lidar2img` (why here it get the lidar2img and in the following code it get another lidar2img)
        # rect = info['calib']['R0_rect'].astype(np.float32)
        # Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        # P0 = info['calib']['P0'].astype(np.float32)
        # lidar2img = P0 @ rect @ Trv2c

        # the Tr_velo_to_cam is computed for all images but not saved in .info for img1-4
        # the size of img0-2: 1280x1920; img3-4: 886x1920. Attention

        # Step 4: get the image paths, lidar2img, intrinsics, sensor2ego for each image
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            intrinsics_rts = []
            sensor2ego_rts = []

            for idx_img in range(self.num_views):
                pose = self.pose_all[scene_idx][frame_idx][idx_img]

                intrinsics = pose['intrinsics'] # sensor2img
                sensor2ego = pose['sensor2ego']
                lidar2img = intrinsics @ np.linalg.inv(sensor2ego)
                ego2global = pose['ego2global']
                
                # Attention! (this code means the pose info dismatch the image data file)
                if idx_img == 2: 
                    image_paths.append(img_filename.replace('image_0', f'image_3'))
                elif idx_img == 3: 
                    image_paths.append(img_filename.replace('image_0', f'image_2'))
                else:
                    image_paths.append(img_filename.replace('image_0', f'image_{idx_img}'))

                lidar2img_rts.append(lidar2img)
                intrinsics_rts.append(intrinsics)
                sensor2ego_rts.append(sensor2ego)

        # Step 5: get the pts filename by function `_get_pts_filename` in class `CustomWaymoDataset`
        pts_filename = self._get_pts_filename(sample_idx)

        # Step 6: pack the data info into a dict
        input_dict = dict(
            sample_idx=sample_idx,
            pts_filename=pts_filename,
            img_prefix=None,
        )

        if self.modality['use_camera']:
            input_dict['img_filename'] = image_paths
            input_dict['lidar2img'] = lidar2img_rts
            input_dict['cam_intrinsic'] = intrinsics_rts
            input_dict['sensor2ego'] = sensor2ego_rts
            ego2global = self.pose_all[scene_idx][frame_idx][0]['ego2global']
            input_dict['ego2global'] = ego2global
            input_dict['global_to_curr_lidar_rt'] = np.linalg.inv(pose['ego2global'])

        # Step 7: get the annos info
        annos = self.get_ann_info(index)
        input_dict['ann_info'] = annos

        # Step 8: get the can_bus info (In `waymo` dataset, we do not have can_bus info)
        can_bus = np.zeros(9)
        input_dict['can_bus'] = can_bus

        return input_dict

    def get_ann_info(self, index):
        '''
        get the annotation info according to the index.
        Args:
            index (Int): the index of the data.
        Returns:
            annos (Dict): the annotation info dict.
        '''

        if self.test_mode == True:
            info = self.data_infos[index]
        else: info = self.data_infos_full[index]
        
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)

        annos = info['annos']
        # we need other objects to avoid collision when sample
        annos = self.remove_dontcare(annos)

        loc = annos['location']
        dims = annos['dimensions']
        rots = annos['rotation_y']
        gt_names = annos['name']
        gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1).astype(np.float32)

        gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d).convert_to(
            self.box_mode_3d, np.linalg.inv(rect @ Trv2c))


        gt_bboxes = annos['bbox']

        selected = self.drop_arrays_by_name(gt_names, ['DontCare'])
        gt_bboxes = gt_bboxes[selected].astype('float32')
        gt_names = gt_names[selected]
        gt_labels = []
        for cat in gt_names:
            if cat in self.CLASSES:
                gt_labels.append(self.CLASSES.index(cat))
            else:
                gt_labels.append(-1)
        gt_labels = np.array(gt_labels).astype(np.int64)
        gt_labels_3d = copy.deepcopy(gt_labels)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            bboxes=gt_bboxes,
            labels=gt_labels,
            gt_names=gt_names)
        return anns_results

    def evaluate(self, occ_results, metric='mIoU', runner=None, **eval_kwargs):
        '''
        This function will be called by `tools/test.py` to evaluate the results.
        Args:
            occ_results (List[Dict]): the results of the model. 
                                      function `forward_test` of `occformer_waymo.py` return the occ_result. 
                                      Then function `custom_multi_gpu_test` of `projects/mmdet3d_plugin/bevformer/apis/test.py` will pack them. 
            metric (Str): the evaluation metric. By default, it is `mIoU`.
        Returns: 
            None. `occ_eval_metrics.print()` will directly print the result to the terminal. 
        '''

        def eval(occ_eval_metrics, runner=None):
            print('\nStarting Evaluation...')
            for index, occ_result in enumerate(tqdm(occ_results)):
                CDist_tensor = occ_result.get('CDist_tensor', None)
                count_matrix = occ_result['count_matrix']
                scene_id = occ_result['scene_id']
                frame_id = occ_result['frame_id']
                occ_eval_metrics.add_batch(CDist_tensor, count_matrix, scene_id, frame_id)
            occ_eval_metrics.print(runner=runner)

        if "mIoU" in metric:
            # Step 1: initialize the `metric_mIoU`
            occ_eval_metrics = Metric_mIoU(**self.evaluation_kwargs)

            # Step 2: evaluate the results by `eval` function
            # Because we have move most calculation to `forward test` and directly return the `count_matrix`, the eval here will be fast. 
            eval(occ_eval_metrics, runner=runner)

        elif "FScore" in metric:
            occ_eval_metrics = Metric_FScore()
            eval(occ_eval_metrics)

        else:
            raise NotImplementedError
