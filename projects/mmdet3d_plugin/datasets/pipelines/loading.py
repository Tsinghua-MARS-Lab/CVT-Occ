import numpy as np
from numpy import random
import mmcv
from mmdet.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC
import os
from PIL import Image



@PIPELINES.register_module()
class MyLoadMultiViewImageFromFiles(object):
    """
    This image file loader is for Waymo dataset. 
    Load multi channel images from a list of separate channel files.
    Expects results['img_filename'] to be a list of filenames.
    note that we read image in BGR style to align with opencv.imread
    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, img_scale=None, color_type='unchanged'):
        self.to_float32 = to_float32
        self.img_scale = img_scale
        self.color_type = color_type

    def pad(self, img):
        # to pad the 5 input images into a same size (for Waymo)
        if img.shape[0] != self.img_scale[0]:
            padded = np.zeros((self.img_scale[0],self.img_scale[1],3))
            padded[0:img.shape[0], 0:img.shape[1], :] = img
            img = padded
        return img

    def __call__(self, results):
        """
        Call function to load multi-view image from files.
        Args:
            results (dict): Result dict containing multi-view image filenames.
        Returns:
            dict: The result dict containing the multi-view image data.
                  Added keys and values are described below.
                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        
        # Step 1: load image according to the filename
        filename = results['img_filename']
        img = [np.asarray(Image.open(name))[...,::-1] for name in filename]

        # Step 2: record the original shape of the image
        results['ori_shape'] = [img_i.shape for img_i in img]

        # Step 3: pad the image
        if self.img_scale is not None:
            img = [self.pad(img_i) for img_i in img]

        # Step 4: stack the image
        img = np.stack(img, axis=-1)

        # Step 5: convert the image to float32
        if self.to_float32:
            img = img.astype(np.float32)

        # Step 6: record the filename, image, image shape, and image normalization configuration
        results['filename'] = filename
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(mean=np.zeros(num_channels, dtype=np.float32), 
                                       std=np.ones(num_channels, dtype=np.float32), 
                                       to_rgb=False) # This will be replaced in `NormalizeMultiviewImage`
        
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return "{} (to_float32={}, color_type='{}')".format(self.__class__.__name__, self.to_float32, self.color_type)


@PIPELINES.register_module()
class LoadOccGTFromFileWaymo(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    note that we read image in BGR style to align with opencv.imread
    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
            self,
            data_root,
            use_larger=True,
            crop_x=False,
            use_infov_mask=True, 
            use_lidar_mask=False, 
            use_camera_mask=True,
            FREE_LABEL=None, 
            num_classes=None,
        ):
        self.use_larger=use_larger
        self.data_root = data_root # this is occ_gt_data_root in config file
        self.crop_x = crop_x
        self.use_infov_mask = use_infov_mask
        self.use_lidar_mask = use_lidar_mask
        self.use_camera_mask = use_camera_mask
        self.FREE_LABEL = FREE_LABEL
        self.num_classes = num_classes

    def __call__(self, results):
        # Step 1: get the occupancy ground truth file path
        pts_filename = results['pts_filename']
        basename = os.path.basename(pts_filename)
        seq_name = basename[1:4]
        frame_name = basename[4:7]
        if self.use_larger:
            file_path = os.path.join(self.data_root, seq_name,  '{}_04.npz'.format(frame_name))
        else:
            file_path = os.path.join(self.data_root, seq_name, '{}.npz'.format(frame_name))
        
        # Step 2: load the file
        occ_labels = np.load(file_path)
        semantics = occ_labels['voxel_label']
        mask_infov = occ_labels['infov'].astype(bool)
        mask_lidar = occ_labels['origin_voxel_state'].astype(bool)
        mask_camera = occ_labels['final_voxel_state'].astype(bool)

        # Step 3: crop the x axis
        if self.crop_x: # default is False
            w, h, d = semantics.shape
            semantics = semantics[w//2:, :, :]
            mask_infov = mask_infov[w//2:, :, :]
            mask_lidar = mask_lidar[w//2:, :, :]
            mask_camera = mask_camera[w//2:, :, :]

        # Step 4: unify the mask
        mask = np.ones_like(semantics).astype(bool) # 200, 200, 16
        if self.use_infov_mask:
            mask = mask & mask_infov
        if self.use_lidar_mask:
            mask = mask & mask_lidar
        if self.use_camera_mask:
            mask = mask & mask_camera
        mask = mask.astype(bool)
        results['valid_mask'] = mask

        # Step 5: change the FREE_LABEL to num_classes-1
        if self.FREE_LABEL is not None:
            semantics[semantics == self.FREE_LABEL] = self.num_classes - 1
        results['voxel_semantics'] = semantics

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return "{} (data_root={}')".format(
            self.__class__.__name__, self.data_root)

@PIPELINES.register_module()
class LoadOccGTFromFileNuScenes(object):
    """
    Load multi channel images from a list of separate channel files.
    Expects results['img_filename'] to be a list of filenames.
    note that we read image in BGR style to align with opencv.imread
    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
            self,
            data_root,
        ):
        self.data_root = data_root

    def __call__(self, results):
        # Step 1: get the occ_gt_path
        occ_gt_path = results['occ_gt_path']
        occ_gt_path = os.path.join(self.data_root, occ_gt_path)

        # Step 2: parse the scene idx
        parts = occ_gt_path.split('/')
        scene_part = [part for part in parts if 'scene-' in part]
        if scene_part:
            scene_number = scene_part[0].split('-')[1]
        results['scene_idx'] = int(scene_number)

        # Step 3: load the occ_gt file
        occ_labels = np.load(occ_gt_path)
        semantics = occ_labels['semantics']
        mask_lidar = occ_labels['mask_lidar'].astype(bool)
        mask_camera = occ_labels['mask_camera'].astype(bool)

        results['voxel_semantics'] = semantics
        results['mask_lidar'] = mask_lidar
        results['mask_camera'] = mask_camera

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return "{} (data_root={}')".format(
            self.__class__.__name__, self.data_root)