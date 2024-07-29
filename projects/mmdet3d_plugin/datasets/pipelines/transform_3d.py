import numpy as np
from numpy import random
import mmcv
from mmdet.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC
import os




@PIPELINES.register_module()
class PadMultiViewImage(object):
    """
    Pad the multi-view image and add keys "pad_shape".
    There are two padding modes: (1) pad to a fixed size
    (2) pad to the minimum size that is divisible by some number.
    
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to `self.size`."""
        if self.size is not None:
            padded_img = [mmcv.impad(img, shape=self.size, pad_val=self.pad_val) for img in results['img']]
        elif self.size_divisor is not None:
            padded_img = [mmcv.impad_to_multiple(img, self.size_divisor, pad_val=self.pad_val) for img in results['img']]
        
        results['img'] = padded_img
        results['img_shape'] = [img.shape for img in padded_img]
        results['pad_shape'] = [img.shape for img in padded_img]

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """
    Normalize the image and add key "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb


    def __call__(self, results):
        """
        Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        results['img'] = [mmcv.imnormalize(img, self.mean, self.std, self.to_rgb) for img in results['img']]
        results['img_norm_cfg'] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results['img']
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, \
                'PhotoMetricDistortion needs the input image of dtype np.float32,'\
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta,
                                    self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(self.saturation_lower,
                                            self.saturation_upper)

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results['img'] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str

@PIPELINES.register_module()
class RandomScaleImageMultiViewImage(object):
    """
    Random scale the image
    Args:
        scales (list): List of scales to choose from.
                       If the list contains only one scale, then the scale is fixed.
    """

    def __init__(self, scales=[]):
        self.scales = scales

    def __call__(self, results):
        """
        Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """

        # Step 1: sample a random scale
        rand_scale = np.random.choice(self.scales)

        # Step 2: scale the image
        y_size = [int(img.shape[0] * rand_scale) for img in results['img']]
        x_size = [int(img.shape[1] * rand_scale) for img in results['img']]
        results['img'] = [mmcv.imresize(img, (x_size[idx], y_size[idx]), return_scale=False) for idx, img in enumerate(results['img'])]

        # Step 3: update the `lidar2img` and `intrinsic` transformation matrix
        scale_factor = np.eye(4)
        scale_factor[0, 0] *= rand_scale
        scale_factor[1, 1] *= rand_scale                  
        lidar2img = [scale_factor @ l2i for l2i in results['lidar2img']]
        results['lidar2img'] = lidar2img
        cam_intrinsic = [scale_factor @ cam_intr for cam_intr in results['cam_intrinsic']]
        results['cam_intrinsic'] = cam_intrinsic

        # Step 4: update the image shape
        results['img_shape'] = [img.shape for img in results['img']]

        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.scales}, '
        return repr_str
    
@PIPELINES.register_module()
class CustomCollect3D(object):
    """
    Collect data from the loader relevant to the specific task.
    This is usually the last stage of the data loader pipeline. 
    Args:
        keys (Sequence[str]): Keys of results to be collected in `data`.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            `mmcv.DataContainer` and collected in `data[img_metas]`.
    """

    def __init__(self, keys=None, meta_keys=None,):
        self.keys = keys
        self.meta_keys = meta_keys
        assert self.meta_keys is not None, 'meta_keys must be set'

    def __call__(self, results):
        """
        Call function to collect keys in results. The keys in `meta_keys`
        will be converted to :obj:`mmcv.DataContainer`.
        Args:
            results (dict): Result dict contains the data to collect.
        Returns:
            dict: The result dict contains the following keys in `self.keys` and `img_metas`
        """

        if 'rots' in self.meta_keys:
            sensor2ego_list = results['sensor2ego'] # list of 4x4 matrices
            rots_list = [sensor2ego[:3, :3] for sensor2ego in sensor2ego_list] # list of 3x3 matrices
            rots = np.stack(rots_list, axis=0) # (num_cams, 3, 3)
            results['rots'] = rots
            trans_list = [sensor2ego[:3, 3] for sensor2ego in sensor2ego_list] # list of 3x1 vectors
            trans = np.stack(trans_list, axis=0) # (num_cams, 3)
            results['trans'] = trans
            
            cam_intrinsic_list = results['cam_intrinsic'] # list of 4x4 matrices
            intrins_list = [cam_intrinsic[:3, :3] for cam_intrinsic in cam_intrinsic_list] # list of 3x3 matrices
            intrins = np.stack(intrins_list, axis=0) # (num_cams, 3, 3)
            results['intrins'] = intrins

            post_rots_list = [np.eye(3) for sensor2ego in sensor2ego_list] # list of 3x3 matrices
            post_rots = np.stack(post_rots_list, axis=0) # (num_cams, 3, 3)
            results['post_rots'] = post_rots
            post_trans_list = [np.zeros(3) for sensor2ego in sensor2ego_list] # list of 3x1 vectors
            post_trans = np.stack(post_trans_list, axis=0) # (num_cams, 3)
            results['post_trans'] = post_trans
            
            frame_id = results['sample_idx'] % 1000
            if frame_id == 0:
                results['start_of_sequence'] = 1
            else:
                results['start_of_sequence'] = 0
            results['sequence_group_idx'] = results['sample_idx'] % 1000000 // 1000

        data = {}
        img_metas = {}

        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]

        data['img_metas'] = DC(img_metas, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data


    def __repr__(self):
        """str: Return a string that describes the module."""
        return self.__class__.__name__ + \
               f'(keys={self.keys}, meta_keys={self.meta_keys})'