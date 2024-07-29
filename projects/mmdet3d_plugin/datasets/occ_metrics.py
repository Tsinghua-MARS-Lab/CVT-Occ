import os
import math

from tqdm import tqdm
import numpy as np
from datetime import datetime
from functools import reduce
from sklearn.neighbors import KDTree
from termcolor import colored

np.seterr(divide='ignore', invalid='ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def pcolor(string, color, on_color=None, attrs=None):
    """
    Produces a colored string for printing

    Parameters
    ----------
    string : str
        String that will be colored
    color : str
        Color to use
    on_color : str
        Background color to use
    attrs : list of str
        Different attributes for the string

    Returns
    -------
    string: str
        Colored string
    """
    return colored(string, color, on_color, attrs)


def getCellCoordinates(points, voxelSize):
    return (points / voxelSize).astype(np.int)


def getNumUniqueCells(cells):
    M = cells.max() + 1
    return np.unique(cells[:, 0] + M * cells[:, 1] + M ** 2 * cells[:, 2]).shape[0]

class Metric_mIoU():
    def __init__(self,
                 point_cloud_range,
                 voxel_size,
                 CLASS_NAMES,
                 use_CDist=False,
                 **kwargs,
                 ):
        self.use_CDist = use_CDist
        self.num_classes = len(CLASS_NAMES)
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.CLASS_NAMES = CLASS_NAMES
        self.occ_xdim = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.voxel_size[0])
        self.occ_ydim = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.voxel_size[1])
        self.occ_zdim = int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.voxel_size[2])
        self.voxel_num = self.occ_xdim * self.occ_ydim * self.occ_zdim
        self.hist = np.zeros((self.num_classes, self.num_classes))
        self.cnt = 0
        self.class_voxel_count_pred = {}
        self.class_voxel_count_gt = {}
        self.CDist_tensor = np.zeros(self.num_classes-1)

    def hist_info(self, n_cl, pred, gt):
        """
        This matrix is called by the function `compute_mIoU`. 
        But I move the count matrix process to the `forward_test` function in the model. 
        So this function will not be used.
        build confusion matrix
        Args:
            n_cl (int): num_classes_occupancy
            pred (1-d array): pred_occupancy_label
            gt (1-d array): gt_occupancu_label
        Returns:
            tuple:(hist, correctly number_predicted_labels, num_labelled_sample) 
        """

        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
            ).reshape(n_cl, n_cl),
            correct,
            labeled,
        )

    def per_class_iu(self, hist):
        '''
        iou = TP / (TP + FP + FN). 
        '''
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def per_class_recall(self, hist):
       '''
        recall = TP / (TP + FN). But I do not use it, it can be added easily.
       '''
       return np.diag(hist) / hist.sum(0)
  
    def per_class_precision(self, hist):
       '''
        precision = TP / (TP + FP). But I do not use it, it can be added easily.
       '''
       return np.diag(hist) / hist.sum(1)

    def compute_mIoU(self, pred, label, n_classes): 
        '''
        Because I move the compute count matrix process to the `forward_test` function in the model, this function will not be used.
        '''
        hist = np.zeros((n_classes, n_classes))
        new_hist, correct, labeled = self.hist_info(n_classes, pred.flatten(), label.flatten()) 
        hist += new_hist
        mIoUs = self.per_class_iu(hist)
        return round(np.nanmean(mIoUs) * 100, 2), hist

    def add_batch(self, CDist_tensor=None, count_matrix=None, scene_id=None, frame_id=None):
        '''
        The main evalution function. Called by function `eval` in dataset class.
        Args:
            CDist_tensor (np.array): (num_classes-1, )
            count_matrix (np.array): (num_classes, num_classes)
            with scene_id(int) and frame_id(int), we can change our evaluation strategy with a little bit of code. 
            For example we can only evaluation a specific subset of scenes.
        '''
        self.cnt += 1
        self.hist += count_matrix
        if self.use_CDist:
            self.CDist_tensor += CDist_tensor
    
    def print_iou(self, hist, mIoU_type_str=None, runner=None):
        '''
        print the final IoU results. Called by function `print`. Use mIoU_type_str to control the output. 
        Args:
            hist (np.array): (num_classes, num_classes) count matrix.
            mIoU_type_str (str): for flexible control. I do not use it now.
        '''

        print(f"===> {mIoU_type_str} mIoU: ")
        # Step 1: count mIoU for each class
        mIoU = self.per_class_iu(hist)
        file_path = "work_dirs/result.txt"

        # Step 2: print IoU for each class
        for ind_class in range(self.num_classes):
            class_name = self.CLASS_NAMES[ind_class]
            iou_value = mIoU[ind_class]
            if not math.isnan(iou_value):
                print(f"{class_name} IoU: {round(iou_value * 100, 2)}")
                if runner is not None:
                    runner.log_buffer.output[f"{class_name} IoU"] = round(iou_value * 100, 2)
                else:
                    with open(file_path, "a") as file:
                        file.write(f"{class_name} IoU: {round(iou_value * 100, 2)}\n")

        # Step 3: print mIoU

        # mIoU_go_free = round(np.nanmean(mIoU[1:-1]) * 100, 2)
        mIoU_free = round(np.nanmean(mIoU[:-1]) * 100, 2)
        # mIoU_all = round(np.nanmean(mIoU) * 100, 2)
        # IoU_go_motor_free = np.concatenate((mIoU[1:9], mIoU[10:-1]), axis=0)
        # mIoU_go_motor_free = round(np.nanmean(IoU_go_motor_free) * 100, 2)
        IoU_motor_free = np.concatenate((mIoU[0:9], mIoU[10:-1]), axis=0)
        mIoU_motor_free = round(np.nanmean(IoU_motor_free) * 100, 2)
        # print(f'===> mIoU without general object and free classes: ' + str(mIoU_go_free)) # mIoU without general object and free classes
        print(f'===> mIoU of non free class: ' + str(mIoU_free)) # mIoU of non-free classes
        # print(f'===> mIoU: ' + str(mIoU_all)) # mIoU of all classes
        # print(f'===> mIoU without general object, MOTORCYCLE and free class: ' + str(mIoU_go_motor_free))
        print(f'===> mIoU without MOTORCYCLE class(only for waymo): ' + str(mIoU_motor_free))

        if runner is not None:
            # write the results to the log file and log.json file in work directory
            runner.log_buffer.output['mIoU of non free'] = mIoU_free
            runner.log_buffer.output['mIoU without MOTORCYCLE(waymo)'] = mIoU_motor_free
            runner.log_buffer.ready = True
        else:
            with open(file_path, "a") as file:
                # file.write(f'===> mIoU without general object and free classes: ' + str(mIoU_go_free) + '\n')
                file.write(f'===> mIoU of non free class: ' + str(mIoU_free) + '\n')
                # file.write(f'===> mIoU: ' + str(mIoU_all) + '\n')
                # file.write(f'===> mIoU without general object and MOTORCYCLE class: ' + str(mIoU_go_motor_free) + '\n')
                file.write(f'===> mIoU without MOTORCYCLE class: ' + str(mIoU_motor_free) + '\n')

    def print(self, runner=None): # this is important
        '''
        compute and print the final results. Called by function `eval` in dataset class.
        Args: 
            None. All information are stored in the member variables by function `add_batch`
        Returns:
            I want to add some. 
        '''
        
        # Step 1: prepare some parameters
        gt_count = self.hist.sum(1)
        pred_count = self.hist.sum(0)
        total_count = gt_count + pred_count

        # Step 2: compute CDist (controlled by `self.use_CDist`)
        if self.use_CDist:
            new_array = total_count[:15]
            self.CDist_tensor /= new_array
            # print CDist_tensor here
            for ind_class in range(self.num_classes-1): # without free
                class_name = self.CLASS_NAMES[ind_class]
                print(f"{class_name} CDist: {round(self.CDist_tensor[ind_class] * 100, 2)}")

        # Step 3: print mIoU and save to file
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_path = "work_dirs/result.txt"
        if os.path.exists(file_path):
            with open(file_path, "a") as file:
                file.write(current_time + "\n")
        else:
            with open(file_path, "w") as file:
                file.write(current_time + "\n")
        
        self.print_iou(self.hist, mIoU_type_str='all', runner=runner)

        return
            
    def eval_from_file(self, pred_path, gt_path, load_interval=1):
        gts_dict = {}
        for scene in os.listdir(gt_path):
            for frame in os.listdir(os.path.join(gt_path, scene)):
                scene_token = frame
                gts_dict[scene_token] = os.path.join(gt_path, scene, frame, 'labels.npz')
        print('number of gt samples = {}'.format(len(gts_dict)))

        dirs_list = [
            "work_dirs/bevformer_base_occ_conv3d_waymo_allgift/results_epoch8/",
            "work_dirs/bevformer_base_occ_conv3d_waymo_ambiguous/results_epoch8/",
            "work_dirs/bevformer_base_occ_conv3d_waymo_noohem/results_epoch8/",
            "work_dirs/bevformer_base_occ_conv3d_waymo_no_cross_atten/results_epoch8/",
        ]
        pred_path = dirs_list[2] # "work_dirs/bevformer_base_occ_conv3d_waymo_ambiguous/results_epoch6/"
        union_files = set(os.listdir(dirs_list[0]))
        print(pred_path)
        for _dir in dirs_list:
            union_files = union_files.intersection(set(os.listdir(_dir)))

        preds_dict = {}
        for file in os.listdir(pred_path)[::load_interval]:
            if file not in union_files: continue
            if '.npz' not in file: continue

            scene_token = file.split('.npz')[0]
            preds_dict[scene_token] = os.path.join(pred_path, file)
        print('number of pred samples = {}'.format(len(preds_dict)))
        return gts_dict, preds_dict
    
    def __call__(self):
        gts_dict, preds_dict = self.eval_from_file()
        # _mIoU = 0.        
        for scene_token in tqdm(preds_dict.keys()):
            cnt += 1
            # gt = np.load(gts_dict[scene_token])
            # bs,H,W,Z
            self.add_batch()
            # _mIoU += _miou

        results = self.print()
        return results


class Metric_FScore():
    def __init__(self,
                 leaf_size=10,
                 threshold_acc=0.6,
                 threshold_complete=0.6,
                 voxel_size=[0.4, 0.4, 0.4],
                 range=[-40, -40, -1, 40, 40, 5.4],
                 void=[17, 255],
                 use_lidar_mask=False,
                 use_camera_mask=False, 
        ) -> None:

        self.leaf_size = leaf_size
        self.threshold_acc = threshold_acc
        self.threshold_complete = threshold_complete
        self.voxel_size = voxel_size
        self.range = range
        self.void = void
        self.use_lidar_mask = use_lidar_mask
        self.use_camera_mask = use_camera_mask
        self.cnt=0
        self.tot_acc = 0.
        self.tot_cmpl = 0.
        self.tot_f1_mean = 0.
        self.eps = 1e-8
        raise NotImplementedError

    def voxel2points(self, voxel):
        # occIdx = torch.where(torch.logical_and(voxel != FREE, voxel != NOT_OBSERVED))
        # if isinstance(voxel, np.ndarray): voxel = torch.from_numpy(voxel)
        mask = np.logical_not(reduce(np.logical_or, [voxel == self.void[i] for i in range(len(self.void))]))
        occIdx = np.where(mask)

        points = np.concatenate((occIdx[0][:, None] * self.voxel_size[0] + self.voxel_size[0] / 2 + self.range[0], \
                                 occIdx[1][:, None] * self.voxel_size[1] + self.voxel_size[1] / 2 + self.range[1], \
                                 occIdx[2][:, None] * self.voxel_size[2] + self.voxel_size[2] / 2 + self.range[2]),
                                axis=1)
        return points

    def add_batch(self,semantics_pred,semantics_gt,mask_lidar,mask_camera):

        # for scene_token in tqdm(preds_dict.keys()):
        self.cnt += 1

        if self.use_camera_mask:

            semantics_gt[mask_camera == False] = 255
            semantics_pred[mask_camera == False] = 255
        elif self.use_lidar_mask:
            semantics_gt[mask_lidar == False] = 255
            semantics_pred[mask_lidar == False] = 255
        else:
            pass

        ground_truth = self.voxel2points(semantics_gt)
        prediction = self.voxel2points(semantics_pred)
        if prediction.shape[0] == 0:
            accuracy=0
            completeness=0
            fmean=0

        else:
            prediction_tree = KDTree(prediction, leaf_size=self.leaf_size)
            ground_truth_tree = KDTree(ground_truth, leaf_size=self.leaf_size)
            complete_distance, _ = prediction_tree.query(ground_truth)
            complete_distance = complete_distance.flatten()

            accuracy_distance, _ = ground_truth_tree.query(prediction)
            accuracy_distance = accuracy_distance.flatten()

            # evaluate completeness
            complete_mask = complete_distance < self.threshold_complete
            completeness = complete_mask.mean()

            # evalute accuracy
            accuracy_mask = accuracy_distance < self.threshold_acc
            accuracy = accuracy_mask.mean()

            fmean = 2.0 / (1 / (accuracy+self.eps) + 1 / (completeness+self.eps))

        self.tot_acc += accuracy
        self.tot_cmpl += completeness
        self.tot_f1_mean += fmean

    def count_fscore(self,):
        base_color, attrs = 'red', ['bold', 'dark']
        print(pcolor('######## F score: {} #######'.format(self.tot_f1_mean / self.cnt), base_color, attrs=attrs))