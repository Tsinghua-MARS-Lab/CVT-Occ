import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

class Metric_CDist:
    def __init__(self, 
                 num_classes = 16,
                 FREE_LABEL = 23,
                 use_CDist = True,
                 use_mIoU = True,
                 use_infov_mask = True,
                 use_lidar_mask = False,
                 use_camera_mask = True,
                 use_binary_mask = False,
                 use_dynamic_object_mask = True,
                 ):
        self.num_classes = num_classes
        self.FREE_LABEL = FREE_LABEL
        self.use_CDist = use_CDist
        self.use_mIoU = use_mIoU
        self.use_infov_mask = use_infov_mask
        self.use_lidar_mask = use_lidar_mask
        self.use_camera_mask = use_camera_mask
        self.use_binary_mask = use_binary_mask
        self.use_dynamic_object_mask = use_dynamic_object_mask
        self.CLASS_NAMES = [
                        'GO',
                        'TYPE_VEHICLE', "TYPE_BICYCLIST", "TYPE_PEDESTRIAN", "TYPE_SIGN",
                        'TYPE_TRAFFIC_LIGHT', 'TYPE_POLE', 'TYPE_CONSTRUCTION_CONE', 'TYPE_BICYCLE', 'TYPE_MOTORCYCLE',
                        'TYPE_BUILDING', 'TYPE_VEGETATION', 'TYPE_TREE_TRUNK', 
                        'TYPE_ROAD', 'TYPE_WALKABLE',
                        'TYPE_FREE',
                    ],

    def get_mask(self, voxel_semantics, mask_infov, mask_lidar, mask_camera):
        mask=torch.ones_like(voxel_semantics) # shape (bs, w, h, z)
        if self.use_infov_mask:
            mask = torch.logical_and(mask_infov, mask)
        if self.use_lidar_mask:
            mask = torch.logical_and(mask_lidar, mask)
        if self.use_camera_mask:
            mask = torch.logical_and(mask_camera, mask)
        if self.use_binary_mask:
            mask_binary = torch.logical_or(voxel_semantics == 0, voxel_semantics == self.num_classes-1) # 0: general object, 15: free
            mask = torch.logical_and(mask_binary, mask)
        if self.use_dynamic_object_mask:
            classname = self.CLASS_NAMES 
            dynamic_class = ['TYPE_VEHICLE', 'TYPE_BICYCLIST', 'TYPE_PEDESTRIAN', 'TYPE_BICYCLE', 'TYPE_MOTORCYCLE']
            class_to_index = {class_name: index for index, class_name in enumerate(classname)}
            dynamic_semantics_label_list = [class_to_index[class_name] for class_name in dynamic_class]
            mask_dynamic_object = torch.ones_like(voxel_semantics, dtype=torch.bool)
            for label in dynamic_semantics_label_list:
                # for each dynamic object, mask out the corresponding class
                mask_dynamic_object = torch.logical_and(mask_dynamic_object, voxel_semantics != label)
            mask = torch.logical_and(mask_dynamic_object, mask)
        mask = mask.bool() # ensure the mask is boolean, (bs, bev_w, bev_h, bev_z)
        return mask

    def compute_CDist(self, gtocc, predocc, mask):
        alpha = 1.0  # Hyperparameter
        
        # Squeeze dimensions
        gtocc = gtocc.squeeze(0)
        predocc = predocc.squeeze(0)
        mask = mask.squeeze(0)
        
        # Use mask to change unobserved into 16 (out of range)
        gtocc = torch.where(mask, gtocc, torch.ones_like(gtocc) * self.num_classes)
        predocc = torch.where(mask, predocc, torch.ones_like(predocc) * self.num_classes)
        
        # Get all unique class labels
        labels_tensor = torch.unique(torch.cat((gtocc, predocc), dim=0))
        labels_list = labels_tensor.tolist()
        labels_list = [x for x in labels_list if x < (self.num_classes-1)] # skip free type
        
        CDist_tensor = torch.zeros((self.num_classes-1), device='cuda')
        for label in labels_list:
            
            # Extract points for the current class
            labeled_gtocc = torch.nonzero(gtocc == label).float()  # (N_1, 3)
            labeled_predocc = torch.nonzero(predocc == label).float() # (N_2, 3)
            
            if labeled_gtocc.shape[0] == 0 or labeled_predocc.shape[0] == 0:
                # CDist_tensor[label] = 2
                CDist_tensor[label] = labeled_gtocc.shape[0] + labeled_predocc.shape[0]
                continue

            # convert tensor to numpy
            labeled_gtocc_np = labeled_gtocc.cpu().numpy()
            labeled_predocc_np = labeled_predocc.cpu().numpy()

            # Use sklearn's NearestNeighbors to find nearest neighbors
            reference_gt = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(labeled_gtocc_np)
            reference_pred = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(labeled_predocc_np)

            dist_pred_to_gt, _ = reference_gt.kneighbors(labeled_predocc_np)
            dist_gt_to_pred, _ = reference_pred.kneighbors(labeled_gtocc_np)

            dist_pred_to_gt = torch.from_numpy(dist_pred_to_gt).squeeze().to('cuda')
            dist_gt_to_pred = torch.from_numpy(dist_gt_to_pred).squeeze().to('cuda')
            
            exp_dist1 = 1 - torch.exp(-dist_pred_to_gt * alpha)
            exp_dist2 = 1 - torch.exp(-dist_gt_to_pred * alpha)
            chamfer_distance = torch.sum(exp_dist1) + torch.sum(exp_dist2)
            
            CDist_tensor[label] = chamfer_distance.item()
            
        return CDist_tensor

    def compute_count_matrix(self, gtocc, predocc):
        n_cl = self.num_classes
        count_matrix = torch.zeros((n_cl, n_cl), device='cuda')
        assert gtocc.shape == predocc.shape, "ground truth and predition result should share the same shape and same mask"

        # filter out out of bound semantics
        correct_idx = (gtocc >= 0) & (gtocc < n_cl)
        count_matrix = torch.bincount(n_cl * gtocc[correct_idx].to(torch.int) + predocc[correct_idx].to(torch.int), 
                                        weights=None, minlength=n_cl ** 2).reshape(n_cl, n_cl)
        return count_matrix

    def eval_metrics(self, voxel_semantics, voxel_semantics_preds, mask_lidar, mask_infov, mask_camera):
        """
        Args:
            voxel_semantic: bs, w, h, z
            other four are the same
        """
        # process the data
        voxel_semantics[voxel_semantics==self.FREE_LABEL] = self.num_classes-1
        voxel_semantics_preds[voxel_semantics_preds==self.FREE_LABEL] = self.num_classes-1
        mask = self.get_mask(voxel_semantics, mask_infov, mask_lidar, mask_camera)
        
        # compute chamfer distance
        if self.use_CDist:
            CDist_tensor = self.compute_CDist(gtocc=voxel_semantics, predocc=voxel_semantics_preds, mask=mask)
        else:
            CDist_tensor = torch.zeros((self.num_classes), device='cuda')
        
        if self.use_mIoU:
            # compute mIoU
            masked_semantics_gt = voxel_semantics[mask]
            masked_semantics_pred = voxel_semantics_preds[mask]
            count_matrix = self.compute_count_matrix(gtocc=masked_semantics_gt, predocc=masked_semantics_pred)
        else:
            count_matrix = torch.zeros((self.num_classes, self.num_classes), device='cuda')

        # count dict
        # use count matrix is the same
        # gt_count = torch.sum(count_matrix, dim=1)
        # pred_count = torch.sum(count_matrix, dim=0)

        occ_results = { "CDist_tensor": CDist_tensor.cpu().numpy(),
                        "count_matrix": count_matrix.cpu().numpy(), }

        return occ_results