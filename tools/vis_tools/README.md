# Visualization Tools
## vis_ref.py
### usage
This script processes the `Waymo` or `NuScenes` datasets and generates images with 3D reference points. The dataset configuration can be switched by setting the `IS_WAYMO` variable.

### config parameter
IS_WAYMO: Boolean value, set to `True` to process the Waymo dataset, set to `False` to process the NuScenes dataset.
Other configuration parameters like NOT_OBSERVED, FREE, OCCUPIED, etc., are defined in the script for specific settings of different datasets.

### input
ref_file: Path to the reference file, select the corresponding file based on the value of IS_WAYMO (e.g., work_dirs/ref_waymo.pkl or work_dirs/ref_nuscene.pkl).
metas_file: Path to the metadata file, select the corresponding file based on the value of IS_WAYMO (e.g., work_dirs/metas_waymo.pkl or work_dirs/metas_nuscene.pkl).

### output
"work_dirs/waymo{}_debug_{}.jpg".format(IS_WAYMO, cam_id)

### command line
```sh
python tools/vis_tools/vis_ref.py
```

## vis_ref_dataloader.py & vis_ref_dataloader_nus.py
### usage
Similar to the previous file, but this one uses a dataloader to fetch data directly instead of using saved data. Split the files into `waymo` and `nuscenes` two version. 

### config parameter
config: Path to the required config file.

### output
"work_dirs/", "{}_{}.jpg".format("ref", img_idx)

### command line
```sh
python -m tools.vis_tools.vis_ref_dataloader # waymo
python -m tools.vis_tools.vis_ref_dataloader_nus # nuscenes
```

## vis_occ.py

### usage
This file visualizes the ground truth source files of Occ3d-Waymo. Simple modifications can meet various occupancy visualization needs.


### config parameter
data_dir: Directory of the npz files to be visualized. It will visualize npz files from 0 to 99 in the directory. The npz files contain keys "voxel_label", "origin_voxel_state", "final_voxel_state", "infov", and "ego2global".

### command line
```sh
python vis_occ.py
```

## vis_pose.py

### usage
This script projects point cloud data onto images and generates image files with the point cloud visualization results.

### config parameter
#### constant

- `point_cloud_range`：Defines the coordinate range of the point cloud.
- `voxel_size`：Defines the size of the voxel.
- `IS_WAYMO`：Flag to indicate if the dataset is Waymo. Different parameters are set based on this flag.

#### directory
- `pcd_path`：Path to the point cloud data.
- `img_path`：Path to the image data.
- `voxel_path`：Path to the voxel label data.

### output
"work_dirs/", "{}_{}.jpg".format("src", cam_idx)

### command line
```sh
python -m tools.vis_tools.vis_pose
```

## utils.py

### usage
This is a utility file defining various functions used by other vis files, including `get_cv_color`, `get_open3d_color`, `display_laser_on_image` and `volume2points`
