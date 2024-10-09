# Preparing Dataset

We use datasets Occ3D-Waymo and Occ3D-NuScenes proposed by [Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D/)

## Occ3D-Waymo

| Type |  Info |
| :----: | :----: |
| train           | 798 scenes|
| val             | 202 scenes|
| Time Span       | 20s |
| Frame           | 200 per scene|
| Time Interval   | 0.1s|
| cameras         | 5 |
| voxel size      | [0.4m, 0.4m, 0.4m] |
| range           | [-40m, -40m, -1m, 40m, 40m, 5.4m] |
| volume size     | [200, 200, 16] |
| classes label        | 0 - 14, 23 |

- sensor:

    - 5 cameras. Front(image_0), front left(image_1), front right(image_2), side left(image_3), side right(image_4). I write the coresponding data file folder in the brackets. But the pose info idx dismatch the image data file. I write a code in `get_data_info` function in `waymo_temporal_zlt.py` to fix this bug. 

    - the size of img0-2: 1280x1920; img3-4: 886x1920. All of them will be reshaped and padded to 640x960.

- coordinate: 
    
    - The whole data set coordinate system obeys the `right-hand rule`. 
    
    - The global coordinate system: the Up(z) axis is consistent with the direction of gravity, and upward is positive; East(x) points due east along latitude, and North(y) points to the North Pole. 
    
    - The vehicle coordinate system moves with the car, with the X-axis pointing forward, the Y-axis pointing to the left, and the Z-axis pointing up positive. 
    
    - The sensor coordinates can be obtained from the vehicle coordinates via the rotation matrix, which can be viewed as an external parameter matrix.

- Voxel semantics for each sample frame is given as `[semantics]` in the `labels.npz`. Please note that there is a slight difference between the Occupancy classes and the classes used in the [Waymo LiDAR segmentation](https://github.com/waymo-research/waymo-open-dataset/blob/bae19fa0a36664da18b691349955b95b29402713/waymo_open_dataset/protos/segmentation.proto#L20).

- The dataset contains 15 classes. The definition of classes from 0 to 14 is `TYPE_GENERALOBJECT, TYPE_VEHICLE, TYPE_PEDESTRIAN, TYPE_SIGN, TYPE_CYCLIST, TYPE_TRAFFIC_LIGHT, TYPE_POLE, TYPE_CONSTRUCTION_CONE, TYPE_BICYCLE, TYPE_MOTORCYCLE, TYPE_BUILDING, TYPE_VEGETATION, TYPE_TREE_TRUNK, TYPE_ROAD, TYPE_WALKABLE`.

- The label 15 category represents voxels that are not occupied by anything, which is named as `free`. Indeed `free` label is `23` in ground truth file. It is converted to `15` in dataloader.  

**1. Prepare Waymo dataset**

Download Waymo v1.3.1 full dataset from [Waymo website](https://waymo.com/open/download/). 

**2. Prepare 3D Occupancy ground truth**

Download the gts with voxel size 0.4m, annotation file(`waymo_infos_{train, val}.pkl`), and pose file(`cam_infos.pkl` and `cam_infos_vali.pkl`) we provided in [HERE](https://drive.google.com/drive/folders/13WxRl9Zb_AshEwvD96Uwz8cHjRNrtfQk) and organize your folder structure as below:

```
└── Occ3D-Waymo
    ├── waymo_infos_train.pkl
    ├── waymo_infos_val.pkl
    ├── cam_infos.pkl
    ├── cam_infos_vali.pkl
    ├── training
    |   ├── 000
    |   |   ├── 000_04.npz
    |   |   ├── 001_04.npz
    |   |   ├── 002_04.npz
    |   |   └── ...
    |   |     
    |   ├── 001
    |   |   ├── 000_04.npz
    |   |   └── ...
    |   ├── ...
    |   |
    |   └── 797
    |       ├── 000_04.npz
    |       └── ...
    |
    ├── validation
    |   ├── 000
    |   |   ├── 000_04.npz
    |   |   └── ...
    |   ├── ...
    |   |
    |   └── 201
    |       ├── 000_04.npz
    |       └── ...
```

- `training` and `validation`contains data for each scene. Each scene includes corresponding ground truth of each frame.

- `*.npz` contains `[voxel_label]`, `[origin_voxel_state]`, `[final_voxel_state]` , and `[infov]` for each frame. 

    - `[voxel_label]`: semantic ground truth. 

    - `[origin_voxel_state]`: lidar mask.

    - `[final_voxel_state]`: camera mask. Since we focus on a vision-centric task, we provide a binary voxel mask `[mask_camera]`, indicating whether the voxels are observed or not in the current camera view. 
    
    - `[infov]`: infov mask. Since Waymo only has 5 cameras and does not provide a 360-degree surround view, we additionally provide `[mask_fov]`. 

- `*_04.npz` represents the data with a voxel size of 0.4m.

## Occ3D-NuScenes

| Type |  Info |
| :----: | :----: |
| train           | 600 scenes|
| val             | 150 scenes|
| Time Span       | 20s |
| Frame           | 40 per scene |
| Time Interval   | 0.5s|
| cameras         | 6 |
| voxel size      | [0.4m, 0.4m, 0.4m] |
| range           | [-40m, -40m, -1m, 40m, 40m, 5.4m] |
| volume size     | [200, 200, 16]|
| classes         | 0 - 17 |

- sensor:
    
    - 6 cameras. Front, Front Right, Front Left, Back, Back Right, Back Left. 

    - size of image: 1600x900

- The dataset contains 18 classes. The definition of classes from 0 to 16 is the same as the [nuScenes-lidarseg](https://github.com/nutonomy/nuscenes-devkit/blob/fcc41628d41060b3c1a86928751e5a571d2fc2fa/python-sdk/nuscenes/eval/lidarseg/README.md) dataset. The label 17 category represents `free`. Voxel semantics for each sample frame is given as `[semantics]` in the labels.npz. 

**1. Prepare NuScenes dataset**

Download nuScenes V1.0 full dataset and can bus data from [NuScenes website](https://www.nuscenes.org/download). Organize the folder structure:

```
cvtocc
├── project code/
├── data/
│   ├── can_bus/
│   ├── occ3d-nus/
│   │   ├── maps/
│   │   ├── samples/
|   |   |   ├── CAM_BACK
|   |   |   |   ├── n015-2018-07-18-11-07-57+0800__CAM_BACK__1531883530437525.jpg
|   |   |   |   └── ...
|   |   |   ├── CAM_BACK_LEFT
|   |   |   |   ├── n015-2018-07-18-11-07-57+0800__CAM_BACK_LEFT__1531883530447423.jpg
|   |   |   |   └── ...
|   |   |   └── ...
│   │   ├── v1.0-trainval
```

- samples/ contains images captured by various cameras.

**2. Prepare 3D Occupancy ground truth**

Download the gts and annotations.json we provided in [HERE](https://drive.google.com/drive/folders/1Xarc91cNCNN3h8Vum-REbI-f0UlSf5Fc) and organize your folder structure as below:

```
cvtocc
├── data/
│   ├── can_bus/
│   ├── occ3d-nus/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── v1.0-trainval/
│   │   ├── gts/
|   |   |   ├── [scene_name]
|   |   |   |   ├── [frame_token]
|   |   |   |   |   └── labels.npz
|   |   |   |   └── ...
|   |   |   └── ...
│   │   └── annotations.json
```

- gts/ contains the ground truth of each sample. [scene_name] specifies a sequence of frames, and [frame_token] specifies a single frame in a sequence. `labels.npz` contains [semantics], [mask_lidar], and [mask_camera] for each frame.

- annotations.json contains meta infos of the dataset.

**3. Generate the info files for training and validation:**

```shell
python tools/create_data.py occ --root-path ./data/occ3d-nus --out-dir ./data/occ3d-nus --extra-tag occ --version v1.0 --canbus ./data --occ-path ./data/occ3d-nus
```

Using the above code will generate the following files `data/occ3d-nus/occ_infos_temporal_{train, val}.pkl`
