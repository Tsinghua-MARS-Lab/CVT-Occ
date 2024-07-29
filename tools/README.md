# Tools
## test_data_pipeline.py
### usage
This file allows you to run the entire dataloader pipeline without requiring CUDA, making it very convenient for debugging.

### config parameter
config: Path to the required config file.

### command line
```sh
python -m tools.test_data_pipeline
```

## condition_benchmark.py
### usage
This script calls the pose file of the specified dataset and divides each scene based on the average speed of the vehicle. Its output is a list containing all scenes with ego car speed larger than the threshold. 

### config parameter
threshold: The standard for speed division.
pose_file: Path to the dataset pose parameter file.

### command line
```sh
python tools/condition_benchmark.py
```

## train.py
### usage
called by `dist_train.sh` and `slurm_train.sh`. Run the training process. 

## test.py
### usage 
called by `dist_test.sh` and `slurm_test.sh`. Run the training process. We can directly get the metric results(`--eval mIoU`) or save the occupancy prediction results into files(`--out`). 

## dist_train.sh
### usage
Run the training process. 

### command line
```sh
./tools/dist_train.sh path/to/config/file GPU_NUM
```

## dist_test.sh
### usage
Run the evaluation process. 

### command line
```sh
./tools/dist_test.sh path/to/config/file path/to/checkpoint/file GPU_NUM --eval mIoU
./tools/dist_test.sh path/to/config/file path/to/checkpoint/file GPU_NUM --out
```

## slurm_train.sh
### usage
Run the training process by `srun`.

### command line
```sh
GPUS=${NUM_GPUS} ./tools/slurm_train.sh brie1 train path/to/config/file
```
Explanation: 
PARTITION: `brie1`
JOB_NAME: `train`
NUM_GPUS: The code logic of this script requires the input to be a multiple of 8. If you want to run the code with a different number of GPUs, please modify the script according to the srun cluster documentation.

## slrum_test.sh
### usage
Run the evaluation process by `srun`.

### command line
```sh
GPUS=${NUM_GPUS} ./tools/slurm_test.sh brie1 test path/to/config/file path/to/checkpoint/file --eval mIoU
GPUS=${NUM_GPUS} ./tools/slurm_test.sh brie1 test path/to/config/file path/to/checkpoint/file --out
```
Explanation: 
PARTITION: `brie1`
JOB_NAME: `test`
NUM_GPUS: The code logic of this script requires the input to be a multiple of 8. If you want to run the code with a different number of GPUs, please modify the script according to the srun cluster documentation. 

## create_data.py
### usage
In the dataset preparing process, we need to use this script to generate info files. For more information, you can see 'https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction/blob/main/docs/getting_started.md'

### command line
```sh
python tools/create_data.py occ --root-path ./data/occ3d-nus --out-dir ./data/occ3d-nus --extra-tag occ --version v1.0 --canbus ./data --occ-path ./data/occ3d-nus
```