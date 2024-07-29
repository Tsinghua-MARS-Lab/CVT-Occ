# Training & Evaluation

## training
### single GPU
```sh
./tools/dist_train.sh projects/configs/cvtocc/cvtocc_waymo.py 1
```

### single machine
```sh
./tools/dist_train.sh projects/configs/cvtocc/cvtocc_waymo.py 8
```

## evaluation
### single GPU
```sh
./tools/dist_test.sh projects/configs/cvtocc/cvtocc_waymo.py work_dirs/cvtocc_waymo/latest.pth 1 --eval mIoU
```
### single machine
```sh
./tools/dist_test.sh projects/configs/cvtocc/cvtocc_waymo.py work_dirs/cvtocc_waymo/latest.pth 8 --eval mIoU
```

## save results
### single machine
```sh
./tools/dist_test.sh projects/configs/cvtocc/cvtocc_waymo.py work_dirs/cvtocc_waymo/latest.pth 8 --out work_dirs/cvtocc_waymo/results.pkl
```