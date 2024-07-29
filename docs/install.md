# Installation instructions

Following https://mmdetection3d.readthedocs.io/en/v0.17.1/getting_started.html#installation

**a. Create a conda virtual environment and activate it.**
```shell
conda create -n cvtocc python=3.8 -y
conda activate cvtocc
```

**b. Install PyTorch, torchvision and torchaudio following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

**c. Install mmcv-full.**
```shell
pip install mmcv-full==1.4.0
```

**d. Install mmdet and mmseg.**
```shell
pip install mmdet==2.22.0
pip install mmsegmentation==0.14.1
```

**e. Install mmdet3d from source code.**
```shell
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1
python setup.py install
```
**g. Clone cvtocc.**
```
git clone git@github.com:Tsinghua-MARS-Lab/CVT-Occ.git
```

**h. Prepare pretrained models.**
```shell
cd cvtocc
mkdir ckpts

cd ckpts & wget https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth
```