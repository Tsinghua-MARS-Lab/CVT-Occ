# Installation instructions

**a. Create a conda virtual environment and activate it.**
```shell
conda create -n cvtocc python=3.8
conda activate cvtocc
```

**b. Install PyTorch, torchvision and torchaudio following the [official instructions](https://pytorch.org/).**
pip install
```shell
pip install numpy==1.19.5 # (mmdetetion3d need numpy<1.20.0)
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113
```

conda install
```shell
conda install numpy=1.19.5 # (mmdetetion3d need numpy<1.20.0)
conda install pytorch==1.10.0 torchvision==0.11.1 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch
```

**c. Install mmcv-full.**
```shell
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
```
I encountered a small bug here. One of the dependency of mmcv-full==1.4.0 is yapf==0.40.2. It should be degraded to yapf==0.33.0. 

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
rm requirements/optional.txt # don't install the optional packages
pip install -r requirements.txt
python setup.py install # This will take some time to compile
```

If you encounter this bug "module 'distutils' has no attribute 'version'" after finishing all steps above, degrade the setuptools to 59.5.0.

If you do not intend to run the SOLOFusion model, the current environment setup should suffice. However, import statements in the SOLOFusion module in the code may cause interruptions. I recommend removing or commenting out those import statements to avoid any issues during execution.

You can also build the environment directly from the file [environment.yaml](../environment.yaml) or [requirements.txt](../requirements.txt). 

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