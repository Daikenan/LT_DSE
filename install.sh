#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "ERROR! Illegal number of parameters. Usage: bash install.sh conda_install_path environment_name"
    exit 0
fi

conda_install_path=$1
conda_env_name=$2

source $conda_install_path/etc/profile.d/conda.sh
echo "****************** Creating conda environment ${conda_env_name} python=3.6 ******************"
conda create -y --name $conda_env_name python=3.6

echo ""
echo ""
echo "****************** Activating conda environment ${conda_env_name} ******************"
conda activate $conda_env_name

echo ""
echo ""
echo "****************** Installing pytorch with cuda9 ******************"
conda install -y cudatoolkit=9.0 torchvision pytorch=0.4.1 -c pytorch


echo ""
echo ""
echo "****************** Installing matplotlib 2.2.2 ******************"
conda install -y matplotlib=2.2.2

echo ""
echo ""
echo "****************** Installing pandas ******************"
conda install -y pandas

echo ""
echo ""
echo "****************** Installing opencv ******************"
pip install opencv-python

echo ""
echo ""
echo "****************** Installing tensorboardX ******************"
pip install tensorboardX

echo ""
echo ""
echo "****************** Installing cython ******************"
conda install -y cython

echo ""
echo ""
echo "****************** Installing coco toolkit ******************"
pip install pycocotools

echo ""
echo ""
echo "****************** Installing jpeg4py python wrapper ******************"
pip install jpeg4py 
conda install -y tensorflow-gpu=1.9
conda install -y -c https://conda.binstar.org/menpo opencv
conda install -y scipy
conda install -y scikit-learn
pip install mmcv

cd mmdetection

echo "Building roi align op..."
cd mmdet/ops/roi_align
if [ -d "build" ]; then
    rm -r build
fi
python setup.py build_ext --inplace

echo "Building roi pool op..."
cd ../roi_pool
if [ -d "build" ]; then
    rm -r build
fi
python setup.py build_ext --inplace

echo "Building nms op..."
cd ../nms
make clean
python setup.py build_ext --inplace

echo "Building dcn..."
cd ../dcn
if [ -d "build" ]; then
    rm -r build
fi
python setup.py build_ext --inplace

cd ../../../

pip install .
cd ../

echo ""
echo ""
echo "****************** Installing ninja-build to compile PreROIPooling ******************"
sudo apt-get install ninja-build

echo ""
echo ""
echo "****************** ATOM Network ******************"
bash pytracking/utils/gdrive_download 1ZTdQbZ1tyN27UIwUnUrjHChQb5ug2sxr pytracking/networks/atom_default.pth

echo ""
echo ""
echo "****************** Downloading other networks ******************"
bash pytracking/utils/gdrive_download 1sZ_j5sre7356nSGSdY7djX7ygpxPyzPN model.zip
unzip model.zip

cd SiamMask/sm_utils/pyvotkit
python setup.py build_ext --inplace
cd ../../../

cd SiamMask/sm_utils/pysot/utils/
python setup.py build_ext --inplace
cd ../../../../

cd SiamMask/experiments/siammask
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT_LD.pth
cd ../../../

cd mmdetection
wget https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth
cd ../

echo ""
echo ""
echo "****************** Installation complete! ******************"

