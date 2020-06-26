#!/bin/bash

echo ""
echo "****************** ATOM Network ******************"
bash pytracking/utils/gdrive_download 1ZTdQbZ1tyN27UIwUnUrjHChQb5ug2sxr pytracking/networks/atom_default.pth

echo ""
echo ""
echo "****************** Downloading other networks ******************"
bash pytracking/utils/gdrive_download 1sZ_j5sre7356nSGSdY7djX7ygpxPyzPN model.zip
unzip model.zip


cd SiamMask/experiments/siammask
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT_LD.pth
cd ../../../

cd mmdetection
wget https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth
cd ../

echo ""
echo ""
echo "****************** Download complete! ******************"

