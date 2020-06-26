#!/bin/bash

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

cd SiamMask/sm_utils/pyvotkit
python setup.py build_ext --inplace
cd ../../../

cd SiamMask/sm_utils/pysot/utils/
python setup.py build_ext --inplace
cd ../../../../

cd ltr/external/PreciseRoIPooling/pytorch/prroi_pool
if [ -d "_prroi_pooling" ]; then
    rm -r _prroi_pooling
fi
bash travis.sh

