# LT_DSE

### :star2:The Winner Tracker of the VOT-2019 LT Challenge.:star2: 
### News!The installation script has been updated for easier testing.

If you failed to install and run this tracker, please email me (<dkn2014@mail.dlut.edu.cn>)

# Prerequisites

* ubuntu 16.04
* anaconda3
* gcc 5.4
* g++ 5.4
# Installation
1. Clone the GIT repository:
```
 $ git clone https://github.com/Daikenan/LT_DSE.git
```
2. Clone the submodules.  
   In the repository directory, run the commands:
```
   $ git submodule init  
   $ git submodule update
```
3. Run the install script. 
```
conda env create -f LTDSE.yaml
source activate LTDSE
bash compile.sh
```
4.Download models
```
bash download_models.sh
```
You can also download it manually.

[atom model](https://docs.google.com/uc?export=download&id=1VNyr-Ds0khjM0zaq6lU-xfY74-iWxBvU)
```
|—— pytracking
  |—— networks
    |—— atom_default.pth
```
[det model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth)
```
|—— mmdetection
  |—— faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth
```
[siammask model](http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT_LD.pth)
```
|—— SiamMask
  |—— experiments
    |—— siammask
      |—— SiamMask_VOT_LD.pth
```
[other models](https://docs.google.com/uc?export=download&id=1sZ_j5sre7356nSGSdY7djX7ygpxPyzPN)
```
|—— model
  |—— R_model
    |—— xxx.ckpt/checkpoint
  |—— mobilenet_v1_1.0_224_2017_06_14
    |—— xxx.ckpt
  |—— ssd_mobilenet_tracking.config
  |—— ssd_mobilenet_video.config
```

5. modify ``local_path.py``:

``toolkit_path`` is not needed if you don't test on VOT toolkit.

6. Run the demo script to test the tracker:
```
source activate LTDSE
python LT_DSE_Demo.py
```

# Integrate into VOT-2019LT

## VOT-toolkit
Before running the toolkit, please change the environment path to use the python in the conda environment "votenvs".
For example, in my computer, I add  `export PATH=/home/daikenan/anaconda3/envs/votenvs/bin:$PATH` to the `~/.bashrc` file.  

The interface for integrating the tracker into the vot evaluation tool kit is implemented in the module `tracker_vot.py`. The script `tracker_LT_DSE.m` is needed to be copied to vot-tookit. 

Since the vot-toolkit may be not compatible with pytorch-0.4.1, I always change the line  `command = sprintf('%s %s -c "%s"', python_executable, argument_string, python_script);` to `command = sprintf('env -i %s %s -c "%s"', python_executable, argument_string, python_script);` in `generate_python_command.m`. 

# Introduction
This algorithm divides each long-term sequence into several short episodes and tracks the target in each episode using short-term tracking techniques. Whether the target is visible or not is judged by the outputs from the short-term local tracker and the classification-based verifier updated online. If the target disappears, the image-wide re-detection, which is carried out by sliding window method, will be conducted and output the possible location and size of the target. Based on these, the tracker crops the local search region that may include the target and sends it to the RPN based regression network. Then, the candidate proposals from the regression network will be scored by the online learned verifier. If the candidate with the maximum score is above the pre-defined threshold, the tracker will regard it as the target and re-initialize the short-term components. Finally, the tracker conducts short-term tracking until the target disappears again. 
The short-term local tracker contains two components. One is for target localization and based on ATOM algorithm[1]. It uses ResNet18 as the backbone network and adds two convolutional layers above it. The input of it is the local search region and it outputs a single response map, in which the center of the target has the highest response. The parameter update method is the same as that of ATOM. The other component is the SiamMask network[2] and used for refining the bounding box after locating the center of the target. It also takes the local search region as the input and outputs the tight bounding boxes of candidate proposals. 
For the verifier, we adopts RT-MDNet network[3] which uses VGGM as the backbone and is pre-trained on ILSVRC VID dataset. The input of it is the local search region as well and in each frame, we crop the feature of the search region outputted by the third convolutional layer via its improved ROIAlign pooling method to get the feature of the tracking result. The classification score is finally obtained by sending the tracking result's feature to three fully connected layers. 

The architecture of the region-proposal network is the same as that used in [4]. It uses MobileNet architecture as the feature extractor and takes the local search region as input. The RPN network consists of three convolutional layers and outputs the bounding boxes of candidate proposals. The network is trained using LaSOT dataset[5] and ILSVRC image detection dataset. 
```
[1] @inproceedings{danelljan2018atom,
  title={ATOM: Accurate Tracking by Overlap Maximization},
  author={Danelljan, Martin and Bhat, Goutam and Khan, Fahad Shahbaz and Felsberg, Michael},
  booktitle={CVPR},
  year={2019}
}
[2] @inproceedings{wang2019fast,
  title={Fast online object tracking and segmentation: A unifying approach},
  author={Wang, Qiang and Zhang, Li and Bertinetto, Luca and Hu, Weiming and Torr, Philip HS},
  booktitle={CVPR},
  pages={1328--1338},
  year={2019}
}


[3] @inproceedings{jung2018real,
  title={Real-time mdnet},
  author={Jung, Ilchae and Son, Jeany and Baek, Mooyeol and Han, Bohyung},
  booktitle={ECCV},
  pages={83--98},
  year={2018}
}


[4] @article{zhang2018learning,
  title={Learning regression and verification networks for long-term visual tracking},
  author={Zhang, Yunhua and Wang, Dong and Wang, Lijun and Qi, Jinqing and Lu, Huchuan},
  journal={arXiv preprint arXiv:1809.04320},
  year={2018}
}


[5] @inproceedings{fan2018lasot,
  title={Lasot: A high-quality benchmark for large-scale single object tracking},
  author={Fan, Heng and Lin, Liting and Yang, Fan and Chu, Peng and Deng, Ge and Yu, Sijia and Bai, Hexin and Xu, Yong and Liao, Chunyuan and Ling, Haibin},
  booktitle={CVPR},
  year={2019}
}
```
