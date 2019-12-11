import os
import sys
import argparse
import torch
import numpy as np
from pytracking.evaluation.otbdataset import OTBDataset
from numpy.random import seed
from tensorflow import set_random_seed
seed(1)
set_random_seed(2)
torch.manual_seed(456)
torch.cuda.manual_seed(789)

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)
from pytracking.evaluation import Tracker
# dataset = OTBDataset()
# sequence = dataset['Bolt']
dataset_path = '/home/daikenan/dataset/VOT2019/lt2019'
video = 'deer'
img_list = os.listdir(os.path.join(dataset_path, video, 'color'))
img_list.sort()
groundtruth = np.loadtxt(os.path.join(dataset_path, video, 'groundtruth.txt'), dtype=np.float64, delimiter=',')
atom_tracker = Tracker('atom', 'default', None)
atom = atom_tracker.tracker_class(atom_tracker.parameters)
image = atom._read_image(os.path.join(dataset_path, video, 'color', img_list[0]))
atom.initialize(image, groundtruth[0])
atom.init_visualization()
atom.visualize(image, groundtruth[0])

tracked_bb = [groundtruth[0]]
for id, frame in enumerate(img_list[1:]):
    image = atom._read_image(os.path.join(dataset_path, video, 'color', frame))
    state, flag = atom.track(image)
    tracked_bb.append(state)
    atom.visualize(image, state)
    print("%s: %d / %d" % (video, id, len(img_list)))
    if flag is not None:
        print(flag)


