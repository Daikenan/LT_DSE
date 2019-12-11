#coding=utf-8
import cv2 as cv
import os
import tensorflow as tf
import numpy as np
from local_path import base_path
from local_path import toolkit_path
import sys
sys.path.append(toolkit_path + 'native/trax/support/python')
sys.path.append(os.path.join(base_path, 'lib'))
sys.path.append(os.path.join(base_path, 'lib/slim'))
sys.path.append(os.path.join(base_path, 'RT_MDNet'))
sys.path.append(os.path.join(base_path, 'mmdetection'))
sys.path.insert(0,os.path.join(base_path, 'RT_MDNet/modules'))
sys.path.append(os.path.join(base_path, 'SiamMask'))
sys.path.append(os.path.join(base_path, 'SiamMask/experiments/siammask'))

# rtmdnet
from rtmdnet_utils import *
sys.path.insert(0,os.path.join(base_path, 'RT_MDNet/modules'))
from rt_sample_generator import *
from data_prov import *

from rtmdnet_model import *
from rtmdnet_options import *
from img_cropper import *

from roi_align.modules.roi_align import RoIAlignAvg,RoIAlignMax,RoIAlignAdaMax,RoIAlignDenseAdaMax

from bbreg import *

from RT_MDNet.tracker import set_optimizer, rt_train

# atom
import argparse
from pytracking.libs.tensorlist import TensorList
from pytracking.utils.plotting import show_tensor
from pytracking.features.preprocessing import numpy_to_torch
env_path = os.path.join(os.path.dirname(__file__))
if env_path not in sys.path:
    sys.path.append(env_path)
from pytracking.evaluation import Tracker
# siammask
from custom import Custom
from tools.test import *
# mmdetection
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000
import vot
from tracking_utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"





class p_config(object):
    name = 'a'
    R_loss_thr = 0.3
    Verification = "rtmdnet"
    Regressor = "mrpn"
    visualization = True
    R_candidates = 20
    confidence_pool_num = 200
    R_model_path = base_path + 'model/R_model'

def _compute_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    if xA < xB and yA < yB:
        # compute the area of intersection rectangle
        interArea = (xB - xA) * (yB - yA)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
    else:
        iou = 0

    assert iou >= 0
    assert iou <= 1.01

    return iou
def get_mmresult(img, result, dataset='coco', score_thr=0.3):
    # class_names = get_classes(dataset)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)
    bboxes = np.vstack(result)
    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
    return bboxes, labels
class Region:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

class MobileTracker(object):
    def __init__(self, image, region, video=None, p=None, groundtruth=None):
        self.p = p
        self.i = 0
        self.globalmode = True
        if groundtruth is not None:
            self.groundtruth = groundtruth
        self.confidence_pool = np.ones(self.p.confidence_pool_num)
        self.V_pool = np.zeros(self.p.confidence_pool_num)
        init_training = True
        config_file = os.path.join(base_path, 'model/ssd_mobilenet_tracking.config')
        checkpoint_dir = os.path.join(base_path, self.p.R_model_path)

        model_config, train_config, input_config, eval_config = get_configs_from_pipeline_file(config_file)
        model = build_man_model(model_config=model_config, is_training=False)

        model_scope = 'model'
        self.initFeatOp, self.initInputOp = build_init_graph(model, model_scope, reuse=None)
        self.initConstantOp = tf.placeholder(tf.float32, [1,1,1,512])
        self.pre_box_tensor, self.scores_tensor, self.input_cur_image = build_box_predictor(model, model_scope, self.initConstantOp, reuse=None)

        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.4
        self.sess = tf.Session(config=tfconfig)
        self.sess.run(tf.global_variables_initializer())
        #if not init_training:
        variables_to_restore = tf.global_variables()
        restore_model(self.sess, model_scope, checkpoint_dir, variables_to_restore)

        init_img = Image.fromarray(image)
        init_gt1 = [region.x,region.y,region.width,region.height]
        # init_gt1 = [region[0], region[1], region[2], region[3]]
        init_gt = [init_gt1[1], init_gt1[0], init_gt1[1]+init_gt1[3], init_gt1[0]+init_gt1[2]] # ymin xmin ymax xmax
        init_img_array = np.array(init_img)
        self.expand_channel = False
        if init_img_array.ndim < 3:
            init_img_array = np.expand_dims(init_img_array, axis=2)
            init_img_array = np.repeat(init_img_array, repeats=3, axis=2)
            init_img = Image.fromarray(init_img_array)
            self.expand_channel = True

        gt_boxes = np.zeros((1,4))
        gt_boxes[0,0] = init_gt[0] / float(init_img.height)
        gt_boxes[0,1] = init_gt[1] / float(init_img.width)
        gt_boxes[0,2] = init_gt[2] / float(init_img.height)
        gt_boxes[0,3] = init_gt[3] / float(init_img.width)

        img1_xiaobai = np.array(init_img)
        pad_x = 36.0 / 264.0 * (gt_boxes[0, 3] - gt_boxes[0, 1]) * init_img.width
        pad_y = 36.0 / 264.0 * (gt_boxes[0, 2] - gt_boxes[0, 0]) * init_img.height
        cx = (gt_boxes[0, 3] + gt_boxes[0, 1]) / 2.0 * init_img.width
        cy = (gt_boxes[0, 2] + gt_boxes[0, 0]) / 2.0 * init_img.height
        startx = gt_boxes[0, 1] * init_img.width - pad_x
        starty = gt_boxes[0, 0] * init_img.height - pad_y
        endx = gt_boxes[0, 3] * init_img.width + pad_x
        endy = gt_boxes[0, 2] * init_img.height + pad_y
        left_pad = max(0, int(-startx))
        top_pad = max(0, int(-starty))
        right_pad = max(0, int(endx - init_img.width + 1))
        bottom_pad = max(0, int(endy - init_img.height + 1))

        startx = int(startx + left_pad)
        starty = int(starty + top_pad)
        endx = int(endx + left_pad)
        endy = int(endy + top_pad)

        if top_pad or left_pad or bottom_pad or right_pad:
            r = np.pad(img1_xiaobai[:, :, 0], ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant',
                       constant_values=128)
            g = np.pad(img1_xiaobai[:, :, 1], ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant',
                       constant_values=128)
            b = np.pad(img1_xiaobai[:, :, 2], ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant',
                       constant_values=128)
            r = np.expand_dims(r, 2)
            g = np.expand_dims(g, 2)
            b = np.expand_dims(b, 2)


            img1_xiaobai = np.concatenate((r, g, b), axis=2)
        img1_xiaobai = Image.fromarray(img1_xiaobai)
        im = np.array(init_img)
        # gt_boxes resize
        init_img_crop = img1_xiaobai.crop(np.int32([startx, starty, endx, endy]))
        init_img_crop = init_img_crop.resize([128,128], resample=Image.BILINEAR)
        self.last_gt = init_gt

        self.init_img_array = np.array(init_img_crop)
        self.init_feature_maps = self.sess.run(self.initFeatOp, feed_dict={self.initInputOp:self.init_img_array})

        if self.p.Verification == "rtmdnet":
            self.init_rtmdnet(image, init_gt1)
        else:
            ValueError()
        self.local_init(image, init_gt1)
        # mmdetection
        self.cfg = mmcv.Config.fromfile(os.path.join(base_path, 'mmdetection/configs/faster_rcnn_r50_fpn_1x.py'))
        self.cfg.model.pretrained = None
        self.mm_model = build_detector(self.cfg.model, test_cfg=self.cfg.test_cfg)
        _ = load_checkpoint(self.mm_model, os.path.join(base_path, 'mmdetection/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'))
        result = inference_detector(self.mm_model, cv.cvtColor(image, cv.COLOR_RGB2BGR), self.cfg)
        bboxes, labels = get_mmresult(image, result)
        boxes = bboxes[:, :4]
        mmscore = bboxes[:, -1]
        boxes = np.array([boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]])
        iou = np.zeros((boxes.shape[1],))
        for i in range(boxes.shape[1]):
            iou[i] = _compute_iou(boxes[:, i], init_gt1)
        if iou.shape[0] == 0:
            self.label = None
        elif max(iou) > 0.4 or max(iou) > 0.1 and mmscore[np.argmax(iou)] > 0.5:
            self.label = labels[np.argmax(iou)]
        else:
            self.label = None
        self.v = 0.3
        if self.label is not None:
            index = labels == self.label
            candicate_boxes = bboxes[index]
            if candicate_boxes.shape[0] > 10:
                self.v = 0.8
            elif candicate_boxes.shape[0] > 3:
                self.v = 0.5
            # print(video+":"+str(candicate_boxes.shape[0]))
        if self.label is None:
            self.v = 0.3

        # siammask
        self.siammask_init(image, init_gt1)
        self.V_reliable_pool = np.ones(self.p.confidence_pool_num) * self.first_score
        self.target_w = init_gt[3] - init_gt[1]
        self.target_h = init_gt[2] - init_gt[0]
        self.last_reliable_h = self.target_h
        self.last_reliable_w = self.target_w

        self.first_w = init_gt[3] - init_gt[1]
        self.first_h = init_gt[2] - init_gt[0]
        self.pos_regions_record = []
        self.neg_regions_record = []
        self.startx = 0
        self.starty = 0
        self.count = 0

    def init_rtmdnet(self, image, init_bbox):
        target_bbox = np.array(init_bbox)
        self.last_result = target_bbox
        self.rtmodel = RTMDNet(rt_opts['model_path'])
        if rt_opts['adaptive_align']:
            align_h = self.rtmodel.roi_align_model.aligned_height
            align_w = self.rtmodel.roi_align_model.aligned_width
            spatial_s = self.rtmodel.roi_align_model.spatial_scale
            self.rtmodel.roi_align_model = RoIAlignAdaMax(align_h, align_w, spatial_s)
        if rt_opts['use_gpu']:
            self.rtmodel = self.rtmodel.cuda()

            self.rtmodel.set_learnable_params(rt_opts['ft_layers'])

        # Init image crop model
        self.img_crop_model = imgCropper(1.)
        if rt_opts['use_gpu']:
            self.img_crop_model.gpuEnable()

        # Init criterion and optimizer
        self.criterion = BinaryLoss()
        init_optimizer = set_optimizer(self.rtmodel, rt_opts['lr_init'])
        self.rtupdate_optimizer = set_optimizer(self.rtmodel, rt_opts['lr_update'])

        tic = time.time()
        # Load first image
        cur_image = Image.fromarray(image).convert('RGB')
        cur_image = np.asarray(cur_image)

        # Draw pos/neg samples
        ishape = cur_image.shape
        pos_examples = gen_samples(RT_SampleGenerator('gaussian', (ishape[1], ishape[0]), 0.1, 1.2),
                                   target_bbox, rt_opts['n_pos_init'], rt_opts['overlap_pos_init'])
        neg_examples = gen_samples(RT_SampleGenerator('uniform', (ishape[1], ishape[0]), 1, 2, 1.1),
                                   target_bbox, rt_opts['n_neg_init'], rt_opts['overlap_neg_init'])
        neg_examples = np.random.permutation(neg_examples)

        # cur_bbreg_examples = gen_samples(SampleGenerator('uniform', (ishape[1], ishape[0]), 0.3, 1.5, 1.1),
        #                                  target_bbox, rt_opts['n_bbreg'], rt_opts['overlap_bbreg'], rt_opts['scale_bbreg'])

        # compute padded sample
        padded_x1 = (neg_examples[:, 0] - neg_examples[:, 2] * (rt_opts['padding'] - 1.) / 2.).min()
        padded_y1 = (neg_examples[:, 1] - neg_examples[:, 3] * (rt_opts['padding'] - 1.) / 2.).min()
        padded_x2 = (neg_examples[:, 0] + neg_examples[:, 2] * (rt_opts['padding'] + 1.) / 2.).max()
        padded_y2 = (neg_examples[:, 1] + neg_examples[:, 3] * (rt_opts['padding'] + 1.) / 2.).max()
        padded_scene_box = np.reshape(np.asarray((padded_x1, padded_y1, padded_x2 - padded_x1, padded_y2 - padded_y1)),
                                      (1, 4))

        scene_boxes = np.reshape(np.copy(padded_scene_box), (1, 4))
        if rt_opts['jitter']:
            ## horizontal shift
            jittered_scene_box_horizon = np.copy(padded_scene_box)
            jittered_scene_box_horizon[0, 0] -= 4.
            jitter_scale_horizon = 1.

            ## vertical shift
            jittered_scene_box_vertical = np.copy(padded_scene_box)
            jittered_scene_box_vertical[0, 1] -= 4.
            jitter_scale_vertical = 1.

            jittered_scene_box_reduce1 = np.copy(padded_scene_box)
            jitter_scale_reduce1 = 1.1 ** (-1)

            ## vertical shift
            jittered_scene_box_enlarge1 = np.copy(padded_scene_box)
            jitter_scale_enlarge1 = 1.1 ** (1)

            ## scale reduction
            jittered_scene_box_reduce2 = np.copy(padded_scene_box)
            jitter_scale_reduce2 = 1.1 ** (-2)
            ## scale enlarge
            jittered_scene_box_enlarge2 = np.copy(padded_scene_box)
            jitter_scale_enlarge2 = 1.1 ** (2)

            scene_boxes = np.concatenate(
                [scene_boxes, jittered_scene_box_horizon, jittered_scene_box_vertical, jittered_scene_box_reduce1,
                 jittered_scene_box_enlarge1, jittered_scene_box_reduce2, jittered_scene_box_enlarge2], axis=0)
            jitter_scale = [1., jitter_scale_horizon, jitter_scale_vertical, jitter_scale_reduce1,
                            jitter_scale_enlarge1, jitter_scale_reduce2, jitter_scale_enlarge2]
        else:
            jitter_scale = [1.]

            self.rtmodel.eval()
        for bidx in range(0, scene_boxes.shape[0]):
            crop_img_size = (scene_boxes[bidx, 2:4] * (
                    (rt_opts['img_size'], rt_opts['img_size']) / target_bbox[2:4])).astype(
                'int64') * jitter_scale[bidx]
            cropped_image, cur_image_var = self.img_crop_model.crop_image(cur_image,
                                                                          np.reshape(scene_boxes[bidx], (1, 4)),
                                                                          crop_img_size)
            cropped_image = cropped_image - 128.

            feat_map = self.rtmodel(cropped_image, out_layer='conv3')

            rel_target_bbox = np.copy(target_bbox)
            rel_target_bbox[0:2] -= scene_boxes[bidx, 0:2]

            batch_num = np.zeros((pos_examples.shape[0], 1))
            cur_pos_rois = np.copy(pos_examples)
            cur_pos_rois[:, 0:2] -= np.repeat(np.reshape(scene_boxes[bidx, 0:2], (1, 2)), cur_pos_rois.shape[0], axis=0)
            scaled_obj_size = float(rt_opts['img_size']) * jitter_scale[bidx]
            cur_pos_rois = samples2maskroi(cur_pos_rois, self.rtmodel.receptive_field,
                                           (scaled_obj_size, scaled_obj_size),
                                           target_bbox[2:4], rt_opts['padding'])
            cur_pos_rois = np.concatenate((batch_num, cur_pos_rois), axis=1)
            cur_pos_rois = Variable(torch.from_numpy(cur_pos_rois.astype('float32'))).cuda()
            cur_pos_feats = self.rtmodel.roi_align_model(feat_map, cur_pos_rois)
            cur_pos_feats = cur_pos_feats.view(cur_pos_feats.size(0), -1).data.clone()

            batch_num = np.zeros((neg_examples.shape[0], 1))
            cur_neg_rois = np.copy(neg_examples)
            cur_neg_rois[:, 0:2] -= np.repeat(np.reshape(scene_boxes[bidx, 0:2], (1, 2)), cur_neg_rois.shape[0], axis=0)
            cur_neg_rois = samples2maskroi(cur_neg_rois, self.rtmodel.receptive_field,
                                           (scaled_obj_size, scaled_obj_size),
                                           target_bbox[2:4], rt_opts['padding'])
            cur_neg_rois = np.concatenate((batch_num, cur_neg_rois), axis=1)
            cur_neg_rois = Variable(torch.from_numpy(cur_neg_rois.astype('float32'))).cuda()
            cur_neg_feats = self.rtmodel.roi_align_model(feat_map, cur_neg_rois)
            cur_neg_feats = cur_neg_feats.view(cur_neg_feats.size(0), -1).data.clone()

            # ## bbreg rois
            # batch_num = np.zeros((cur_bbreg_examples.shape[0], 1))
            # cur_bbreg_rois = np.copy(cur_bbreg_examples)
            # cur_bbreg_rois[:, 0:2] -= np.repeat(np.reshape(scene_boxes[bidx, 0:2], (1, 2)), cur_bbreg_rois.shape[0],
            #                                     axis=0)
            # scaled_obj_size = float(rt_opts['img_size']) * jitter_scale[bidx]
            # cur_bbreg_rois = samples2maskroi(cur_bbreg_rois, self.rtmodel.receptive_field, (scaled_obj_size, scaled_obj_size),
            #                                  target_bbox[2:4], rt_opts['padding'])
            # cur_bbreg_rois = np.concatenate((batch_num, cur_bbreg_rois), axis=1)
            # cur_bbreg_rois = Variable(torch.from_numpy(cur_bbreg_rois.astype('float32'))).cuda()
            # cur_bbreg_feats = self.rtmodel.roi_align_model(feat_map, cur_bbreg_rois)
            # cur_bbreg_feats = cur_bbreg_feats.view(cur_bbreg_feats.size(0), -1).data.clone()

            self.rtfeat_dim = cur_pos_feats.size(-1)

            if bidx == 0:
                pos_feats = cur_pos_feats
                neg_feats = cur_neg_feats
                ##bbreg feature
                # bbreg_feats = cur_bbreg_feats
                # bbreg_examples = cur_bbreg_examples
            else:
                pos_feats = torch.cat((pos_feats, cur_pos_feats), dim=0)
                neg_feats = torch.cat((neg_feats, cur_neg_feats), dim=0)
                ##bbreg feature
                # bbreg_feats = torch.cat((bbreg_feats, cur_bbreg_feats), dim=0)
                # bbreg_examples = np.concatenate((bbreg_examples, cur_bbreg_examples), axis=0)

        if pos_feats.size(0) > rt_opts['n_pos_init']:
            pos_idx = np.asarray(range(pos_feats.size(0)))
            np.random.shuffle(pos_idx)
            pos_feats = pos_feats[pos_idx[0:rt_opts['n_pos_init']], :]
        if neg_feats.size(0) > rt_opts['n_neg_init']:
            neg_idx = np.asarray(range(neg_feats.size(0)))
            np.random.shuffle(neg_idx)
            neg_feats = neg_feats[neg_idx[0:rt_opts['n_neg_init']], :]

        # ##bbreg
        # if bbreg_feats.size(0) > rt_opts['n_bbreg']:
        #     bbreg_idx = np.asarray(range(bbreg_feats.size(0)))
        #     np.random.shuffle(bbreg_idx)
        #     bbreg_feats = bbreg_feats[bbreg_idx[0:rt_opts['n_bbreg']], :]
        #     bbreg_examples = bbreg_examples[bbreg_idx[0:rt_opts['n_bbreg']], :]
        #     # print bbreg_examples.shape

        ## open images and crop patch from obj
        extra_obj_size = np.array((rt_opts['img_size'], rt_opts['img_size']))
        extra_crop_img_size = extra_obj_size * (rt_opts['padding'] + 0.6)
        replicateNum = 100
        for iidx in range(replicateNum):
            extra_target_bbox = np.copy(target_bbox)

            extra_scene_box = np.copy(extra_target_bbox)
            extra_scene_box_center = extra_scene_box[0:2] + extra_scene_box[2:4] / 2.
            extra_scene_box_size = extra_scene_box[2:4] * (rt_opts['padding'] + 0.6)
            extra_scene_box[0:2] = extra_scene_box_center - extra_scene_box_size / 2.
            extra_scene_box[2:4] = extra_scene_box_size

            extra_shift_offset = np.clip(2. * np.random.randn(2), -4, 4)
            cur_extra_scale = 1.1 ** np.clip(np.random.randn(1), -2, 2)

            extra_scene_box[0] += extra_shift_offset[0]
            extra_scene_box[1] += extra_shift_offset[1]
            extra_scene_box[2:4] *= cur_extra_scale[0]

            scaled_obj_size = float(rt_opts['img_size']) / cur_extra_scale[0]

            cur_extra_cropped_image, _ = self.img_crop_model.crop_image(cur_image, np.reshape(extra_scene_box, (1, 4)),
                                                                        extra_crop_img_size)
            cur_extra_cropped_image = cur_extra_cropped_image.detach()

            cur_extra_pos_examples = gen_samples(RT_SampleGenerator('gaussian', (ishape[1], ishape[0]), 0.1, 1.2),
                                                 extra_target_bbox, rt_opts['n_pos_init'] // replicateNum,
                                                 rt_opts['overlap_pos_init'])
            cur_extra_neg_examples = gen_samples(RT_SampleGenerator('uniform', (ishape[1], ishape[0]), 0.3, 2, 1.1),
                                                 extra_target_bbox, rt_opts['n_neg_init'] // replicateNum // 4,
                                                 rt_opts['overlap_neg_init'])

        torch.cuda.empty_cache()
        self.rtmodel.zero_grad()

        # Initial training
        rt_train(self.rtmodel, self.criterion, init_optimizer, pos_feats, neg_feats, rt_opts['maxiter_init'])
        self.first_score = self.rtmdnet_eval(np.reshape(target_bbox, (1, 4)), cur_image)
        self.first_score = self.first_score[0, 1].data.cpu().numpy().reshape(1)[0]

        if pos_feats.size(0) > rt_opts['n_pos_update']:
            pos_idx = np.asarray(range(pos_feats.size(0)))
            np.random.shuffle(pos_idx)
            self.rtpos_feats_all = [
                pos_feats.index_select(0, torch.from_numpy(pos_idx[0:rt_opts['n_pos_update']]).cuda())]
        if neg_feats.size(0) > rt_opts['n_neg_update']:
            neg_idx = np.asarray(range(neg_feats.size(0)))
            np.random.shuffle(neg_idx)
            self.rtneg_feats_all = [
                neg_feats.index_select(0, torch.from_numpy(neg_idx[0:rt_opts['n_neg_update']]).cuda())]

        spf_total = time.time() - tic
        self.trans_f = rt_opts['trans_f']
        return


    def rtmdnet_track(self, image):
        self.i += 1
        cur_image = Image.fromarray(image).convert('RGB')
        cur_image = np.asarray(cur_image)

        target_bbox = self.last_result
        # Estimate target bbox
        ishape = cur_image.shape
        samples = gen_samples(SampleGenerator('gaussian', (ishape[1], ishape[0]), self.trans_f, rt_opts['scale_f'], valid=True),
                              target_bbox, rt_opts['n_samples'])
        sample_scores, sample_feats = self.rtmdnet_eval(samples, cur_image, target_bbox)
        top_scores, top_idx = sample_scores[:, 1].topk(5)
        top_idx = top_idx.data.cpu().numpy()
        target_score = top_scores.data.mean()
        target_bbox = samples[top_idx].mean(axis=0)

        success = target_score > rt_opts['success_thr']

        if not success:
            target_bbox = self.last_result
        else:
            self.last_result = target_bbox
        # # Expand search area at failure
        if success:
            self.trans_f = rt_opts['trans_f']
        else:
            self.trans_f = rt_opts['trans_f_expand']

        ## Bbox regression
        if success:
            bbreg_feats = sample_feats[top_idx, :]
            bbreg_samples = samples[top_idx]
            bbreg_samples = self.bbreg.predict(bbreg_feats.data, bbreg_samples)
            bbreg_bbox = bbreg_samples.mean(axis=0)
        else:
            bbreg_bbox = target_bbox


        # Data collect
        if success:
            self.collect_samples_rtmdnet(cur_image, target_bbox)

        self.rtmdnet_update(use_short_update=success)

        return target_bbox, bbreg_bbox

    def rtmdnet_eval(self, samples, cur_image):
        try:
            target_bbox = np.array(
                [self.detection_box[1], self.detection_box[0], self.detection_box[3] - self.detection_box[1],
                 self.detection_box[2] - self.detection_box[0]])
        except:
            target_bbox = np.array(
                [self.last_gt[1], self.last_gt[0], self.last_gt[3] - self.last_gt[1],
                 self.last_gt[2] - self.last_gt[0]])
        cur_image = np.asarray(cur_image)

        padded_x1 = (samples[:, 0] - (3*samples[:, 2]+1*samples[:, 3])/2.0 * (rt_opts['padding'] - 1.) / 2.).min()
        padded_y1 = (samples[:, 1] - (3*samples[:, 3]+1*samples[:, 2])/2.0 * (rt_opts['padding'] - 1.) / 2.).min()
        padded_x2 = (samples[:, 0] + (3*samples[:, 2]+1*samples[:, 3])/2.0 * (rt_opts['padding'] + 1.) / 2.).max()
        padded_y2 = (samples[:, 1] + (3*samples[:, 3]+1*samples[:, 2])/2.0 * (rt_opts['padding'] + 1.) / 2.).max()
        padded_scene_box = np.asarray((padded_x1, padded_y1, padded_x2 - padded_x1, padded_y2 - padded_y1))

        if padded_scene_box[0] > cur_image.shape[1]:
            padded_scene_box[0] = cur_image.shape[1] - 1
        if padded_scene_box[1] > cur_image.shape[0]:
            padded_scene_box[1] = cur_image.shape[0] - 1
        if padded_scene_box[0] + padded_scene_box[2] < 0:
            padded_scene_box[2] = -padded_scene_box[0] + 1
        if padded_scene_box[1] + padded_scene_box[3] < 0:
            padded_scene_box[3] = -padded_scene_box[1] + 1

        crop_img_size = (
                    padded_scene_box[2:4] * ((rt_opts['img_size'], rt_opts['img_size']) / target_bbox[2:4])).astype(
            'int64')
        crop_img_size[0] = np.clip(crop_img_size[0], 84, 2000)
        crop_img_size[1] = np.clip(crop_img_size[1], 84, 2000)
        cropped_image, cur_image_var = self.img_crop_model.crop_image(cur_image, np.reshape(padded_scene_box, (1, 4)),
                                                                      crop_img_size)
        cropped_image = cropped_image - 128.

        self.rtmodel.eval()
        feat_map = self.rtmodel(cropped_image, out_layer='conv3')

        # relative target bbox with padded_scene_box
        rel_target_bbox = np.copy(target_bbox)
        rel_target_bbox[0:2] -= padded_scene_box[0:2]

        # Extract sample features and get target location
        batch_num = np.zeros((samples.shape[0], 1))
        sample_rois = np.copy(samples)
        sample_rois[:, 0:2] -= np.repeat(np.reshape(padded_scene_box[0:2], (1, 2)), sample_rois.shape[0], axis=0)
        sample_rois = samples2maskroi(sample_rois, self.rtmodel.receptive_field,
                                      (rt_opts['img_size'], rt_opts['img_size']),
                                      target_bbox[2:4], rt_opts['padding'])
        sample_rois = np.concatenate((batch_num, sample_rois), axis=1)
        sample_rois = Variable(torch.from_numpy(sample_rois.astype('float32'))).cuda()
        sample_feats = self.rtmodel.roi_align_model(feat_map, sample_rois)
        sample_feats = sample_feats.view(sample_feats.size(0), -1).clone()
        sample_scores = self.rtmodel(sample_feats, in_layer='fc4')
        return sample_scores

    def collect_samples_rtmdnet(self, cur_image):
        cur_image = np.asarray(cur_image)
        target_bbox = np.array(
            [self.detection_box[1], self.detection_box[0], self.detection_box[3] - self.detection_box[1],
             self.detection_box[2] - self.detection_box[0]])
        # Draw pos/neg samples
        ishape = cur_image.shape
        pos_examples = gen_samples(
            RT_SampleGenerator('gaussian', (ishape[1], ishape[0]), 0.1, 1.2), target_bbox,
            rt_opts['n_pos_update'],
            rt_opts['overlap_pos_update'])
        neg_examples = gen_samples(
            RT_SampleGenerator('uniform', (ishape[1], ishape[0]), 1.5, 1.2), target_bbox,
            rt_opts['n_neg_update'],
            rt_opts['overlap_neg_update'])
        if pos_examples.shape[0] == rt_opts['n_pos_update'] and neg_examples.shape[0] == rt_opts['n_neg_update']:
            padded_x1 = (neg_examples[:, 0] - neg_examples[:, 2] * (rt_opts['padding'] - 1.) / 2.).min()
            padded_y1 = (neg_examples[:, 1] - neg_examples[:, 3] * (rt_opts['padding'] - 1.) / 2.).min()
            padded_x2 = (neg_examples[:, 0] + neg_examples[:, 2] * (rt_opts['padding'] + 1.) / 2.).max()
            padded_y2 = (neg_examples[:, 1] + neg_examples[:, 3] * (rt_opts['padding'] + 1.) / 2.).max()
            padded_scene_box = np.reshape(
                np.asarray((padded_x1, padded_y1, padded_x2 - padded_x1, padded_y2 - padded_y1)), (1, 4))

            scene_boxes = np.reshape(np.copy(padded_scene_box), (1, 4))
            jitter_scale = [1.]

            for bidx in range(0, scene_boxes.shape[0]):
                crop_img_size = (scene_boxes[bidx, 2:4] * (
                        (rt_opts['img_size'], rt_opts['img_size']) / target_bbox[2:4])).astype('int64') * jitter_scale[
                                    bidx]
                cropped_image, cur_image_var = self.img_crop_model.crop_image(cur_image,
                                                                              np.reshape(scene_boxes[bidx], (1, 4)),
                                                                              crop_img_size)
                cropped_image = cropped_image - 128.

                feat_map = self.rtmodel(cropped_image, out_layer='conv3')

                rel_target_bbox = np.copy(target_bbox)
                rel_target_bbox[0:2] -= scene_boxes[bidx, 0:2]

                batch_num = np.zeros((pos_examples.shape[0], 1))
                cur_pos_rois = np.copy(pos_examples)
                cur_pos_rois[:, 0:2] -= np.repeat(np.reshape(scene_boxes[bidx, 0:2], (1, 2)), cur_pos_rois.shape[0],
                                                  axis=0)
                scaled_obj_size = float(rt_opts['img_size']) * jitter_scale[bidx]
                cur_pos_rois = samples2maskroi(cur_pos_rois, self.rtmodel.receptive_field,
                                               (scaled_obj_size, scaled_obj_size),
                                               target_bbox[2:4], rt_opts['padding'])
                cur_pos_rois = np.concatenate((batch_num, cur_pos_rois), axis=1)
                cur_pos_rois = Variable(torch.from_numpy(cur_pos_rois.astype('float32'))).cuda()
                cur_pos_feats = self.rtmodel.roi_align_model(feat_map, cur_pos_rois)
                cur_pos_feats = cur_pos_feats.view(cur_pos_feats.size(0), -1).data.clone()

                batch_num = np.zeros((neg_examples.shape[0], 1))
                cur_neg_rois = np.copy(neg_examples)
                cur_neg_rois[:, 0:2] -= np.repeat(np.reshape(scene_boxes[bidx, 0:2], (1, 2)), cur_neg_rois.shape[0],
                                                  axis=0)
                cur_neg_rois = samples2maskroi(cur_neg_rois, self.rtmodel.receptive_field,
                                               (scaled_obj_size, scaled_obj_size),
                                               target_bbox[2:4], rt_opts['padding'])
                cur_neg_rois = np.concatenate((batch_num, cur_neg_rois), axis=1)
                cur_neg_rois = Variable(torch.from_numpy(cur_neg_rois.astype('float32'))).cuda()
                cur_neg_feats = self.rtmodel.roi_align_model(feat_map, cur_neg_rois)
                cur_neg_feats = cur_neg_feats.view(cur_neg_feats.size(0), -1).data.clone()

                feat_dim = cur_pos_feats.size(-1)

                if bidx == 0:
                    pos_feats = cur_pos_feats  ##index select
                    neg_feats = cur_neg_feats
                else:
                    pos_feats = torch.cat((pos_feats, cur_pos_feats), dim=0)
                    neg_feats = torch.cat((neg_feats, cur_neg_feats), dim=0)

            if pos_feats.size(0) > rt_opts['n_pos_update']:
                pos_idx = np.asarray(range(pos_feats.size(0)))
                np.random.shuffle(pos_idx)
                pos_feats = pos_feats.index_select(0, torch.from_numpy(pos_idx[0:rt_opts['n_pos_update']]).cuda())
            if neg_feats.size(0) > rt_opts['n_neg_update']:
                neg_idx = np.asarray(range(neg_feats.size(0)))
                np.random.shuffle(neg_idx)
                neg_feats = neg_feats.index_select(0, torch.from_numpy(neg_idx[0:rt_opts['n_neg_update']]).cuda())

            self.rtpos_feats_all.append(pos_feats)
            self.rtneg_feats_all.append(neg_feats)

            if len(self.rtpos_feats_all) > rt_opts['n_frames_long']:
                del self.rtpos_feats_all[0]
            if len(self.rtneg_feats_all) > rt_opts['n_frames_short']:
                del self.rtneg_feats_all[0]


    def rtmdnet_update(self, use_short_update=False):
        # Short term update
        if use_short_update:
            nframes = min(rt_opts['n_frames_short'], len(self.rtpos_feats_all))
            pos_data = torch.stack(self.rtpos_feats_all[-nframes:], 0).view(-1, self.rtfeat_dim)
            neg_data = torch.stack(self.rtneg_feats_all, 0).view(-1, self.rtfeat_dim)
            rt_train(self.rtmodel, self.criterion, self.rtupdate_optimizer, pos_data, neg_data, rt_opts['maxiter_update'])

        # Long term update
        if self.local_Tracker.frame_num % rt_opts['long_interval'] == 0:
            pos_data = torch.stack(self.rtpos_feats_all, 0).view(-1, self.rtfeat_dim)
            neg_data = torch.stack(self.rtneg_feats_all, 0).view(-1, self.rtfeat_dim)
            rt_train(self.rtmodel, self.criterion, self.rtupdate_optimizer, pos_data, neg_data, rt_opts['maxiter_update'])

    def reselect_R_candidates_by_V(self, detection_box_ori, cur_ori_img, cur_ori_img_array):
        search_box1 = detection_box_ori[:self.p.R_candidates]
        search_box = np.zeros_like(search_box1)
        search_box[:, 1] = search_box1[:, 0]
        search_box[:, 0] = search_box1[:, 1]
        search_box[:, 2] = search_box1[:, 3]
        search_box[:, 3] = search_box1[:, 2]  # xmin, ymin, xmax, ymax
        haha = np.ones_like(search_box[:, 2]) * 3
        search_box[:, 2] = search_box[:, 2] - search_box[:, 0]  # w
        search_box[:, 3] = search_box[:, 3] - search_box[:, 1]  # h
        search_box[:, 2] = np.maximum(search_box[:, 2], haha)
        search_box[:, 3] = np.maximum(search_box[:, 3], haha)  # make sure w,h >=3 pixels
        haha2 = np.zeros_like(search_box[:, 0])
        search_box[:, 0] = np.maximum(search_box[:, 0], haha2)
        search_box[:, 1] = np.maximum(search_box[:, 1], haha2)
        haha = np.ones_like(search_box[:, 2]) * cur_ori_img.width - 1 - search_box[:, 2]
        search_box[:, 0] = np.minimum(search_box[:, 0], haha)
        haha2 = np.ones_like(search_box[:, 3]) * cur_ori_img.height - 1 - search_box[:, 3]
        search_box[:, 1] = np.minimum(search_box[:, 1], haha2)  # make sure search_box do not out of boundary

        if self.p.Verification == "tfmdnet":
            search_regions = extract_regions(cur_ori_img_array, search_box)
            search_regions = search_regions[:, :, :, ::-1]
            mdnet_scores = self.sess.run(self.outputsOp, feed_dict={self.imageOp: search_regions})
            mdnet_scores = mdnet_scores[:, 1]
        elif self.p.Verification == "pymdnet":
            mdnet_scores = forward_samples(self.pymodel, cur_ori_img, search_box, out_layer='fc6')
            mdnet_scores = mdnet_scores[:, 1].cpu().numpy()
        elif self.p.Verification == "rtmdnet":
            mdnet_scores = self.rtmdnet_eval(search_box, cur_ori_img)
            mdnet_scores = mdnet_scores[:, 1].data.cpu().numpy()
        else:
            ValueError()

        return mdnet_scores

    def research_from_specific_gt(self, cur_ori_img, search_gt, cur_ori_img_array, R_thre=0.8):
        cropped_img1, last_gt_norm1, win_loc1, scale1 = crop_search_region(cur_ori_img, search_gt, 300,
                                                                           mean_rgb=128)
        cur_img_array = np.array(cropped_img1)
        detection_box_ori1, scores1 = self.sess.run([self.pre_box_tensor, self.scores_tensor],
                                                    feed_dict={self.input_cur_image: cur_img_array,
                                                               self.initConstantOp: self.init_feature_maps})
        if scores1[0, 0] > R_thre:
            detection_box_ori1[:, 0] = detection_box_ori1[:, 0] * scale1[0] + win_loc1[0]
            detection_box_ori1[:, 1] = detection_box_ori1[:, 1] * scale1[1] + win_loc1[1]
            detection_box_ori1[:, 2] = detection_box_ori1[:, 2] * scale1[0] + win_loc1[0]
            detection_box_ori1[:, 3] = detection_box_ori1[:, 3] * scale1[1] + win_loc1[1]
            detection_box_ori = detection_box_ori1.copy()
            # max_idx = 0
            search_box1 = detection_box_ori[0]

            search_box1[0] = np.clip(search_box1[0], 0, cur_ori_img.height - 1)
            search_box1[2] = np.clip(search_box1[2], 0, cur_ori_img.height - 1)
            search_box1[1] = np.clip(search_box1[1], 0, cur_ori_img.width - 1)
            search_box1[3] = np.clip(search_box1[3], 0, cur_ori_img.width - 1)
            if (search_box1[0] == search_box1[2]) or (search_box1[1] == search_box1[3]):
                self.score_max = -20.0
            else:
                search_box1 = [search_box1[1], search_box1[0], search_box1[3] - search_box1[1],
                               search_box1[2] - search_box1[0]]
                search_box1 = np.reshape(search_box1, (1, 4))
                if self.p.Verification == "tfmdnet":
                    search_regions = extract_regions(cur_ori_img_array, search_box1)
                    search_regions = search_regions[:, :, :, ::-1]
                    self.score_max = self.sess.run(self.outputsSingleOp,
                                                   feed_dict={self.imageSingleOp: search_regions})
                    self.score_max = self.score_max[0, 1]
                elif self.p.Verification == "pymdnet":
                    self.score_max = forward_samples(self.pymodel, cur_ori_img, search_box1, out_layer='fc6')
                    self.score_max = self.score_max[0, 1]
                elif self.p.Verification == "rtmdnet":
                    self.score_max = self.rtmdnet_eval(search_box1, cur_ori_img)
                    self.score_max = self.score_max[0, 1].data.cpu().numpy().reshape(1)[0]
                    # if self.score_max < 0:
                    rtbox, checkFlag = self.rtmdnet_check(cur_ori_img_array, search_box1)
                else:
                    ValueError()

            if self.score_max >= 0. * self.first_score:
                self.max_idx = 0
                self.scores = scores1.copy()

                self.detection_box = detection_box_ori[self.max_idx]
                self.flag = 'found'

            if self.score_max < 0. * self.first_score:
                mdnet_scores = self.reselect_R_candidates_by_V(detection_box_ori, cur_ori_img, cur_ori_img_array)
                max_idx1 = np.argmax(mdnet_scores)
                if mdnet_scores[max_idx1] > 0 and scores1[0, max_idx1] > self.p.R_loss_thr:
                    self.score_max = mdnet_scores[max_idx1]
                    self.max_idx = max_idx1
                    self.scores = scores1.copy()
                    self.detection_box = detection_box_ori[self.max_idx]
                    self.flag = 'found'

        return

    def local_init(self, image, init_bbox):
        local_tracker = Tracker('atom', 'default', None)
        self.local_Tracker = local_tracker.tracker_class(local_tracker.parameters)
        self.local_Tracker.initialize(image, init_bbox)
        # if self.p.visualization:
        #     show_res(cv.cvtColor(image, cv.COLOR_RGB2BGR), np.array(self.last_gt, dtype=np.int32), '2', groundtruth=self.groundtruth,frame_id=self.i)

    def locate(self, image):

        # Convert image
        im = numpy_to_torch(image)
        self.local_Tracker.im = im  # For debugging only

        # ------- LOCALIZATION ------- #

        # Get sample
        sample_pos = self.local_Tracker.pos.round()
        sample_scales = self.local_Tracker.target_scale * self.local_Tracker.params.scale_factors
        test_x = self.local_Tracker.extract_processed_sample(im, self.local_Tracker.pos, sample_scales, self.local_Tracker.img_sample_sz)

        # Compute scores
        scores_raw = self.local_Tracker.apply_filter(test_x)
        translation_vec, scale_ind, s, flag = self.local_Tracker.localize_target(scores_raw)
        return translation_vec, scale_ind, s, flag, sample_pos, sample_scales, test_x
    def local_update(self, sample_pos, translation_vec, scale_ind, sample_scales, s, test_x):

        # Check flags and set learning rate if hard negative
        update_flag = self.flag not in ['not_found', 'uncertain']
        hard_negative = (self.flag == 'hard_negative')
        learning_rate = self.local_Tracker.params.hard_negative_learning_rate if hard_negative else None

        if update_flag:
            # Get train sample
            train_x = TensorList([x[scale_ind:scale_ind + 1, ...] for x in test_x])

            # Create label for sample
            train_y = self.local_Tracker.get_label_function(sample_pos, sample_scales[scale_ind])

            # Update memory
            self.local_Tracker.update_memory(train_x, train_y, learning_rate)

        # Train filter
        if hard_negative:
            self.local_Tracker.filter_optimizer.run(self.local_Tracker.params.hard_negative_CG_iter)
        elif (self.local_Tracker.frame_num - 1) % self.local_Tracker.params.train_skipping == 0:
            self.local_Tracker.filter_optimizer.run(self.local_Tracker.params.CG_iter)

    def rtmdnet_check(self, cur_ori_img_array, local_state):
        target_bbox = np.reshape(local_state, (4,))
        ishape = cur_ori_img_array.shape
        samples = gen_samples(
            RT_SampleGenerator('gaussian', (ishape[1], ishape[0]), rt_opts['trans_f'], rt_opts['scale_f'], valid=True),
            target_bbox, rt_opts['n_samples'])
        padded_x1 = (samples[:, 0] - samples[:, 2] * (rt_opts['padding'] - 1.) / 2.).min()
        padded_y1 = (samples[:, 1] - samples[:, 3] * (rt_opts['padding'] - 1.) / 2.).min()
        padded_x2 = (samples[:, 0] + samples[:, 2] * (rt_opts['padding'] + 1.) / 2.).max()
        padded_y2 = (samples[:, 1] + samples[:, 3] * (rt_opts['padding'] + 1.) / 2.).max()
        padded_scene_box = np.asarray((padded_x1, padded_y1, padded_x2 - padded_x1, padded_y2 - padded_y1))

        if padded_scene_box[0] > cur_ori_img_array.shape[1]:
            padded_scene_box[0] = cur_ori_img_array.shape[1] - 1
        if padded_scene_box[1] > cur_ori_img_array.shape[0]:
            padded_scene_box[1] = cur_ori_img_array.shape[0] - 1
        if padded_scene_box[0] + padded_scene_box[2] < 0:
            padded_scene_box[2] = -padded_scene_box[0] + 1
        if padded_scene_box[1] + padded_scene_box[3] < 0:
            padded_scene_box[3] = -padded_scene_box[1] + 1

        crop_img_size = (
                    padded_scene_box[2:4] * ((rt_opts['img_size'], rt_opts['img_size']) / target_bbox[2:4])).astype(
            'int64')
        cropped_image, cur_image_var = self.img_crop_model.crop_image(cur_ori_img_array,
                                                                      np.reshape(padded_scene_box, (1, 4)),
                                                                      crop_img_size)
        cropped_image = cropped_image - 128.

        self.rtmodel.eval()
        feat_map = self.rtmodel(cropped_image, out_layer='conv3')

        batch_num = np.zeros((samples.shape[0], 1))
        sample_rois = np.copy(samples)
        sample_rois[:, 0:2] -= np.repeat(np.reshape(padded_scene_box[0:2], (1, 2)), sample_rois.shape[0], axis=0)
        sample_rois = samples2maskroi(sample_rois, self.rtmodel.receptive_field,
                                      (rt_opts['img_size'], rt_opts['img_size']), target_bbox[2:4], rt_opts['padding'])
        sample_rois = np.concatenate((batch_num, sample_rois), axis=1)
        sample_rois = Variable(torch.from_numpy(sample_rois.astype('float32'))).cuda()
        sample_feats = self.rtmodel.roi_align_model(feat_map, sample_rois)
        sample_feats = sample_feats.view(sample_feats.size(0), -1).clone()
        sample_scores = self.rtmodel(sample_feats, in_layer='fc4')
        top_scores, top_idx = sample_scores[:, 1].topk(5)
        top_idx = top_idx.data.cpu().numpy()
        target_score = top_scores.data.mean()
        rt_box = samples[top_idx].mean(axis=0)
        iou = np.zeros((5, 1))
        check_flag = False
        for tt in range(5):
            iou[tt] = _compute_iou(target_bbox, samples[top_idx[tt]])
        if min(iou) > 0.5 and min(top_scores) > 0:
            self.score_max = target_score.data.cpu().numpy()
            check_flag = True
        return samples[top_idx], check_flag
    def redect_check(self, box, mdsocre, cannum, imshape):
        check = False
        dis = np.sqrt(pow(box[1]-self.last_gt[1], 2) + pow(box[0]-self.last_gt[0], 2))
        sz = np.sqrt(pow(self.last_gt[3]-self.last_gt[1], 2) + pow(self.last_gt[2]-self.last_gt[0], 2))
        shape = np.sqrt(pow(imshape[1], 2)+pow(imshape[1], 2))
        if cannum > 5:
            v = 0.9
        elif cannum > 2:
            v = 0.5
        else:
            v = 0.0
        if self.count>5:
            if mdsocre > v*self.first_score:
                check = True
        if self.count<=5:
            if (dis<1.5*sz or dis<0.24*shape) and mdsocre > v*self.first_score:
                check = True
        return check

    def siammask_init(self, im, init_gt):
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

        parser.add_argument('--resume', default=base_path+'SiamMask/experiments/siammask/SiamMask_VOT_LD.pth', type=str,
                            metavar='PATH', help='path to latest checkpoint (default: none)')
        parser.add_argument('--config', dest='config', default=base_path+'SiamMask/experiments/siammask/config_vot19lt.json',
                            help='hyper-parameter of SiamMask in json format')
        args = parser.parse_args()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True

        # Setup Model
        cfg = load_config(args)
        self.siammask = Custom(anchors=cfg['anchors'])
        if args.resume:
            assert isfile(args.resume), '{} is not a valid file'.format(args.resume)
            self.siammask = load_pretrain(self.siammask, args.resume)

        self.siammask.eval().to(device)
        x = init_gt[0]
        y = init_gt[1]
        w = init_gt[2]
        h = init_gt[3]
        target_pos = np.array([x + w / 2, y + h / 2])
        target_sz = np.array([w, h])
        self.siamstate = siamese_init(im, target_pos, target_sz, self.siammask, cfg['hp'])

    def siammask_track(self, im):
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        self.siamstate = siamese_track(self.siamstate, im, mask_enable=True, refine_enable=True)  # track
        # pdb.set_trace()
        score = np.max(self.siamstate['score'])
        location = self.siamstate['ploygon'].flatten()
        mask = self.siamstate['mask'] > self.siamstate['p'].seg_thr

        return score, mask

    def tracking(self, image):
        self.i += 1
        cur_ori_img = Image.fromarray(image)
        cur_ori_img_array = np.array(cur_ori_img)
        candicate_boxes = []
        rtbox = []
        mask = 0
        self.local_Tracker.pos = torch.FloatTensor([(self.last_gt[0]+self.last_gt[2]-1)/2,(self.last_gt[1]+self.last_gt[3]-1)/2])
        self.local_Tracker.target_sz = torch.FloatTensor([(self.last_gt[2]-self.last_gt[0]),(self.last_gt[3]-self.last_gt[1])])
        translation_vec, scale_ind, s, self.flag, sample_pos, sample_scales, test_x = self.locate(image)
        self.local_score = torch.max(s[scale_ind,...]).item()
        self.local_Tracker.update_state(sample_pos + translation_vec)
        local_state = torch.cat((self.local_Tracker.pos[[1, 0]] - (self.local_Tracker.target_sz[[1, 0]] - 1) / 2, self.local_Tracker.target_sz[[1, 0]])).tolist()
        # local_state, self.flag, self.local_score = self.local_Tracker.track(image)
        local_state = np.reshape(local_state, (1, 4))  # [x, y, w, h]

        self.score_max = self.rtmdnet_eval(local_state, cur_ori_img)
        self.score_max = self.score_max[0, 1].data.cpu().numpy().reshape(1)[0]
        if self.score_max < 0:
            rtbox, checkFlag = self.rtmdnet_check(cur_ori_img_array, local_state)

        if (self.score_max >= 0 and self.flag != 'not_found'):
            self.local_Tracker.frame_num += 1
            self.local_Tracker.refine_target_box(sample_pos, sample_scales[scale_ind], scale_ind, True)
            self.local_Tracker.pos = self.local_Tracker.pos_iounet.clone()

            self.siamstate['target_pos'] = self.local_Tracker.pos.numpy()[::-1]
            self.siamstate['target_sz'] = self.local_Tracker.target_sz.numpy()[::-1]
            siamscore, mask = self.siammask_track(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            self.local_Tracker.pos = torch.FloatTensor(self.siamstate['target_pos'][::-1].copy())
            self.local_Tracker.target_sz = torch.FloatTensor(self.siamstate['target_sz'][::-1].copy())

            self.target_bbox = torch.cat((self.local_Tracker.pos[[1, 0]] - (
                        self.local_Tracker.target_sz[[1, 0]] - 1) / 2, self.local_Tracker.target_sz[[1, 0]])).tolist()
            self.detection_box = np.array(
                [self.target_bbox[1], self.target_bbox[0], self.target_bbox[1] + self.target_bbox[3],
                 self.target_bbox[0] + self.target_bbox[2]])
            self.flag = 'found'

        ##------------------------------------------------------##

        ##------------------------------------------------------##

        if (self.flag == 'not_found' or self.score_max < 0.0) and self.label is not None:
            self.count += 1

            V_best = 0
            result = inference_detector(self.mm_model, image, self.cfg)
            bboxes, labels = get_mmresult(image, result)
            boxes = bboxes[:, :4]
            mmscore = bboxes[:, -1]
            boxes = np.array([boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]])
            index = labels == self.label
            candicate_boxes = bboxes[index]
            for i in range(candicate_boxes.shape[0]):
                search_gt = np.array([candicate_boxes[i, 1], candicate_boxes[i, 0], candicate_boxes[i, 3], candicate_boxes[i, 2]])
                self.research_from_specific_gt(cur_ori_img, search_gt, cur_ori_img_array, R_thre=0.5)
                if self.flag == 'found' and self.score_max > V_best:
                    V_best = self.score_max
                    redet_best = self.detection_box
            # reinit local tracker
            if self.flag == 'found':
                detcheck = self.redect_check(redet_best, V_best, candicate_boxes.shape[0], cur_ori_img_array.shape)
                if detcheck:
                    self.detection_box = redet_best
                    self.local_Tracker.pos = torch.Tensor(
                        [(self.detection_box[0] + self.detection_box[2]) / 2,
                         (self.detection_box[1] + self.detection_box[3]) / 2])
                    self.local_Tracker.target_sz = torch.Tensor(
                        [self.detection_box[2] - self.detection_box[0], self.detection_box[3] - self.detection_box[1]])
                else:
                    self.flag = 'not_found'


        if (self.flag == 'not_found' or self.score_max < 0) and (self.label is None):  # and score_max < 20.0:
            search_gt = (np.array(self.last_gt)).copy()
            # search_gt = last_gt.copy()
            search_gt[0] = cur_ori_img.height / 2.0 - (self.last_gt[2] - self.last_gt[0]) / 2.0
            search_gt[2] = cur_ori_img.height / 2.0 + (self.last_gt[2] - self.last_gt[0]) / 2.0
            search_gt[1] = cur_ori_img.width / 2.0 - (self.last_gt[3] - self.last_gt[1]) / 2.0
            search_gt[3] = cur_ori_img.width / 2.0 + (self.last_gt[3] - self.last_gt[1]) / 2.0

            self.research_from_specific_gt(cur_ori_img, search_gt, cur_ori_img_array)

            if self.score_max < 0:
                search_gt[0] = cur_ori_img.height / 2.0 - self.first_h / 2.0
                search_gt[2] = cur_ori_img.height / 2.0 + self.first_h / 2.0
                search_gt[1] = cur_ori_img.width / 2.0 - self.first_w / 2.0
                search_gt[3] = cur_ori_img.width / 2.0 + self.first_w / 2.0

                self.research_from_specific_gt(cur_ori_img, search_gt, cur_ori_img_array)

            if self.score_max < 0:
                search_gt[0] = cur_ori_img.height / 2.0 - self.first_h / 2.0 / 2.0
                search_gt[2] = cur_ori_img.height / 2.0 + self.first_h / 2.0 / 2.0
                search_gt[1] = cur_ori_img.width / 2.0 - self.first_w / 2.0 / 2.0
                search_gt[3] = cur_ori_img.width / 2.0 + self.first_w / 2.0 / 2.0

                self.research_from_specific_gt(cur_ori_img, search_gt, cur_ori_img_array)

            if self.score_max < 0:
                search_gt[0] = cur_ori_img.height / 2.0 - self.first_h / 2.0 * 2.0
                search_gt[2] = cur_ori_img.height / 2.0 + self.first_h / 2.0 * 2.0
                search_gt[1] = cur_ori_img.width / 2.0 - self.first_w / 2.0 * 2.0
                search_gt[3] = cur_ori_img.width / 2.0 + self.first_w / 2.0 * 2.0

                self.research_from_specific_gt(cur_ori_img, search_gt, cur_ori_img_array)
            # reinit local tracker
            if self.flag == 'found':
                self.local_Tracker.pos = torch.Tensor(
                    [(self.detection_box[0]+self.detection_box[2])/2, (self.detection_box[1]+self.detection_box[3])/2])
                self.local_Tracker.target_sz = torch.Tensor(
                    [self.detection_box[2]-self.detection_box[0], self.detection_box[3]-self.detection_box[1]])

        # global search
        if (self.flag == 'not_found') and self.label is None:
            self.count += 1
            if self.count >= 0:
                if self.globalmode:
                    last_reliable_w = self.last_reliable_w#self.first_w
                    last_reliable_h = self.last_reliable_h#self.first_h
                else:
                    last_reliable_w = self.first_w  # self.first_w
                    last_reliable_h = self.first_w  # self.first_h
                # self.globalmode = not self.globalmode
                count_research = 0

                search_list = []
                #top left corner
                self.startx = last_reliable_w * 2.0
                self.starty = last_reliable_h * 2.0
                search_gt = np.int32(
                    [self.starty - last_reliable_h / 2.0, self.startx - last_reliable_w / 2.0,
                     self.starty + last_reliable_h / 2.0,
                     self.startx + last_reliable_w / 2.0])
                search_gt = np.expand_dims(search_gt, axis=0)
                search_list.append(search_gt)

                #top right corner
                self.startx = cur_ori_img.width - 1 - last_reliable_w * 2.0
                self.starty = last_reliable_h * 2.0
                search_gt = np.int32(
                    [self.starty - last_reliable_h / 2.0, self.startx - last_reliable_w / 2.0,
                     self.starty + last_reliable_h / 2.0,
                     self.startx + last_reliable_w / 2.0])
                search_gt = np.expand_dims(search_gt, axis=0)
                search_list.append(search_gt)

                #bottom left corner
                self.startx = last_reliable_w * 2.0
                self.starty = cur_ori_img.height - 1 - last_reliable_h * 2.0
                search_gt = np.int32(
                    [self.starty - last_reliable_h / 2.0, self.startx - last_reliable_w / 2.0,
                     self.starty + last_reliable_h / 2.0,
                     self.startx + last_reliable_w / 2.0])
                search_gt = np.expand_dims(search_gt, axis=0)
                search_list.append(search_gt)

                #bottom right corner
                self.startx = cur_ori_img.width - 1 - last_reliable_w * 2.0
                self.starty = cur_ori_img.height - 1 - last_reliable_h * 2.0
                search_gt = np.int32(
                    [self.starty - last_reliable_h / 2.0, self.startx - last_reliable_w / 2.0,
                     self.starty + last_reliable_h / 2.0,
                     self.startx + last_reliable_w / 2.0])
                search_gt = np.expand_dims(search_gt, axis=0)
                search_list.append(search_gt)

                self.startx = 2* last_reliable_w
                self.starty = 2* last_reliable_h
                self.starty = 3.5 * last_reliable_h + self.starty
                #whole image
                while (self.startx < cur_ori_img.width - 1) and (self.starty < cur_ori_img.height - 1):
                    search_gt = np.int32(
                        [self.starty - last_reliable_h / 2.0, self.startx - last_reliable_w / 2.0, self.starty + last_reliable_h / 2.0,
                         self.startx + last_reliable_w / 2.0])
                    search_gt = np.expand_dims(search_gt, axis=0)
                    search_list.append(search_gt)

                    self.starty = 3.5 * last_reliable_h + self.starty
                    if self.starty >= cur_ori_img.height - 1  and self.startx < cur_ori_img.width - 1:
                        self.starty = 0
                        self.startx = 3.5 * last_reliable_w + self.startx

                searchLen = len(search_list)
                search_list = np.concatenate(search_list, axis=0)
                if searchLen > 8:
                    tmp = np.random.randint(4,searchLen,min(10,int(np.round(searchLen/4))))#20
                    search_list = np.concatenate((search_list[:4], search_list[tmp]), axis=0)

                for haha in range(search_list.shape[0]):
                    search_gt = search_list[haha]
                    # search_gt = np.int32(
                    #     [self.starty - last_reliable_h / 2.0, self.startx - last_reliable_w / 2.0, self.starty + last_reliable_h / 2.0,
                    #      self.startx + last_reliable_w / 2.0])
                    cropped_img1, last_gt_norm1, win_loc1, scale1 = crop_search_region(cur_ori_img, search_gt, 300,
                                                                                       mean_rgb=128)
                    cur_img_array1 = np.array(cropped_img1)
                    detection_box1, scores1 = self.sess.run([self.pre_box_tensor, self.scores_tensor],
                                                            feed_dict={self.input_cur_image: cur_img_array1,
                                                                       self.initConstantOp: self.init_feature_maps})
                    # print(scores1[0,0])

                    if scores1[0, 0] > 0.5:

                        detection_box1[:, 0] = detection_box1[:, 0] * scale1[0] + win_loc1[0]
                        detection_box1[:, 1] = detection_box1[:, 1] * scale1[1] + win_loc1[1]
                        detection_box1[:, 2] = detection_box1[:, 2] * scale1[0] + win_loc1[0]
                        detection_box1[:, 3] = detection_box1[:, 3] * scale1[1] + win_loc1[1]
                        detection_box_ori = detection_box1.copy()
                        # max_idx = 0
                        search_box1 = detection_box_ori[0]
                        search_box1[0] = np.clip(search_box1[0], 0, cur_ori_img.height - 1)
                        search_box1[2] = np.clip(search_box1[2], 0, cur_ori_img.height - 1)
                        search_box1[1] = np.clip(search_box1[1], 0, cur_ori_img.width - 1)
                        search_box1[3] = np.clip(search_box1[3], 0, cur_ori_img.width - 1)
                        if (search_box1[0] == search_box1[2]) or (search_box1[1] == search_box1[3]):
                            self.score_max = -20.0
                        else:
                            search_box1 = [search_box1[1], search_box1[0], search_box1[3] - search_box1[1],
                                           search_box1[2] - search_box1[0]]
                            search_box1 = np.reshape(search_box1, (1, 4))
                            if self.p.Verification == "tfmdnet":
                                search_regions = extract_regions(cur_ori_img_array, search_box1)
                                search_regions = search_regions[:, :, :, ::-1]
                                self.score_max = self.sess.run(self.outputsSingleOp,
                                                               feed_dict={self.imageSingleOp: search_regions})
                                self.score_max = self.score_max[0, 1]
                            elif self.p.Verification == "pymdnet":
                                self.score_max = forward_samples(self.pymodel, cur_ori_img, search_box1, out_layer='fc6')
                                self.score_max = self.score_max[0, 1]
                            elif self.p.Verification == "rtmdnet":
                                self.score_max = self.rtmdnet_eval(search_box1, cur_ori_img)
                                self.score_max = self.score_max[0, 1].data.cpu().numpy().reshape(1)[0]
                            else:
                                ValueError()

                        if self.score_max >= 0:
                            self.scores = scores1.copy()
                            self.max_idx = 0
                            self.startx = 0
                            self.starty = 0
                            self.flag = 'found'
                            self.detection_box = detection_box_ori[self.max_idx]
                            break

                        if self.score_max < 0:
                            mdnet_scores = self.reselect_R_candidates_by_V(detection_box_ori, cur_ori_img,
                                                                           cur_ori_img_array)
                            max_idx1 = np.argmax(mdnet_scores)
                            self.score_max = mdnet_scores[max_idx1]
                            if mdnet_scores[max_idx1] > 0 and scores1[0, max_idx1] > 0.5:
                                self.scores = scores1.copy()
                                self.max_idx = max_idx1
                                self.detection_box = detection_box_ori[self.max_idx]
                                self.startx = 0
                                self.starty = 0
                                self.flag = 'uncertain'
                                break
                # reinit local tracker
                if self.flag == 'found':
                    self.local_Tracker.pos = torch.Tensor(
                        [(self.detection_box[0]+self.detection_box[2])/2, (self.detection_box[1]+self.detection_box[3])/2])
                    self.local_Tracker.target_sz = torch.Tensor(
                        [self.detection_box[2]-self.detection_box[0], self.detection_box[3]-self.detection_box[1]])
                    self.count = 0

        # collecting samples
        if self.flag == 'found' and self.score_max>self.first_score*self.v:

            if self.i > 0:
                self.local_update(sample_pos, translation_vec, scale_ind, sample_scales, s, test_x)
            self.count = 0
            self.last_gt = self.detection_box
            if self.p.Verification == "rtmdnet":
                self.collect_samples_rtmdnet(cur_ori_img)
            else:
                ValueError()
        try:
            outputs = self.detection_box
        except:
            outputs = self.last_gt
        # update mdnet
        if self.p.Verification == "rtmdnet":
            self.rtmdnet_update(use_short_update=False)

        # record reliable state
        if self.flag == 'found':
            self.last_reliable_w = self.detection_box[3] - self.detection_box[1]
            self.last_reliable_h = self.detection_box[2] - self.detection_box[0]
            self.V_reliable_pool[:self.p.confidence_pool_num - 1] = self.V_reliable_pool[1:]
            self.V_reliable_pool[-1] = self.score_max

        width = self.last_gt[3] - self.last_gt[1]
        height = self.last_gt[2] - self.last_gt[0]



        if self.flag == 'found' and self.score_max > 0:
            confidence_score = 0.99
        elif self.flag == 'not_found':
            confidence_score = 0.0
        else:
            confidence_score = np.clip((self.local_score+np.arctan(0.2*self.score_max)/math.pi+0.5)/2, 0, 1)

        if self.i <= self.p.confidence_pool_num:
            self.confidence_pool[self.i - 1] = confidence_score
            self.V_pool[self.i - 1] = self.score_max
        else:
            self.confidence_pool[:self.p.confidence_pool_num-1] = self.confidence_pool[1:]
            self.confidence_pool[-1] = confidence_score
            self.V_pool[:self.p.confidence_pool_num-1] = self.V_pool[1:]
            self.V_pool[-1] = self.score_max

        if self.p.visualization:
            show_res(cv.cvtColor(image, cv.COLOR_RGB2BGR), np.array(self.last_gt, dtype=np.int32), '2',
                     confidence=confidence_score, groundtruth=self.groundtruth,
                     frame_id=self.i, can=candicate_boxes, mask=mask)

        print("frame: " + "%d  " % self.i + "Region: " + "%.2f" % float(self.last_gt[1]) + ",%.2f" % float(
            self.last_gt[0]) + ",%.2f" % float(width) + ",%.2f" % float(height))
        return vot.Rectangle(float(outputs[1]), float(outputs[0]), float(outputs[3]-outputs[1]), float(outputs[2]-outputs[0])),confidence_score#scores[0,max_idx]


handle = vot.VOT("rectangle")
selection = handle.region()
imagefile = handle.frame()
p = p_config()
image = cv.cvtColor(cv.imread(imagefile), cv.COLOR_BGR2RGB)
tracker = MobileTracker(image, selection, p=p)

while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    image = cv.cvtColor(cv.imread(imagefile), cv.COLOR_BGR2RGB)
    region, confidence = tracker.tracking(image)
    handle.report(region, confidence)
