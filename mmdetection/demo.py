import mmcv
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result
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

def show_res(im, r, win_name='2', frame_id=None, groundtruth=None):
    cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)
    obj_num = len(r)
    for i in range(0, obj_num):
        # cls = r[i][0]
        score = r[i][-1]
        box = r[i][:4]
        box = [int(s) for s in box]
        cv2.rectangle(im, (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])), [0, 255, 255], 2)
        # cv2.putText(im, str(cls), (int(box[0]-box[2]/2), int(box[1]-box[3]/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        cv2.putText(im, str(score), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    if groundtruth is not None and not groundtruth[frame_id][0]==np.nan:
        groundtruth = groundtruth.astype("int16")
        cv2.rectangle(im, (groundtruth[frame_id][0], groundtruth[frame_id][1]),
                      (groundtruth[frame_id][0]+groundtruth[frame_id][2], groundtruth[frame_id][1]+groundtruth[frame_id][3]), [0, 0, 255], 2)

    #cv2.imwrite("/home/xiaobai/Desktop/MBMD_vot_code/figure/%05d.jpg"%frame_id, im[:, :, -1::-1])
    cv2.imshow(win_name, im)
    cv2.waitKey(0)

cfg = mmcv.Config.fromfile('configs/faster_rcnn_r50_fpn_1x.py')
cfg.model.pretrained = None

# construct the model and load checkpoint
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
_ = load_checkpoint(model, './faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth')

# test a single image
# img = mmcv.imread('./demo/coco_test_12510.jpg')
# result = inference_detector(model, img, cfg)
# show_result(img, result)

data_dir = '/home/daikenan/dataset/VOT2019/lt2019'
sequence_list = os.listdir(data_dir)
sequence_list.sort()
sequence_list = [title for title in sequence_list if not title.endswith("txt")]
# r = detect(net, meta, "data/dog.jpg".encode('utf-8'))
sequence_list = ['longboard']
for seq_id, video in enumerate(sequence_list):
    # video = 'boat'
    sequence_dir = data_dir + '/' + video + '/color/'

    gt_dir = data_dir + '/' + video + '/groundtruth.txt'
    image_list = os.listdir(sequence_dir)
    image_list.sort()
    image_list = [im for im in image_list if im.endswith("jpg") or im.endswith("jpeg")]
    try:
        groundtruth = np.loadtxt(gt_dir, delimiter=',')
    except:
        groundtruth = np.loadtxt(gt_dir)
    imagefile = sequence_dir + image_list[0]
    img = mmcv.imread(imagefile)
    result = inference_detector(model, img, cfg)
    bboxes, labels = show_result(img, result)
    boxes = bboxes[:, :4]
    mmscore = bboxes[:, -1]
    boxes = np.array([boxes[:, 0], boxes[:, 1], boxes[:, 2]-boxes[:, 0], boxes[:, 3]-boxes[:, 1]])
    iou = np.zeros((boxes.shape[1],))
    for i in range(boxes.shape[1]):
        iou[i] = _compute_iou(boxes[:, i], groundtruth[0])
    if max(iou) > 0.4:
        label = labels[np.argmax(iou)]
    for id, imagefile in enumerate(image_list):
        imagefile = sequence_dir + image_list[id*5]
        img = mmcv.imread(imagefile)
        result = inference_detector(model, img, cfg)
        bboxes, labels = show_result(img, result)
        index = labels == label
        candicate_boxes = bboxes[index]
        show_res(img, candicate_boxes, frame_id=5*id, groundtruth=groundtruth)


# # test a list of images
# imgs = ['test1.jpg', 'test2.jpg']
# for i, result in enumerate(inference_detector(model, imgs, cfg, device='cuda:0')):
#     print(i, imgs[i])
#     show_result(imgs[i], result)