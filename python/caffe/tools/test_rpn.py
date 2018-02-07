import sys
import _init_paths
import caffe
import glob
import os
import cv2
from caffe.frcnn.fast_rcnn.test import _get_blobs
from caffe.frcnn.fast_rcnn.config import cfg
import numpy as np
import xml.etree.ElementTree as ET
from caffe.frcnn.utils.cython_bbox import bbox_overlaps

def load_annotation(path):
    tree = ET.parse(path)
    objs = tree.findall('object')
    num_objs = len(objs)
    boxes = np.zeros((num_objs, 4), dtype=np.float32)
    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1
        boxes[ix, :] = [x1, y1, x2, y2]
    return boxes


cfg.TEST.HAS_RPN = True
cfg.TEST.SCALE_MULTIPLE_OF = 32
cfg.TEST.SCALES = (640,)
cfg.TEST.MAX_SIZE = 2000
cfg.TEST.RPN_PRE_NMS_TOP_N =  12000
cfg.TEST.RPN_POST_NMS_TOP_N =  2000
cfg.TEST.RPN_NMS_THRESH = 0.4

IOU_THRESHOLD = 0.5

IMAGE_PATH = '/home/alec.tu/0815training_data/JPEGImages'
ANNOPATH = '/home/alec.tu/0815training_data/Annotations'

ims = glob.glob(os.path.join(IMAGE_PATH, '*.jpg'))
cfg.GPU_ID = 3

caffe.set_mode_gpu()

caffe.set_device(3)
net = caffe.Net('/home/alec.tu/mycaffe/models/pvanet/faster_rcnn_alt_opt/pt/rpn_test.prototxt', '/home/alec.tu/pva_ohem/pvanet_stage1_rpn_iter_136438.caffemodel', caffe.TEST)

rois = net.blobs['rois'].data

num_gt_box = 0
num_pred = 0

for i, im_name in enumerate(ims):
    print im_name
    im = cv2.imread(im_name)
    blobs, im_scales = _get_blobs(im, None)
    print blobs['data'].shape
    im_blob = blobs['data']
    blobs['im_info'] = np.array(
        [np.hstack((im_blob.shape[2], im_blob.shape[3], im_scales[0]))],
        dtype=np.float32)
    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    # do forward
    net.blobs['data'].data[...] = blobs['data']
    net.blobs['im_info'].data[...] = blobs['im_info']
    blobs_out = net.forward()

    base_name = os.path.basename(im_name).split('.')[0]
    f_name = os.path.join(ANNOPATH, os.path.basename(im_name).split('.')[0] + '.xml')
    gt_bboxes = load_annotation(f_name)
    rois = blobs_out['rois']
    rois = rois[:,1:5] / im_scales[0]
    print rois.shape

    #  for j in range(rois.shape[0]):
        #  cv2.rectangle(im, (int(rois[j][0]), int(rois[j][1])), (int(rois[j][2]),int(rois[j][3])), (255, 0, 0), 2)

    #  if i < 10:
        #  cv2.imwrite('/home/alec.tu/mycaffe/{}_draw.jpg'.format(base_name), im)

    #  if i  >= 10:
        #  break


    overlaps = bbox_overlaps(
        np.ascontiguousarray(rois, dtype=np.float),
        np.ascontiguousarray(gt_bboxes, dtype=np.float))

    argmax_overlaps = overlaps.argmax(axis=0)
    overlaps = overlaps[argmax_overlaps, np.arange(gt_bboxes.shape[0])]
    num_pred += len(np.where(overlaps > IOU_THRESHOLD)[0])
    num_gt_box += gt_bboxes.shape[0]

print 'recall {}'.format(float(num_pred)/num_gt_box)
