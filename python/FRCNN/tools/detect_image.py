#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from fast_rcnn.test import im_detect, bbox_vote
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import glob
import cv2

CLASSES = ('__background__', # always index 0
           'people', 'car', 'motorbike','bicycle',
           'traffic light','sign')


#CLASSES = ('__background__', # always index 0
#           'traffic light')


#CLASSES = ('__background__', # always index 0
#           'green','yellow','red')
colors=[(255,0,0),(0,255,0),(255,255,0),(0,255,255),(0,0,255),(255,255,255)]

_t = {'im_preproc': Timer(), 'im_net' : Timer(), 'im_postproc': Timer(), 'misc' : Timer()}

#CLASSES = ('__background__', # always index 0
#           'P1','P2','P3','S',
#           'S1','S2','S3','S4',
#           'S5','S6','S7','S8',
#           'S9','W1','W2','W3')

#colors=[(255,0,0)]

#colors=[(0,255,0),(0,255,255),(0,0,255)]
def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2] - bbox[0], bbox[3] - bbox[1]), (0, 255, 0))
        cv2.putText(im, class_name, ((int(bbox[2] - bbox[0]) / 2), int((bbox[3] - bbox[1]) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0))

def vis_detections2(im, class_name, dets, thresh = 0.5):
    #  num, _ = dets.shape

    #  for i in xrange(num):
        #  bbox = dets[i, :4]
        #  score = dets[i, -1]

        #  cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2] - bbox[0], bbox[3] - bbox[1]), (0, 255, 0))
        #  cv2.putText(im, class_name, ((int(bbox[2] - bbox[0]) / 2), int((bbox[3] - bbox[1]) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0))
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
	    for j in range(0,len(CLASSES)-1):
                if class_name == CLASSES[j+1]:
                    #if class_name == 'traffic light':
                    #    RG=im[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2]),:]
                    #    if np.mean(RG[:,:,1])-np.mean(RG[:,:,2])>0: 
	            #        class_name='green'
                    #    else:
                    #        class_name='red'   
                    #  cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2] - bbox[0], bbox[3] - bbox[1]), (0, 255, 0))
                    cv2.rectangle(im, (bbox[0],bbox[1]), (bbox[2],bbox[3]), colors[j], 3)

                    #  cv2.putText(im, class_name, ((int(bbox[2] - bbox[0]) / 2), int((bbox[3] - bbox[1]) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0))
                    class_name2='%s(%s)'%(class_name,str(score)[0:4])
                    cv2.putText(im, class_name2, (int(bbox[0]), int(bbox[1] - 2)) , cv2.FONT_HERSHEY_SIMPLEX, 1, colors[j],3)


def demo(net, im_file):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(im_file)

    orig_im = im.copy()

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im, _t)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.9
    NMS_THRESH = 0.4
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(orig_im, cls, dets, thresh=CONF_THRESH)

    #  print('/home/tumh/SJ/pva-faster-rcnn/results/{0}'.format(im_file.split('/')[-1]))

    cv2.imwrite('/home/tumh/SJ/pva-faster-rcnn/results/{0}.png'.format(im_file.split('/')[-1]), im)


def test_all(net, im_files):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(im_files)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(len(CLASSES))]
    thresh=0.01
    max_per_image = 100
    # timers
    _t = {'im_preproc': Timer(), 'im_net' : Timer(), 'im_postproc': Timer(), 'misc' : Timer()}
    orig_ims = []

    for i in xrange(len(im_files)):
        # filter out any ground truth boxes
        if cfg.TEST.HAS_RPN:
            box_proposals = None
        else:
            # The roidb may contain ground-truth rois (for example, if the roidb
            # comes from the training or val split). We only want to evaluate
            # detection on the *non*-ground-truth rois. We select those the rois
            # that have the gt_classes field set to 0, which means there's no
            # ground truth.
            box_proposals = roidb[i]['boxes'][roidb[i]['gt_classes'] == 0]
        im = cv2.imread(im_files[i])     
        #im=cv2.resize(im,(0,0),fx=0.4333,fy=0.4444)  
        im_copy = im.copy()

        scores, boxes = im_detect(net, im, _t, box_proposals)

        _t['misc'].tic()
        # skip j = 0, because it's the background class
        for j in xrange(1, len(CLASSES)):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, cfg.TEST.NMS)

            dets_NMSed = cls_dets[keep, :]
            if cfg.TEST.BBOX_VOTE:
                cls_dets = bbox_vote(dets_NMSed, cls_dets)
            else:
                cls_dets = dets_NMSed

            #  if vis:
                #  vis_detections(im_orig, CLASSES[j], cls_dets)
            all_boxes[j][i] = cls_dets

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, len(CLASSES))])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, len(CLASSES)):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        for j in xrange(1, len(CLASSES)):
            vis_detections2(im_copy, CLASSES[j],all_boxes[j][i])
            cv2.imwrite('/home/tumh/SJ/pva-faster-rcnn/results_%s/'%data_file+'{0}'.format(im_files[i].split('/')[-1]), im_copy)


        _t['misc'].toc()


        print 'im_detect: {:d}/{:d}  net {:.3f}s  preproc {:.3f}s  postproc {:.3f}s  misc {:.3f}s' \
              .format(i + 1, num_images, _t['im_net'].average_time,
                      _t['im_preproc'].average_time, _t['im_postproc'].average_time,
                      _t['misc'].average_time)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    cfg_from_file('/home/tumh/SJ/pva-faster-rcnn/models/pvanet/cfgs/submit_1019.yml')

    args = parse_args()

    caffe.set_mode_gpu()
    caffe.set_device(0)


    prototxt = '/home/tumh/SJ/pva-faster-rcnn/test_300000_v3.prototxt'
    caffemodel = '/home/tumh/SJ/pva-faster-rcnn/pvanet_300000_v3.caffemodel'
    
    #prototxt = '/home/acer/pva-faster-rcnn/test_TL_merge.prototxt'
    #caffemodel = '/home/acer/pva-faster-rcnn/pvanet_TL_merge.caffemodel'

    #prototxt = '/home/acer/pva-faster-rcnn/test_lightGYR.prototxt'
    #caffemodel = '/home/acer/pva-faster-rcnn/pvanet_lightGYR_200000.caffemodel'

    #prototxt = '/home/acer/pva-faster-rcnn/test_v0621.prototxt'
    #caffemodel = '/home/acer/pva-faster-rcnn/test_v0621.caffemodel'

    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    data_file='demo'
    os.makedirs('/home/tumh/SJ/pva-faster-rcnn/results_%s/'%data_file)
    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im, _t)
    
    im_files = glob.glob(os.path.join(cfg.DATA_DIR, data_file, '*'))
    
    test_all(net, im_files)

    # im = cv2.capture(0)
    # test_all(net, im)

    #  im_files = glob.glob('/home/tumh/SJ/pva-faster-rcnn/single/*.jpg')

    #  for im_file in im_files:
        #  print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        #  print 'Demo for data/demo/{}'.format(im_file)
        #  demo2(net, im_file)
