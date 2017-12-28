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
                         'traffic light')

colors=[(255,0,0),(0,255,0),(255,255,0),(0,255,255),(0,0,255),(255,255,255)]


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    for i in range(dets.shape[0]):
        bbox = dets[i, :4]
        score = dets[i, -1]

        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3] ), (0, 255, 0))
        cv2.putText(im, class_name, (int(bbox[0]) , int(bbox[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)


def test_net(net, im_file, timer):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(im_file)
    im_copy = im.copy()

    # Detect all object classes and regress object bounds
    scores, boxes = im_detect(net, im, timer)

    timer['misc'].tic()

    thresh = 0.5
    max_per_image = 100

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[] for _ in xrange(len(CLASSES))]

    # apply nms and bounding box voting
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

        all_boxes[j] = cls_dets

    #  Limit to max_per_image detections *over all classes*
    if max_per_image > 0:
        image_scores = np.hstack([all_boxes[j][:, -1]
                                  for j in xrange(1, len(CLASSES))])
        if len(image_scores) > max_per_image:
            image_thresh = np.sort(image_scores)[-max_per_image]
            for j in xrange(1, len(CLASSES)):
                keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                all_boxes[j] = all_boxes[j][keep, :]

    timer['misc'].toc()

    for j in xrange(1, len(CLASSES)):
        vis_detections(im_copy, CLASSES[j],all_boxes[j])


    return im_copy


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]', default=0, type=int)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input_image', type=str, help='input image path')
    group.add_argument('--input_dir', type=str, help='input dir')
    parser.add_argument('--out_dir', dest='out_dir', help='output dir', type=str)
    parser.add_argument('--arch', dest='arch', help='network architecture', type=str)
    parser.add_argument('--weight', dest='weight', help='model weight', type=str)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    cfg_from_file(os.path.join(cfg.ROOT_DIR, 'models', 'pvanet', 'cfgs', 'submit_1019.yml'))

    args = parse_args()

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    net = caffe.Net(args.arch, args.weight, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(args.weight)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    _t = {'im_preproc': Timer(), 'im_net' : Timer(), 'im_postproc': Timer(), 'misc' : Timer()}
    for i in xrange(10):
        im_detect(net, im, _t)

    timer = {'im_preproc': Timer(), 'im_net' : Timer(), 'im_postproc': Timer(), 'misc' : Timer()}

    if args.out_dir:
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)

    if args.input_image:
        im = test_net(net, args.input_image, timer)
        cv2.imshow(args.input_image, im)
        cv2.waitKey(0)
        if args.out_dir:
            im_name = args.input_image.split('/')[-1]
            cv2.imwrite(os.path.join(args.out_dir, im_name), im)

    elif args.input_dir:
        im_files = glob.glob(os.path.join(args.input_dir, '*'))
        for im_file in im_files[0:50]:
            print 'test {0}'.format(im_file)
            im = test_net(net, im_file, _t)
            if args.out_dir:
                im_name = im_file.split('/')[-1]
                cv2.imwrite(os.path.join(args.out_dir, im_name), im)

        for im_file in im_files[0:50]:
            print 'test {0}'.format(im_file)
            im = test_net(net, im_file, timer)
            if args.out_dir:
                im_name = im_file.split('/')[-1]
                cv2.imwrite(os.path.join(args.out_dir, im_name), im)

    print 'im_detect: net {:.3f}s  preproc {:.3f}s  postproc {:.3f}s  misc {:.3f}s' \
          .format(_t['im_net'].average_time,
                  _t['im_preproc'].average_time, _t['im_postproc'].average_time,
                  _t['misc'].average_time)
