# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

import caffe
from fast_rcnn.config import cfg
import roi_data_layer.roidb as rdl_roidb
from utils.timer import Timer
import numpy as np
import os
import sys
import torchvision.transforms
from tensorboardX import SummaryWriter
import signal

from caffe.proto import caffe_pb2
import google.protobuf as pb2
import google.protobuf.text_format
import cv2

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, solver_prototxt, roidb, output_dir,
                 pretrained_model=None):
        """Initialize the SolverWrapper."""

        # register signal handelr to handel ctrl-C event
        signal.signal(signal.SIGINT, self.signal_handler)

        self.writer = SummaryWriter()

        self.output_dir = output_dir

        if (cfg.TRAIN.HAS_RPN and cfg.TRAIN.BBOX_REG and
            cfg.TRAIN.BBOX_NORMALIZE_TARGETS):
            # RPN can only use precomputed normalization because there are no
            # fixed statistics to compute a priori
            assert cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED

        if cfg.TRAIN.BBOX_REG:
            print 'Computing bounding-box regression targets...'
            self.bbox_means, self.bbox_stds = \
                    rdl_roidb.add_bbox_regression_targets(roidb)
            print 'done'

        self.solver = caffe.SGDSolverWrapper(solver_prototxt)
        if pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

        self.solver.net.layers[0].set_roidb(roidb)

    def snapshot(self):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.solver.net

        scale_bbox_params = (cfg.TRAIN.BBOX_REG and
                             cfg.TRAIN.BBOX_NORMALIZE_TARGETS and
                             net.params.has_key('bbox_pred'))

        if scale_bbox_params:
            # save original values
            orig_0 = net.params['bbox_pred'][0].data.copy()
            orig_1 = net.params['bbox_pred'][1].data.copy()

            # scale and shift with bbox reg unnormalization; then save snapshot
            net.params['bbox_pred'][0].data[...] = \
                    (net.params['bbox_pred'][0].data *
                     self.bbox_stds[:, np.newaxis])
            net.params['bbox_pred'][1].data[...] = \
                    (net.params['bbox_pred'][1].data *
                     self.bbox_stds + self.bbox_means)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = self.solver_param.snapshot_prefix + infix + \
                    '_iter_{:d}'.format(self.solver.iter)

        #  print filename

        # FIXME: filename not expected
        self.solver.snapshot_solverstate(str(filename + '.solverstate'))
        #  self.solver.snapshot_solverstate('11.solverstate')

        #  filename = os.path.join(self.output_dir, filename)


        net.save(str(filename) + '.caffemodel')
        print 'Wrote snapshot to: {:s}'.format(filename)

        if scale_bbox_params:
            # restore net to original state
            net.params['bbox_pred'][0].data[...] = orig_0
            net.params['bbox_pred'][1].data[...] = orig_1
        return filename

    def signal_handler(self, signal, frame):
        print ('Received Ctrl-C')
        if self.last_snapshot_iter != self.solver.iter:
            self.snapshot()

        self.writer.close()
        sys.exit(0)


    def train_model(self):
        """Network training loop."""
        self.last_snapshot_iter = -1
        transformer = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
        print self.solver.lr
        print cfg.TRAIN.PLATEAU_LR
        # FIXME: initialize lr to base lr
        while self.solver.iter == 0 or self.solver.lr >= cfg.TRAIN.PLATEAU_LR:
        #  while self.solver.iter < 100:
            loss = 0
            rpn_cls_loss = 0
            rpn_bbox_loss = 0
            bbox_loss = 0
            cls_loss = 0
            for i in range(self.solver.param.iter_size):
                loss += self.solver.forward_backward()
                rpn_cls_loss += self.solver.net.blobs['rpn_loss_cls'].data
                rpn_bbox_loss += self.solver.net.blobs['rpn_loss_bbox'].data
                bbox_loss += self.solver.net.blobs['loss_bbox'].data
                cls_loss += self.solver.net.blobs['loss_cls'].data

            loss = loss / self.solver.param.iter_size
            rpn_cls_loss = rpn_cls_loss / self.solver.param.iter_size
            rpn_bbox_loss = rpn_bbox_loss / self.solver.param.iter_size
            bbox_loss = bbox_loss / self.solver.param.iter_size
            cls_loss = cls_loss / self.solver.param.iter_size

            smooth_loss = self.solver.update_smoothloss(loss, self.solver.param.average_loss)

            if self.solver.iter != 0:
                if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                    self.last_snapshot_iter = self.solver.iter
                    self.snapshot()

            # send summary data to tensorboard
            if self.solver.iter != 0 and cfg.TRAIN.SCALAR_SUMMARY_ITERS > 0:
                if self.solver.iter % cfg.TRAIN.SCALAR_SUMMARY_ITERS == 0:
                    self.writer.add_scalar('data/total_loss', smooth_loss, self.solver.iter)
                    self.writer.add_scalar('data/rpn_cls_loss', rpn_cls_loss, self.solver.iter)
                    self.writer.add_scalar('data/rpn_bbox_loss', rpn_bbox_loss, self.solver.iter)
                    self.writer.add_scalar('data/cls_loss', cls_loss, self.solver.iter)
                    self.writer.add_scalar('data/bbox_loss', bbox_loss, self.solver.iter)
                    self.writer.add_scalar('data/lr', self.solver.lr, self.solver.iter)

            if self.solver.iter != 0 and cfg.TRAIN.IMAGE_SUMMARY_ITERS > 0:
                if self.solver.iter % cfg.TRAIN.IMAGE_SUMMARY_ITERS == 0:
                    # monitor input
                    input_im = self.solver.net.blobs['data'].data
                    assert (input_im.shape[0] == 1)
                    input_im[0, 0,:,:] += cfg.PIXEL_MEANS[0][0][0]
                    input_im[0, 1,:,:] += cfg.PIXEL_MEANS[0][0][1]
                    input_im[0, 2,:,:] += cfg.PIXEL_MEANS[0][0][2]
                    bgr_im = np.transpose(np.squeeze(input_im), (1,2,0))
                    # FIXME: error when add rgb order
                    #  rgb_im = bgr_im[:,:,::-1]
                    #  print rgb_im.shape

                    # monitor hard rois
                    hard_rois = self.solver.net.blobs['rois_hard'].data
                    assert (hard_rois.shape[0] <= cfg.TRAIN.BATCH_SIZE), '{} <= {}'.format(hard_rois.shape[0], cfg.TRAIN.BATCH_SIZE)
                    roi_coords = hard_rois[:, 1:]
                    assert (roi_coords.shape[0] <= cfg.TRAIN.BATCH_SIZE)
                    assert (roi_coords.shape[1] == 4)
                    hard_rois = []
                    bgr_im = bgr_im.astype(np.uint8).copy()
                    for i in range(roi_coords.shape[0]):
                        roi_coord = map(int, roi_coords[i, :])
                        assert (len(roi_coord) == 4)
                        cv2.rectangle(bgr_im, (roi_coord[0], roi_coord[1]), (roi_coord[2], roi_coord[3]), (255, 0, 0))
                    #  bgr_im = bgr_im.astype(np.float32).copy()
                    self.writer.add_image('Image', transformer(bgr_im), self.solver.iter)

            self.solver.apply_update()


def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    print 'Preparing training data...'
    rdl_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb

def filter_roidb(roidb):
    """Remove roidb entries that have no usable RoIs."""

    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print 'Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                       num, num_after)
    return filtered_roidb

def train_net(solver_prototxt, roidb, output_dir, pretrained_model=None):
    """Train a Fast R-CNN network."""

    roidb = filter_roidb(roidb)
    sw = SolverWrapper(solver_prototxt, roidb, output_dir,
                       pretrained_model=pretrained_model)

    print 'Solving...'
    sw.train_model()
    print 'done solving'
