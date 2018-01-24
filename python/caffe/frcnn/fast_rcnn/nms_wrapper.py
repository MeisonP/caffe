# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from caffe.frcnn.fast_rcnn.config import cfg
from caffe.frcnn.nms.gpu_nms import gpu_nms
from caffe.frcnn.nms.cpu_nms import cpu_nms
from caffe.frcnn.nms.cpu_nms import cpu_soft_nms

def nms(dets, thresh, method=0, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []

    if method > 0:
        return cpu_soft_nms(dets, threshold=thresh, method=soft_method)
    if cfg.USE_GPU_NMS and not force_cpu:
        return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
    else:
        return cpu_nms(dets, thresh)
