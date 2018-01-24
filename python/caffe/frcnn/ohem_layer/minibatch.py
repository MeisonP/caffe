import numpy as np
from caffe.frcnn.fast_rcnn.config import cfg
from caffe.frcnn.fast_rcnn.nms_wrapper import nms

def get_ohem_minibatch(loss, rois, labels, bbox_targets=None,
                       bbox_inside_weights=None, bbox_outside_weights=None):
    """Given rois and their loss, construct a minibatch using OHEM."""
    loss = np.array(loss)

    if cfg.TRAIN.OHEM_USE_NMS:
        # Do NMS using loss for de-dup and diversity
        keep_inds = []
        nms_thresh = cfg.TRAIN.OHEM_NMS_THRESH
        source_img_ids = [roi[0] for roi in rois]
        for img_id in np.unique(source_img_ids):
            for label in np.unique(labels):
                sel_indx = np.where(np.logical_and(labels == label, \
                                    source_img_ids == img_id))[0]
                if not len(sel_indx):
                    continue
                boxes = np.concatenate((rois[sel_indx, 1:],
                        loss[sel_indx][:,np.newaxis]), axis=1).astype(np.float32)
                keep_inds.extend(sel_indx[nms(boxes, nms_thresh)])

        hard_keep_inds = select_hard_examples(loss[keep_inds])
        hard_inds = np.array(keep_inds)[hard_keep_inds]
    else:
        hard_inds = select_hard_examples(loss)

    blobs = {'rois_hard': rois[hard_inds, :].copy(),
             'labels_hard': labels[hard_inds].copy()}
    if bbox_targets is not None:
        assert cfg.TRAIN.BBOX_REG
        blobs['bbox_targets_hard'] = bbox_targets[hard_inds, :].copy()
        blobs['bbox_inside_weights_hard'] = bbox_inside_weights[hard_inds, :].copy()
        blobs['bbox_outside_weights_hard'] = bbox_outside_weights[hard_inds, :].copy()

    return blobs

def select_hard_examples(loss):
    """Select hard rois."""
    # Sort and select top hard examples.
    sorted_indices = np.argsort(loss)[::-1]
    hard_keep_inds = sorted_indices[0:np.minimum(len(loss), cfg.TRAIN.BATCH_SIZE)]
    # (explore more ways of selecting examples in this function; e.g., sampling)

    return hard_keep_inds
