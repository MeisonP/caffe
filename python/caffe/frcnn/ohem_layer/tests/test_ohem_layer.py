from caffe.frcnn.ohem_layer import layer
import pytest
import tempfile
import os
import caffe
import numpy as np

class SimpleLayer(caffe.Layer):
    """A layer that just multiplies by ten"""

    def setup(self, bottom, top):
        # suppose we have two classes (bg + fg)
        # cls_prob_readonly
        num_classes = 2
        num_rois = 5
        # cls_prob_readonly
        top[0].reshape(num_rois, num_classes)
        # bbox_pred_readonly
        top[1].reshape(num_rois, 4 * num_classes)
        # rois
        top[2].reshape(num_rois, 5)
        # labels
        top[3].reshape(num_rois, 1)
        # bbox_targets
        top[4].reshape(num_rois, 4 * num_classes)
        # bbox_inside_weights
        top[5].reshape(num_rois, 4 * num_classes)
        # bbox_outside_weights
        top[6].reshape(num_rois, 4 * num_classes)


    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        top[3].data[:,0] = 1
        # fill bbox_inside_weights
        top[5].data.fill(1)

    def backward(self, top, propagate_down, bottom):
        pass

def python_net_file():
    f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.write("""name: 'pythonnet' force_backward: true
    layer { type: 'Python' name: 'ohem_input' top: 'cls_prob_readonly' top: 'bbox_pred_readonly'
           top: 'rois', top: 'labels' top: 'bbox_targets' top: 'bbox_inside_weights'
           top: 'bbox_outside_weights'
      python_param { module: 'caffe.frcnn.ohem_layer.tests.test_ohem_layer' layer: 'SimpleLayer' } }
    layer { type: 'Python' name: 'three' bottom: 'cls_prob_readonly' bottom: 'bbox_pred_readonly'
            bottom: 'rois', bottom: 'labels' bottom: 'bbox_targets' bottom: 'bbox_inside_weights'
            bottom: 'bbox_outside_weights' top: 'rois_hard' top: 'labels_hard' top: 'bbox_targets_hard'
            top: 'bbox_inside_weights_hard' top: 'bbox_outside_weights_hard'
      python_param { module: 'caffe.frcnn.ohem_layer.layer' layer: 'OHEMDataLayer'  param_str: "'num_classes': 2"}}""")
    f.close()
    return f.name

@pytest.fixture
def init_net():
    net_file = python_net_file()
    net = caffe.Net(net_file, caffe.TRAIN)
    yield net
    os.remove(net_file)

def test_forward(init_net):
    init_net.forward()
