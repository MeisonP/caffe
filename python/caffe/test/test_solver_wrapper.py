import unittest
import tempfile
import os
import numpy as np
import six

import caffe
from test_net import simple_net_file
from test_solver import TestSolver


class TestSolverWrapper(TestSolver):
    def setUp(self):
        self.num_output = 13
        net_f = simple_net_file(self.num_output)
        f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        f.write("""net: '""" + net_f + """'
        test_iter: 10 test_interval: 10 base_lr: 0.01 momentum: 0.9
        weight_decay: 0.0005 lr_policy: 'inv' gamma: 0.0001 power: 0.75
        display: 100 max_iter: 100 snapshot_after_train: false
        snapshot_prefix: "model" """)
        f.close()
        self.solver = caffe.SGDSolverWrapper(f.name)
        # also make sure get_solver runs
        caffe.get_solver(f.name)
        caffe.set_mode_cpu()
        # fill in valid labels
        self.solver.net.blobs['label'].data[...] = \
                np.random.randint(self.num_output,
                    size=self.solver.net.blobs['label'].data.shape)
        self.solver.test_nets[0].blobs['label'].data[...] = \
                np.random.randint(self.num_output,
                    size=self.solver.test_nets[0].blobs['label'].data.shape)
        os.remove(f.name)
        os.remove(net_f)
