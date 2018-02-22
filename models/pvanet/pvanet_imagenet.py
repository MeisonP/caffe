import sys
import os.path as osp
this_dir = osp.dirname(__file__)
caffe_path = osp.join(this_dir, '..', '..', 'python')
sys.path.insert(0, caffe_path)
import caffe
import caffe.proto.caffe_pb2
from caffe import layers as L
from caffe import params as P
from caffe import NetSpec
from caffe import name_scope, arg_scope, get_scope_name, get_scope_arg

num_classes = 6

n = NetSpec()

#  n.data, n.im_info, n.gt_boxes = L.Python(ntop=3, python_param=dict(module="caffe.frcnn.roi_data_layer.layer", layer="RoIDataLayer", param_str='num_classes: {}'.format(num_classes)), name = 'input_data')
n.data = L.Input(shape=dict(dim = [1, 3, 320, 320]), name='input')

use_global_stats = True

def init_c_relu(n, bottom):
    n.conv = L.Convolution(bottom, name='conv', pad_h=3, pad_w=3, kernel_h=7, kernel_w=7, stride_h=2, stride_w=2, num_output=16, bias_term=False)
    # batch normalization
    n.bn = L.BatchNorm(n.conv, name='bn')
    n.neg = L.Power(n.bn, power=1, scale=-1.0, shift=0, name='neg')
    # concat
    n.concat = L.Concat(n.bn, n.neg, name='concat')
    # scale
    n.scale = L.Scale(n.concat, in_place=True, name='scale')
    # relu
    n.relu = L.ReLU(n.scale, in_place=True, name='relu')
    return n.relu

def c_relu_block_1(n, last_layer, last_layer_out=None, num_output=24, with_bn=False, new_depth=False, zero=False):
    with name_scope('1'):
        if with_bn:
            n.pre = L.BatchNorm(last_layer, name='bn', in_place=False)
            #scale
            scale = L.Scale(n.pre, name='bn_scale')
            # relu
            relu = L.ReLU(scale, name='relu')
            last_layer = relu

        if new_depth:
            stride = 2
        else:
            stride = 1
        # 1 X 1 convoltion
        n._ = L.Convolution(last_layer, pad_h=0, pad_w=0, kernel_h=1, kernel_w=1, stride_h=stride, stride_w=stride, num_output=num_output, name='conv')
        return n._


def c_relu_block_2(n, last_layer, bn_inplace=True, last_layer_out=None, num_output=24):
    with name_scope('2'):
        bn = n.pre = L.BatchNorm(last_layer, name='bn', in_place=False)
        #scale
        scale = L.Scale(bn, name='bn_scale')
        # relu
        relu = L.ReLU(scale, in_place=True, name='relu')
        # convolution
        conv = L.Convolution(relu, pad_h=1, pad_w=1, kernel_h=3, kernel_w=3, stride_h=1, stride_w=1, num_output=num_output, name='conv')
        return conv

def c_relu_block_3(n, last_layer, last_layer_out=None, num_output=24):
    with name_scope('3'):
        bn = n.pre = L.BatchNorm(last_layer, name='bn', in_place=False)
        # negation
        neg = L.Power(bn, power=1, scale=-1.0, shift=0, name='neg')
        # concat
        concat = L.Concat(bn, neg, name='concat')
        # scale
        scale = L.Scale(concat, name='scale')
        # relu
        relu = L.ReLU(scale, name='relu')

        conv = n._ = L.Convolution(relu, pad_h=0, pad_w=0, kernel_h=1, kernel_w=1, stride_h=1, stride_w=1, num_output=num_output, name='conv')

    return conv

def residual_c_relu_block(n, num_output, last_layer, residual_mode='conv', with_batch_normalization_on_left=False, replace_last_layer=False, to_seperable=False, new_depth=False, zero=False):

    if new_depth:
        with name_scope('1'):
            n.pre = L.BatchNorm(last_layer, name='bn', in_place=False)
            n.bn_scale = L.Scale(n.pre, name='bn_scale')
            n.relu = L.ReLU(n.bn_scale, name='relu')
            last_layer = n.relu

    c1 = c_relu_block_1(n, last_layer, num_output=num_output[0], with_bn=with_batch_normalization_on_left, new_depth=new_depth)
    c2 = c_relu_block_2(n, c1, num_output=num_output[1])
    c3 = c_relu_block_3(n, c2, num_output=num_output[2])

    if replace_last_layer:
        assert(with_batch_normalization_first==True)
        right_branch = left_branch
    else:
        right_branch = last_layer

    # proj conv
    if residual_mode != 'power':
        if new_depth:
            stride = 2
        else:
            stride = 1
        conv_proj = n.proj = L.Convolution(last_layer, pad_h=0, pad_w=0, kernel_h=1, kernel_w=1, stride_h=stride, stride_w=stride, num_output=num_output[2], name='proj')
    else:
        conv_proj = n.input = L.Power(right_branch, power=1, scale=1, shift=0, name='input')
    # elewise
    # residual
    eltwise = n._ = L.Eltwise(c3, conv_proj, operation=P.Eltwise.SUM, coeff=[1, 1])
    return eltwise

def inception_module(n, conv_param, last_layer, last_layer_out=None):

    # conv
    conv = L.Convolution(last_layer, convolution_param=conv_param, name='conv')
    # batch normalization
    bn = L.BatchNorm(conv, name='bn')
    # scale
    bn_scale = L.Scale(bn, name='bn_scale')
    # relu
    relu = L.ReLU(bn_scale, name='relu')

    return relu


def inception_block(n, last_layer, module_params, pool_param, residual_mode='conv', num_output=256, last=False, new_depth=False):

    with name_scope('incep'):
        bn = n.pre = L.BatchNorm(last_layer, name='bn', in_place=False)
        bn_scale = L.Scale(bn, name='bn_scale')
        relu = L.ReLU(bn_scale, name='relu')
    # first module
        with name_scope('0'):
            inception_param_0 = module_params['incep_0']
            incep_0 = inception_module(n, inception_param_0, relu)
    # second module
        with name_scope('1_reduce'):
            inception_param_reduce_1 = module_params['incep_reduce_1']
            reduce_out = inception_module(n, inception_param_reduce_1, relu)
        with name_scope('1_0'):
            inception_param_1_0 = module_params['incep_1_0']
            incep_1_0 = inception_module(n, inception_param_1_0, reduce_out, last_layer_out=module_params['incep_reduce_1']['num_output'])
    # third module
        with name_scope('2_reduce'):
            inception_param_reduce_2 = module_params['incep_reduce_2']
            reduce_out = inception_module(n, inception_param_reduce_2, relu)
        with name_scope('2_0'):
            inception_param_2_0 = module_params['incep_2_0']
            incep_2_0 = inception_module(n, inception_param_2_0, reduce_out, last_layer_out=module_params['incep_reduce_2']['num_output'])
        with name_scope('2_1'):
            inception_param_2_1 = module_params['incep_2_1']
            incep_2_1 = inception_module(n, inception_param_2_1, incep_2_0, last_layer_out=module_params['incep_2_1']['num_output'])

        concat_layers = [incep_0, incep_1_0, incep_2_1]

        if bool(pool_param):
            incep_pool = L.Pooling(relu, pooling_param=pool_param, name='pool')
            # inception module after pool
            inception_param_poolproj = module_params['incep_poolproj']
            with name_scope('poolproj'):
                poolproj = inception_module(n, inception_param_poolproj, incep_pool)
            concat_layers.append(poolproj)
        # concat
        concat = L.Concat(*concat_layers)


    with name_scope('out'):
        # residual
        conv_param = dict(pad_h=0, pad_w=0, kernel_h=1, kernel_w=1, stride_h=1, stride_w=1, num_output=num_output)
        if last:
            conv_param['bias_term'] = False
        conv_incep_out = L.Convolution(concat, convolution_param=conv_param, name='conv')

        left_branch = conv_incep_out

        if last:
            # batch normalization
            out_bn = L.BatchNorm(conv_incep_out, name='bn')
            #scale
            out_scale = L.Scale(out_bn, name='bn_scale')
            left_branch = out_scale

    if residual_mode != 'power':
        conv_proj = L.Convolution(last_layer, pad_h=0, pad_w=0, kernel_h=1, kernel_w=1, stride_h=2, stride_w=2, num_output=num_output, name='proj')
    else:
         conv_proj = L.Power(last_layer, name='input')


    out = n._ = L.Eltwise(left_branch, conv_proj, operation=P.Eltwise.SUM, coeff=[1, 1])

    if last:
        # batch normalization
        last_bn = L.BatchNorm(out, name='last_bn')
        #scale
        last_scale = L.Scale(last_bn, name='last_bn_scale')
        last_relu = L.ReLU(last_scale, name='last_relu')
        out = last_relu

    return out

# init c_relu
with arg_scope(['BatchNorm'], use_global_stats=use_global_stats, param=[dict(lr_mult=0),dict(lr_mult=0),dict(lr_mult=0)], in_place=True):
    with arg_scope(['Scale'], param=[dict(lr_mult=0.1, decay_mult=0), dict(lr_mult=0.1, decay_mult=0)], bias_term=True, in_place=True):
        with arg_scope(['Convolution'], param=[dict(lr_mult=0.1, decay_mult=0.1)]):
            with arg_scope(['ReLU'], in_place=True):
                inp = n.data
                with name_scope('conv1_1'):
                    c1 = init_c_relu(n, inp)
                pool = n.pool1 = L.Pooling(c1, pool=P.Pooling.MAX, kernel_size=3, stride=2, pad=0, name = 'pool1')

                with name_scope('conv2_1'):
                    c_2_1 = residual_c_relu_block(n, [24, 24, 64], pool)
                with name_scope('conv2_2'):
                    c_2_2 = residual_c_relu_block(n, [24, 24, 64], c_2_1, with_batch_normalization_on_left=True, residual_mode='power')

                with name_scope('conv2_3'):
                    c_2_3 = residual_c_relu_block(n, [24, 24, 64], c_2_2, with_batch_normalization_on_left=True, residual_mode='power')
                # stage 3
                with name_scope('conv3_1'):
                    c_3_1 = residual_c_relu_block(n, [48, 48, 128], c_2_3, new_depth=True, with_batch_normalization_on_left=False)
                with name_scope('conv3_2'):
                    c_3_2 = residual_c_relu_block(n, [48, 48, 128], c_3_1, with_batch_normalization_on_left=True, residual_mode='power')
                with name_scope('conv3_3'):
                    c_3_3 = residual_c_relu_block(n, [48, 48, 128], c_3_2, with_batch_normalization_on_left=True, residual_mode='power')
                with name_scope('conv3_4'):
                    c_3_4 = residual_c_relu_block(n, [48, 48, 128], c_3_3, with_batch_normalization_on_left=True, residual_mode='power')

                # Inception block
                module_params = {}
                # incep_4_1
                # from leftmost to rightmost
                # 1X1 convoltion
                with name_scope('conv4_1'):
                    module_params['incep_0'] = dict(pad_h=0, pad_w=0, kernel_h=1, kernel_w=1, stride_h=2, stride_w=2, num_output=64, bias_term=False)
                    # 3X3 convoltion 48-128
                    module_params['incep_reduce_1'] = dict(pad_h=0, pad_w=0, kernel_h=1, kernel_w=1, stride_h=2, stride_w=2, num_output=48, bias_term=False)
                    module_params['incep_1_0'] = dict(pad_h=1, pad_w=1, kernel_h=3, kernel_w=3, stride_h=1, stride_w=1, num_output=128, bias_term=False)
                    # 5 X 5 convoltion
                    module_params['incep_reduce_2'] = dict(pad_h=0, pad_w=0, kernel_h=1, kernel_w=1, stride_h=2, stride_w=2, num_output=24, bias_term=False)
                    module_params['incep_2_0'] = dict(pad_h=1, pad_w=1, kernel_h=3, kernel_w=3, stride_h=1, stride_w=1, num_output=48, bias_term=False)
                    module_params['incep_2_1'] = dict(pad_h=1, pad_w=1, kernel_h=3, kernel_w=3, stride_h=1, stride_w=1, num_output=48, bias_term=False)
                    # pool param
                    module_params['incep_poolproj'] = dict(pad_h=0, pad_w=0, kernel_h=1, kernel_w=1, stride_h=1, stride_w=1, num_output=128, bias_term=False)
                    pool_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2, pad=0)
                    incep_4_1 = inception_block(n, c_3_4, module_params, pool_param, new_depth=True)

                with name_scope('conv4_2'):
                    # incep_4_2
                    # 1X1 convoltion
                    module_params['incep_0'] = dict(pad_h=0, pad_w=0, kernel_h=1, kernel_w=1, stride_h=1, stride_w=1, num_output=64, bias_term=False)
                    # 3X3 convoltion 48-128
                    module_params['incep_reduce_1'] = dict(pad_h=0, pad_w=0, kernel_h=1, kernel_w=1, stride_h=1, stride_w=1, num_output=64, bias_term=False)
                    module_params['incep_1_0'] = dict(pad_h=1, pad_w=1, kernel_h=3, kernel_w=3, stride_h=1, stride_w=1, num_output=128, bias_term=False)
                    # 5 X 5 convoltion
                    module_params['incep_reduce_2'] = dict(pad_h=0, pad_w=0, kernel_h=1, kernel_w=1, stride_h=1, stride_w=1, num_output=24, bias_term=False)
                    module_params['incep_2_0'] = dict(pad_h=1, pad_w=1, kernel_h=3, kernel_w=3, stride_h=1, stride_w=1, num_output=48, bias_term=False)
                    module_params['incep_2_1'] = dict(pad_h=1, pad_w=1, kernel_h=3, kernel_w=3, stride_h=1, stride_w=1, num_output=48, bias_term=False)
                    incep_4_2 = inception_block(n, incep_4_1, module_params, {}, residual_mode='power')

                with name_scope('conv4_3'):
                    # incep_4_3
                    incep_4_3 = inception_block(n, incep_4_2, module_params, {}, residual_mode='power')

                with name_scope('conv4_4'):
                    # incep_4_4
                    incep_4_4 = inception_block(n, incep_4_3, module_params, {}, residual_mode='power')

                with name_scope('conv5_1'):
                    # 1X1 convoltion
                    module_params['incep_0'] = dict(pad_h=0, pad_w=0, kernel_h=1, kernel_w=1, stride_h=2, stride_w=2, num_output=64, bias_term=False)
                    # 3X3 convoltion 48-128
                    module_params['incep_reduce_1'] = dict(pad_h=0, pad_w=0, kernel_h=1, kernel_w=1, stride_h=2, stride_w=2, num_output=96, bias_term=False)
                    module_params['incep_1_0'] = dict(pad_h=1, pad_w=1, kernel_h=3, kernel_w=3, stride_h=1, stride_w=1, num_output=192, bias_term=False )
                    # 5 X 5 convoltion
                    module_params['incep_reduce_2'] = dict(pad_h=0, pad_w=0, kernel_h=1, kernel_w=1, stride_h=2, stride_w=2, num_output=32, bias_term=False)
                    module_params['incep_2_0'] = dict(pad_h=1, pad_w=1, kernel_h=3, kernel_w=3, stride_h=1, stride_w=1, num_output=64, bias_term=False)
                    module_params['incep_2_1'] = dict(pad_h=1, pad_w=1, kernel_h=3, kernel_w=3, stride_h=1, stride_w=1, num_output=64, bias_term=False)
                    # pool param
                    module_params['incep_poolproj'] = dict(pad_h=0, pad_w=0, kernel_h=1, kernel_w=1, stride_h=1, stride_w=1, num_output=128, bias_term=False)
                    pool_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2, pad=0)
                    incep_5_1 = inception_block(n, incep_4_4, module_params, pool_param, num_output=384)

                with name_scope('conv5_2'):
                    # incep_5_2
                    # 1X1 convoltion
                    module_params['incep_0'] = dict(pad_h=0, pad_w=0, kernel_h=1, kernel_w=1, stride_h=1, stride_w=1, num_output=64, bias_term=False)
                    # 3X3 convoltion 48-128
                    module_params['incep_reduce_1'] = dict(pad_h=0, pad_w=0, kernel_h=1, kernel_w=1, stride_h=1, stride_w=1, num_output=96, bias_term=False)
                    module_params['incep_1_0'] = dict(pad_h=1, pad_w=1, kernel_h=3, kernel_w=3, stride_h=1, stride_w=1, num_output=192, bias_term=False)
                    # 5 X 5 convoltion
                    module_params['incep_reduce_2'] = dict(pad_h=0, pad_w=0, kernel_h=1, kernel_w=1, stride_h=1, stride_w=1, num_output=32, bias_term=False)
                    module_params['incep_2_0'] = dict(pad_h=1, pad_w=1, kernel_h=3, kernel_w=3, stride_h=1, stride_w=1, num_output=64, bias_term=False)
                    module_params['incep_2_1'] = dict(pad_h=1, pad_w=1, kernel_h=3, kernel_w=3, stride_h=1, stride_w=1, num_output=64, bias_term=False)
                    incep_5_2 = inception_block(n, incep_5_1, module_params, {}, residual_mode='power', num_output=384)

                with name_scope('conv5_3'):
                    incep_5_3 = inception_block(n, incep_5_2, module_params, {}, residual_mode='power', num_output=384)

                with name_scope('conv5_4'):
                    n.incep_5_4 = inception_block(n, incep_5_3, module_params, {}, residual_mode='power', num_output=384, last=True)




open('train.prototxt', 'w').write(str(n.to_proto()))
caffe.Net('./train.prototxt', '/home/tumh/pva9.1_preAct_train_iter_1900000.caffemodel', caffe.TEST)
