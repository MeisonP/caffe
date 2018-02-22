import sys
sys.path.append('/home/tumh/mycaffe/python')
import caffe
import caffe.proto.caffe_pb2
from caffe import layers as L
from caffe import params as P
from caffe import NetSpec
from caffe import name_scope, arg_scope, get_scope_name, get_scope_arg

num_classes = 6

n = NetSpec()

n.data, n.im_info, n.gt_boxes = L.Python(ntop=3, python_param=dict(module="caffe.frcnn.roi_data_layer.layer", layer="RoIDataLayer", param_str='num_classes: {}'.format(num_classes)))

use_global_stats = True

def init_c_relu(n, bottom):
    conv = L.Convolution(n.data, name='conv', pad_h=3, pad_w=3, kernel_h=7, kernel_w=7, stride_h=2, stride_w=2, num_output=16, bias_term=False, weight_filler=dict(type='xavier'))
    # batch normalization
    bn = L.BatchNorm(conv, in_place=True, name='bn')
    neg = L.Power(bn, power=1, scale=-1.0, shift=0, name='neg')
    # concat
    concat = L.Concat(bn, neg, name='concat')
    # scale
    scale = L.Scale(concat, in_place=True, name='scale')
    # relu
    relu = L.ReLU(scale, in_place=True, name='relu')
    return relu

def residual_c_rulu_block(n, num_output, last_layer, residual_mode='conv', with_batch_normalization_on_left=False, replace_last_layer=False, to_seperable=False, new_depth=False, zero=False):

    if new_depth:
        bn = L.BatchNorm(last_layer, name='bn')
        bn_scale = L.Scale(bn, name='bn_scale')
        relu = L.ReLU(bn_scale, name='relu')
        last_layer = relu

    c1 = build_c_relu_block_1(n, last_layer, bn_inplace=True, num_output=num_output[0], with_bn=with_batch_normalization_on_left, new_depth=new_depth)
    c2 = build_c_relu_block_2(n, stage, block, c1, use_global_stats, bn_inplace=False,  num_output=num_output[1])
    c3 = build_c_relu_block_3(n, stage, block, c2, use_global_stats, bn_inplace=False,  num_output=num_output[2])

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
        n['conv{0}_{1}/proj'.format(stage, block)] = conv_proj = L.Convolution(last_layer, convolution_param=dict(pad_h=0, pad_w=0, kernel_h=1, kernel_w=1, stride_h=stride, stride_w=stride, num_output=num_output[2], weight_filler=dict(type='xavier'), bias_filler = dict(type='constant', value=0.1)), param=param)
    else:
        n['conv{0}_{1}/input'.format(stage, block)] = conv_proj = L.Power(right_branch, power_param=dict(power=1, scale=1, shift=0))
    # elewise
    # residual
    n['conv{0}_{1}'.format(stage, block)] = eltwise = L.Eltwise(c3, conv_proj, eltwise_param=dict(operation=P.Eltwise.SUM, coeff=[1, 1]))
    return eltwise

# init c_relu
with arg_scope(['BatchNorm'], use_global_stats=use_global_stats, param=[dict(lr_mult=0),dict(lr_mult=0),dict(lr_mult=0)]):
    with arg_scope(['Scale'], param=[dict(lr_mult=0.1, decay_mult=0), dict(lr_mult=0.1, decay_mult=0)], bias_term=True):
        with arg_scope(['Convolution'], param=[dict(lr_mult=0.1, decay_mult=0.1)]):
            with name_scope('conv_1_1'):
                c1 = init_c_relu(n, n.data)
            pool = L.Pooling(c1, pool=P.Pooling.MAX, kernel_size=3, stride=2, pad=0)

        c_2_1 = residual_c_rulu_block(n, [24, 24, 64], pool)
        c_2_2 = residual_c_rulu_block(n, [24, 24, 64], c_2_1, with_batch_normalization_on_left=True, residual_mode='power', to_seperable=to_seperable, zero=zero)


open('train.prototxt', 'w').write(str(n.to_proto()))
