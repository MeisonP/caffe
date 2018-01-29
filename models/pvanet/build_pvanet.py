import sys
import os.path as osp
this_dir = osp.dirname(__file__)
caffe_path = osp.join(this_dir, '..', '..', 'python')
sys.path.insert(0, caffe_path)

from caffe.proto import caffe_pb2
import google.protobuf as pb
from caffe import layers as L
from caffe import params as P
import caffe
from argparse import ArgumentParser

parser = ArgumentParser(description=""" This script generates imagenet alexnet train_val.prototxt files""")
parser.add_argument('-o', '--output_folder', help="""Train and Test prototxt will be generated as train.prototxt and test.prototxt""")

def build_init_c_relu(n, stage, block, bottom, num_output, use_global_stats, zero=False):
    if zero:
        param=[dict(lr_mult=0, decay_mult=0)]
    else:
        param=[dict(lr_mult=0.1, decay_mult=0.1)]

    if zero:
        scale_param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    else:
        scale_param=[dict(lr_mult=0.1, decay_mult=0), dict(lr_mult=0.1, decay_mult=0)]

    n['conv{0}_{1}/conv'.format(stage, block)] = conv = L.Convolution(bottom, convolution_param=dict(pad_h=3, pad_w=3, kernel_h=7, kernel_w=7, stride_h=2, stride_w=2, num_output=16, bias_term=False, weight_filler=dict(type='xavier')), param=param)
    # batch normalization
    n['conv{0}_{1}/bn'.format(stage, block)] = bn = L.BatchNorm(conv, param=[dict(lr_mult=0),dict(lr_mult=0),dict(lr_mult=0)], in_place=True, use_global_stats=use_global_stats)
    n['conv{0}_{1}/neg'.format(stage, block)] = neg = L.Power(bn, power_param=dict(power=1, scale=-1.0, shift=0))
    # concat
    n['conv{0}_{1}'.format(stage, block)] = concat = L.Concat(bn, neg, name='conv{0}_{1}/concat'.format(stage, block) )
    # scale
    n['conv{0}_{1}/scale'.format(stage, block)] = scale = L.Scale(concat, scale_param=dict(bias_term=True), in_place=True, param=scale_param)
    # relu
    last = n['conv{0}_{1}/relu'.format(stage, block)] = relu = L.ReLU(scale, in_place=True)
    return relu

def build_c_relu_block_1(n, stage, block, last_layer, use_global_stats, bn_inplace=True, to_seperable=False, last_layer_out=None, num_output=24, with_bn=False, new_depth=False, zero=False):

    if zero:
        param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0.0)]
    else:
        param=[dict(lr_mult=0.1, decay_mult=0.1), dict(lr_mult=0.2, decay_mult=0.0)]

    if zero:
        scale_param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    else:
        scale_param=[dict(lr_mult=0.1, decay_mult=0), dict(lr_mult=0.1, decay_mult=0)]

    if with_bn:
        n['conv{0}_{1}/1/pre'.format(stage, block)] = bn = L.BatchNorm(last_layer, param=[dict(lr_mult=0),dict(lr_mult=0),dict(lr_mult=0)], in_place=False, name='conv{0}_{1}/1/bn'.format(stage, block), use_global_stats=use_global_stats)
        #scale
        n['conv{0}_{1}/1/bn_scale'.format(stage, block)] = scale = L.Scale(bn, scale_param=dict(bias_term=True), in_place=True, param=scale_param)
        # relu
        n['conv{0}_{1}/1/relu'.format(stage, block)] = relu = L.ReLU(scale, in_place=True)
        last_layer = relu

    if new_depth:
        stride = 2
    else:
        stride = 1
    # 1 X 1 convoltion
    n['conv{0}_{1}/1'.format(stage, block)] = conv = L.Convolution(last_layer, convolution_param=dict(pad_h=0, pad_w=0, kernel_h=1, kernel_w=1, stride_h=stride, stride_w=stride, num_output=num_output, weight_filler=dict(type='xavier'), bias_filler = dict(type='constant', value=0.1)), name='conv{0}_{1}/1/conv'.format(stage, block), param=param)
    return conv


def build_c_relu_block_2(n, stage, block, last_layer, use_global_stats, bn_inplace=True, to_seperable=False, last_layer_out=None, num_output=24, zero=False):
    #  batch normalization
    if zero:
        param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0.0)]
    else:
        param=[dict(lr_mult=0.1, decay_mult=0.1), dict(lr_mult=0.2, decay_mult=0.0)]

    if zero:
        scale_param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    else:
        scale_param=[dict(lr_mult=0.1, decay_mult=0), dict(lr_mult=0.1, decay_mult=0)]

    n['conv{0}_{1}/2/pre'.format(stage, block)] = bn = L.BatchNorm(last_layer, param=[dict(lr_mult=0),dict(lr_mult=0),dict(lr_mult=0)], in_place=False, name='conv{0}_{1}/2/bn'.format(stage, block), use_global_stats=use_global_stats)
    #scale
    n['conv{0}_{1}/2/bn_scale'.format(stage, block)] = scale = L.Scale(bn, scale_param=dict(bias_term=True), in_place=True, param=scale_param)
    # relu
    n['conv{0}_{1}/2/relu'.format(stage, block)] = relu = L.ReLU(scale, in_place=True)

    # convolution
    if to_seperable:
        conv = get_seperable(n, last_layer, conv_param, last_layer_out, 'conv{0}_{1}/2/conv'.format(stage, block))
    else:
        conv = n['conv{0}_{1}/2'.format(stage, block)] = L.Convolution(relu, convolution_param=dict(pad_h=1, pad_w=1, kernel_h=3, kernel_w=3, stride_h=1, stride_w=1, num_output=num_output, weight_filler=dict(type='xavier'), bias_filler = dict(type='constant', value=0.1)), name='conv{0}_{1}/2/conv'.format(stage, block), param=param)
    return conv

def build_c_relu_block_3(n, stage, block, last_layer, use_global_stats, bn_inplace=True, to_seperable=False, last_layer_out=None, num_output=24, zero=False):
    # batch normalization
    if zero:
        param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0.0)]
    else:
        param=[dict(lr_mult=0.1, decay_mult=0.1), dict(lr_mult=0.2, decay_mult=0.0)]

    if zero:
        scale_param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    else:
        scale_param=[dict(lr_mult=0.1, decay_mult=0), dict(lr_mult=0.1, decay_mult=0)]

    if not bn_inplace:
        bn = n['conv{0}_{1}/3/pre'.format(stage, block)] = L.BatchNorm(last_layer, param=[dict(lr_mult=0),dict(lr_mult=0),dict(lr_mult=0)], in_place=bn_inplace, name='conv{0}_{1}/3/bn'.format(stage, block), use_global_stats=use_global_stats)
    else:
        bn = n['conv{0}_{1}/bn'.format(stage, block)] = L.BatchNorm(conv, param=[dict(lr_mult=0),dict(lr_mult=0),dict(lr_mult=0)], in_place=bn_inplace, use_global_stats=use_global_stats)

    # negation
    n['conv{0}_{1}/3/neg'.format(stage, block)] = neg = L.Power(bn, power_param=dict(power=1, scale=-1.0, shift=0))
    # concat
    n['conv{0}_{1}/3/preAct'.format(stage, block)] = concat = L.Concat(bn, neg, name='conv{0}_{1}/3/concat'.format(stage, block))
    # scale
    n['conv{0}_{1}/3/scale'.format(stage, block)] = scale = L.Scale(concat, scale_param=dict(bias_term=True), in_place=True, param=scale_param)
    # relu
    n['conv{0}_{1}/3/relu'.format(stage, block)] = relu = L.ReLU(scale, in_place=True)

    conv = n['conv{0}_{1}/3'.format(stage, block)] = L.Convolution(relu, convolution_param=dict(pad_h=0, pad_w=0, kernel_h=1, kernel_w=1, stride_h=1, stride_w=1, num_output=num_output, weight_filler=dict(type='xavier'), bias_filler = dict(type='constant', value=0.1)), name='conv{0}_{1}/3/conv'.format(stage, block), param=param)

    return conv


def build_residual_c_rulu_block(n, stage, block, use_global_stats, num_output, last_layer, residual_mode='conv', with_batch_normalization_on_left=False, replace_last_layer=False, to_seperable=False, new_depth=False, zero=False):

    if zero:
        param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0.0)]
    else:
        param=[dict(lr_mult=0.1, decay_mult=0.1), dict(lr_mult=0.2, decay_mult=0.0)]

    if zero:
        scale_param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    else:
        scale_param=[dict(lr_mult=0.1, decay_mult=0), dict(lr_mult=0.1, decay_mult=0)]

    if new_depth:
        n['conv{0}_{1}/1/pre'.format(stage, block)] = bn = L.BatchNorm(last_layer, param=[dict(lr_mult=0),dict(lr_mult=0),dict(lr_mult=0)], in_place=False, name='conv{0}_{1}/1/bn'.format(stage, block), use_global_stats=use_global_stats)
        n['conv{0}_{1}/1/bn_scale'.format(stage, block)] = bn_scale = L.Scale(bn, scale_param=dict(bias_term=True), in_place=True, name='conv{0}_{1}/1/bn_scale'.format(stage, block), param=scale_param)
        n['conv{0}_{1}/1/relu'.format(stage, block)] = relu = L.ReLU(bn_scale, in_place=True, name='conv{0}_{1}/1/relu'.format(stage, block))
        last_layer = relu

    c1 = build_c_relu_block_1(n, stage, block, last_layer, use_global_stats, bn_inplace=True, num_output=num_output[0], with_bn=with_batch_normalization_on_left, new_depth=new_depth)
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

def get_seperable(netspec, last_layer, conv_param, M, name):
    kernel_h = conv_param['kernel_h']
    kernel_w = conv_param['kernel_w']
    stride_h = conv_param['stride_h']
    stride_w = conv_param['stride_w']
    pad_h = conv_param['pad_h']
    pad_w = conv_param['pad_w']
    assert(kernel_h > 0 and kernel_w > 0)
    assert(M > 0)
    keep_output = conv_param['num_output']
    dw_conv = BaseLegoFunction('Convolution', dict(name='{0}/dw'.format(name), convolution_param=dict(pad_h=pad_h, pad_w=pad_w, kernel_h=kernel_h, kernel_w=kernel_w, stride_h=stride_h, stride_w=stride_w, num_output=M, group=M))).attach(netspec, [last_layer])
    pw_param  = dict(pad_h=0, pad_w=0, kernel_h=1, kernel_w=1, stride_h=1, stride_w=1, num_output=keep_output)
    conv = BaseLegoFunction('Convolution', dict(name=name, convolution_param=pw_param)).attach(netspec, [dw_conv])
    return conv


def build_inception_module(n, stage, block, name, order, conv_param, use_global_stats, last_layer, to_seperable=False, last_layer_out=None, zero=False):

    conv_param['weight_filler'] = dict(type='xavier')

    if zero:
        param=[dict(lr_mult=0, decay_mult=0)]
    else:
        param=[dict(lr_mult=0.1, decay_mult=0.1)]

    if zero:
        scale_param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    else:
        scale_param=[dict(lr_mult=0.1, decay_mult=0), dict(lr_mult=0.1, decay_mult=0)]

    # conv
    if to_seperable:
        print 'to_seperable'
        n[top_name] = get_seperable(n, last_layer, conv_param, last_layer_out, conv_name)
    else:
        n['conv{0}_{1}/{2}/{3}'.format(stage, block, name, order)] = conv = L.Convolution(last_layer, convolution_param=conv_param, name='conv{0}_{1}/{2}/{3}/conv'.format(stage, block, name, order), param=param)
    # batch normalization
    n['conv{0}_{1}/{2}/{3}/bn'.format(stage, block, name, order)] = bn = L.BatchNorm(conv, param=[dict(lr_mult=0),dict(lr_mult=0),dict(lr_mult=0)], name='conv{0}_{1}/{2}/{3}/bn'.format(stage, block, name, order).format(stage, block, name, order), in_place=True, use_global_stats=use_global_stats)
    # scale
    n['conv{0}_{1}/{2}/{3}/bn_scale'.format(stage, block, name, order)] = bn_scale = L.Scale(bn, scale_param=dict(bias_term=True), name='conv{0}_{1}/{2}/{3}/bn_scale'.format(stage, block, name, order), in_place=True, param=scale_param)
    # relu
    relu_name = 'conv{0}_{1}/{2}/{3}/relu'.format(stage, block, name, order)
    n['conv{0}_{1}/{2}/{3}/relu'.format(stage, block, name, order)] = relu = L.ReLU(bn_scale, name='conv{0}_{1}/{2}/{3}/relu'.format(stage, block, name, order), in_place=True)

    return relu


def build_inception_block(n, stage, block, last_layer, use_global_stats, module_params, pool_param, residual_mode='conv', num_output=256, last=False, to_seperable=False, new_depth=False, zero=False):

    if zero:
        scale_param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    else:
        scale_param=[dict(lr_mult=0.1, decay_mult=0), dict(lr_mult=0.1, decay_mult=0)]
    n['conv{0}_{1}/incep/pre'.format(stage, block)] = bn = L.BatchNorm(last_layer, param=[dict(lr_mult=0),dict(lr_mult=0),dict(lr_mult=0)], in_place=False, name='conv{0}_{1}/incep/bn'.format(stage, block), use_global_stats=use_global_stats)
    n['conv{0}_{1}/incep/bn_scale'.format(stage, block)] = bn_scale = L.Scale(bn, scale_param=dict(bias_term=True), in_place=True, name='conv{0}_{1}/incep/bn_scale'.format(stage, block), param=scale_param)
    n['conv{0}_{1}/incep/relu'.format(stage, block)] = relu = L.ReLU(bn_scale, in_place=True, name='conv{0}_{1}/incep/relu'.format(stage, block))

    # first module
    inception_param_0 = module_params['incep_0']
    incep_0 = build_inception_module(n, stage, block, 'incep', '0', inception_param_0, use_global_stats, relu)
    # second module
    inception_param_reduce_1 = module_params['incep_reduce_1']
    reduce_out = build_inception_module(n, stage, block, 'incep', '1_reduce', inception_param_reduce_1, use_global_stats, relu)
    inception_param_1_0 = module_params['incep_1_0']
    incep_1_0 = build_inception_module(n, stage, block, 'incep', '1_0', inception_param_1_0, use_global_stats, reduce_out, to_seperable=to_seperable, last_layer_out=module_params['incep_reduce_1']['num_output'])
    # third module
    inception_param_reduce_2 = module_params['incep_reduce_2']
    reduce_out = build_inception_module(n, stage, block, 'incep', '2_reduce', inception_param_reduce_2, use_global_stats, relu, to_seperable=to_seperable)
    inception_param_2_0 = module_params['incep_2_0']
    incep_2_0 = build_inception_module(n, stage, block, 'incep', '2_0', inception_param_2_0, use_global_stats, reduce_out, to_seperable=to_seperable, last_layer_out=module_params['incep_reduce_2']['num_output'])
    inception_param_2_1 = module_params['incep_2_1']
    incep_2_1 = build_inception_module(n, stage, block, 'incep', '2_1', inception_param_2_1, use_global_stats, incep_2_0, to_seperable=to_seperable, last_layer_out=module_params['incep_2_1']['num_output'])

    concat_layers = [incep_0, incep_1_0, incep_2_1]

    if bool(pool_param):
        n['conv{0}_{1}/incep/pool'.format(stage, block)] = incep_pool = L.Pooling(relu, pooling_param=pool_param)
        # inception module after pool
        inception_param_poolproj = module_params['incep_poolproj']
        n['conv{0}_{1}/incep/poolproj'.format(stage, block)] = poolproj = build_inception_module(n, stage, block, 'incep', 'poolproj', inception_param_poolproj, use_global_stats, incep_pool)
        concat_layers.append(poolproj)
    # concat
    n['conv{0}_{1}/incep'.format(stage, block)] = concat = L.Concat(*concat_layers)


    # residual
    conv_param = dict(pad_h=0, pad_w=0, kernel_h=1, kernel_w=1, stride_h=1, stride_w=1, num_output=num_output, weight_filler = dict(type='xavier'))
    if last:
        conv_param['bias_term'] = False

    if last and zero:
        param=[dict(lr_mult=0, decay_mult=0)]
    elif last and not zero:
        param=[dict(lr_mult=0.1, decay_mult=0.1)]
    elif not last and zero:
        param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    else:
        param=[dict(lr_mult=0.1, decay_mult=0.1), dict(lr_mult=0.2, decay_mult=0)]

    n['conv{0}_{1}/out'.format(stage, block)] = conv_incep_out = L.Convolution(concat, convolution_param=conv_param, name='conv{0}_{1}/out/conv'.format(stage, block), param=param)

    left_branch = conv_incep_out

    if last:
        # batch normalization
        out_bn = L.BatchNorm(conv_incep_out, param=[dict(lr_mult=0),dict(lr_mult=0),dict(lr_mult=0)], in_place=True, name='conv{0}_{1}/out/bn'.format(stage, block), use_global_stats=use_global_stats)
        #scale
        out_scale = L.Scale(out_bn, scale_param=dict(bias_term=True), in_place=True, name='conv{0}_{1}/out/bn_scale'.format(stage, block), param=scale_param)
        left_branch = out_scale

    if residual_mode != 'power':
        if zero:
            param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
        else:
            param=[dict(lr_mult=0.1, decay_mult=0.1), dict(lr_mult=0.2, decay_mult=0)]
        n['conv{0}_{1}/proj'.format(stage, block)] = conv_proj = L.Convolution(last_layer, convolution_param=dict(pad_h=0, pad_w=0, kernel_h=1, kernel_w=1, stride_h=2, stride_w=2, num_output=num_output, weight_filler=dict(type='xavier')),  param=param)
    else:
        n['conv{0}_{1}/input'.format(stage, block)] = conv_proj = L.Power(last_layer)


    n['conv{0}_{1}'.format(stage, block)] = out = L.Eltwise(left_branch, conv_proj, eltwise_param=dict(operation=P.Eltwise.SUM, coeff=[1, 1]))

    if last:
        # batch normalization
        print 'last bn'
        n['conv{0}_{1}/last_bn'.format(stage, block)] = last_bn = L.BatchNorm(out, param=[dict(lr_mult=0),dict(lr_mult=0),dict(lr_mult=0)], in_place=True, name='conv{0}_{1}/last_bn'.format(stage, block), use_global_stats=use_global_stats)
        #scale
        n['conv{0}_{1}/last_bn_scale'.format(stage, block)] = last_scale = L.Scale(last_bn, scale_param=dict(bias_term=True), in_place=True, name='conv{0}_{1}/last_bn_scale'.format(stage, block), param=scale_param)
        n['conv{0}_{1}/last_relu'.format(stage, block)] = last_relu = L.ReLU(last_scale, in_place=True, name='conv{0}_{1}/last_relu'.format(stage, block))
        out = last_relu


    return out

def make_fully(n, name, num_output, last_layer, use_global_stats):
    n[name] = ip = L.InnerProduct(last_layer, name=name, inner_product_param=dict(num_output=num_output, weight_filler = dict(type = 'xavier'), bias_filler = dict(type = 'constant', value = 0.1)))
    #  # batch normalization
    n['{}/bn'.format(name)] = bn = L.BatchNorm(ip, param=[dict(lr_mult=0),dict(lr_mult=0),dict(lr_mult=0)], in_place=True, use_global_stats=use_global_stats)
    #scale
    scale_param=[dict(lr_mult=1, decay_mult=0), dict(lr_mult=1, decay_mult=0)]
    n['{}/scale'.format(name)] = scale = L.Scale(bn, scale_param=dict(bias_term=True), in_place=True, param=scale_param)
    # dropout
    n['{}/dropout'.format(name)] = dropout = L.Dropout(scale, in_place=True, dropout_param=dict(dropout_ratio=0.5))
    # relu
    n['{}/relu'.format(name)] = relu = L.ReLU(dropout, in_place=True)

    return relu

def write_prototxt(is_train, source, output_folder, to_seperable=False):
    n = netspec = caffe.NetSpec()
    if is_train:
        include = 'train'
        use_global_stats = False
        batch_size = 128
    else:
        include = 'test'
        use_global_stats = True
        batch_size = 1

    use_global_stats = True
    zero = False

    n.data, n.im_info, n.gt_boxes = L.Python(name='input-data', ntop=3, python_param=dict(module="caffe.frcnn.roi_data_layer.layer", layer="RoIDataLayer", param_str='num_classes: 21'))
    #  netspec.data, netspec.im_info, netspec.gt_boxes = BaseLegoFunction('Python', params).attach(netspec, [])

    #  params = dict(name='data', ntop = 2, input_param=dict(shape=dict(dim=[1, 3, 320, 320])))
    #  netspec.data, netspec.label = BaseLegoFunction('Input', params).attach(netspec, [])
    #  netspec.data = BaseLegoFunction('Input', params).attach(netspec, [])
    #  netspec.data = L.Input(input_param=dict(shape=dict(dim=[1,3,224,224])))
    # stage1
    # C.RELU block
    # initial c.RELU block
    # 7X7 convoltion
    c1 = build_init_c_relu(netspec, stage=1, block=1, bottom=n.data, num_output=3, use_global_stats=use_global_stats, zero=zero)
    # max pool 3_3
    netspec['pool1'] = max_pool_3_3 = L.Pooling(c1, pooling_param=dict(pool=P.Pooling.MAX, kernel_size=3, stride=2, pad=0))
    # stage2
    c_2_1 = build_residual_c_rulu_block(netspec, 2, 1, use_global_stats, [24, 24, 64], max_pool_3_3, to_seperable=to_seperable, zero=zero)
    c_2_2 = build_residual_c_rulu_block(netspec, 2, 2, use_global_stats, [24, 24, 64], c_2_1, with_batch_normalization_on_left=True, residual_mode='power', to_seperable=to_seperable, zero=zero)
    c_2_3 = build_residual_c_rulu_block(netspec, 2, 3, use_global_stats, [24, 24, 64], c_2_2, with_batch_normalization_on_left=True, residual_mode='power', to_seperable=to_seperable, zero=zero)
    # stage 3
    c_3_1 = build_residual_c_rulu_block(netspec, 3, 1, use_global_stats, [48, 48, 128], c_2_3, to_seperable=to_seperable, new_depth=True, with_batch_normalization_on_left=False, zero=zero)
    c_3_2 = build_residual_c_rulu_block(netspec, 3, 2, use_global_stats, [48, 48, 128], c_3_1, with_batch_normalization_on_left=True, residual_mode='power', to_seperable=to_seperable, zero=zero)
    c_3_3 = build_residual_c_rulu_block(netspec, 3, 3, use_global_stats, [48, 48, 128], c_3_2, with_batch_normalization_on_left=True, residual_mode='power', to_seperable=to_seperable, zero=zero)
    c_3_4 = build_residual_c_rulu_block(netspec, 3, 4, use_global_stats, [48, 48, 128], c_3_3, with_batch_normalization_on_left=True, residual_mode='power', to_seperable=to_seperable, zero=zero)

    # Inception block
    module_params = {}
    # incep_4_1
    # from leftmost to rightmost
    # 1X1 convoltion
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
    incep_4_1 = build_inception_block(netspec, 4, 1, c_3_4, use_global_stats, module_params, pool_param, to_seperable=to_seperable, new_depth=True, zero=zero)

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
    incep_4_2 = build_inception_block(netspec, 4, 2, incep_4_1, use_global_stats, module_params, {}, residual_mode='power', to_seperable=to_seperable, zero=zero)
    # incep_4_3
    incep_4_3 = build_inception_block(netspec, 4, 3, incep_4_2, use_global_stats, module_params, {}, residual_mode='power', to_seperable=to_seperable, zero=zero)
    # incep_4_4
    incep_4_4 = build_inception_block(netspec, 4, 4, incep_4_3, use_global_stats, module_params, {}, residual_mode='power', to_seperable=to_seperable, zero=zero)

    # incep_5_1
    # from leftmost to rightmost
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
    incep_5_1 = build_inception_block(netspec, 5, 1, incep_4_4, use_global_stats, module_params, pool_param, num_output=384, to_seperable=to_seperable, zero=zero)

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
    incep_5_2 = build_inception_block(netspec, 5, 2, incep_5_1, use_global_stats, module_params, {}, residual_mode='power', num_output=384, to_seperable=to_seperable, zero=zero)
    # incep_4_3
    incep_5_3 = build_inception_block(netspec, 5, 3, incep_5_2, use_global_stats, module_params, {}, residual_mode='power', num_output=384, to_seperable=to_seperable, zero=zero)
    # incep_4_4
    incep_5_4 = build_inception_block(netspec, 5, 4, incep_5_3, use_global_stats, module_params, {}, residual_mode='power', num_output=384, last=True, to_seperable=to_seperable, zero=zero)

    if zero:
        param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    else:
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2.0, decay_mult=0)]

    n = netspec

    # upsample
    netspec['upsample'] = upsample = L.Deconvolution(incep_5_4, param=[dict(lr_mult=0, decay_mult=0)], convolution_param=dict(pad=1, stride=2 , kernel_size=4, group=384, num_output=384, bias_term=False, weight_filler=dict(type='bilinear')))
    # downsample
    netspec['downsample'] = downsample = L.Pooling(c_3_4, pooling_param=dict(pad=0, stride=2 , kernel_size=3, pool=P.Pooling.MAX))
    # concat 
    netspec['concat'] = concat = L.Concat(upsample, downsample, incep_4_4, concat_param=dict(axis=1))

    # reduce channels
    netspec['convf_rpn'] = convf_rpn = L.Convolution(concat, param=param, convolution_param=dict(pad=0, stride=1 , kernel_size=1, num_output=128, weight_filler=dict(type='xavier', std=0.1), bias_filler=dict(type = 'constant', value=0.1)))
    netspec['reluf_rpn'] = reluf_rpn = L.ReLU(convf_rpn, in_place=True)

    # reduce channels
    netspec['convf_2'] = convf2 = L.Convolution(concat, param=param, convolution_param=dict(pad=0, stride=1 , kernel_size=1, num_output=384, weight_filler=dict(type='xavier', std=0.1), bias_filler=dict(type = 'constant', value=0.1)))
    netspec['reluf_2'] = reluf2 = L.ReLU(convf2, in_place=True)

    # concat 
    netspec['concat_convf'] = concat_conf = L.Concat(reluf_rpn, reluf2, concat_param=dict(axis=1))

    anchor_num = 25


    #  # train rpn
    n['rpn_conv1'] = rpn_conv1 = L.Convolution(convf_rpn, param=param, convolution_param=dict(pad=1, stride=1 , kernel_size=3, num_output=384, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0)))
    n['rpn_relu1'] = rpn_relu1 = L.ReLU(rpn_conv1, in_place=True)

    n['rpn_cls_score'] = rpn_cls_score = L.Convolution(rpn_relu1, param=[dict(lr_mult=1.0, decay_mult=1.0), dict(lr_mult=2.0, decay_mult=0)], convolution_param=dict(pad=0, stride=1 , kernel_size=1, num_output=anchor_num * 2, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type = 'constant', value=0)))
    n['rpn_bbox_pred'] = rpn_bbox_pred = L.Convolution(rpn_relu1, param=[dict(lr_mult=1.0, decay_mult=1.0), dict(lr_mult=2.0, decay_mult=0)], convolution_param=dict(pad=0, stride=1 , kernel_size=1, num_output=anchor_num * 4, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type = 'constant', value=0)))

    n['rpn_cls_score_reshape'] = rpn_cls_score_reshape = L.Reshape(rpn_cls_score, reshape_param=dict(shape=dict(dim=[0, 2, -1, 0])))

    n.rpn_labels, n.rpn_bbox_targets, n.rpn_bbox_inside_weights, n.rpn_bbox_outside_weights =  \
            L.Python(rpn_cls_score, n.gt_boxes, n.im_info, n.data, name='rpn-data', python_param=dict(module='caffe.frcnn.rpn.anchor_target_layer', layer='AnchorTargetLayer', param_str="{'feat_stride': 16, 'ratios': [0.5, 0.667, 1.0, 1.5, 2.0], 'scales': [3, 6, 9, 16, 25]}"), ntop=4)

    n.rpn_cls_score = L.SoftmaxWithLoss(rpn_cls_score_reshape, n.rpn_labels, propagate_down=[1,0], name='rpn_loss_cls', loss_weight=1, loss_param=dict(ignore_label=-1, normalize=True))

    n.rpn_loss_bbox = L.SmoothL1Loss(rpn_bbox_pred, n.rpn_bbox_targets, n.rpn_bbox_inside_weights, n.rpn_bbox_outside_weights, loss_weight=1, smooth_l1_loss_param=dict(sigma=3.0))

    # DummyLayers
    n.dummy_roi_pool_conv5 = L.DummyData(dummy_data_param=dict(shape=dict(dim =[1,18432]), data_filler=dict(type='constant', value=0)))


    #  fc6
    fc6 = make_fully(netspec, 'fc6', 4096, n.dummy_roi_pool_conv5, use_global_stats)
    #  fc7
    fc7 = make_fully(netspec, 'fc7', 4096, fc6, use_global_stats)
    # silence
    L.Silence(fc7, name='silence_fc7')

    #  #  imagenet
    #  fc8 = netspec['fc8'] = BaseLegoFunction('InnerProduct', dict(name='fc8', param=[dict(lr_mult=1.0, decay_mult=1.0)], inner_product_param=dict(num_output=1000))).attach(netspec, [fc7])
    #  #  softmax
    #  #  netspec['loss'] = BaseLegoFunction('SoftmaxWithLoss', dict(name='loss')).attach(netspec, [fc8, netspec.label])

    #  filename = 'train.prototxt' if is_train else 'test.prototxt'
    #  filepath = output_folder + '/' + filename
    #  fp = open(filepath, 'w')
    #  print >> fp, netspec.to_proto()
    #  fp.close()
    return netspec


if __name__ == '__main__':
    args = parser.parse_args()
    netspec = write_prototxt(True, 'train', args.output_folder, to_seperable=False)
    filepath = './stage1_rpn_train.prototxt'
    open(filepath, 'w').write(str(netspec.to_proto()))
    #  net = caffe.Net(filepath, "/home/tumh/pva9.1_pretrained_no_fc6.caffemodel", caffe.TEST)
    net = caffe.Net(filepath, "/home/tumh/SJ/pva-faster-rcnn/pvanet_1000000.caffemodel", caffe.TEST)

    # Also print out the network complexity
    #  params, flops = get_complexity(prototxt_file=filepath)
    #  print 'Number of params: ', (1.0 * params) / 1000000.0, ' Million'
    #  print 'Number of flops: ', (1.0 * flops) / 1000000.0, ' Million'

