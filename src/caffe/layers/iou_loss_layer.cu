// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <cfloat>
#include "caffe/layers/iou_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void IOULossForward(const int nthreads, const Dtype* pred_data, const Dtype* gt_data, Dtype *loss, Dtype* count, const int height, const int width) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / (height * width);
    int h = (index % (width * height)) / width;
    int w = index % width;
    const int channels = 4;
    Dtype p_xt = pred_data[((n * channels + 0) * height + h) * width + w];
    Dtype p_xb = pred_data[((n * channels + 1) * height + h) * width + w];
    Dtype p_xl = pred_data[((n * channels + 2) * height + h) * width + w];
    Dtype p_xr = pred_data[((n * channels + 3) * height + h) * width + w];

    Dtype g_xt = gt_data[((n * channels + 0) * height + h) * width + w];
    Dtype g_xb = gt_data[((n * channels + 1) * height + h) * width + w];
    Dtype g_xl = gt_data[((n * channels + 2) * height + h) * width + w];
    Dtype g_xr = gt_data[((n * channels + 3) * height + h) * width + w];

    if (p_xt == 0 && p_xb == 0 && p_xl == 0 && p_xr == 0) {
      count[index] = 0;
      continue;
    }

    if (g_xt == 0 && g_xb == 0 && g_xl == 0 && g_xr == 0) {
      count[index] = 0;
      continue;
    }


    // area_
    Dtype x_tb = p_xt + p_xb;
    Dtype x_lr = p_xl + p_xr;
    // x_tb_data[offset] = x_tb;
    // x_lr_data[offset] = x_lr;

    Dtype p_x = x_tb * x_lr;
    Dtype g_x = (g_xt + g_xb) * (g_xt + g_xb);

    // intersection
    Dtype ih = min(p_xt, g_xt) + min(p_xb, g_xb);
    // ih_data[offset] = ih;

    Dtype iw = min(p_xl, g_xl) + min(p_xr, g_xr);
    // iw_data[offset] = iw;
    Dtype i = ih * iw;
    // i_data[offset] = std::max(i, Dtype(FLT_MIN);

    Dtype u = p_x + g_x - i;
    Dtype iou = (i / max(u, Dtype(FLT_MIN)));
    // u_data[offset] = std::max(u, Dtype(FLT_MIN);
    loss[index] = -log(iou);
    count[index] = 1;
  }
}

template <typename Dtype>
void IouLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  const int nthreads = num * height * width; 

  const Dtype* pred_data = bottom[0]->gpu_data();
  const Dtype* gt_data = bottom[1]->gpu_data();

  Dtype *loss_data = loss_.mutable_gpu_data();
  Dtype *count_data = count_.mutable_gpu_data();

  IOULossForward<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      nthreads, pred_data, gt_data, loss_data, count_data, height, width);
  CUDA_POST_KERNEL_CHECK;

  Dtype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss);

  Dtype valid_count;
  caffe_gpu_asum(nthreads, count_data, &valid_count);

  top[0]->mutable_cpu_data()[0] = loss / valid_count;
}

template <typename Dtype>
__global__ void IOULossBackward(const int nthreads, const Dtype* pred_data, const Dtype* gt_data, Dtype *bottom_diff, Dtype* count, const int height, const int width) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / (height * width);
    int h = (index % (width * height)) / width;
    int w = index % width;
    const int channels = 4;
    Dtype p_xt = pred_data[((n * channels + 0) * height + h) * width + w];
    Dtype p_xb = pred_data[((n * channels + 1) * height + h) * width + w];
    Dtype p_xl = pred_data[((n * channels + 2) * height + h) * width + w];
    Dtype p_xr = pred_data[((n * channels + 3) * height + h) * width + w];

    Dtype g_xt = gt_data[((n * channels + 0) * height + h) * width + w];
    Dtype g_xb = gt_data[((n * channels + 1) * height + h) * width + w];
    Dtype g_xl = gt_data[((n * channels + 2) * height + h) * width + w];
    Dtype g_xr = gt_data[((n * channels + 3) * height + h) * width + w];

    if (p_xt == 0 && p_xb == 0 && p_xl == 0 && p_xr == 0) {
      bottom_diff[((n * channels + 0) * height + h) * width + w] = 0;
      bottom_diff[((n * channels + 1) * height + h) * width + w] = 0;
      bottom_diff[((n * channels + 2) * height + h) * width + w] = 0;
      bottom_diff[((n * channels + 3) * height + h) * width + w] = 0;
      count[index] = 0;
      continue;
    }

    if (g_xt == 0 && g_xb == 0 && g_xl == 0 && g_xr == 0) {
      bottom_diff[((n * channels + 0) * height + h) * width + w] = 0;
      bottom_diff[((n * channels + 1) * height + h) * width + w] = 0;
      bottom_diff[((n * channels + 2) * height + h) * width + w] = 0;
      bottom_diff[((n * channels + 3) * height + h) * width + w] = 0;
      count[index] = 0;
      continue;
    }


    // area_
    Dtype x_tb = p_xt + p_xb;
    Dtype x_lr = p_xl + p_xr;

    Dtype p_x = x_tb * x_lr;
    Dtype g_x = (g_xt + g_xb) * (g_xt + g_xb);

    // intersection
    Dtype ih = min(p_xt, g_xt) + min(p_xb, g_xb);
    Dtype iw = min(p_xl, g_xl) + min(p_xr, g_xr);
    Dtype i = ih * iw;

    Dtype u = max(p_x + g_x - i, Dtype(FLT_MIN)); 
    // diff_x_t
    if (p_xt < g_xt) {
      bottom_diff[((n * channels + 0) * height + h) * width + w] = (1 / u) * x_lr  - ((u + i) / (u * i)) * iw;
    } else {
      bottom_diff[((n * channels + 0) * height + h) * width + w] = (1 / u) * x_lr;
    }
    // diff_x_b
    if (p_xb < g_xb) {
      bottom_diff[((n * channels + 1) * height + h) * width + w] = (1 / u) * x_lr  - ((u + i) / (u * i)) * iw;
    } else {
      bottom_diff[((n * channels + 1) * height + h) * width + w] = (1 / u) * x_lr;
    }
    // diff_x_l
    if (p_xl < g_xl) {
      bottom_diff[((n * channels + 2) * height + h) * width + w] = (1 / u) * x_tb  - ((u + i) / (u * i)) * ih;
    } else {
      bottom_diff[((n * channels + 2) * height + h) * width + w] = (1 / u) * x_tb;
    }
    // diff_x_r
    if (p_xr < g_xr) {
      bottom_diff[((n * channels + 3) * height + h) * width + w] = (1 / u) * x_tb  - ((u + i) / (u * i)) * ih;
    } else {
      bottom_diff[((n * channels + 3) * height + h) * width + w] = (1 / u) * x_tb;
    }
    count[index] = 1;
  }
}

template <typename Dtype>
void IouLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  const int nthreads = num * height * width; 

  const Dtype* pred_data = bottom[0]->gpu_data();
  const Dtype* gt_data = bottom[1]->gpu_data();

  Dtype *bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype *count_data = count_.mutable_gpu_data();


  IOULossBackward<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      nthreads, pred_data, gt_data, bottom_diff, count_data, height, width);
  CUDA_POST_KERNEL_CHECK;

  Dtype valid_count;
  caffe_gpu_asum(nthreads, count_data, &valid_count);

  const Dtype loss_weight = top[0]->cpu_diff()[0] / valid_count;
  caffe_gpu_scal(bottom[0]->count(), loss_weight , bottom_diff);
                          
}

INSTANTIATE_LAYER_GPU_FUNCS(IouLossLayer);

}  // namespace caffe

