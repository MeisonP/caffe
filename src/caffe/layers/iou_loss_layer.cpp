// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <cfloat>
#include "caffe/layers/iou_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void IouLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void IouLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());

  CHECK_EQ(bottom[0]->channels(), 4);
  vector<int> shape = bottom[0]->shape();
  shape[1] = 1;
  loss_.Reshape(shape);
  count_.Reshape(shape);
  // x_lr_.Reshape(shape);
  // x_tb_.Reshape(shape);
  // ih_.Reshape(shape);
  // iw_.Reshape(shape);
  // i_.Reshape(shape);
  // u_.Reshape(shape);
}


template <typename Dtype>
void IouLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  const Dtype* pred_data = bottom[0]->cpu_data();
  const Dtype* gt_data = bottom[1]->cpu_data();

  // Dtype * x_lr_data = x_lr_.mutable_cpu_data();
  // Dtype * x_tb_data = x_tb_.mutable_cpu_data();
  // Dtype * ih_data = ih_.mutable_cpu_data();
  // Dtype * iw_data = iw_.mutable_cpu_data();
  // Dtype * i_data = i_.mutable_cpu_data();
  // Dtype * u_data = u_.mutable_cpu_data();


  Dtype loss = 0;
  int count = 0;

  for (int n = 0; n < num; n++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {

        // int offset = n * height * width + h * width + width;

        Dtype p_xt = pred_data[((n * channels + 0) * height + h) * width + w];
        Dtype p_xb = pred_data[((n * channels + 1) * height + h) * width + w];
        Dtype p_xl = pred_data[((n * channels + 2) * height + h) * width + w];
        Dtype p_xr = pred_data[((n * channels + 3) * height + h) * width + w];

        Dtype g_xt = gt_data[((n * channels + 0) * height + h) * width + w];
        Dtype g_xb = gt_data[((n * channels + 1) * height + h) * width + w];
        Dtype g_xl = gt_data[((n * channels + 2) * height + h) * width + w];
        Dtype g_xr = gt_data[((n * channels + 3) * height + h) * width + w];

        if (p_xt == 0 && p_xb == 0 && p_xl == 0 && p_xr == 0) {
          // x_tb_data[offset] = 0;
          // x_lr_data[offset] = 0;
          // i_data[offset] = 1;
          // u_data[offset] = 1;
          continue;
        }

        if (g_xt == 0 && g_xb == 0 && g_xl == 0 && g_xr == 0) {
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
        Dtype ih = std::min(p_xt, g_xt) + std::min(p_xb, g_xb);
        // ih_data[offset] = ih;

        Dtype iw = std::min(p_xl, g_xl) + std::min(p_xr, g_xr);
        // iw_data[offset] = iw;
        Dtype i = ih * iw;
        // i_data[offset] = std::max(i, Dtype(FLT_MIN);

        Dtype u = p_x + g_x - i;
        Dtype iou = (i / std::max(u, Dtype(FLT_MIN)));
        // u_data[offset] = std::max(u, Dtype(FLT_MIN);
        loss -= log(iou);
        count++;
      }
    }
  }

  top[0]->mutable_cpu_data()[0] = loss / std::max(count, 1);
}

template <typename Dtype>
void IouLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }

  int count = 0;
  int num = bottom[0]->num();
  int height = bottom[0]->height();
  int channels = bottom[0]->channels();
  int width = bottom[0]->width();

  const Dtype* pred_data = bottom[0]->cpu_data();
  const Dtype* gt_data = bottom[1]->cpu_data();

  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    for (int n = 0; n < num; n++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {

          // int offset = n * height * width + h * width + width;

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
            continue;
          }

          if (g_xt == 0 && g_xb == 0 && g_xl == 0 && g_xr == 0) {
            bottom_diff[((n * channels + 0) * height + h) * width + w] = 0;
            bottom_diff[((n * channels + 1) * height + h) * width + w] = 0;
            bottom_diff[((n * channels + 2) * height + h) * width + w] = 0;
            bottom_diff[((n * channels + 3) * height + h) * width + w] = 0;
            continue;
          }


          // area_
          Dtype x_tb = p_xt + p_xb;
          Dtype x_lr = p_xl + p_xr;

          Dtype p_x = x_tb * x_lr;
          Dtype g_x = (g_xt + g_xb) * (g_xt + g_xb);

          // intersection
          Dtype ih = std::min(p_xt, g_xt) + std::min(p_xb, g_xb);
          Dtype iw = std::min(p_xl, g_xl) + std::min(p_xr, g_xr);
          Dtype i = ih * iw;

          Dtype u = std::max(p_x + g_x - i, Dtype(FLT_MIN)); 
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
          count++;
        }
      }
    }
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / std::max(count, 1);
    caffe_scal(bottom[0]->count(), loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(IouLossLayer);
#endif

INSTANTIATE_CLASS(IouLossLayer);
REGISTER_LAYER_CLASS(IouLoss);

}  // namespace caffe

