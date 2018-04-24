// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include "caffe/layers/upsample_layer.hpp"

namespace caffe {

template <typename Dtype>
void UpsampleLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  UpsampleParameter upsample_param = this->layer_param_.upsample_param();
  scale_ = upsample_param.scale();
}

template <typename Dtype>
void UpsampleLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> out_shape;
  for (int i = 0; i < bottom[0]->num_axes(); i++) {
    out_shape.push_back(bottom[0]->shape(i));
  }

  out_shape[bottom[0]->num_axes() - 1] *= scale_;
  out_shape[bottom[0]->num_axes() - 2] *= scale_;
  top[0]->Reshape(out_shape);

}

template <typename Dtype>
void UpsampleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void UpsampleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(UpsampleLayer);
#endif

INSTANTIATE_CLASS(UpsampleLayer);
REGISTER_LAYER_CLASS(Upsample);

}  // namespace caffe

