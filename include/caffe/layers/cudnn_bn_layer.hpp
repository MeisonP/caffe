#ifndef CAFFE_CUDNN_BN_LAYER_HPP_
#define CAFFE_CUDNN_BN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layers/bn_layer.hpp"
#include "caffe/proto/caffe.pb.h"
/**
 * @brief Batch normalization the input blob along the channel axis while
 *        averaging over the spatial axes.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
namespace caffe {
/**
 * @brief cuDNN implementation of BNLayer.
 *        Fallback to BNLayer for CPU mode.
 */
#ifdef USE_CUDNN
template <typename Dtype>
class CuDNNBNLayer : public BNLayer<Dtype> {
 public:
  explicit CuDNNBNLayer(const LayerParameter& param)
      : BNLayer<Dtype>(param), handles_setup_(false) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual ~CuDNNBNLayer();

  virtual inline const char* type() const { return "BN"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  bool handles_setup_;
  cudnnHandle_t handle_;
  cudnnTensorDescriptor_t bottom_desc_;
  cudnnTensorDescriptor_t top_desc_;
  cudnnTensorDescriptor_t bn_param_desc_;

  Blob<Dtype> save_mean_;
  Blob<Dtype> save_inv_variance_;
};
#endif
}
#endif  // CAFFE_BATCHNORM_LAYER_HPP_

