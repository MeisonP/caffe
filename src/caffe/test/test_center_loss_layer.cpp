#include <cmath>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/center_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using std::unique_ptr;

namespace caffe {

template <typename TypeParam>
class CenterLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  CenterLossLayerTest()
      : blob_bottom_data_(new TBlob<Dtype>(10, 5, 1, 1)),
        blob_bottom_label_(new TBlob<Dtype>(10, 1, 1, 1)),
        blob_top_loss_(new TBlob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_std(tol<Dtype>(10., 0.3));
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = caffe_rng_rand() % 10;
    }
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~CenterLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }
  TBlob<Dtype>* const blob_bottom_data_;
  TBlob<Dtype>* const blob_bottom_label_;
  TBlob<Dtype>* const blob_top_loss_;
  vector<Blob*> blob_bottom_vec_;
  vector<Blob*> blob_top_vec_;
};

TYPED_TEST_CASE(CenterLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(CenterLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  if (!is_precise<Dtype>()) {
    return;
  }
  LayerParameter layer_param;
  CenterLossParameter* center_loss_param = layer_param.mutable_center_loss_param();
  FillerParameter* filler_param = center_loss_param->mutable_center_filler();
  filler_param->set_type("xavier");
  center_loss_param->set_num_output(10);
  layer_param.add_loss_weight(3);
  CenterLossLayer<Dtype, Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(tol<Dtype>(1e-2, 1e-1), tol<Dtype>(1e-2, 1e-2));
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0, true);
}
}
