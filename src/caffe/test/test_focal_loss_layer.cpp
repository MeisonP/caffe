#include <cmath>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/util/io.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/focal_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

template <typename TypeParam>
class FocalLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  FocalLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 5, 2, 3)),
        blob_bottom_label_(new Blob<Dtype>(10, 1, 2, 3)),
        blob_top_loss_(new Blob<Dtype>())
        {

    MakeTempFilename(&filename_);
    std::ofstream outfile(filename_.c_str(), std::ofstream::out);

    outfile << 0.20 << std::endl;
    outfile << 0.20 << std::endl;
    outfile << 0.20 << std::endl;
    outfile << 0.20 << std::endl;
    outfile << 0.20 << std::endl;
    // fill the values
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = caffe_rng_rand() % 5;
    }
    // for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      // LOG(ERROR) << "label " << blob_bottom_label_[i];
    // }
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }

  virtual ~FocalLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  string filename_;
};

TYPED_TEST_CASE(FocalLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(FocalLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  FocalLossParameter* loss_param =
      layer_param.mutable_focal_loss_param();
  loss_param->set_alpha(this->filename_);
  layer_param.add_loss_weight(3);
  FocalLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(FocalLossLayerTest, TestForwardIgnoreLabel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_loss_param()->set_normalize(false);
  FocalLossParameter* loss_param =
      layer_param.mutable_focal_loss_param();
  loss_param->set_alpha(this->filename_);
  // First, compute the loss with all labels
  scoped_ptr<FocalLossLayer<Dtype> > layer(
      new FocalLossLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype full_loss = this->blob_top_loss_->cpu_data()[0];
  // Now, accumulate the loss, ignoring each label in {0, ..., 4} in turn.
  Dtype accum_loss = 0;
  for (int label = 0; label < 5; ++label) {
    layer_param.mutable_loss_param()->set_ignore_label(label);
    layer.reset(new FocalLossLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    accum_loss += this->blob_top_loss_->cpu_data()[0];
  }
  // Check that each label was included all but once.
  EXPECT_NEAR(4 * full_loss, accum_loss, 1e-4);
}

TYPED_TEST(FocalLossLayerTest, TestGradientIgnoreLabel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // labels are in {0, ..., 4}, so we'll ignore about a fifth of them
  layer_param.mutable_loss_param()->set_ignore_label(0);
  FocalLossParameter* loss_param =
      layer_param.mutable_focal_loss_param();
  loss_param->set_alpha(this->filename_);
  FocalLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(FocalLossLayerTest, TestGradientUnnormalized) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_loss_param()->set_normalize(false);
  FocalLossParameter* loss_param =
      layer_param.mutable_focal_loss_param();
  loss_param->set_alpha(this->filename_);
  FocalLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe

