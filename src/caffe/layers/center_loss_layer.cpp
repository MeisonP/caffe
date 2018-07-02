#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/center_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void CenterLossLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  const int num_output = this->layer_param_.center_loss_param().num_output();
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.center_loss_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    // Intialize the weight
    const Type btype = blobs_type();
    vector<int> center_shape(2);
    center_shape[0] = N_;
    center_shape[1] = K_;
    this->blobs_[0] = Blob::create(btype, btype);
    // fill the weights
    shared_ptr<Filler<Ftype> > center_filler(GetFiller<Ftype>(
        this->layer_param_.center_loss_param().center_filler()));
    center_filler->Fill(this->blobs_[0].get());
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Ftype, typename Btype>
void CenterLossLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  M_ = bottom[0]->num();
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  LossLayer<Ftype, Btype>::Reshape(bottom, top);
  distance_.ReshapeLike(*bottom[0]);
  variation_sum_.ReshapeLike(*this->blobs_[0]);
  vector<int> count_shape(1);
  count_shape[0] = N_;
  count_.Reshape(count_shape);
}

template <typename Ftype, typename Btype>
void CenterLossLayer<Ftype, Btype>::Forward_cpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->cpu_data<Ftype>();
  const Ftype* label = bottom[1]->cpu_data<Ftype>();
  const Ftype* center = this->blobs_[0]->template cpu_data<Ftype>();
  Ftype* distance_data = distance_.template mutable_cpu_data<Ftype>();
  // the i-th distance_data
  for (int i = 0; i < M_; i++) {
    const int label_value = static_cast<int>(label[i]);
    // D(i,:) = X(i,:) - C(y(i),:)
    caffe_sub<Ftype>(K_, bottom_data + i * K_,
              center + label_value * K_, distance_data + i * K_);
  }
  Ftype dot = caffe_cpu_dot<Ftype>(M_ * K_, distance_.cpu_data(),
                            distance_.cpu_data());
  Ftype loss = dot / M_ / Ftype(2);
  top[0]->mutable_cpu_data<Ftype>()[0] = loss;
}

template <typename Ftype, typename Btype>
void CenterLossLayer<Ftype, Btype>::Backward_cpu(const vector<Blob*>& top,
                                          const vector<bool>& propagate_down,
                                          const vector<Blob*>& bottom) {
  // Gradient with respect to centers
  if (this->param_propagate_down_[0]) {
    const Ftype* label = bottom[1]->cpu_data<Ftype>();
    Ftype* center_diff = this->blobs_[0]->template mutable_cpu_diff<Ftype>();
    Ftype* variation_sum_data = variation_sum_.template mutable_cpu_data<Ftype>();
    int* count_data = count_.template mutable_cpu_data<int>();

    const Ftype* distance_data = distance_.template cpu_data<Ftype>();

    // \sum_{y_i==j}
    caffe_set<Ftype>(N_ * K_, (Ftype)0., variation_sum_.template mutable_cpu_data<Ftype>());
    caffe_set<int>(N_, 0 , count_.template mutable_cpu_data<int>());

    for (int m = 0; m < M_; m++) {
      const int label_value = static_cast<int>(label[m]);
      caffe_sub(K_, variation_sum_data + label_value * K_,
                distance_data + m * K_, variation_sum_data + label_value * K_);
      count_data[label_value]++;
    }
    for (int m = 0; m < M_; m++) {
      const int n = static_cast<int>(label[m]);
      caffe_cpu_axpby<Ftype>(K_, (Ftype)1./ (count_data[n] + (Ftype)1.),
                      variation_sum_data + n * K_,
                      (Ftype)0., center_diff + n * K_);
    }
  }
  // Gradient with respect to bottom data
  if (propagate_down[0]) {
    caffe_copy<Ftype>(M_ * K_, distance_.template cpu_data<Ftype>(),
               bottom[0]->template mutable_cpu_diff<Ftype>());
    caffe_scal<Ftype>(M_ * K_, top[0]->cpu_diff<Ftype>()[0] / M_,
               bottom[0]->mutable_cpu_diff<Ftype>());
  }
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
}


INSTANTIATE_CLASS_FB(CenterLossLayer);
REGISTER_LAYER_CLASS(CenterLoss);

}  // namespace caffe
