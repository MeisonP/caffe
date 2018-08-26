#include "caffe/blob.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
__global__ void AdamUpdate(int N, Dtype* g, Dtype* m, Dtype* v, Dtype *max_v,
    Dtype beta1, Dtype beta2, Dtype eps_hat, Dtype corrected_local_rate, float partial) {
  CUDA_KERNEL_LOOP(i, N) {
    float gi = g[i];
    float mi = m[i] = m[i]*beta1 + gi*(1-beta1);
    float vi = v[i] = v[i]*beta2 + gi*gi*(1-beta2);
    if (max_v) {
      float maxvi = v[i] > max_v[i]? v[i]: max_v[i];
      g[i] = corrected_local_rate * mi / (powf(maxvi, partial) + eps_hat);
    } else {
      g[i] = corrected_local_rate * mi / (powf(vi, partial) + eps_hat);
    }
  }
}
template <typename Dtype>
void adam_update_gpu(int N, Dtype* g, Dtype* m, Dtype* v, Blob<Dtype>* max_v, Dtype beta1,
    Dtype beta2, Dtype eps_hat, Dtype corrected_local_rate, float partial) {
  Dtype* max_v_data = NULL;
  if (max_v) {
    max_v_data = max_v->mutable_gpu_data();
  }
  AdamUpdate<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, g, m, v, max_v_data, beta1, beta2, eps_hat, corrected_local_rate, partial);
  CUDA_POST_KERNEL_CHECK;
}
template void adam_update_gpu<float>(int, float*, float*, float*, Blob<float>*,
    float, float, float, float, float);
template void adam_update_gpu<double>(int, double*, double*, double*, Blob<double>*,
    double, double, double, double, float);

}  // namespace caffe
