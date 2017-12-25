// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include "caffe/layers/FRCNN/frcnn_proposal_layer.hpp"
#include "caffe/FRCNN/util/frcnn_utils.hpp"
#include "caffe/FRCNN/util/frcnn_helper.hpp"
#include "caffe/FRCNN/util/frcnn_param.hpp"  
#include "caffe/FRCNN/util/gpu_nms.hpp"  
#include "caffe/util/benchmark.hpp"
#include <fstream>



namespace caffe {

namespace Frcnn {

using std::vector;

template <typename Dtype>
void FrcnnProposalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
  const vector<Blob<Dtype> *> &top) {
  // caffe::Timer time_;
  // time_.Start();

#ifndef CPU_ONLY
  // LOG(INFO) << "proposal anchor size " << FrcnnParam::anchors.size();

  // FrcnnParam::anchor_ratios
  CUDA_CHECK(cudaMalloc(&anchors_, sizeof(float) * FrcnnParam::anchors.size()));
  CUDA_CHECK(cudaMemcpy(anchors_, &(FrcnnParam::anchors[0]),
                        sizeof(float) * FrcnnParam::anchors.size(), cudaMemcpyHostToDevice));

  const int rpn_pre_nms_top_n = 
    this->phase_ == TRAIN ? FrcnnParam::rpn_pre_nms_top_n : FrcnnParam::test_rpn_pre_nms_top_n;;
  CUDA_CHECK(cudaMalloc(&transform_bbox_, sizeof(float) * rpn_pre_nms_top_n * 4));
  CUDA_CHECK(cudaMalloc(&selected_flags_, sizeof(int) * rpn_pre_nms_top_n));

  const int rpn_post_nms_top_n = 
    this->phase_ == TRAIN ? FrcnnParam::rpn_post_nms_top_n : FrcnnParam::test_rpn_post_nms_top_n;
  CUDA_CHECK(cudaMalloc(&gpu_keep_indices_, sizeof(int) * rpn_post_nms_top_n));

#endif
  CHECK(top.size() == 3);
  top[0]->Reshape(1, 5, 1, 1);
  top[1]->Reshape(1, 1, 1, 1);
  top[2]->Reshape(1, 1, 1, 1);
  // LOG(INFO) << "Proposal layer setup cost " << time_.MilliSeconds() << " ms."; 
}

template <typename Dtype>
void FrcnnProposalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                            const vector<Blob<Dtype> *> &top) {
  caffe::Timer time_;
  time_.Start();

  // LOG(ERROR) << "========== enter proposal layer";
  const Dtype *bottom_rpn_score = bottom[0]->cpu_data();  // rpn_cls_prob_reshape
  // LOG(INFO) << "bottom[0]" << bottom[0]->shape_string();
  const Dtype *bottom_rpn_bbox = bottom[1]->cpu_data();   // rpn_bbox_pred
  // LOG(INFO) << "bottom[1]" << bottom[1]->shape_string();
  const Dtype *bottom_im_info = bottom[2]->cpu_data();    // im_info

  // LOG(INFO) << "bottom_rpn_score shape " << bottom[0]->shape_string();
  // LOG(INFO) << "bottom_rpn_bbox shape " << bottom[1]->shape_string();
  // LOG(INFO) << "bottom_im_info shape " << bottom[2]->shape_string();

  const int num = bottom[1]->num();
  // LOG(INFO) << "batch_size " << num;   
  const int channes = bottom[1]->channels();
  const int height = bottom[1]->height();
  const int width = bottom[1]->width();
  // CHECK(num == 1) << "only single item batches are supported";
  CHECK(channes % 4 == 0) << "rpn bbox pred channels should be divided by 4";

  const float im_height = bottom_im_info[0];
  const float im_width = bottom_im_info[1];

  int rpn_pre_nms_top_n;
  int rpn_post_nms_top_n;
  float rpn_nms_thresh;
  int rpn_min_size;
  if (this->phase_ == TRAIN) {
    rpn_pre_nms_top_n = FrcnnParam::rpn_pre_nms_top_n;
    rpn_post_nms_top_n = FrcnnParam::rpn_post_nms_top_n;
    rpn_nms_thresh = FrcnnParam::rpn_nms_thresh;
    rpn_min_size = FrcnnParam::rpn_min_size;
  } else {
    rpn_pre_nms_top_n = FrcnnParam::test_rpn_pre_nms_top_n;
    rpn_post_nms_top_n = FrcnnParam::test_rpn_post_nms_top_n;
    rpn_nms_thresh = FrcnnParam::test_rpn_nms_thresh;
    rpn_min_size = FrcnnParam::test_rpn_min_size;
  }
  const int config_n_anchors = FrcnnParam::anchors.size() / 4;
  LOG_IF(ERROR, rpn_pre_nms_top_n <= 0 ) << "rpn_pre_nms_top_n : " << rpn_pre_nms_top_n;
  LOG_IF(ERROR, rpn_post_nms_top_n <= 0 ) << "rpn_post_nms_top_n : " << rpn_post_nms_top_n;
  if (rpn_pre_nms_top_n <= 0 || rpn_post_nms_top_n <= 0 ) return;

  CHECK(top.size() == 3) << "rois, scores and boundary are required";


  vector<vector<Point4f<Dtype> >> anchors(num);
  typedef pair<Dtype, int> sort_pair;
  vector<vector<sort_pair>> sort_vector(num);

  const Dtype bounds[4] = { im_width - 1, im_height - 1, im_width - 1, im_height -1 };
  const Dtype min_size = bottom_im_info[2] * rpn_min_size;

  DLOG(ERROR) << "========== generate anchors";

  vector<vector<Point4f<Dtype>>> box_final(num);
  vector<vector<Dtype>> scores_final(num);
  int box_count = 0;

  // fstream file;
  // file.open("b2.txt", ios::out | ios::trunc);

  for (int n = 0; n < num; n++) {
    // LOG(INFO) << "n:" << n;
    for (int j = 0; j < height; j++) {
      for (int i = 0; i < width; i++) {
        for (int k = 0; k < config_n_anchors; k++) {

          Dtype score = bottom_rpn_score[(n * (2 * config_n_anchors) * height * width) + config_n_anchors * height * width +
                                          + k * height * width + j * width + i];
          //const int index = i * height * config_n_anchors + j * config_n_anchors + k;
          // LOG(INFO) << " " << score;
          // if (n == 1) {
            // file << score << std::endl;
          // }

          Point4f<Dtype> anchor(
              FrcnnParam::anchors[k * 4 + 0] + i * FrcnnParam::feat_stride,  // shift_x[i][j];
              FrcnnParam::anchors[k * 4 + 1] + j * FrcnnParam::feat_stride,  // shift_y[i][j];
              FrcnnParam::anchors[k * 4 + 2] + i * FrcnnParam::feat_stride,  // shift_x[i][j];
              FrcnnParam::anchors[k * 4 + 3] + j * FrcnnParam::feat_stride); // shift_y[i][j];
          // LOG(INFO) << "(" << j << "," << i << "," << k << ") " << anchor.to_string();

          Point4f<Dtype> box_delta(
              bottom_rpn_bbox[(n * config_n_anchors * 4 * height * width) + (k * 4 + 0) * height * width + j * width + i],
              bottom_rpn_bbox[(n * config_n_anchors * 4 * height * width) + (k * 4 + 1) * height * width + j * width + i],
              bottom_rpn_bbox[(n * config_n_anchors * 4 * height * width) + (k * 4 + 2) * height * width + j * width + i],
              bottom_rpn_bbox[(n * config_n_anchors * 4 * height * width) + (k * 4 + 3) * height * width + j * width + i]);
          // LOG(INFO) << "(" << j << "," << i << "," << k << ") " << box_delta.to_string();

          Point4f<Dtype> cbox = bbox_transform_inv(anchor, box_delta);
          //FIXME: very little precision error when batch size > 1
          // LOG(INFO) << "(" << j << "," << i << "," << k << ") " << cbox.to_string();
          // if (n == 0) {
            // file<<box_delta.to_string()<<std::endl;
          // }
          
          // 2. clip predicted boxes to image
          for (int q = 0; q < 4; q++) {
            cbox.Point[q] = std::max(Dtype(0), std::min(cbox[q], bounds[q]));
          }
          // LOG(INFO) << "clib boxes (" << j << "," << i << "," << k << ") " << cbox.to_string();
          // 3. remove predicted boxes with either height or width < threshold
          if((cbox[2] - cbox[0] + 1) >= min_size && (cbox[3] - cbox[1] + 1) >= min_size) {
            const int now_index = sort_vector[n].size();
            sort_vector[n].push_back(sort_pair(score, now_index)); 
            anchors[n].push_back(cbox);
            // LOG(INFO) << count;
          }
        }
      }
    }
    // if (n == 1) {
      // file.close();
    // }

    // LOG(INFO) << "========== after clip and remove size < threshold box " << (int)sort_vector[n].size() << "," << min_size;

    std::sort(sort_vector[n].begin(), sort_vector[n].end(), std::greater<sort_pair>());
    const int n_anchors = std::min((int)sort_vector[n].size(), rpn_pre_nms_top_n);
    // FIXME: Not sure if this is needed, the accuracy differs a little
    // sort_vector[n].erase(sort_vector[n].begin() + n_anchors, sort_vector[n].end());
    //anchors.erase(anchors.begin() + n_anchors, anchors.end());
    std::vector<bool> select(n_anchors, true);

    // apply nms
    //
    std::vector<int> tmps;
    // LOG(INFO) << "========== apply nms, pre nms number is : " << n_anchors;
    for (int i = 0; i < n_anchors && box_final[n].size() < rpn_post_nms_top_n; i++) {
      if (select[i]) {
        const int cur_i = sort_vector[n][i].second;
        for (int j = i + 1; j < n_anchors; j++)
          if (select[j]) {
            const int cur_j = sort_vector[n][j].second;
            if (get_iou(anchors[n][cur_i], anchors[n][cur_j]) >= rpn_nms_thresh) {
              select[j] = false;
            }
          }
        box_final[n].push_back(anchors[n][cur_i]);
        scores_final[n].push_back(sort_vector[n][i].first);
        // tmps.push_back(cur_i);
      }
    }
    // LOG(INFO) << "box_final [n]" << box_final[n].size();
    box_count += box_final[n].size();
  }

  top[0]->Reshape(box_count, 5, 1, 1);
  top[1]->Reshape(box_count, 1, 1, 1);
  top[2]->Reshape(num, 1, 1, 1);

  Dtype *bbox_data = top[0]->mutable_cpu_data();
  Dtype *score_data = top[1]->mutable_cpu_data();
  Dtype *boundary_data = top[2]->mutable_cpu_data();

  // LOG(ERROR) << "rpn number after nms: " <<  box_count;

  DLOG(ERROR) << "========== copy to top";
  CHECK(box_final.size() == num);
  int count = 0;
  for (int n = 0; n < num; n++) {
    CHECK_EQ(box_final[n].size(), scores_final[n].size());
    for (size_t i = 0; i < box_final[n].size(); i++) {
      Point4f<Dtype> &box = box_final[n][i];
      bbox_data[count * 5 + i * 5] = n;
      for (int j = 1; j < 5; j++) {
        bbox_data[count * 5 + i * 5 + j] = box[j - 1];
      }
      score_data[count + i] = scores_final[n][i];
    }
    boundary_data[n] = box_final[n].size();
    count += box_final[n].size();
  }

  DLOG(ERROR) << "========== exit proposal layer";
  // LOG(INFO) << "Proposal layer  cost " << time_.MilliSeconds() << " ms."; 
}

template <typename Dtype>
void FrcnnProposalLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
    const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom) {
  for (int i = 0; i < propagate_down.size(); ++i) {
    if (propagate_down[i]) {
      NOT_IMPLEMENTED;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(FrcnnProposalLayer);
#endif

INSTANTIATE_CLASS(FrcnnProposalLayer);
REGISTER_LAYER_CLASS(FrcnnProposal);

} // namespace frcnn

} // namespace caffe
