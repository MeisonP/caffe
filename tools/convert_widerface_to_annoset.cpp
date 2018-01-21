// This program converts a set of images and annotations to a lmdb/leveldb by
// storing them as AnnotatedDatum proto buffers.
// Usage:
//   convert_annoset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images and
// annotations, and LISTFILE should be a list of files as well as their labels
// or label files.
// For classification task, the file should be in the format as
//   imgfolder1/img1.JPEG 7
//   ....
// For detection task, the file should be in the format as
//   imgfolder1/img1.JPEG annofolder1/anno1.xml
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "boost/variant.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
    "The backend {lmdb, leveldb} for storing the result");
DEFINE_string(anno_type, "classification",
    "The type of annotation {classification, detection}.");
DEFINE_string(label_type, "xml",
    "The type of annotation file format.");
DEFINE_string(label_map_file, "",
    "A file with LabelMap protobuf message.");
DEFINE_bool(check_label, false,
    "When this option is on, check that there is no duplicated name/label.");
DEFINE_int32(min_dim, 0,
    "Minimum dimension images are resized to (keep same aspect ratio)");
DEFINE_int32(max_dim, 0,
    "Maximum dimension images are resized to (keep same aspect ratio)");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, true,
    "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "jpg",
    "Optional: What type should we encode the image as ('png','jpg',...).");

bool ReadWiderFaceBBox(const std::string labelfile, std::vector<std::string> bboxes, const int img_height, const int img_width, AnnotatedDatum* anno_datum) {
  // Parse annotation.
  int instance_id = 0;
  int label = 1;
  int bbox_num = 0;
  for (int i = 0; i < bboxes.size(); i++) {
    bool difficult = false;
    // If there is no such annotation_group, create a new one.
    stringstream ss(bboxes[i]);
    int xmin, ymin, xmax, ymax, w, h;
    ss >> xmin >> ymin >> w >> h;
    xmax = xmin + w;
    ymax = ymin + h;
    if (xmin > img_width) {
      LOG(WARNING) << labelfile <<
          " bounding box exceeds image boundary.";
    } else if (ymin > img_height) {
      LOG(WARNING) << labelfile <<
          " bounding box exceeds image boundary.";
    } else if (xmax > img_width) {
      LOG(WARNING) << labelfile <<
          " bounding box exceeds image boundary.";
    } else if (ymax > img_height) {
      LOG(WARNING) << labelfile <<
          " bounding box exceeds image boundary.";
    } else if (xmin < 0) {
      LOG(WARNING) << labelfile <<
          " bounding box exceeds image boundary.";
    } else if (ymin < 0) {
      LOG(WARNING) << labelfile <<
          " bounding box exceeds image boundary.";
    
    } else if (xmax < 0) {
      LOG(WARNING) << labelfile <<
          " bounding box exceeds image boundary.";
    } else if (ymax < 0) {
      LOG(WARNING) << labelfile <<
          " bounding box exceeds image boundary.";
    } else if (xmin > xmax) {
      LOG(WARNING) << labelfile <<
          " bounding box irregular.";
    } else if (ymin > ymax) {
      LOG(WARNING) << labelfile <<
          " bounding box irregular.";
    } else {
      Annotation* anno = NULL;
      if (anno_datum->annotation_group_size() == 0) {
        AnnotationGroup * anno_group = anno_datum->add_annotation_group();
        anno_group->set_group_label(label);
        anno = anno_group->add_annotation();
      } else {
        AnnotationGroup * anno_group = anno_datum->mutable_annotation_group(0);
        anno = anno_group->add_annotation();
      }
      anno->set_instance_id(instance_id++);
      // Store the normalized bounding box.
      NormalizedBBox* bbox = anno->mutable_bbox();
      bbox->set_xmin(static_cast<float>(xmin) / img_width);
      bbox->set_ymin(static_cast<float>(ymin) / img_height);
      bbox->set_xmax(static_cast<float>(xmax) / img_width);
      bbox->set_ymax(static_cast<float>(ymax) / img_height);
      bbox->set_difficult(difficult);
      bbox_num++;
    }
  }
  if (bbox_num > 0) {
    return true;
  } else {
    return false;
  }
}


int main(int argc, char** argv) {
#ifdef USE_OPENCV
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images and annotations to the "
        "leveldb/lmdb format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_annoset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // if (argc < 3) {
    // gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_annoset");
    // return 1;
  // }

  const bool is_color = !FLAGS_gray;
  // const bool check_size = FLAGS_check_size;
  // const bool encoded = FLAGS_encoded;
  // const string encode_type = FLAGS_encode_type;
  // const string anno_type = FLAGS_anno_type;
  AnnotatedDatum_AnnotationType type;
  // const string label_type = FLAGS_label_type;
  // const string label_map_file = FLAGS_label_map_file;
  // const bool check_label = FLAGS_check_label;
  // std::map<std::string, int> name_to_label;

  int min_dim = std::max<int>(0, FLAGS_min_dim);
  int max_dim = std::max<int>(0, FLAGS_max_dim);
  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);
  const string encode_type = FLAGS_encode_type;

  std::string train_image_dir = "/home/tumh/WIDER_train/images/";
  std::string train_txt = "/home/tumh/wider_face_split/wider_face_train_bbx_gt.txt";

  std::string db_out = "/home/tumh/wider_lmdb";

  type = AnnotatedDatum_AnnotationType_BBOX;

  AnnotatedDatum anno_datum;
  Datum* datum = anno_datum.mutable_datum();

  std::ifstream infile(train_txt.c_str());
  std::string fname, num_str;
  int count = 0;

  // Create new DB
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(db_out, db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  while(std::getline(infile, fname) && std::getline(infile, num_str)) {
    fname  =  train_image_dir + fname;
    LOG(INFO) << fname << " " << num_str;
    int num = atoi(num_str.c_str());
    std::string bbox;
    std::vector<std::string> bbox_list;
    for (int i = 0; i < num; i++) {
      std::getline(infile, bbox);
      bbox_list.push_back(bbox);
    }
    int img_height, img_width;
    GetImageSize(fname, &img_height, &img_width);
    AnnotatedDatum anno_datum;
    anno_datum.set_type(AnnotatedDatum_AnnotationType_BBOX);

    bool status = ReadImageToDatum(fname, -1, resize_height, resize_width,
                                   min_dim, max_dim, is_color, encode_type,
                                   anno_datum.mutable_datum());
    if (!status) {
      continue;
    }

    bool has_box = ReadWiderFaceBBox(fname, bbox_list, img_height, img_width, &anno_datum);

    if (!has_box) {
      continue;
    }

    // sequential
    string key_str = caffe::format_int(count, 8) + "_" + fname;

    // Put in db
    string out;
    CHECK(anno_datum.SerializeToString(&out));
    txn->Put(key_str, out);

    if (++count % 1000 == 0) {
      // Commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(INFO) << "Processed " << count << " files.";
    }
  }

  // write the last batch
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }

#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}

