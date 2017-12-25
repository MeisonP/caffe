#ifndef FRCNN_BENCHMARK_H_
#define FRCNN_BENCHMARK_H_

#include "caffe/util/benchmark.hpp"

namespace API {
  class Timer : public caffe::Timer {
    public:
     Timer();
     ~Timer() {}
     void init();
     void tic();
     void toc();
     float average();
     void reset();
    protected:
     float total_time = 0;
     int calls = 0;
     float average_time = 0;
  };

  class Benchmark {
    // benchmark
    public:
      static const std::string IM_NET;
      static const std::string IM_PREPROC;
      static const std::string IM_POSTPROC;
  };
}
#endif   // CAFFE_UTIL_BENCHMARK_H_
