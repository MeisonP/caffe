#include "caffe/FRCNN/util/timer.hpp"

namespace API {
  Timer::Timer() {
    // LOG(INFO) << " api timer constructor";
    // caffe::Timer::Init();
    init();
  }

  void Timer::init() {
    this->total_time = 0;
    this->calls = 0;
    this->average_time = 0;
  }
    


  void Timer::tic() {
    caffe::Timer::Start();
    // caffe::CPUTimer::Start();
  }

  void Timer::toc() {
    this->total_time += caffe::Timer::MilliSeconds();
    // this->total_time += caffe::CPUTimer::MilliSeconds();
    this->calls += 1;
    this->average_time = this->total_time / this->calls;
  }

  float Timer::average() {
    return this->average_time;
  }

  void Timer::reset() {
    init();
  }

  const std::string Benchmark::IM_PREPROC="im_preproc" ;
  const std::string Benchmark::IM_NET="im_net" ;
  const std::string Benchmark::IM_POSTPROC="im_postproc" ;
}
