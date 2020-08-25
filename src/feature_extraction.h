
#ifndef VISUAL_ODOMETRY_FEATURE_EXTRACTION
#define VISUAL_ODOMETRY_FEATURE_EXTRACTION

#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <NvInfer.h>
#include <NvCaffeParser.h>

namespace feature_tracker
{

class FeatureExtraction
{
    typedef std::vector<std::pair<int, int>> matching_info;

public:
    //EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  FeatureExtraction(std::string module_path,
                     float image_h_,
                     float image_w_);

  cv::Mat getDesc(const cv::Mat &img);
  cv::Mat getDesc1D(const cv::Mat &img);

private:
    nvinfer1::IExecutionContext* context_;
    nvinfer1::ICudaEngine *engine_;
    void* buffer_[2];
    
    int image_w_;
    int image_h_;


};


class Logger : public nvinfer1::ILogger {
 public:
  void log(nvinfer1::ILogger::Severity severity, const char* msg) override {
    switch (severity) {
      case Severity::kINFO:
        LOG(INFO) << msg;
        break;
      case Severity::kWARNING:
        LOG(WARNING) << msg;
        break;
      case Severity::kINTERNAL_ERROR:
      case Severity::kERROR:
        LOG(ERROR) << msg;
        break;
      default:
        break;
    }
  }
};


} // namespace visual_odometry

#endif