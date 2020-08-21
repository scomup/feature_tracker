#ifndef FEATURE_TRACKER_H
#define FEATURE_TRACKER_H


#include <ros/ros.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "sensor_msgs/Image.h"
#include "superpoint_frontend.h"


namespace feature_tracker {

class FeatureTracker
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    FeatureTracker(const Eigen::Matrix3f &Rvc,
                   const Eigen::Matrix3f &K,
                   const Eigen::Vector4f &rect,
                   const int scale,
                   SuperpointFrontend *kp_frontend);
    void track(const cv::Mat& img);
    void show(const cv::Mat &img) const;

  private:
    void precompute();
    std::vector<Eigen::Matrix<float, 1, 2>, Eigen::aligned_allocator<Eigen::Matrix<float, 1, 2>>> dataGradient(const cv::Mat &data) const;
    cv::Mat shiftFrame(const cv::Mat frame, const int pixels, const int direction) const;

    float residuals(const cv::Mat &data1, const cv::Mat &data2, Eigen::VectorXf &res);
    cv::Mat getLivData(Eigen::Matrix3f& H) const;
    Eigen::Matrix4f exp(const Eigen::Vector3f& x) const;
    Eigen::Matrix<float, 3, 4> M1_;
    Eigen::Matrix<float, 4, 3> M2_;
    Eigen::Matrix3f Hroi_;
    Eigen::Matrix3f getH() const;


    std::vector<Eigen::Matrix4f,Eigen::aligned_allocator<Eigen::Matrix4f> > A_;
    std::vector<Eigen::Matrix<float, 2, 3>,Eigen::aligned_allocator<Eigen::Matrix<float, 2, 3>> > JwJg_;
    const Eigen::Vector4f rect_;
    const float scale_;
    SuperpointFrontend* kp_frontend_;
    Eigen::Matrix4f T0_;
    cv::Mat ref_data_;
    cv::Mat liv_data_;
};

}

#endif