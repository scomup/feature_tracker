#ifndef FEATURE_TRACKER_H
#define FEATURE_TRACKER_H


#include <ros/ros.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "sensor_msgs/Image.h"
#include "feature_extraction.h"


namespace feature_tracker {

class FeatureTracker
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    FeatureTracker(const Eigen::Matrix3f &Rvc,
                   const Eigen::Matrix3f &K,
                   const Eigen::Vector4f &rect,
                   const int scale,
                   FeatureExtraction *kp_frontend);
    bool track(const cv::Mat& img);
    void show(const cv::Mat &img) const;
    Eigen::Matrix4f getTwvl() const;

  private:
    void precompute();
    std::vector<Eigen::Matrix<float, 1, 2>, Eigen::aligned_allocator<Eigen::Matrix<float, 1, 2>>> dataGradient(const cv::Mat &data) const;

    float residuals(const cv::Mat &data1, const cv::Mat &data2, Eigen::VectorXf &res) const;
    cv::Mat getLivData(Eigen::Matrix3f& H) const;
    Eigen::Matrix4f exp(const Eigen::Vector3f& x) const;
    Eigen::Matrix<float, 3, 4> M1_;
    Eigen::Matrix<float, 4, 3> M2_;
    Eigen::Matrix3f getH() const;


    std::vector<Eigen::Matrix4f,Eigen::aligned_allocator<Eigen::Matrix4f> > A_;
    std::vector<Eigen::Matrix<float, 2, 3>,Eigen::aligned_allocator<Eigen::Matrix<float, 2, 3>> > JwJg_;
    const Eigen::Vector4f rect_;
    const float scale_;
    FeatureExtraction* kp_frontend_;
    Eigen::Matrix4f Tvlvr_;//Consider vl('live' vehicle) as origin. The coordinates of vr('ref' vehicle)
    Eigen::Matrix4f Twvl_;//The pose of vl('live' vehicle pose) in world coordinate.//what we want to know
    cv::Mat ref_data_;
    cv::Mat liv_data_;
};

}

#endif