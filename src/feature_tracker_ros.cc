
#include <chrono>
#include <thread>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <Eigen/Dense>

#include "feature_tracker_ros.h"
#include "common/math.h"

namespace feature_tracker
{

    FeatureTrackerROS::FeatureTrackerROS()
    {
        ros::NodeHandle private_nh("~");
        ros::NodeHandle nh;

        float pitch, roll, yaw;
        private_nh.param<float>("pitch", pitch, 0.8674);
        private_nh.param<float>("roll", roll, -0.0275);
        private_nh.param<float>("yaw", yaw, -0.0127);

        float fx, fy, cx, cy;
        private_nh.param<float>("fx", fx, 329.749817);
        private_nh.param<float>("fy", fy, 401.966949);
        private_nh.param<float>("cx", cx, 341.199105);
        private_nh.param<float>("cy", cy, 178.383924);

        std::string module_path;
        int width, height;
        private_nh.param<std::string>("module_path", module_path, "/home/liu/catkin_ws/src/feature_tracker/weight/model_trt_fp16.engine");
        private_nh.param<int>("width", width, 640);
        private_nh.param<int>("height", height, 360);

        Eigen::Matrix3f K = Eigen::Matrix3f::Identity();

        float scale_inv = 1;
        K(0, 0) = fx;
        K(1, 1) = fy;
        K(0, 2) = cx;
        K(1, 2) = cy;

        Eigen::Matrix3f Rvc = common::EulertoMatrix3d<float>(Eigen::Vector3f(pitch, roll, yaw));

        SuperpointFrontend* superpoint = new SuperpointFrontend(module_path, height, width);
        int win_size = 160;
        tracker_ = new FeatureTracker(Rvc, K, Eigen::Vector4f((width - win_size)/2, (height - win_size)/2, win_size, win_size), scale_inv, superpoint);
        //debug
#if 0
/*
        {
            cv::Mat img_gray0;
            cv::Mat img0 = cv::imread("/home/liu/bag/wlo60/frame1487.png");
            cv::cvtColor(img0, img_gray0, cv::COLOR_BGR2GRAY);

            cv::Mat img_gray1;
            cv::Mat img1 = cv::imread("/home/liu/bag/wlo60/frame1488.png");
            cv::cvtColor(img1, img_gray1, cv::COLOR_BGR2GRAY);

            //imshow("w0",img_gray0);
            //imshow("w1",img_gray1);
            //cv::waitKey();
            tracker_->track(img_gray0);
            tracker_->track(img_gray1);
            tracker_->show(img_gray1);
            //cv::waitKey();

        }
    */
        for (int i = 0; i < 1487; i++)
        {
            char fn[200];
            sprintf(fn,"/home/liu/bag/wlo60/frame%04d.png",i);
            //printf(fn);
            cv::Mat img = cv::imread(fn);
            cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

            tracker_->track(img);
            tracker_->show(img);
        }
#endif

        img_sub_ = nh.subscribe("web_camera/image_rect", 1000, &FeatureTrackerROS::imageCallback, this);
        std::cout<<"Waiting image."<<std::endl;


    }
    void FeatureTrackerROS::imageCallback(const sensor_msgs::ImageConstPtr &msg)
    {
        //double time = msg->header.stamp.toSec();

        cv_bridge::CvImageConstPtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
        }
        catch (cv_bridge::Exception &e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
        cv::Mat img;
        cv_ptr->image.copyTo(img);

        tracker_->track(img);
        tracker_->show(img);

    }
} // namespace feature_tracker
