
#include <chrono>
#include <thread>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <Eigen/Dense>

#include "feature_tracker_ros.h"
#include <nav_msgs/Odometry.h>
#include <tf/transform_broadcaster.h>
#include "common/math.h"

namespace feature_tracker
{

    FeatureTrackerROS::FeatureTrackerROS()
    {
        ros::NodeHandle private_nh("~");
        ros::NodeHandle nh;

        float pitch, roll, yaw;
        //./visual_odometry -b 200 -s 1 -f /home/liu/bag/vodom/angle30.bag  -l calib.txt -n 100 -m 0
        private_nh.param<float>("pitch", pitch, 0.672059);
        private_nh.param<float>("roll", roll, -0.0156828);
        private_nh.param<float>("yaw", yaw, -0.00547772);


        //./visual_odometry -b 200 -s 1 -f /home/liu/bag/kadoma/wloloop30.bag  -l odom.txt -n 100 -m 0
        //private_nh.param<float>("pitch", pitch, 0.485258);
        //private_nh.param<float>("roll", roll, 0.0093976);
        //private_nh.param<float>("yaw", yaw, 0.00232597);

        float fx, fy, cx, cy;
        private_nh.param<float>("fx", fx, 329.749817);
        private_nh.param<float>("fy", fy, 401.966949);
        private_nh.param<float>("cx", cx, 341.199105);
        private_nh.param<float>("cy", cy, 178.383924);

        std::string module_path;
        int width, height;
        private_nh.param<std::string>("module_path", module_path, "/home/liu/catkin_ws/src/feature_tracker/weight/feature_net.engine");
        private_nh.param<int>("width", width, 640);
        private_nh.param<int>("height", height, 360);
        odom_pub_ = nh.advertise<nav_msgs::Odometry>("visual_odom", 50);

        Eigen::Matrix3f K = Eigen::Matrix3f::Identity();

        float scale_inv = 8;
        K(0, 0) = fx;
        K(1, 1) = fy;
        K(0, 2) = cx;
        K(1, 2) = cy;
        Twvr_ = Eigen::Matrix4f::Identity();
        Eigen::Matrix3f Rvc = common::EulertoMatrix3d<float>(Eigen::Vector3f(pitch, roll, yaw));
        FeatureExtraction* net = new FeatureExtraction(module_path, height, width);
        int win_size = 200;
        //tracker_ = new FeatureTracker(Rvc, K, Eigen::Vector4f((width - win_size)/2, (height - win_size)/2, win_size, win_size), scale_inv, net);
        tracker_ = new FeatureTracker(Rvc, K, Eigen::Vector4f(220, 100, win_size, win_size), scale_inv, net);
        //debug
#if 1
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
        for (int i = 200; i < 300; i++)
        {
            char fn[200];
            sprintf(fn,"/home/liu/bag/vodom/30/frame%04d.png",i);
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

        bool state = tracker_->track(img);

        if (state)
        {
            tracker_->show(img);
            odomPublish(msg->header.stamp);
        }
        else
        {
            last_time_ = msg->header.stamp;
        }
    }
    void FeatureTrackerROS::odomPublish(const ros::Time& t)
    {
        auto Twvl = tracker_->getTwvl();
        //auto Tvlvr = Twvl.inverse() * Twvr_;
        //auto Tvrvl = Twvr_.inverse() * Twvl;
        //double dt = (t - last_time_).toSec();
        //double dth = std::atan2(Tvrvl(1, 0), Tvrvl(0, 0));
        //double dx = Tvrvl(0, 3);
        //double dy = Tvrvl(1, 3);


        //double linearx_velocity = dx/dt;
        //double lineary_velocity = dy/dt;
        //double angular_velocity = dth/dt;

        Twvr_ = Twvl;

        double th = std::atan2(Twvl(1, 0), Twvl(0, 0));
        geometry_msgs::Quaternion odom_quat = tf::createQuaternionMsgFromYaw(-th);

        nav_msgs::Odometry odom;
        odom.header.stamp = t;
        odom.header.frame_id = "odom";

        //set the position
        odom.pose.pose.position.x = Twvl(1,3);
        odom.pose.pose.position.y = Twvl(0,3);
        odom.pose.pose.position.z = 0.0;
        odom.pose.pose.orientation = odom_quat;

        ////set the velocity
        //odom.child_frame_id = "base_link";
        //odom.twist.twist.linear.x = linearx_velocity;
        //odom.twist.twist.linear.y = lineary_velocity;
        //odom.twist.twist.angular.z = angular_velocity;

        //publish the message
        odom_pub_.publish(odom);

        
    }

} // namespace feature_tracker
