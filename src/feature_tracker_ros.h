#ifndef FEATURE_TRACKER_ROS_H
#define FEATURE_TRACKER_ROS_H


#include <ros/ros.h>
#include "sensor_msgs/Image.h"
#include "feature_tracker.h"


namespace feature_tracker {

class FeatureTrackerROS
{
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    FeatureTrackerROS();

  private:
    void imageCallback(const sensor_msgs::ImageConstPtr& msg);
    void odomPublish(const ros::Time& t);
    FeatureTracker*  tracker_;
    ros::Subscriber img_sub_;
    ros::Publisher odom_pub_;

    ros::Time last_time_;
    Eigen::Matrix4f Twvr_;

};

}

#endif