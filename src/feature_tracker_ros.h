#ifndef FEATURE_TRACKER_ROS_H
#define FEATURE_TRACKER_ROS_H


#include <ros/ros.h>
#include "sensor_msgs/Image.h"
#include "feature_tracker.h"


namespace feature_tracker {

class FeatureTrackerROS
{
  public:
    FeatureTrackerROS();

  private:
    void imageCallback(const sensor_msgs::ImageConstPtr& msg);
    FeatureTracker*  tracker_;
    ros::Subscriber img_sub_;
};

}

#endif