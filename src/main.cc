#include <ros/ros.h>
#include "feature_tracker_ros.h"


int main(int argc, char **argv)
{
    
    ros::init(argc, argv, "feature_tracker");
    feature_tracker::FeatureTrackerROS feature_tracker;
    ros::spin();
}