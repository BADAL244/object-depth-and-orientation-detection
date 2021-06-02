#ifndef IMAGEDETECT_H
#define IMAGEDETECT_H

#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include <iostream>
#include "sensor_msgs/image_encoding.h"


using namespace std;


class Controller{
    public:
        explicit Controller(ros::NodeHandle nh);



    private:
        ros::NodeHandle m_nh;
        

};

#endif