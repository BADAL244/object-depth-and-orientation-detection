#include "../include/imagerecognition/imagedetect.h"

int main(int agrc , char** argv){
    ros::init(argc , argv , "imagedetect");
    ros::NodeHandle nh;
    Controller controller(nh);
    ros::spin();
    return 0;
}