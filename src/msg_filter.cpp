#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Float32MultiArray.h>
#include <vector>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>




using namespace std;


using namespace sensor_msgs;
using namespace message_filters;

void callback(const ImageConstPtr& image, const CameraInfoConstPtr& cam_info , const ImageConstPtr& new_image)
{

    cv_bridge::CvImagePtr cv_ptr;
    cv_bridge::CvImagePtr img_ptr_depth;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);
      img_ptr_depth = cv_bridge::toCvCopy(*new_image, sensor_msgs::image_encodings::TYPE_32FC1);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    cv::Mat output;
    output = cv_ptr->image;

    float* depths = (float*)(&new_image->data[0]);

    int u = new_image->width / 2;
    int v = new_image->height / 2;

        // Linear index of the center pixel
    int centerIdx = u + new_image->width * v;

    //cout << depths[centerIdx] <<  "-----------" <<  depths[100 ,100] << endl;
    auto newda = boost::get_c_array<double>(cam_info->K);

    cout << newda[0] << endl;
    
    string windowName = "Object classification";
    cv::namedWindow(windowName, 1);
  //cv::resize(visImg , visImg , cv::Size(940 ,540));
    cv::imshow(windowName, output);
    cv::waitKey(3);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "vision_node");

  ros::NodeHandle nh;

  message_filters::Subscriber<Image> image_sub(nh, "/zed2/zed_node/rgb/image_rect_color", 1);
  message_filters::Subscriber<CameraInfo> info_sub(nh, "/zed2/zed_node/depth/camera_info", 1);
  message_filters::Subscriber<Image> depth_sub(nh, "/zed2/zed_node/depth/depth_registered", 1);
  TimeSynchronizer<Image, CameraInfo , Image> sync(image_sub, info_sub, depth_sub,10);
  sync.registerCallback(boost::bind(&callback, _1, _2 , _3));

  ros::spin();

  return 0;
}