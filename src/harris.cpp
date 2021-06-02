#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/features2d.hpp>
#include "dataStructures.h"
#include "matching2D.hpp"
#include "camFusion.hpp"
#include "objectDetection2D.hpp"


static const std::string yoloBasePath = "../dat/yolov3-coco /";
static const std::string yoloClassesFile = "/home/agx1/catkin_ws/src/imagerecognition/dat/yolov3-coco /coco.names";
static const std::string modelConfiguration = "/home/agx1/catkin_ws/src/imagerecognition/dat/yolov3-coco /yolov3.cfg";
static const std::string modelWeights = "/home/agx1/catkin_ws/src/imagerecognition/dat/yolov3-coco /yolov3.weights";
size_t imgIndex = 0;
using namespace std;
double ttcCamera ;
cv::Mat output , imgGray;
int dataBufferSize = 2;
vector<cv::Mat> buffer_to_display;
vector<pair<int, double>> ttc_results_camera;
vector<DataFrame> dataBuffer;
DataFrame frame;
double sensorFrameRate = 10.0 ;
const bool bVis = true;
class ImageConverter
{
  public:
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  void imageCb(const sensor_msgs::ImageConstPtr& msg);

  public:
  ImageConverter()
    : it_(nh_)
  {
    // Subscrive to input video feed and publish output video feed
    
    image_sub_ = it_.subscribe("/usb_cam/image_raw", 1, &ImageConverter::imageCb, this);
   

    //cv::namedWindow(OPENCV_WINDOW);
  }

  ~ImageConverter()
  {
    //cv::destroyWindow(OPENCV_WINDOW);
  }


};
void ImageConverter::imageCb(const sensor_msgs::ImageConstPtr& msg){
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    float confThreshold = 0.2;
    float nmsThreshold = 0.4;
    
    output = cv_ptr->image;


    string descriptorType = "BRISK";
    string descriptorContentType = "DES_BINARY";
    string matcherType = "MAT_BF";
    string selectorType = "SEL_NN";

    frame.cameraImg = output;
    if(dataBuffer.size() == dataBufferSize){
      cout << "sach me " << endl;
      dataBuffer.erase(dataBuffer.begin());
      buffer_to_display.erase(buffer_to_display.begin());
    }
    dataBuffer.push_back(frame);
    buffer_to_display.push_back(output);

    detectObjects((dataBuffer.end() - 1)->cameraImg,
                  (dataBuffer.end() - 1)->boundingBoxes, confThreshold,
                  nmsThreshold, yoloBasePath, yoloClassesFile,
                  modelConfiguration, modelWeights, true);

    cv::cvtColor((dataBuffer.end()-1)->cameraImg , imgGray , cv::COLOR_BGR2GRAY);
    vector<cv::KeyPoint> keypoints;
    detKeypointsHarris(keypoints , imgGray , false);
    int maxKeypoints = 50;
    cv::KeyPointsFilter::retainBest(keypoints , maxKeypoints);
    (dataBuffer.end() -1)->keypoints = keypoints;
    cout << "#5 : DETECT KEYPOINTS done" << endl;
    cv::Mat descriptors;
    descKeypoints((dataBuffer.end() - 1)->keypoints,
                  (dataBuffer.end() - 1)->cameraImg, descriptors,
                  descriptorType);

    // push descriptors for current frame to end of data buffer
    (dataBuffer.end() - 1)->descriptors = descriptors;

    cout << "#6 : EXTRACT DESCRIPTORS done" << endl;

    if (dataBuffer.size() > 1) {

      /* MATCH KEYPOINT DESCRIPTORS */

    vector<cv::DMatch> matches;  // MAT_BF, MAT_FLANN
    matchDescriptors((dataBuffer.end() - 2)->keypoints,
                      (dataBuffer.end() - 1)->keypoints,
                      (dataBuffer.end() - 2)->descriptors,
                      (dataBuffer.end() - 1)->descriptors, matches,
                      descriptorContentType, matcherType, selectorType);

      // store matches in current data frame
    (dataBuffer.end() - 1)->kptMatches = matches;
    //cv::Mat matchImg = output.clone();
    //cv::drawMatches((dataBuffer.end()-2)->cameraImg, (dataBuffer.end()-2)->keypoints, (dataBuffer.end()-1)->cameraImg, (dataBuffer.end()-1)->keypoints,(dataBuffer.end()-1)->kptMatches,
    //                matchImg, cv::Scalar::all(-1), cv::Scalar::all(-1), vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    //string windowName = "Matching keypoints between two camera images (best 50)";
    //cv::namedWindow(windowName, 1);
    //cv::imshow(windowName, matchImg);
    //cv::waitKey(3);
    cout << "#7 : MATCH KEYPOINT DESCRIPTORS done" << endl;
    map<int, int> bbBestMatches;
    matchBoundingBoxes(matches, bbBestMatches, *(dataBuffer.end() - 2),
                         *(dataBuffer.end() - 1));
      //// EOF STUDENT ASSIGNMENT

      // store matches in current data frame
    (dataBuffer.end() - 1)->bbMatches = bbBestMatches;

    cout << "#8 : TRACK 3D OBJECT BOUNDING BOXES done" << endl;

    for (auto it1 = (dataBuffer.end() - 1)->bbMatches.begin();
           it1 != (dataBuffer.end() - 1)->bbMatches.end(); ++it1) {
        // find bounding boxes associates with current match
      BoundingBox *prevBB, *currBB;
      for (auto it2 = (dataBuffer.end() - 1)->boundingBoxes.begin();
             it2 != (dataBuffer.end() - 1)->boundingBoxes.end(); ++it2) {
          // check whether current match partner corresponds to this BB
        if (it1->second == it2->boxID) { currBB = &(*it2); }
        }

      for (auto it2 = (dataBuffer.end() - 2)->boundingBoxes.begin();
             it2 != (dataBuffer.end() - 2)->boundingBoxes.end(); ++it2) {
          // check whether current match partner corresponds to this BB
          if (it1->first == it2->boxID) { prevBB = &(*it2); }
      }
      double ttcCamera;
      clusterKptMatchesWithROI(*currBB, (dataBuffer.end() - 2)->keypoints,
                                   (dataBuffer.end() - 1)->keypoints,
                                   (dataBuffer.end() - 1)->kptMatches);
      
          if (bVis) {
            computeTTCCamera((dataBuffer.end() - 2)->keypoints,
                             (dataBuffer.end() - 1)->keypoints,
                             currBB->kptMatches, sensorFrameRate, ttcCamera,
                             *(buffer_to_display.end() - 2),
                             *(buffer_to_display.end() - 1));
          } else {
            computeTTCCamera((dataBuffer.end() - 2)->keypoints,
                             (dataBuffer.end() - 1)->keypoints,
                             currBB->kptMatches, sensorFrameRate, ttcCamera);
          }
          ttc_results_camera.push_back(make_pair(imgIndex, ttcCamera));
          //// EOF STUDENT ASSIGNMENT

          if (bVis) {
            cv::Mat visImg = (dataBuffer.end() - 1)->cameraImg.clone();
  


            char str[200];
            sprintf(str, "TTC Lidar : %f s",ttcCamera);
                    
            putText(visImg, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2,
                    cv::Scalar(0, 0, 255));

            string windowName =
                "Final Results : TTC - Frame #" + to_string(imgIndex + 1);
            cv::namedWindow(windowName, 4);
            cv::imshow(windowName, visImg);
        
            cv::waitKey(3);
          }

    }                        

    

    // cv::Mat visImage = (dataBuffer.end() -1)->cameraImg.clone();
    // cv::drawKeypoints(imgGray, keypoints, visImage, cv::Scalar::all(-1),
    //                   cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // string windowName = "Object classification";
    // cv::namedWindow(windowName, 1);
    // cv::imshow(windowName, visImage);

  }
  cout << "Showing results - TTC Camera\n" << descriptorType << "_ttc_times = [";
  for (size_t i = 0; i < ttc_results_camera.size(); i++) {
    cout << ttc_results_camera[i].second;
    if (i != ttc_results_camera.size() - 1) {
      cout << ", ";
    } else {
      cout << "]\n";
    }
  }
        
  

}

int main(int argc, char** argv){
  ros::init(argc, argv, "image_converter");
  ImageConverter ic;
  ros::spin();
  return 0;
}