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
#include "dataStructures.h"

using namespace std;

static const std::string OPENCV_WINDOW = "Image window";
static const std::string yoloBasePath = "../dat/yolov3-coco /";
static const std::string yoloClassesFile = yoloBasePath + "coco.names";


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
    
    image_sub_ = it_.subscribe("/zed2/zed_node/rgb/image_rect_color", 1, &ImageConverter::imageCb, this);
   

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

    cv::Mat output;
    output = cv_ptr->image;
    string yoloClassesFile = "/home/badal/catkin_ws/src/imagerecognition/dat/yolov3-coco/coco.names";

    vector<string> classes;
    ifstream ifs(yoloClassesFile.c_str());
    string line;
    while (getline(ifs, line))
      classes.push_back(line);
   
   
    cv::Mat blob;


    
    double scalefactor = 1/255.0;
    cv::Size size = cv::Size(416, 416);
    cv::Scalar mean = cv::Scalar(0,0,0);
    bool swapRB = false;
    bool crop = false;
    cv::dnn::blobFromImage(output, blob, scalefactor, size, mean, swapRB, crop);
   
    cv::dnn::Net net =
      cv::dnn::readNetFromDarknet("/home/badal/catkin_ws/src/imagerecognition/dat/yolov3-coco/yolov3.cfg", "/home/badal/catkin_ws/src/imagerecognition/dat/yolov3-coco/yolov3.weights");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    vector<cv::String> names;
    vector<int> outLayers = net.getUnconnectedOutLayers();
    vector<cv::String> layersNames = net.getLayerNames();
    names.resize(outLayers.size());
  // Get the names of the output layers in names
  // YOLOv3 has three output layers to help facilitate better
  // multiscale detection - these are at output layers 82, 94 and 106
    for (size_t i = 0; i < outLayers.size(); ++i) {
      names[i] = layersNames[outLayers[i] - 1];
    }

  // invoke forward propagation through network
    vector<cv::Mat> netOutput;
    net.setInput(blob);
    net.forward(netOutput, names);
    float confThreshold = 0.20;
    vector<int> classIds;
    vector<float> confidences;
    vector<cv::Rect> boxes;
    for (size_t i = 0; i < netOutput.size(); ++i) {
      float* data = (float*)netOutput[i].data;
      for (int j = 0; j < netOutput[i].rows; ++j, data += netOutput[i].cols) {
        cv::Mat scores = netOutput[i].row(j).colRange(5, netOutput[i].cols);
        cv::Point classId;
        double confidence;

        // Get the value and location of the maximum score
        cv::minMaxLoc(scores, 0, &confidence, 0, &classId);
        if (confidence > confThreshold) {
          cv::Rect box;
          int cx, cy;
          cx = (int)(data[0] * output.cols);
          cy = (int)(data[1] * output.rows);
          box.width = (int)(data[2] * output.cols);
          box.height = (int)(data[3] * output.rows);
          box.x = cx - box.width / 2;   // left
          box.y = cy - box.height / 2;  // top

          boxes.push_back(box);
          classIds.push_back(classId.x);
          confidences.push_back((float)confidence);
        }
      }
    }
    float nmsThreshold = 0.4;  // Non-maximum suppression threshold
    vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    std::vector<BoundingBox> bBoxes;
    for (auto it = indices.begin(); it != indices.end(); ++it) {
      BoundingBox bBox;
      bBox.roi = boxes[*it];
      bBox.classID = classIds[*it];
      bBox.confidence = confidences[*it];
    // zero-based unique identifier for this bounding box
      bBox.boxID = (int)bBoxes.size();

      bBoxes.push_back(bBox);
    }
    cv::Mat visImg = output.clone();
  for (auto it = bBoxes.begin(); it != bBoxes.end(); ++it) {
    // Draw rectangle displaying the bounding box
    int top, left, width, height;
    top = (*it).roi.y;
    left = (*it).roi.x;
    width = (*it).roi.width;
    height = (*it).roi.height;
    float distance = (2 * 3.14 * 180) / (width + height * 360) * 1000 + 3;
    distance = distance * 2.54;

    cv::rectangle(visImg, cv::Point(left, top),
                  cv::Point(left + width, top + height), cv::Scalar(0, 255, 0),
                  2);

    string label = cv::format("%.2f", (*it).confidence);
    label = classes[((*it).classID)] + ":" + label;

    // Display label at the top of the bounding box
    int baseLine;
    cv::Size labelSize = getTextSize(label, cv::FONT_ITALIC, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(visImg, cv::Point(left, top - round(1.5 * labelSize.height)),
              cv::Point(left + round(1.5 * labelSize.width), top + baseLine),
              cv::Scalar(255, 255, 255), cv::FILLED);
    cv::putText(visImg, label, cv::Point(left, top), cv::FONT_ITALIC, 0.75,
                cv::Scalar(0, 0, 0), 1);
    char str[200];
    sprintf(str,"%f ",distance);
    cv::putText(visImg, str, cv::Point2f(left +10,top+40), cv::FONT_HERSHEY_PLAIN, 2,  cv::Scalar(0,0,255,255));
  }

  string windowName = "Object classification";
  cv::namedWindow(windowName, 1);
  //cv::resize(visImg , visImg , cv::Size(940 ,540));
  cv::imshow(windowName, visImg);
        
  

    cv::waitKey(3);

  
}


        
int main(int argc, char** argv){
  ros::init(argc, argv, "image_converter");
  ImageConverter ic;
  ros::spin();
  return 0;
}