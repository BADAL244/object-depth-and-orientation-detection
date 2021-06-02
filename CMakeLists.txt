cmake_minimum_required(VERSION 3.0.2)
project(imagerecognition)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall" "-pedantic")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf
  tf2
  visualization_msgs
  image_transport
)
find_package(Boost REQUIRED COMPONENTS thread)
set(OpenCV_DIR "/usr/local/include/OpenCV4/")
find_package(OpenCV 4.1 REQUIRED)
find_package(yaml-cpp REQUIRED)



catkin_package(
  INCLUDE_DIRS include
#  LIBRARIES imagecontrol
 CATKIN_DEPENDS cv_bridge image_transport roscpp rospy sensor_msgs std_msgs tf tf2 message_runtime visualization_msgs
 
 DEPENDS Boost OpenCV
)


include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
include
${OpenCV_INCLUDE_DIRS}
${catkin_INCLUDE_DIRS}
${Boost_INCLUDE_DIRS}
${YAML_CPP_INCLUDE_DIR}
)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/imagerecognition.cpp
# )



## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(${PROJECT_NAME}_node src/new_data.cpp)
add_executable(harris src/harris.cpp  src/matching2D_Student.cpp src/camFusion_Student.cpp src/objectDetection2D.cpp)

add_executable(call_back src/msg_filter.cpp)

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} ) 
## Specify libraries to link a library or executable target against

target_link_libraries (${PROJECT_NAME}_node
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
  ${Boost_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
)
target_link_libraries (harris
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}

)
target_link_libraries (call_back
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
  ${Boost_LIBRARIES}
  ${YAML_CPP_LIBRARIES}

)
