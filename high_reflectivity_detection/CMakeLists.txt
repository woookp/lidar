cmake_minimum_required(VERSION 2.8.3)
project(high_reflectivity_detection)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  pcl_ros
  message_generation
)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  DetectionResult.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs sensor_msgs pcl_ros
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Install Python scripts
catkin_install_python(PROGRAMS
  scripts/high_reflectivity_detection_node.py
  scripts/multi_high_reflectivity_detection_node_t.py
  scripts/multi_high_reflectivity_detection_node.py
  scripts/re_high_reflectivity_detection_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
