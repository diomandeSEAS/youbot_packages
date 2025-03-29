cmake_minimum_required(VERSION 3.0.2)
project(youbot_vertical_cavity_detection)

# Find Catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  roslib
  nav_msgs
  tf2_msgs
  tf
  tf_conversions
  gazebo_msgs
  actionlib
  control_msgs
  sensor_msgs
  youbot_description
  tf2_ros
  laser_geometry
  actionlib_msgs
  genmsg
  cv_bridge
  grid_map_ros
  grid_map_msgs
  grid_map_core
  grid_map_filters
  filters
  cavity_detection_api
  cavity_detection_msgs
)

# Find third-party libraries
find_package(Eigen3 REQUIRED)




# Declare the catkin package
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES youbot_vertical_cavity_detection
  CATKIN_DEPENDS 
    roscpp
    rospy
    std_msgs
    geometry_msgs
    roslib
    nav_msgs
    tf2_msgs
    tf
    tf_conversions
    gazebo_msgs
    actionlib
    control_msgs
    sensor_msgs
    youbot_description
    tf2_ros
    laser_geometry
    actionlib_msgs
    genmsg
    cv_bridge
    grid_map_ros
    grid_map_msgs
    grid_map_core
    grid_map_filters
    filters
    cavity_detection_api
    cavity_detection_msgs
  DEPENDS 
    Eigen3 OpenCV 
)

# Include directories
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)


# add_executable(PlaneCalibration 
#   src/PlaneCalibration.cpp
# )

# Optionally set capabilities (you can remove this if not needed)
SET(SUDO_COMMAND gksudo)
IF($ENV{USE_NORMAL_SUDO})
    SET(SUDO_COMMAND sudo)
ENDIF($ENV{USE_NORMAL_SUDO})

IF(USE_SETCAP)
add_custom_command(TARGET youbot_vertical_cavity_detection POST_BUILD
    COMMAND ${SUDO_COMMAND} setcap cap_net_raw+ep $<TARGET_FILE:youbot_vertical_cavity_detection>
)    
endif(USE_SETCAP)

# Install Python scripts
catkin_install_python(PROGRAMS 
  src/BaseTrajectoryFollowerHWServer.py
  src/BaseTrajectoryProcessingClient.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)