cmake_minimum_required(VERSION 3.0.2)
project(youbot_horizontal_cavity_detection)

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
  sensor_msgs
  pcl_ros
  pcl_conversions
  tf2_ros
  laser_geometry
  actionlib_msgs
  genmsg
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
  LIBRARIES youbot_horizontal_cavity_detection
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
    sensor_msgs
    pcl_ros
    pcl_conversions
    tf2_ros
    laser_geometry
    actionlib_msgs
    genmsg
    grid_map_ros
    grid_map_msgs
    grid_map_core
    grid_map_filters
    filters
    cavity_detection_api
    cavity_detection_msgs
  DEPENDS 
    Eigen3 
)

# Include directories
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
)


add_executable(HorizontalCavitiesFiltering 
  src/HorizontalCavityDetection.cpp
)

# Link libraries for `HorizontalCavitiesFiltering`
target_link_libraries(HorizontalCavitiesFiltering
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}  # Explicitly link PCL libraries
  ${OpenCV_LIBRARIES}  # Link OpenCV libraries
)


# Optionally set capabilities (you can remove this if not needed)
SET(SUDO_COMMAND gksudo)
IF($ENV{USE_NORMAL_SUDO})
    SET(SUDO_COMMAND sudo)
ENDIF($ENV{USE_NORMAL_SUDO})

IF(USE_SETCAP)
add_custom_command(TARGET youbot_horizontal_cavity_detection POST_BUILD
    COMMAND ${SUDO_COMMAND} setcap cap_net_raw+ep $<TARGET_FILE:youbot_horizontal_cavity_detection>
)    
endif(USE_SETCAP)
