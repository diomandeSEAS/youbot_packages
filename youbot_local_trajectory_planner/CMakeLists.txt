cmake_minimum_required(VERSION 3.0.2)
project(youbot_local_trajectory_planner)

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
  actionlib_msgs
  genmsg
  message_generation
  cavity_detection_api
  cavity_detection_msgs
)

# Find third-party libraries
find_package(Eigen3 REQUIRED)
find_package(orocos_kdl REQUIRED)
find_package(drake REQUIRED)
find_package(fmt REQUIRED)


# Add action files
add_action_files(
  DIRECTORY action
  FILES BaseTrajectoryFollowing.action
)

add_message_files(
  FILES
  TargetBase.msg
)

# Generate messages
generate_messages(
  DEPENDENCIES
  actionlib_msgs
  geometry_msgs
  std_msgs
)

# Declare the catkin package
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES youbot_local_trajectory_planner
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
    actionlib_msgs
    genmsg
    message_generation
    cavity_detection_api
    cavity_detection_msgs
  DEPENDS 
    Eigen3 orocos_kdl drake fmt 
)

# Include directories
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  ${orocos_kdl_INCLUDE_DIRS}
  ${drake_INCLUDE_DIRS}
  
)

# Link directories (if required)
link_directories(
  ${drake_LIBRARY_DIRS}
)



# Optionally set capabilities (you can remove this if not needed)
SET(SUDO_COMMAND gksudo)
IF($ENV{USE_NORMAL_SUDO})
    SET(SUDO_COMMAND sudo)
ENDIF($ENV{USE_NORMAL_SUDO})

IF(USE_SETCAP)
add_custom_command(TARGET youbot_local_trajectory_planner POST_BUILD
    COMMAND ${SUDO_COMMAND} setcap cap_net_raw+ep $<TARGET_FILE:youbot_local_trajectory_planner>
)    
endif(USE_SETCAP)

# Install Python scripts
catkin_install_python(PROGRAMS 
  src/BaseTrajectoryFollowerHWServer.py
  src/BaseTrajectoryProcessingClient.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)