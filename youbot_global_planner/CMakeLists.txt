cmake_minimum_required(VERSION 3.0.2)
project(youbot_global_planner)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  tf2_msgs
  sensor_msgs
  nav_msgs
  grid_map_ros
  grid_map_msgs
  grid_map_core
  grid_map_filters
  filters
)

find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES youbot_global_planner
  CATKIN_DEPENDS roscpp std_msgs geometry_msgs tf2 tf2_ros tf2_geometry_msgs tf2_msgs sensor_msgs nav_msgs grid_map_ros grid_map_msgs grid_map_core grid_map_filters filters
  DEPENDS yaml-cpp
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${YAML_CPP_INCLUDE_DIR}
)

add_executable(Map_utils_node src/MapUtils.cpp)
target_link_libraries(Map_utils_node
  ${catkin_LIBRARIES}
  yaml-cpp
)