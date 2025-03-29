cmake_minimum_required(VERSION 3.1.3)
project(youbot_arm_moveit_config)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  moveit_core
  moveit_ros_planning
  moveit_ros_planning_interface
  moveit_visual_tools
  eigen_conversions
  tf2_eigen
  geometry_msgs
  message_generation
)

add_message_files(
  FILES
  TargetArm.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
)

catkin_package(
  INCLUDE_DIRS include
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)



add_executable(arm_utils_node src/ArmUtils.cpp)
target_link_libraries(arm_utils_node ${catkin_LIBRARIES})

add_executable(arm_utils_node_v2 src/arm_utils_v2.cpp)
target_link_libraries(arm_utils_node_v2 ${catkin_LIBRARIES})

add_executable(arm_utils_node_v3 src/arm_utils_v3.cpp)
target_link_libraries(arm_utils_node_v3 ${catkin_LIBRARIES})

add_executable(arm_utils_node_v4 src/ArmUtilsLight.cpp)
target_link_libraries(arm_utils_node_v4 ${catkin_LIBRARIES})

install(TARGETS arm_utils_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS arm_utils_node_v2
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS arm_utils_node_v3
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  PATTERN "setup_assistant.launch" EXCLUDE)
install(DIRECTORY config DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
