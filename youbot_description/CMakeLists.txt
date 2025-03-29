cmake_minimum_required(VERSION 2.8.3)
project(youbot_description)

find_package(catkin REQUIRED COMPONENTS
  xacro
)

catkin_package( CATKIN_DEPENDS
  xacro
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(DIRECTORY controller/ DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/controller)
install(DIRECTORY controller/ DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/meshes)
install(DIRECTORY controller/ DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/robots)
install(DIRECTORY controller/ DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/urdf)