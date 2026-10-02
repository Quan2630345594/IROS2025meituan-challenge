cmake_minimum_required(VERSION 3.0.2)
project(goal_detection)
find_package(catkin REQUIRED)
catkin_package()
catkin_python_setup()
catkin_install_python(PROGRAMS scripts/bev_features_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY launch config DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
