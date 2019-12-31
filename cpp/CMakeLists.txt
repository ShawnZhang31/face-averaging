cmake_minimum_required(VERSION 2.8.12)

PROJECT(faceAveraging)



#############  基本库配置  ############
find_package(OpenCV PATHS /Users/zhangxiaomin/Works/libs/opencv-release/cv-412/install/lib/cmake/opencv4/)
# 如果配置了OpenCV的环境变量可以使用下面的语句代替上面的语句
# find_package(OpenCV REQUIRED)

add_definitions(-std=c++11)

include_directories( ${OpenCV_INCLUDE_DIRS})
include(../common/dlib/dlib/cmake)

MACRO(add_example name)
  ADD_EXECUTABLE(${name} ${name}.cpp)
  TARGET_LINK_LIBRARIES(${name} ${OpenCV_LIBS} dlib::dlib)
ENDMACRO()

add_example(faceAveraging)