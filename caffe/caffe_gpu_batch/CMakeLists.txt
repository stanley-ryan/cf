cmake_minimum_required(VERSION 3.5)
project(caffe_gpu)
find_package(CUDA REQUIRED)

add_subdirectory(layers)
include_directories(/usr/local/cuda/include/)
include_directories(${PROJECT_SOURCE_DIR}/layers)
file(GLOB caffe_src "*.cpp")
file(GLOB caffe_head "*.hpp")
SET(CMAKE_CXX_FLAGS "-std=c++11 -O3")
IF(NOT EXISTS libcaffe_gpu_lib.a)
add_library(caffe_gpu_lib  ${caffe_src} ${caffe_head})
ENDIF(NOT EXISTS libcaffe_gpu_lib.a)

target_link_libraries( caffe_gpu_lib caffe_layers_lib)

