cmake_minimum_required(VERSION 3.10)

project(YOLOTensorRT)
set(CMAKE_CXX_STANDARD 17)

find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${TENSORRT_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)

set(SOURCES yolo_v5.cpp)

add_executable(yolo-tensorrt ${SOURCES})

target_link_libraries(yolo-tensorrt
    ${OpenCV_LIBS}
    nvinfer
    nvinfer_plugin
    cudart
    pthread
    dl)
