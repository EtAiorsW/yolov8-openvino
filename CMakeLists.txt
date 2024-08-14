cmake_minimum_required(VERSION 3.10.0)

project(YOLOV8)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Release")
set(Openvino_ROOT /home/gqwang/DL/openvino/l_openvino_toolkit_ubuntu22_2024.3.0.16041.1e3b88e4e3f_x86_64/runtime)
set(OpenVINO_DIR ${Openvino_ROOT}/cmake)

find_package(OpenVINO REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
    ${Openvino_ROOT}/include
    ${Openvino_ROOT}/include/openvino
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
    )

link_directories(
    ${Openvino_ROOT}/lib
    ${Openvino_ROOT}/lib/intel64
)


add_executable(YOLOV8 ov_yolov8.cpp)
target_link_libraries(YOLOV8 openvino ${OpenCV_LIBS})
