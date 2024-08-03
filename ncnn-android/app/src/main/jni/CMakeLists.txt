cmake_minimum_required(VERSION 3.10)
project(tncnn)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencvmobile/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnnvulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

aux_source_directory(. SRC_LIST)
add_library(
        tncnn
        SHARED
        ${SRC_LIST}
)

include_directories(
        ./
)

target_link_libraries(tncnn ncnn ${OpenCV_LIBS} camera2ndk mediandk)
