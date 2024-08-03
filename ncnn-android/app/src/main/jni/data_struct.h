#ifndef NCNN_ANDROID_DATA_STRUCT_H
#define NCNN_ANDROID_DATA_STRUCT_H

#include <opencv2/core/core.hpp>

#ifndef LOG_TAG
#define LOG_TAG "teng_ncnn"

#include <android/log.h>

#define TLOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define TLOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define TLOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define TLOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define TLOGF(...) __android_log_print(ANDROID_LOG_FATAL, LOG_TAG, __VA_ARGS__)

#endif

struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
};

#endif //NCNN_ANDROID_DATA_STRUCT_H
