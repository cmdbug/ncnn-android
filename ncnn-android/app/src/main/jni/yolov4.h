#ifndef YOLOV4_H
#define YOLOV4_H

#include <opencv2/core/mat.hpp>
#include "net.h"

typedef struct BoxInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

class YoloV4 {
public:
    YoloV4();

    ~YoloV4();

    int load(ncnn::Option option, const char *modeltype);

    int load(AAssetManager *mgr, ncnn::Option option, const char *modeltype);

    int detect(const cv::Mat &rgb, std::vector<BoxInfo> &objects, float prob_threshold=0.45f, float nms_threshold=0.65f);

    int draw(cv::Mat &rgb, const std::vector<BoxInfo> &objects);

private:
    std::vector<BoxInfo> decode_infer(ncnn::Mat &data, int img_w, int img_h);

    ncnn::Net net;
    int target_size;
    float mean_vals[3];
    float norm_vals[3];

};

#endif //YOLOV4_H
