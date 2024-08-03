#ifndef NCNN_ANDROID_HUMAN_MATTING_H
#define NCNN_ANDROID_HUMAN_MATTING_H

#include <opencv2/core/core.hpp>

#include <net.h>

namespace humanmatting {
    class HumanMatting {
    public:
        HumanMatting();

        ~HumanMatting();

        int load(ncnn::Option option, const char *modeltype);

        int load(AAssetManager *mgr, ncnn::Option option, const char *modeltype);

        int detect(const cv::Mat &rgb, cv::Mat &matting);

        int draw(cv::Mat &rgb, cv::Mat &matting);

    private:
        ncnn::Net net;
        int target_size;
        float mean_vals[3];
        float norm_vals[3];
    };

}

#endif //NCNN_ANDROID_HUMAN_MATTING_H
