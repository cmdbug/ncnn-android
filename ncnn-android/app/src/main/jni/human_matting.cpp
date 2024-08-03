#include "human_matting.h"
#include "data_struct.h"
#include <map>

namespace humanmatting {

    HumanMatting::HumanMatting() {

    }

    HumanMatting::~HumanMatting() {
        net.clear();
    }

    int HumanMatting::load(ncnn::Option option, const char *modeltype) {
        int r = load(nullptr, option, modeltype);
        return r;
    }

    int HumanMatting::load(AAssetManager *mgr, ncnn::Option option, const char *modeltype) {
        net.opt = option;

        const std::map<std::string, int> _target_sizes = {
                {"humanmatting-mbv2", 512},
        };

        const std::map<std::string, std::vector<float>> _mean_vals = {
                {"humanmatting-mbv2", {127.5f, 127.5f, 127.5f}},
        };

        const std::map<std::string, std::vector<float>> _norm_vals = {
                {"humanmatting-mbv2", {0.0078431f, 0.0078431f, 0.0078431f}},
        };

        target_size = _target_sizes.at(modeltype);
        mean_vals[0] = _mean_vals.at(modeltype)[0];
        mean_vals[1] = _mean_vals.at(modeltype)[1];
        mean_vals[2] = _mean_vals.at(modeltype)[2];
        norm_vals[0] = _norm_vals.at(modeltype)[0];
        norm_vals[1] = _norm_vals.at(modeltype)[1];
        norm_vals[2] = _norm_vals.at(modeltype)[2];

        char parampath[256];
        char modelpath[256];
        sprintf(parampath, "%s.param", modeltype);
        sprintf(modelpath, "%s.bin", modeltype);

        int pr, mr;
        if (mgr != nullptr) {
            pr = net.load_param(mgr, parampath);
            mr = net.load_model(mgr, modelpath);
        } else {
            pr = net.load_param(parampath);
            mr = net.load_model(modelpath);
        }
        if (pr != 0 || mr != 0) {
            TLOGE("load mode failed: %s, param: %d model: %d", modeltype, pr, mr);
        }
        return (pr == 0) && (mr == 0);
    }

    int HumanMatting::detect(const cv::Mat &rgb, cv::Mat &matting) {
        int width = rgb.cols;
        int height = rgb.rows;

        // pad to multiple of 32
        int w = width;
        int h = height;
        float scale = 1.f;
        if (w > h) {
            scale = (float) target_size / w;
            w = target_size;
            h = h * scale;
        } else {
            scale = (float) target_size / h;
            h = target_size;
            w = w * scale;
        }

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB2BGR, width, height, w, h);

        // pad to target_size rectangle
        int wpad = (w + 31) / 32 * 32 - w;
        int hpad = (h + 31) / 32 * 32 - h;
        ncnn::Mat in_pad;
        ncnn::copy_make_border(in,
                               in_pad,
                               hpad / 2,
                               hpad - hpad / 2,
                               wpad / 2,
                               wpad - wpad / 2,
                               ncnn::BORDER_CONSTANT,
                               0.f);

        in_pad.substract_mean_normalize(mean_vals, norm_vals);

        ncnn::Mat out;
        ncnn::Extractor ex = net.create_extractor();
        ex.input("input", in_pad);
        ex.extract("output", out);
        ncnn::Mat alpha;
        ncnn::resize_bilinear(out, alpha, width, height);
        matting = cv::Mat::zeros(cv::Size(width, height), CV_8UC3);

        const int bg_color[3] = {120, 255, 155};
        auto *alpha_data = (float *) alpha.data;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                float alpha_ = alpha_data[i * width + j];
                matting.at<cv::Vec3b>(i, j)[0] = rgb.at<cv::Vec3b>(i, j)[0] * alpha_ + (1 - alpha_) * bg_color[0];
                matting.at<cv::Vec3b>(i, j)[1] = rgb.at<cv::Vec3b>(i, j)[1] * alpha_ + (1 - alpha_) * bg_color[1];
                matting.at<cv::Vec3b>(i, j)[2] = rgb.at<cv::Vec3b>(i, j)[2] * alpha_ + (1 - alpha_) * bg_color[2];
            }
        }
        return 0;
    }

    int HumanMatting::draw(cv::Mat &rgb, cv::Mat &matting) {
        rgb = matting;
        return 0;
    }

}
