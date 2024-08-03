#include <map>
#include <opencv2/imgproc.hpp>
#include "yolov4.h"
#include "data_struct.h"


YoloV4::YoloV4() {

}

YoloV4::~YoloV4() {
    net.clear();
}

int YoloV4::load(ncnn::Option option, const char *modeltype) {
    int r = load(nullptr, option, modeltype);
    return r;
}

int YoloV4::load(AAssetManager *mgr, ncnn::Option option, const char *modeltype) {
    net.opt = option;

    const std::map<std::string, int> _target_sizes = {
            {"yolov4-tiny", 416},
    };

    const std::map<std::string, std::vector<float>> _mean_vals = {
            {"yolov4-tiny", {0.0f, 0.0f, 0.0f}},
    };

    const std::map<std::string, std::vector<float>> _norm_vals = {
            {"yolov4-tiny", {1 / 255.f, 1 / 255.f, 1 / 255.f}},
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
        TLOGE("load mode failed: %s", modeltype);
    }
    return (pr == 0) && (mr == 0);

}

int YoloV4::detect(const cv::Mat &rgb, std::vector<BoxInfo> &objects, float prob_threshold, float nms_threshold) {
    int img_w = rgb.cols;
    int img_h = rgb.rows;

    // letterbox pad to multiple of 32
    int w = img_w;
    int h = img_h;
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

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, img_w, img_h, w, h);

    // pad to target_size rectangle
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in,
                           in_pad,
                           hpad / 2,
                           hpad / 2,
                           wpad / 2,
                           wpad / 2,
                           ncnn::BORDER_CONSTANT,
                           0.f);

    // so for 0-255 input image, rgb_mean should multiply 255 and norm should div by std.
    in_pad.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Mat out;
    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", in_pad);
    ex.extract("output", out);
    auto boxes = decode_infer(out, w + wpad, h + hpad);

    int count = boxes.size();
    objects.resize(count);
    for (int i = 0; i < count; i++) {
        objects[i] = boxes[i];

        // adjust offset to original unpadded
        float x0 = (objects[i].x1 - (wpad / 2)) / scale;
        float y0 = (objects[i].y1 - (hpad / 2)) / scale;
        float x1 = (objects[i].x2 - (wpad / 2)) / scale;
        float y1 = (objects[i].y2 - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float) (img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float) (img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float) (img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float) (img_h - 1)), 0.f);

        objects[i].x1 = x0;
        objects[i].y1 = y0;
        objects[i].x2 = x1;
        objects[i].y2 = y1;
    }

    return 0;
}

int YoloV4::draw(cv::Mat &rgb, const std::vector<BoxInfo> &objects) {
    static const char *class_names[] = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush"
    };
    static const unsigned char colors[19][3] = {
            {54,  67,  244},
            {99,  30,  233},
            {176, 39,  156},
            {183, 58,  103},
            {181, 81,  63},
            {243, 150, 33},
            {244, 169, 3},
            {212, 188, 0},
            {136, 150, 0},
            {80,  175, 76},
            {74,  195, 139},
            {57,  220, 205},
            {59,  235, 255},
            {7,   193, 255},
            {0,   152, 255},
            {34,  87,  255},
            {72,  85,  121},
            {158, 158, 158},
            {139, 125, 96}
    };

    int color_index = 0;

    for (size_t i = 0; i < objects.size(); i++) {
        const BoxInfo &obj = objects[i];

        const unsigned char *color = colors[color_index % 19];
        color_index++;

        cv::Scalar cc(color[0], color[1], color[2]);

        cv::rectangle(rgb,
                      cv::Rect(obj.x1, obj.y1, obj.x2 - obj.x1, obj.y2 - obj.y1),
                      cc,
                      2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.score * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(
                text,
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                1,
                &baseLine);

        int x = obj.x1;
        int y = obj.y1 - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;

        cv::rectangle(rgb,
                      cv::Rect(cv::Point(x, y),
                               cv::Size(label_size.width, label_size.height + baseLine)),
                      cc,
                      -1);

        cv::Scalar textcc = (color[0] + color[1] + color[2] >= 381)
                            ? cv::Scalar(0, 0, 0)
                            : cv::Scalar(255, 255, 255);

        cv::putText(rgb,
                    text,
                    cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    textcc,
                    1);

    }

    return 0;
}

inline float fast_exp(float x) {
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + fast_exp(-x));
}

std::vector<BoxInfo> YoloV4::decode_infer(ncnn::Mat &data, int img_w, int img_h) {
    std::vector<BoxInfo> result;
    for (int i = 0; i < data.h; i++) {
        BoxInfo box;
        const float *values = data.row(i);
        box.label = values[0] - 1;
        box.score = values[1];
        box.x1 = values[2] * (float) img_w;
        box.y1 = values[3] * (float) img_h;
        box.x2 = values[4] * (float) img_w;
        box.y2 = values[5] * (float) img_h;
        result.push_back(box);
    }
    return result;
}

