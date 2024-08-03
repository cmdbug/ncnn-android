#ifndef NCNN_OP_H
#define NCNN_OP_H

#include <net.h>

class YoloV5Focus : public ncnn::Layer {
public:
    YoloV5Focus();

    virtual int forward(const ncnn::Mat &bottom_blob, ncnn::Mat &top_blob, const ncnn::Option &opt) const;
};

static DEFINE_LAYER_CREATOR(YoloV5Focus)

#endif //NCNN_OP_H
