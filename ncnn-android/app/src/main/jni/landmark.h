// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef LANDMARK_H
#define LANDMARK_H

#include <opencv2/core/core.hpp>
#include <net.h>

class LandmarkDetect {
public:
    ~LandmarkDetect();

    int load(AAssetManager *mgr, ncnn::Option option, const char *modeltype);

    int detect(const cv::Mat &rgb, const cv::Mat &trans_mat, std::vector <cv::Point2f> &landmarks);

private:
    ncnn::Net landmark;
};

#endif // LANDMARK_H
