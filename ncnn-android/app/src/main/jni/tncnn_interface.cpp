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

#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>

#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>

#include <platform.h>
#include <benchmark.h>

#include "ndkcamera.h"

#include <net.h>
#include <cpu.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "yolox.h"
#include "nanodet.h"
#include "face_mesh.h"
#include "human_matting.h"
#include "yolov4.h"

#if __ARM_NEON

#include <arm_neon.h>

#endif // __ARM_NEON


static yolox::Yolox *g_yolox = nullptr;
static nanodet::NanoDet *g_nanodet = nullptr;
static Face *g_blazeface = nullptr;
static humanmatting::HumanMatting *g_humanmatting = nullptr;
static YoloV4 *g_yolov4 = nullptr;

static int draw_unsupported(cv::Mat &rgb) {
    const char text[] = "unsupported";

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text,
                                          cv::FONT_HERSHEY_SIMPLEX,
                                          1.0,
                                          1, &baseLine);

    int y = (rgb.rows - label_size.height) / 2;
    int x = (rgb.cols - label_size.width) / 2;

    cv::rectangle(rgb,
                  cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                  cv::Scalar(255, 255, 255),
                  -1);

    cv::putText(rgb,
                text,
                cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX,
                1.0,
                cv::Scalar(0, 0, 0));

    return 0;
}

static int draw_fps(cv::Mat &rgb) {
    // resolve moving average
    float avg_fps = 0.f;
    {
        static double t0 = 0.f;
        static float fps_history[10] = {0.f};

        double t1 = ncnn::get_current_time();
        if (t0 == 0.f) {
            t0 = t1;
            return 0;
        }

        float fps = 1000.f / (t1 - t0);
        t0 = t1;

        for (int i = 9; i >= 1; i--) {
            fps_history[i] = fps_history[i - 1];
        }
        fps_history[0] = fps;

        if (fps_history[9] == 0.f) {
            return 0;
        }

        for (int i = 0; i < 10; i++) {
            avg_fps += fps_history[i];
        }
        avg_fps /= 10.f;
    }

    char text[32];
    sprintf(text, "%dx%d FPS:%05.2f", rgb.cols, rgb.rows, avg_fps);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text,
                                          cv::FONT_HERSHEY_SIMPLEX,
                                          0.5,
                                          1,
                                          &baseLine);

    int y = 0;
    int x = rgb.cols - label_size.width;

//    cv::rectangle(rgb,
//                  cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
//                  cv::Scalar(255, 255, 255),
//                  -1);

    cv::putText(rgb,
                text,
                cv::Point(x, y + label_size.height + 1),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                cv::Scalar(0, 0, 0));
    cv::putText(rgb,
                text,
                cv::Point(x - 1, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                cv::Scalar(255, 255, 255));

    return 0;
}

static void draw_ncnn_version(cv::Mat &rgb) {
    char text[32];
    sprintf(text, "%s", NCNN_VERSION_STRING);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text,
                                          cv::FONT_HERSHEY_SIMPLEX,
                                          0.4,
                                          1,
                                          &baseLine);

    int x = 0;
    int y = rgb.rows - label_size.height;

    cv::putText(rgb,
                text,
                cv::Point(x + 2, y + label_size.height - 2),
                cv::FONT_HERSHEY_SIMPLEX,
                0.4,
                cv::Scalar(0, 0, 0));
    cv::putText(rgb,
                text,
                cv::Point(x + 1, y + label_size.height - 3),
                cv::FONT_HERSHEY_SIMPLEX,
                0.4,
                cv::Scalar(255, 255, 255));
}

static int clear_model() {
    __android_log_print(ANDROID_LOG_DEBUG, "teng_ncnn", "clear model");

    delete g_yolox;
    delete g_nanodet;
    delete g_blazeface;
    delete g_humanmatting;
    delete g_yolov4;
    g_yolox = nullptr;
    g_nanodet = nullptr;
    g_blazeface = nullptr;
    g_humanmatting = nullptr;
    g_yolov4 = nullptr;

    return 0;
}

static int create_model(AAssetManager *mgr, ncnn::Option option, int modelid) {
    // 请与 strings.xml 里面的顺序一致
    const char *modeltypes[] =
            {
                    "m",
                    "m-416",
                    "g",
                    "ELite0_320",
                    "ELite1_416",
                    "ELite2_512",
                    "RepVGG-A0_416",
                    "yolox-nano",
                    "yolox-tiny",
                    "blazeface_192",
                    "blazeface_320",
                    "blazeface_640",
                    "humanmatting-mbv2",
                    "yolov4-tiny",
            };
    const char *modeltype = modeltypes[(int) modelid];

    __android_log_print(ANDROID_LOG_DEBUG, "teng_ncnn", "create model -> load: %s", modeltype);
    if (std::strcmp("yolox-nano", modeltype) == 0
        || std::strcmp("yolox-tiny", modeltype) == 0) {
        g_yolox = new yolox::Yolox();
        g_yolox->load(mgr, option, modeltype);
    } else if (std::strcmp("m", modeltype) == 0
               || std::strcmp("m-416", modeltype) == 0
               || std::strcmp("g", modeltype) == 0
               || std::strcmp("ELite0_320", modeltype) == 0
               || std::strcmp("ELite1_416", modeltype) == 0
               || std::strcmp("ELite2_512", modeltype) == 0
               || std::strcmp("RepVGG-A0_416", modeltype) == 0) {

        g_nanodet = new nanodet::NanoDet();
        g_nanodet->load(mgr, option, modeltype);
    } else if (std::strcmp("blazeface_192", modeltype) == 0
               || std::strcmp("blazeface_320", modeltype) == 0
               || std::strcmp("blazeface_640", modeltype) == 0) {
        g_blazeface = new Face();
        g_blazeface->load(mgr, option, modeltype);
    } else if (std::strcmp("humanmatting-mbv2", modeltype) == 0) {
        g_humanmatting = new humanmatting::HumanMatting();
        g_humanmatting->load(mgr, option, modeltype);
    } else if (std::strcmp("yolov4-tiny", modeltype) == 0) {
        g_yolov4 = new YoloV4();
        g_yolov4->load(mgr, option, modeltype);
    }
    return 0;
}

static int run_model(cv::Mat &rgb) {
    if (g_yolox) {
        std::vector<Object> objects;
        g_yolox->detect(rgb, objects);
        g_yolox->draw(rgb, objects);
    } else if (g_nanodet) {
        std::vector<Object> objects;
        g_nanodet->detect(rgb, objects);
        g_nanodet->draw(rgb, objects);
    } else if (g_blazeface) {
        std::vector<FaceObject> face_objects;
        g_blazeface->detect(rgb, face_objects);
        g_blazeface->draw(rgb, face_objects);
    } else if (g_humanmatting) {
        cv::Mat mat;
        g_humanmatting->detect(rgb, mat);
        g_humanmatting->draw(rgb, mat);
    } else if (g_yolov4) {
        std::vector<BoxInfo> objects;
        g_yolov4->detect(rgb, objects);
        g_yolov4->draw(rgb, objects);
    } else {
        draw_unsupported(rgb);
    }
    return 0;
}

static ncnn::Mutex lock;

class MyNdkCamera : public NdkCameraWindow {
public:
    virtual void on_image_render(cv::Mat &rgb) const;
};

void MyNdkCamera::on_image_render(cv::Mat &rgb) const {
    {
        ncnn::MutexLockGuard g(lock);

        run_model(rgb);
    }
    draw_fps(rgb);
    draw_ncnn_version(rgb);
}

static MyNdkCamera *g_camera = nullptr;
ncnn::UnlockedPoolAllocator blob_pool_allocator;
ncnn::PoolAllocator workspace_pool_allocator;

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "teng_ncnn", "JNI_OnLoad");

    g_camera = new MyNdkCamera;

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM *vm, void *reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "teng_ncnn", "JNI_OnUnload");

    {
        ncnn::MutexLockGuard g(lock);

        clear_model();
    }

    delete g_camera;
    g_camera = nullptr;
}

// public native boolean loadModel(AssetManager mgr, int modelid, int cpugpu);
JNIEXPORT jboolean JNICALL
Java_com_teng_ncnn_Tncnn_loadModel(JNIEnv *env, jobject thiz, jobject assetManager,
                                   jint modelid, jint cpugpu, jint powersave, jint threads, jboolean mempool,
                                   jboolean winograd, jboolean sgemm, jboolean pack4, jboolean bf16s,
                                   jboolean fp16p, jboolean fp16s, jboolean fp16a, jboolean lightmode) {

    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
//    __android_log_print(ANDROID_LOG_DEBUG, "teng_ncnn", "loadModel %p", mgr);

    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(powersave);
    ncnn::set_omp_num_threads((threads == 0) ? ncnn::get_big_cpu_count() : threads);

    ncnn::Option option;
    bool use_gpu = false;
#if NCNN_VULKAN
    use_gpu = (cpugpu == 1) && (ncnn::get_gpu_count() > 0);
    option.use_vulkan_compute = use_gpu;
#endif
    option.num_threads = (threads == 0) ? ncnn::get_big_cpu_count() : threads;
    if (mempool) {
        option.blob_allocator = &blob_pool_allocator;
        option.workspace_allocator = &workspace_pool_allocator;
    }
    option.use_winograd_convolution = winograd;
    option.use_sgemm_convolution = sgemm;
    option.use_bf16_storage = bf16s;
    option.use_fp16_packed = fp16p;
    option.use_fp16_storage = fp16s;
    option.use_fp16_arithmetic = fp16a;
    option.lightmode = lightmode;

    // reload
    {
        ncnn::MutexLockGuard g(lock);

        clear_model();
        create_model(mgr, option, modelid);
    }

    return JNI_TRUE;
}

// public native boolean openCamera(int facing);
JNIEXPORT jboolean JNICALL Java_com_teng_ncnn_Tncnn_openCamera(JNIEnv *env, jobject thiz, jint facing) {
    if (facing < 0 || facing > 1)
        return JNI_FALSE;

    __android_log_print(ANDROID_LOG_DEBUG, "teng_ncnn", "openCamera %d", facing);

    int r = g_camera->open((int) facing);
    if (r != 0) {
        return JNI_FALSE;
    }

    return JNI_TRUE;
}

// public native boolean closeCamera();
JNIEXPORT jboolean JNICALL Java_com_teng_ncnn_Tncnn_closeCamera(JNIEnv *env, jobject thiz) {
    __android_log_print(ANDROID_LOG_DEBUG, "teng_ncnn", "closeCamera");

    g_camera->close();

    return JNI_TRUE;
}

// public native boolean setOutputWindow(Surface surface);
JNIEXPORT jboolean JNICALL Java_com_teng_ncnn_Tncnn_setOutputWindow(JNIEnv *env, jobject thiz, jobject surface) {
    ANativeWindow *win = ANativeWindow_fromSurface(env, surface);

//    __android_log_print(ANDROID_LOG_DEBUG, "teng_ncnn", "setOutputWindow %p", win);

    g_camera->set_window(win);

    return JNI_TRUE;
}

}
