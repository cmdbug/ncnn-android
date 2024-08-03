#include <android/asset_manager_jni.h>
#include <android/log.h>

#include <sys/system_properties.h>

#include <jni.h>

#include <float.h>
#include <string>
#include <vector>
#include <map>

// ncnn
#include "benchmark.h"
#include "c_api.h"
#include "cpu.h"
#include "datareader.h"
#include "net.h"
#include "gpu.h"

#include "benchmarkncnn_op.h"

JNIEnv *myenv;
jobject mythiz;

void change_progress(int p, int max);


class DataReaderFromEmpty : public ncnn::DataReader {
public:
    virtual int scan(const char *format, void *p) const { return 0; }

    virtual size_t read(void *buf, size_t size) const {
        memset(buf, 0, size);
        return size;
    }
};


class BenchmarkNet : public ncnn::Net {
public:
    int run(int loops, double &time_min, double &time_max, double &time_avg, int &width, int &height, int size) {
        time_min = DBL_MAX;
        time_max = -DBL_MAX;
        time_avg = 0;

        // resolve input shape
        const std::vector<const char *> &input_names_x = input_names();
        const std::vector<const char *> &output_names_x = output_names();
        std::vector<ncnn::Mat> input_mats_x;

        {
            for (int i = 0; i < (int) layers().size(); i++) {
                const ncnn::Layer *layer = layers()[i];

                if (layer->type != "Input") {
                    continue;
                }

                for (int j = 0; j < input_names_x.size(); j++) {
                    if (blobs()[layer->tops[0]].name != input_names_x[j]) {
                        continue;
                    }

                    ncnn::Mat in;
                    const ncnn::Mat &shape = layer->top_shapes[0];
                    if (shape.c == 0 || shape.h == 0 || shape.w == 0) {
                        in.create(size, size, 3);
                        width = size;
                        height = size;
                    } else {
                        in.create(shape.w, shape.h, shape.c);
                        width = shape.w;
                        height = shape.h;
                    }
                    in.fill(0.01f);
                    input_mats_x.push_back(in);
                }
            }

            if (input_names_x.empty())
                return -1;
            if (input_mats_x.empty())
                return -1;
            if (input_names_x.size() != input_mats_x.size())
                return -1;
        }

        ncnn::Mat out;

        // warm up
        const int g_warmup_loop_count = 5;  // FIXME hardcode
        for (int i = 0; i < g_warmup_loop_count; i++) {
            change_progress(i + 1, g_warmup_loop_count);

            ncnn::Extractor ex = create_extractor();
            for (int j = 0; j < input_names_x.size(); j++) {
                ex.input(input_names_x[j], input_mats_x[j]);
            }
            for (int j = 0; j < output_names_x.size(); j++) {
                ex.extract(output_names_x[j], out);
            }
        }

        for (int i = 0; i < loops; i++) {
            change_progress(i + 1, loops);

            double start = ncnn::get_current_time();
            {
                ncnn::Extractor ex = create_extractor();
                for (int j = 0; j < input_names_x.size(); j++) {
                    ex.input(input_names_x[j], input_mats_x[j]);
                }
                for (int j = 0; j < output_names_x.size(); j++) {
                    ex.extract(output_names_x[j], out);
                }
            }

            double end = ncnn::get_current_time();

            double time = end - start;

            time_min = std::min(time_min, time);
            time_max = std::max(time_max, time);
            time_avg += time;
        }

        time_avg /= loops;

        return 0;
    }
};

// must be the same order with strings.xml
static const char *g_models[] = {
        "nanodet-m",
        "nanodet-m-416",
        "nanodet-g",
        "nanodet-ELite0_320",
        "nanodet-ELite1_416",
        "nanodet-ELite2_512",
        "nanodet-RepVGG-A0_416",
        "yolox-nano",
        "yolox-tiny",
        "blazeface_192",
        "blazeface_320",
        "blazeface_640",
        "humanmatting-mbv2",
        "yolov4-tiny",
};

static const std::map<std::string, int> g_models_size = {
        {"nanodet-m",             320},
        {"nanodet-m-416",         416},
        {"nanodet-g",             416},
        {"nanodet-ELite0_320",    320},
        {"nanodet-ELite1_416",    416},
        {"nanodet-ELite2_512",    512},
        {"nanodet-RepVGG-A0_416", 416},
        {"yolox-nano",            416},
        {"yolox-tiny",            416},
        {"blazeface_192",         192},
        {"blazeface_320",         320},
        {"blazeface_640",         640},
        {"humanmatting-mbv2",     512},
        {"yolov4-tiny",           416},
};


extern "C" {

static jclass objCls = nullptr;
static jmethodID constructortorId;
static jfieldID retcodeId;
static jfieldID minId;
static jfieldID maxId;
static jfieldID avgId;
static jfieldID widthId;
static jfieldID heightId;

// public native boolean Init();
JNIEXPORT jboolean JNICALL
Java_com_teng_ncnn_BenchmarkNcnn_init(JNIEnv *env, jobject thiz) {
    myenv = env;
    mythiz = env->NewGlobalRef(thiz);
    jclass localObjCls = env->FindClass("com/teng/ncnn/BenchmarkObj");
    objCls = reinterpret_cast<jclass>(env->NewGlobalRef(localObjCls));

    constructortorId = env->GetMethodID(objCls, "<init>", "()V");

    retcodeId = env->GetFieldID(objCls, "retcode", "I");
    minId = env->GetFieldID(objCls, "min", "D");
    maxId = env->GetFieldID(objCls, "max", "D");
    avgId = env->GetFieldID(objCls, "avg", "D");
    widthId = env->GetFieldID(objCls, "width", "I");
    heightId = env->GetFieldID(objCls, "height", "I");

    return JNI_TRUE;
}

// public native String GetPlatform();
JNIEXPORT jstring JNICALL
Java_com_teng_ncnn_BenchmarkNcnn_getPlatform(JNIEnv *env, jobject thiz) {
    char platform[PROP_VALUE_MAX + 1];
    __system_property_get("ro.board.platform", platform);

    return env->NewStringUTF(platform);
}

// public native String GetNcnnVersion();
JNIEXPORT jstring JNICALL
Java_com_teng_ncnn_BenchmarkNcnn_getNcnnVersion(JNIEnv *env, jobject thiz) {
    return env->NewStringUTF(ncnn_version());
}

// public native Obj Run(AssetManager mgr, int threads, int powersave,
//                       boolean mempool, boolean winograd, boolean sgemm, boolean pack4, boolean bf16s,
//                       boolean gpu, boolean gpufp16p, boolean gpufp16s, boolean gpufp16a, boolean gpupack8,
//                       int model, int loops);
JNIEXPORT jobject JNICALL
Java_com_teng_ncnn_BenchmarkNcnn_run(JNIEnv *env, jobject thiz, jobject assetManager, jint threads,
                                     jint powersave,
                                     jboolean mempool, jboolean winograd, jboolean sgemm, jboolean pack4,
                                     jboolean bf16s,
                                     jboolean gpu, jboolean gpufp16p, jboolean gpufp16s, jboolean gpufp16a,
                                     jboolean lightmodel, jboolean gpupack8, jint model, jint loops) {
    __android_log_print(ANDROID_LOG_DEBUG, "BenchmarkNcnn",
                        "threads=%d powersave=%d mempool=%d winograd=%d sgemm=%d pack4=%d bf16s=%d gpu=%d gpufp16p=%d gpufp16s=%d gpufp16a=%d gpupack8=%d model=%d loops=%d",
                        threads, powersave, mempool, winograd, sgemm, pack4, bf16s, gpu, gpufp16p, gpufp16s, gpufp16a,
                        gpupack8, model, loops);

    if (gpu == JNI_TRUE && ncnn::get_gpu_count() == 0) {
        // return result
        jobject jObj = env->NewObject(objCls, constructortorId);

        env->SetIntField(jObj, retcodeId, 1);

        return jObj;
    }

    if (model < 0 || model >= sizeof(g_models) / sizeof(const char *)) {
        // unknown model
        jobject jObj = env->NewObject(objCls, constructortorId);

        env->SetIntField(jObj, retcodeId, 2);

        return jObj;
    }

    ncnn::UnlockedPoolAllocator *blob_pool_allocator = 0;
    ncnn::UnlockedPoolAllocator *workspace_pool_allocator = 0;

    ncnn::VulkanDevice *vkdev = 0;
    ncnn::VkBlobAllocator *blob_vkallocator = 0;
    ncnn::VkStagingAllocator *staging_vkallocator = 0;

    // prepare opt
    ncnn::Option opt;
    opt.lightmode = lightmodel;
    opt.num_threads = (threads == 0) ? ncnn::get_big_cpu_count() : threads;

    if (mempool) {
        blob_pool_allocator = new ncnn::UnlockedPoolAllocator;
        workspace_pool_allocator = new ncnn::UnlockedPoolAllocator;

        opt.blob_allocator = blob_pool_allocator;
        opt.workspace_allocator = workspace_pool_allocator;
    }

    if (gpu) {
        const int gpu_device = 0;// FIXME hardcode
        vkdev = ncnn::get_gpu_device(0);

        blob_vkallocator = new ncnn::VkBlobAllocator(vkdev);
        staging_vkallocator = new ncnn::VkStagingAllocator(vkdev);

        opt.blob_vkallocator = blob_vkallocator;
        opt.workspace_vkallocator = blob_vkallocator;
        opt.staging_vkallocator = staging_vkallocator;
    }

    opt.use_winograd_convolution = winograd;
    opt.use_sgemm_convolution = sgemm;

    opt.use_vulkan_compute = gpu;

    opt.use_fp16_packed = gpufp16p;
    opt.use_fp16_storage = gpufp16s;
    opt.use_fp16_arithmetic = gpufp16a;
    opt.use_int8_storage = true;
    opt.use_int8_arithmetic = false;

    opt.use_shader_pack8 = gpupack8;

    opt.use_bf16_storage = bf16s;

    ncnn::set_cpu_powersave(powersave);

    // load model
    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);

    BenchmarkNet net;

    net.register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);

    net.opt = opt;

    if (gpu) {
        net.set_vulkan_device(vkdev);
    }

    std::vector<std::string> run_models;
    std::vector<int> run_model_sizes;

    std::string param_temp = g_models[model];
    if (param_temp.find("blazeface") != std::string::npos) {  // blazeface_192 / blazeface_320 / blazeface_640
        param_temp = "blazeface";
    }

    std::string param_path = param_temp + std::string(".param");
    run_models.push_back(param_path);
    run_model_sizes.push_back(g_models_size.at(g_models[model]));
    if (param_path.find("blazeface") != std::string::npos) {
        run_models.push_back("face_mesh.param");
        run_model_sizes.push_back(192);
    }

    double time_min = 0;
    double time_max = 0;
    double time_avg = 0;
    int width = 0;
    int height = 0;
    int rr;

    for (int i = 0; i < run_models.size(); i++) {
        double min = 0, max = 0, avg = 0;
        int w = 0, h = 0;
        int rp = net.load_param(mgr, run_models.at(i).c_str());
        __android_log_print(ANDROID_LOG_DEBUG, "teng_ncnn", "load %s: %s",
                            run_models.at(i).c_str(), rp == 0 ? "success" : "fail");

        DataReaderFromEmpty dr;
        int rm = net.load_model(dr);
        __android_log_print(ANDROID_LOG_DEBUG, "teng_ncnn", "load model: %s",
                            rm == 0 ? "success" : "fail");

        rr = net.run(loops, min, max, avg, w, h, run_model_sizes.at(i));
        time_min += min;
        time_max += max;
        time_avg += avg;
        if (width == 0 || height == 0) {  // blazeface_192 / 320 / 640 还要运行 landmark 192
            width = w;
            height = h;
        }
    }

    delete blob_pool_allocator;
    delete workspace_pool_allocator;

    delete blob_vkallocator;
    delete staging_vkallocator;

    if (rr != 0) {
        // runtime error
        jobject jObj = env->NewObject(objCls, constructortorId);

        env->SetIntField(jObj, retcodeId, 3);

        return jObj;
    }

    // return result
    jobject jObj = env->NewObject(objCls, constructortorId);

    env->SetIntField(jObj, retcodeId, 0);
    env->SetDoubleField(jObj, minId, time_min);
    env->SetDoubleField(jObj, maxId, time_max);
    env->SetDoubleField(jObj, avgId, time_avg);
    env->SetIntField(jObj, widthId, width);
    env->SetIntField(jObj, heightId, height);

    return jObj;
}

}


void change_progress(int p, int max) {
    jclass benchmarkNcnn = myenv->FindClass("com/teng/ncnn/BenchmarkNcnn");
    jmethodID pJmethodID = myenv->GetStaticMethodID(benchmarkNcnn, "changeProgress", "(II)V");
    myenv->CallStaticVoidMethod(benchmarkNcnn, pJmethodID, p, max);
    myenv->DeleteLocalRef(benchmarkNcnn);
}
