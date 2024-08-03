package com.teng.ncnn;

import android.content.res.AssetManager;
import android.util.Log;

public class BenchmarkNcnn {
    public native boolean init();

    public native String getPlatform();

    public native String getNcnnVersion();

    public native BenchmarkObj run(AssetManager mgr, int threads, int powersave,
                                   boolean mempool, boolean winograd, boolean sgemm, boolean pack4, boolean bf16s,
                                   boolean gpu, boolean gpufp16p, boolean gpufp16s, boolean gpufp16a, boolean lightmode,
                                   boolean gpupack8, int model, int loops);

    static {
        System.loadLibrary("tncnn");
    }

    // ************************* 进度提示部分 *************************
    private static IProgressCallback iProgressCallback;

    public void setiProgressCallback(IProgressCallback iProgressCallback) {
        BenchmarkNcnn.iProgressCallback = iProgressCallback;
    }

    public static void changeProgress(int progress, int max) {
        if (iProgressCallback == null) {
            return;
        }
        iProgressCallback.changeProgress(progress, max);
    }

    public interface IProgressCallback {
        void changeProgress(int prgress, int max);
    }

}
