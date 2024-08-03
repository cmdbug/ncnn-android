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

package com.teng.ncnn;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.app.ProgressDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.BlurMaskFilter;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.graphics.PorterDuff;
import android.graphics.Rect;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.os.Build;
import android.os.Bundle;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.Spinner;

import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.widget.Toast;

import java.util.Locale;

public class MainActivity extends Activity implements SurfaceHolder.Callback {
    public static final int REQUEST_CAMERA = 100;

    private Tncnn tncnn = new Tncnn();
    private int facing = 1;

    private Spinner spinnerModel;
    private Spinner spinnerCPUGPU;
    private Spinner spinnerPowersave;
    private Spinner spinnerThreads;
    private int current_model = 0;
    private int current_cpugpu = 0;
    private int current_powersave = 0;
    private int current_threads = 0;

    private SurfaceView cameraView;

    private final BenchmarkNcnn benchmarkNcnn = new BenchmarkNcnn();
    private ProgressDialog progressDialog;
    private final int LOOP = 50;  // warmup 是 5，不要一样就行了

    private boolean camera2Support = true;  // 是否支持 Camera2 接口
    private boolean surfaceViewCreated = false;  // 显示控件是否创建完成

    private final String[] items = new String[]{
            "Mempool", "Winograd", "SGEMM", "Pack4", "BF16 storage",
            "FP16 packed", "FP16 storage", "FP16 arithmetic", "Light mode"
    };

    private final boolean[] checks = new boolean[]{
            false, true, true, true, true,
            true, true, false, true
    };

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        initView();
        reload();

        setProgressCallback();
    }

    /**
     * 初始化视图
     */
    private void initView() {
        cameraView = (SurfaceView) findViewById(R.id.cameraview);
        cameraView.getHolder().setFormat(PixelFormat.RGBA_8888);
        cameraView.getHolder().addCallback(this);

        Button buttonSwitchCamera = (Button) findViewById(R.id.buttonSwitchCamera);
        buttonSwitchCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                int new_facing = 1 - facing;
                tncnn.closeCamera();
                openCamera(new_facing);
                facing = new_facing;
            }
        });

        Button buttonOptionChange = (Button) findViewById(R.id.buttonOptionChange);
        buttonOptionChange.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                optionDialog();
            }
        });

        Button buttonBenchmark = (Button) findViewById(R.id.buttonBenchmark);
        buttonBenchmark.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                benchmarkNcnnDialog();
            }
        });

        spinnerModel = (Spinner) findViewById(R.id.spinnerModel);
        spinnerModel.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id) {
                if (position != current_model) {
                    current_model = position;
                    reload();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0) {
            }
        });

        spinnerCPUGPU = (Spinner) findViewById(R.id.spinnerCPUGPU);
        spinnerCPUGPU.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id) {
                if (position != current_cpugpu) {
                    current_cpugpu = position;
                    reload();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0) {
            }
        });

        spinnerPowersave = (Spinner) findViewById(R.id.spinnerPowersave);
        spinnerPowersave.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> adapterView, View view, int position, long id) {
                if (position != current_powersave) {
                    current_powersave = position;
                    reload();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> adapterView) {

            }
        });

        spinnerThreads = (Spinner) findViewById(R.id.spinnerThreads);
        spinnerThreads.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> adapterView, View view, int position, long id) {
                if (position != current_threads) {
                    current_threads = position;
                    reload();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> adapterView) {

            }
        });
    }

    /**
     * 切换模型
     */
    private void reload() {
        boolean ret_init = tncnn.loadModel(getAssets(),
                current_model, current_cpugpu, current_powersave, current_threads,
                checks[0], checks[1], checks[2], checks[3], checks[4], checks[5], checks[6], checks[7], checks[8]
        );
        if (!ret_init) {
            Toast.makeText(this, "加载模型失败", Toast.LENGTH_SHORT).show();
        }
    }

    /**
     * 参数配置弹窗
     */
    private void optionDialog() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setIcon(R.drawable.ncnn_icon);
        builder.setTitle("参数配置");
        builder.setCancelable(false);
        builder.setPositiveButton("确定", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialogInterface, int i) {
                reload();
            }
        });
        builder.setMultiChoiceItems(items, checks, new DialogInterface.OnMultiChoiceClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which, boolean isChecked) {
                checks[which] = isChecked;
            }
        });
        builder.show();
    }

    /**
     * 不支持Camera2接口里显示 icon
     */
    private void drawIcon() {
        if (!surfaceViewCreated) {
            return;
        }
        if (camera2Support) {
            return;
        }

        Canvas canvas = cameraView.getHolder().lockCanvas();

        canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);
        Paint paint = new Paint();
        paint.setAlpha(100);
//        paint.setMaskFilter(new BlurMaskFilter(20, BlurMaskFilter.Blur.SOLID));
        Bitmap icon = BitmapFactory.decodeResource(getResources(), R.drawable.ncnn_icon);
        int w = cameraView.getWidth();
        int h = cameraView.getHeight();
        Rect dst = new Rect(
                (w - icon.getWidth()) / 2,
                (h - icon.getHeight()) / 2,
                (w - icon.getWidth()) / 2 + icon.getWidth(),
                (h - icon.getHeight()) / 2 + icon.getHeight()
        );
        canvas.drawBitmap(icon, null, dst, paint);

        cameraView.getHolder().unlockCanvasAndPost(canvas);
    }

    /**
     * 基准测试弹窗
     */
    private void benchmarkNcnnDialog() {
        tncnn.closeCamera();

        progressDialog = new ProgressDialog(this);
        progressDialog.setIcon(R.drawable.ncnn_icon);
        progressDialog.setTitle("基准测试中...");
        progressDialog.setProgressStyle(ProgressDialog.STYLE_HORIZONTAL);
        progressDialog.setIndeterminate(false);
        progressDialog.setCancelable(false);
        progressDialog.setCanceledOnTouchOutside(false);
        progressDialog.setMax(LOOP);
        progressDialog.show();

        new Thread(new Runnable() {
            @Override
            public void run() {
                benchmarkNcnn.init();
                final BenchmarkObj obj = benchmarkNcnn.run(getAssets(), current_threads, current_powersave,
                        checks[0], checks[1], checks[2], checks[3], checks[4], current_cpugpu == 1, checks[5], checks[6],
                        checks[7], checks[8], false, current_model, LOOP);
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        progressDialog.dismiss();

                        String message = String.format(Locale.CHINESE,
                                "%-7s %s" +
                                        "\n%-7s %s" +
                                        "\n%-7s %dx%d" +
                                        "\n%-7s %.2f ms" +
                                        "\n%-7s %.2f ms" +
                                        "\n%-7s %.2f ms" +
                                        "\n%-7s %.2f" +
                                        "\n%-7s %s" +
                                        "\n%-7s %s" +
                                        "\n%-7s %s" +
                                        "\n%-7s %s",
                                "循环次数:", LOOP,
                                "运行模式:", (current_cpugpu == 1 ? "GPU" : "CPU"),
                                "输入尺寸:", obj.width, obj.height,
                                "最快时间:", obj.min,
                                "最慢时间:", obj.max,
                                "平均时间:", obj.avg,
                                "估计帧数:", 1000 / obj.avg,
                                "库版本号:", benchmarkNcnn.getNcnnVersion(),
                                "设备型号:", Build.MODEL,
                                "系统版本:", Build.VERSION.RELEASE,
                                "硬件信息:", Build.HARDWARE);

                        String[] models = getResources().getStringArray(R.array.model_array);
                        String model = models[current_model];

                        AlertDialog.Builder normalDialog = new AlertDialog.Builder(MainActivity.this);
                        normalDialog.setIcon(R.drawable.ncnn_icon);
                        normalDialog.setTitle(model);
                        normalDialog.setMessage(message);
                        normalDialog.setCancelable(false);
                        normalDialog.setNegativeButton("心里有点数了吧", new DialogInterface.OnClickListener() {
                            @Override
                            public void onClick(DialogInterface dialog, int which) {
                                reload();
                                openCamera(facing);
                            }
                        });
                        normalDialog.show();
                    }
                });
            }
        }, "benchmarkncnn").start();
    }

    /**
     * 设置jni调用的进度接口
     */
    public void setProgressCallback() {
        benchmarkNcnn.setiProgressCallback(new BenchmarkNcnn.IProgressCallback() {
            @Override
            public void changeProgress(int prgress, int max) {
                if (progressDialog == null) {
                    return;
                }
                progressDialog.setIndeterminate(max != LOOP);  // warmup
                progressDialog.setProgress(prgress);
            }
        });
    }

    /**
     * 检查手机相机Camera2接口支持等级
     *
     * @return
     */
    public boolean checkCameraHardwareLevel() {
        CameraManager mCameraManager = (CameraManager) this.getSystemService(Context.CAMERA_SERVICE);
        if (mCameraManager == null) {
            return false;
        }
        try {
            CameraCharacteristics characteristics = mCameraManager.getCameraCharacteristics(String.valueOf(facing));
            Integer level = characteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL);
            if (level == null || level == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY) {
                return false;
            }
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
        return true;
    }

    /**
     * 打开相机
     *
     * @param id
     */
    public void openCamera(int id) {
        if (!tncnn.openCamera(id)) {
            if (!checkCameraHardwareLevel()) {
                Toast.makeText(this, "相机" + id + " Camera2接口驱动不支持", Toast.LENGTH_SHORT).show();
                camera2Support = false;
            } else {
                Toast.makeText(this, "相机" + id + "打开失败", Toast.LENGTH_SHORT).show();
            }
        }
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        tncnn.setOutputWindow(holder.getSurface());
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        surfaceViewCreated = true;
        drawIcon();
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
    }

    @Override
    public void onResume() {
        super.onResume();

        if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA);
        } else {
            openCamera(facing);
        }
    }

    @Override
    public void onPause() {
        super.onPause();
        tncnn.closeCamera();
    }
}
