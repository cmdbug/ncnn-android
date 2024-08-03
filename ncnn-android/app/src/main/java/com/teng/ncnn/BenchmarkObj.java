package com.teng.ncnn;

public class BenchmarkObj {
    // 0 = success
    // 1 = no gpu
    public int retcode;
    public double min;
    public double max;
    public double avg;
    public int width;
    public int height;

    public BenchmarkObj() {
        this.retcode = 0;
        this.min = 0;
        this.max = 0;
        this.avg = 0;
        this.width = 0;
        this.height = 0;
    }
}
