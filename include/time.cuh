#pragma once
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include <chrono>

class ChronoTimer {
public:
    explicit ChronoTimer(std::string name, size_t batch_size)
        : name_(std::move(name)), batch_size_(batch_size) {}

    ~ChronoTimer() {
        const size_t n = samples_us_.size();
        if (n == 0) return;

        const double mean_batch_us = mean(samples_us_);
        const double mean_us_per_item = mean_batch_us / (double)batch_size_;
        const double mean_ops_per_sec = mean_batch_us > 0.0 ? ((double)batch_size_ * 1e6) / mean_batch_us : 0.0;

        std::cout << name_ << "\n"<<"ops/s: " << mean_ops_per_sec << "\n";
    }

    inline void start() {
        t0_ = std::chrono::steady_clock::now();
    }

    inline void stop() {
        auto t1 = std::chrono::steady_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0_).count();
        samples_us_.push_back(us);
    }

    double min_us()  const { return samples_us_.empty() ? 0.0 : min(samples_us_); }
    double mean_us() const { return samples_us_.empty() ? 0.0 : mean(samples_us_); }

private:
    std::string name_;
    size_t batch_size_{1};
    std::chrono::steady_clock::time_point t0_{};
    std::vector<double> samples_us_;

    static double min(const std::vector<double>& v) {
        return *std::min_element(v.begin(), v.end());
    }
    static double mean(const std::vector<double>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0) / (double)v.size();
    }
};


class CUDAEventTimer {
public:
    explicit CUDAEventTimer(std::string name)
        : name_(std::move(name)) {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    ~CUDAEventTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);

        const size_t n = samples_us_.size();
        if (n == 0) return;

        const double minv = min(samples_us_);
        const double medv = median(samples_us_);
        const double sdv  = stddev(samples_us_);

        std::cout << name_ << ","
                  << n << ","
                  << minv << ","
                  << medv << ","
                  << sdv << "\n";
    }

    inline void start() {
        cudaEventRecord(start_, 0);
    }

    inline void stop() {
        cudaEventRecord(stop_, 0);
        cudaEventSynchronize(stop_);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_, stop_);
        samples_us_.push_back((double)ms * 1000.0);
    }

    inline void start(cudaStream_t stream) {
        cudaEventRecord(start_, stream);
    }

    inline void stop(cudaStream_t stream) {
        cudaEventRecord(stop_, stream);
        cudaEventSynchronize(stop_);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_, stop_);
        samples_us_.push_back((double)ms * 1000.0);
    }

private:
    std::string name_;
    cudaEvent_t start_{}, stop_{};
    std::vector<double> samples_us_;

    static double min(const std::vector<double>& v) {
        return *std::min_element(v.begin(), v.end());
    }

    static double median(std::vector<double> v) {
        std::sort(v.begin(), v.end());
        const size_t n = v.size();
        if (n % 2) return v[n / 2];
        return 0.5 * (v[n / 2 - 1] + v[n / 2]);
    }

    static double stddev(const std::vector<double>& v) {
        const size_t n = v.size();
        if (n == 0) return 0.0;
        const double mu = std::accumulate(v.begin(), v.end(), 0.0) / (double)n;
        double acc = 0.0;
        for (double x : v) { double d = x - mu; acc += d * d; }
        return std::sqrt(acc / (double)n);
    }
};
