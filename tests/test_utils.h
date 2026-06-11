#pragma once

// Shared helpers for kernel tests: correctness checks and a benchmark
// harness. Timing reports the median of several samples to ride out
// clock ramping and thermal throttling (significant on laptop GPUs).

#include "core/cuda_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

inline bool compareResults(const float* gpu,
                           const float* cpu,
                           int size,
                           float tolerance = 1e-3f)
{
    for (int i = 0; i < size; ++i) {
        float diff = std::fabs(gpu[i] - cpu[i]);

        if (diff > tolerance) {
            std::printf("Mismatch at %d: GPU = %f, CPU = %f, diff = %f\n",
                        i, gpu[i], cpu[i], diff);
            return false;
        }
    }

    return true;
}

// Relative-error check for scalar results. Reductions reorder float
// additions, so compare against a higher-precision reference with a
// relative tolerance rather than expecting bit equality.
inline bool checkScalar(double gpu, double ref, double tolerance = 1e-4)
{
    double relErr = std::fabs(gpu - ref) / std::fabs(ref);

    if (relErr >= tolerance) {
        std::printf("Scalar mismatch: GPU = %f, ref = %f, relErr = %.2e\n",
                    gpu, ref, relErr);
        return false;
    }

    return true;
}

// Returns the median ms per launch of `launch()`. Takes `reps` timed
// samples of `iters` launches each and reports the median sample.
template <typename LaunchFn>
inline double benchmarkMs(LaunchFn launch, int iters = 50, int reps = 5)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    launch();   // warmup
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> samples(reps);
    for (int r = 0; r < reps; ++r) {
        CUDA_CHECK(cudaEventRecord(start));
        for (int it = 0; it < iters; ++it) {
            launch();
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        samples[r] = ms / iters;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::sort(samples.begin(), samples.end());
    return samples[reps / 2];
}

// Prints ms/iter plus whichever throughput metric applies: effective
// bandwidth for memory-bound kernels (pass bytes, flops = 0) and/or
// GFLOP/s for compute-bound kernels (pass flops, bytes = 0).
inline void reportPerf(const char* name,
                       double msPerIter,
                       double bytes,
                       double flops)
{
    std::printf("%-28s %9.4f ms/iter", name, msPerIter);
    if (bytes > 0.0) {
        std::printf("  %7.1f GB/s", bytes / (msPerIter * 1.0e-3) / 1.0e9);
    }
    if (flops > 0.0) {
        std::printf("  %8.1f GFLOP/s", flops / (msPerIter * 1.0e-3) / 1.0e9);
    }
    std::printf("\n");
}
