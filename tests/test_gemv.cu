#include "core/cuda_utils.h"
#include "linalg/gemv.h"
#include "test_utils.h"

#include <cstdio>
#include <vector>

void gemvCPU(const float* A, const float* x, float* y, int M, int N)
{
    for (int r = 0; r < M; ++r) {
        float sum = 0.0f;
        for (int c = 0; c < N; ++c) {
            sum += A[r * N + c] * x[c];
        }
        y[r] = sum;
    }
}

int main()
{
    // M, N multiples of 32 so the warp mapping divides cleanly. N is the
    // reduction length; keep it modest enough that x fits in shared memory
    // for the shared-x variant (N floats <= 48 KB => N <= 12288).
    int M = 4096;
    int N = 4096;

    std::vector<float> h_A(static_cast<size_t>(M) * N);
    std::vector<float> h_x(N);
    std::vector<float> h_y(M);
    std::vector<float> h_ref(M);

    // Small bounded values: the warp/CPU reductions sum in different orders, so
    // results differ by float rounding. Keeping |A|,|x| tiny bounds |y| (~tens)
    // so that error stays well under the absolute 1e-3 compare, while an
    // indexing bug still shifts terms enough to trip it.
    for (size_t i = 0; i < h_A.size(); ++i)
        h_A[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.01f;
    for (int c = 0; c < N; ++c)
        h_x[c] = static_cast<float>((c % 7) - 3) * 0.1f;

    gemvCPU(h_A.data(), h_x.data(), h_ref.data(), M, N);

    float *d_A, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_A, h_A.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, M * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * sizeof(float),
                          cudaMemcpyHostToDevice));

    struct Variant {
        const char* name;
        void (*launch)(const float*, const float*, float*, int, int);
    };

    const Variant variants[] = {
        { "gemv naive",      launchGemvNaive },
        { "gemv warp",       launchGemvWarp },
        { "gemv warp+smemX", launchGemvWarpSharedX },
    };

    std::printf("M = %d, N = %d, copy-bandwidth ceiling ~192 GB/s\n", M, N);

    // A (M*N) dominates; x is read once-per-row in principle but counted once
    // here as the streamed footprint, plus y written.
    double bytes = (static_cast<double>(M) * N + N + M) * sizeof(float);

    bool allPassed = true;
    for (const Variant& v : variants) {
        CUDA_CHECK(cudaMemset(d_y, 0, M * sizeof(float)));
        v.launch(d_A, d_x, d_y, M, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, M * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // Reduction order differs from the CPU; small absolute slack (see the
        // input-scaling note above for why this stays safe yet bug-sensitive).
        bool passed = compareResults(h_y.data(), h_ref.data(), M, 1e-3f);
        if (passed) {
            double ms = benchmarkMs([&] { v.launch(d_A, d_x, d_y, M, N); });
            reportPerf(v.name, ms, bytes, 0.0);
        } else {
            std::printf("%-28s FAILED (not implemented yet?)\n", v.name);
        }
        allPassed = allPassed && passed;
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    return allPassed ? 0 : 1;
}
