#include "core/cuda_utils.h"
#include "linalg/transpose.h"
#include "test_utils.h"

#include <cstdio>
#include <vector>

void transposeCPU(const float* A, float* B, int rows, int cols)
{
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            B[c * rows + r] = A[r * cols + c];
        }
    }
}

int main()
{
    // Rectangular (and both multiples of 32) so a row/col mix-up is caught
    // and the bandwidth math stays clean.
    int rows = 2048;
    int cols = 4096;
    int n = rows * cols;

    std::vector<float> h_A(n);
    std::vector<float> h_B(n);
    std::vector<float> h_ref(n);

    // Distinct values: any indexing bug shows up as a mismatch. n < 2^24, so
    // every index is exactly representable as a float (transpose is exact).
    for (int i = 0; i < n; ++i) {
        h_A[i] = static_cast<float>(i);
    }
    transposeCPU(h_A.data(), h_ref.data(), rows, cols);

    float *d_A, *d_B;
    CUDA_CHECK(cudaMalloc(&d_A, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));

    struct Variant {
        const char* name;
        void (*launch)(const float*, float*, int, int);
    };

    const Variant variants[] = {
        { "transpose naive",  launchTransposeNaive },
        { "transpose tiled",  launchTransposeTiled },
        { "transpose padded", launchTransposePadded },
    };

    std::printf("rows = %d, cols = %d, copy-bandwidth ceiling ~192 GB/s\n",
                rows, cols);

    bool allPassed = true;
    double bytes = 2.0 * n * sizeof(float);   // read A + write B

    for (const Variant& v : variants) {
        CUDA_CHECK(cudaMemset(d_B, 0, n * sizeof(float)));
        v.launch(d_A, d_B, rows, cols);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_B.data(), d_B, n * sizeof(float),
                              cudaMemcpyDeviceToHost));

        bool passed = compareResults(h_B.data(), h_ref.data(), n, 0.0f);
        if (passed) {
            double ms = benchmarkMs([&] { v.launch(d_A, d_B, rows, cols); });
            reportPerf(v.name, ms, bytes, 0.0);
        } else {
            std::printf("%-28s FAILED (not implemented yet?)\n", v.name);
        }
        allPassed = allPassed && passed;
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));

    return allPassed ? 0 : 1;
}
