#include "core/cuda_utils.h"
#include "linalg/dotprod.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Reference dot product in double precision so the CPU value is essentially
// exact; we then compare the GPU float result against it with a relative
// tolerance (float reduction reorders additions, so bit-equality is wrong).
double dotProdCPU(const float* A, const float* B, int N)
{
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        sum += static_cast<double>(A[i]) * static_cast<double>(B[i]);
    }
    return sum;
}

int main()
{
    const int N = 1 << 24;            // ~16.7M elements
    const int iters = 100;            // timing iterations

    std::vector<float> h_A(N);
    std::vector<float> h_B(N);

    // Small bounded values keep the running sum well-conditioned so the
    // float GPU result stays close to the double reference.
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>((i % 7) + 1) * 0.5f;
        h_B[i] = static_cast<float>((i % 5) + 1) * 0.25f;
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), N * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * sizeof(float),
                          cudaMemcpyHostToDevice));

    // ---- Correctness -------------------------------------------------------
    double ref = dotProdCPU(h_A.data(), h_B.data(), N);

    launchDotProd(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_C = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_C, d_C, sizeof(float), cudaMemcpyDeviceToHost));

    double relErr = std::fabs(h_C - ref) / std::fabs(ref);
    bool passed = relErr < 1e-4;

    std::printf("Dot product: GPU = %.4f, CPU = %.4f, relErr = %.2e -> %s\n",
                h_C, ref, relErr, passed ? "PASSED" : "FAILED");

    // ---- Bandwidth ---------------------------------------------------------
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    launchDotProd(d_A, d_B, d_C, N);     // warmup
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iters; ++it) {
        launchDotProd(d_A, d_B, d_C, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    double msPerIter = ms / iters;

    // Read A and read B: 2 floats moved per element.
    double bytes = 2.0 * static_cast<double>(N) * sizeof(float);
    double gbPerSec = (bytes / (msPerIter * 1.0e-3)) / 1.0e9;

    std::printf("N = %d, %.4f ms/iter, %.1f GB/s effective bandwidth\n",
                N, msPerIter, gbPerSec);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return passed ? 0 : 1;
}
