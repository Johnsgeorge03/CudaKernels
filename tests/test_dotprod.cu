#include "core/cuda_utils.h"
#include "linalg/dotprod.h"
#include "test_utils.h"

#include <cstdio>
#include <vector>

// Reference dot product in double precision so the CPU value is
// essentially exact for the relative-error comparison.
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
    const int N = 1 << 24;   // ~16.7M elements

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

    bool passed = checkScalar(h_C, ref);
    std::printf("Dot product (N = %d): %s\n", N, passed ? "PASSED" : "FAILED");

    // ---- Performance -------------------------------------------------------
    // Memory-bound kernel: report effective bandwidth (reads A and B once).
    double ms = benchmarkMs([&] { launchDotProd(d_A, d_B, d_C, N); });
    reportPerf("dotprod", ms, 2.0 * N * sizeof(float), 0.0);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return passed ? 0 : 1;
}
