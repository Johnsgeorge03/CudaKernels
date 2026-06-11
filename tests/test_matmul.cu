#include "core/cuda_utils.h"
#include "linalg/matmul.h"
#include "test_utils.h"

#include <cstdio>
#include <vector>

void matMulCPU(const float* A,
               const float* B,
               float* C,
               int A_rows,
               int A_cols,
               int B_cols)
{
    for (int row = 0; row < A_rows; ++row) {
        for (int col = 0; col < B_cols; ++col) {
            float sum = 0.0f;

            for (int k = 0; k < A_cols; ++k) {
                sum += A[row * A_cols + k] * B[k * B_cols + col];
            }

            C[row * B_cols + col] = sum;
        }
    }
}

int main()
{
    // ---- Correctness (small size, verified against CPU) --------------------
    int A_rows = 128;
    int A_cols = 96;
    int B_cols = 64;

    int sizeA = A_rows * A_cols;
    int sizeB = A_cols * B_cols;
    int sizeC = A_rows * B_cols;

    std::vector<float> h_A(sizeA);
    std::vector<float> h_B(sizeB);
    std::vector<float> h_C(sizeC);
    std::vector<float> h_C_ref(sizeC);

    for (int i = 0; i < sizeA; ++i) {
        h_A[i] = static_cast<float>((i % 7) + 1);
    }

    for (int i = 0; i < sizeB; ++i) {
        h_B[i] = static_cast<float>((i % 5) + 1);
    }

    float *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMalloc(&d_A, sizeA * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, sizeC * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeA * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), sizeB * sizeof(float),
                          cudaMemcpyHostToDevice));

    matMulCPU(h_A.data(), h_B.data(), h_C_ref.data(),
              A_rows, A_cols, B_cols);

    struct Variant {
        const char* name;
        void (*launch)(const float*, const float*, float*, int, int, int);
    };

    const Variant variants[] = {
        { "matmul naive",          launchMatMulNaive },
        { "matmul shared tiled",   launchMatMulSharedTiled },
        { "matmul register tiled", launchMatMulRegisterTiled },
    };

    bool allPassed = true;
    for (const Variant& v : variants) {
        v.launch(d_A, d_B, d_C, A_rows, A_cols, B_cols);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, sizeC * sizeof(float),
                              cudaMemcpyDeviceToHost));

        bool passed = compareResults(h_C.data(), h_C_ref.data(), sizeC);
        std::printf("%-28s %s\n", v.name, passed ? "PASSED" : "FAILED");
        allPassed = allPassed && passed;
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // ---- Performance (larger size; correctness already verified above) -----
    // Compute-bound kernel: report GFLOP/s (2 * M * K * N flops per matmul).
    int M = 1024, K = 1024, N = 1024;

    std::vector<float> h_PA(M * K);
    std::vector<float> h_PB(K * N);
    for (int i = 0; i < M * K; ++i) h_PA[i] = static_cast<float>((i % 7) + 1);
    for (int i = 0; i < K * N; ++i) h_PB[i] = static_cast<float>((i % 5) + 1);

    float *d_PA, *d_PB, *d_PC;
    CUDA_CHECK(cudaMalloc(&d_PA, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_PB, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_PC, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_PA, h_PA.data(), M * K * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_PB, h_PB.data(), K * N * sizeof(float),
                          cudaMemcpyHostToDevice));

    double flops = 2.0 * M * K * N;
    for (const Variant& v : variants) {
        double ms = benchmarkMs(
            [&] { v.launch(d_PA, d_PB, d_PC, M, K, N); }, 10, 3);
        reportPerf(v.name, ms, 0.0, flops);
    }

    CUDA_CHECK(cudaFree(d_PA));
    CUDA_CHECK(cudaFree(d_PB));
    CUDA_CHECK(cudaFree(d_PC));

    return allPassed ? 0 : 1;
}
