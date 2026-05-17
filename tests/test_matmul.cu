#include "core/cuda_utils.h"
#include "linalg/matmul.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
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

bool compareResults(const float* gpu,
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

int main()
{
    int A_rows = 128;
    int A_cols = 96;
    int B_cols = 64;

    int sizeA = A_rows * A_cols;
    int sizeB = A_cols * B_cols;
    int sizeC = A_rows * B_cols;

    std::vector<float> h_A(sizeA);
    std::vector<float> h_B(sizeB);
    std::vector<float> h_C_naive(sizeC);
    std::vector<float> h_C_tiled(sizeC);
    std::vector<float> h_C_register_tiled(sizeC);
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

    launchMatMulNaive(d_A, d_B, d_C, A_rows, A_cols, B_cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C_naive.data(), d_C, sizeC * sizeof(float),
                          cudaMemcpyDeviceToHost));

    bool naivePassed = compareResults(h_C_naive.data(),
                                      h_C_ref.data(),
                                      sizeC);

    launchMatMulSharedTiled(d_A, d_B, d_C, A_rows, A_cols, B_cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C_tiled.data(), d_C, sizeC * sizeof(float),
                          cudaMemcpyDeviceToHost));

    bool tiledPassed = compareResults(h_C_tiled.data(),
                                      h_C_ref.data(),
                                      sizeC);

    launchMatMulRegisterTiled(d_A, d_B, d_C, A_rows, A_cols, B_cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C_register_tiled.data(), d_C, sizeC * sizeof(float),
                          cudaMemcpyDeviceToHost));

    bool registerTiledPassed = compareResults(h_C_register_tiled.data(),
                                              h_C_ref.data(),
                                              sizeC);

    std::printf("Naive matmul: %s\n", naivePassed ? "PASSED" : "FAILED");
    std::printf("Tiled matmul: %s\n", tiledPassed ? "PASSED" : "FAILED");
    std::printf("Register-tiled matmul: %s\n", registerTiledPassed ? "PASSED" : "FAILED");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return (naivePassed && tiledPassed && registerTiledPassed) ? 0 : 1;
}
