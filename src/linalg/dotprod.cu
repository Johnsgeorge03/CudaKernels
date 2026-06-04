#include "linalg/dotprod.h"
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void dotProdKernel(const float* A, 
                              const float* B, 
                                float* C,
                                int N)
{

}

void launchDotProd(const float* d_A,
                    const float* d_B, 
                    float* d_C, 
                    int N)
{
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dotProdKernel<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_C, N);
}