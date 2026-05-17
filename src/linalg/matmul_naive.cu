#include "linalg/matmul.h"
#include<cuda_runtime.h>

__global__ void matMulNaiveKernel( const float* A, 
                                   const float* B, 
                                         float* C, 
                                         int A_rows, 
                                         int A_cols, 
                                         int B_cols )
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ( row < A_rows && col < B_cols ) {
        float value = 0.0f;
        for ( int k = 0; k < A_cols; k++) {
            value += A[row * A_cols + k] * B[k * B_cols + col];
        }
        C[row * B_cols + col] = value;
    }
}

void launchMatMulNaive( const float* A,
                        const float* B,
                              float* C,
                              int A_rows,
                              int A_cols,
                              int B_cols )
{
    dim3 blockSize(16, 16);
    dim3 gridSize( (B_cols + blockSize.x - 1) / blockSize.x, 
                   (A_rows + blockSize.y - 1) / blockSize.y );

    matMulNaiveKernel<<<gridSize, blockSize>>>(A, B, C, A_rows, A_cols, B_cols);
}