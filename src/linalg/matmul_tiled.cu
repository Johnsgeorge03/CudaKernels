#include "linalg/matmul.h"
#include<cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void matMulTiledKernel( const float* A, 
                                   const float* B, 
                                         float* C, 
                                         int A_rows, 
                                         int A_cols, 
                                         int B_cols )
{
    __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    float value = 0.0f;
    int numTiles = (A_cols + TILE_WIDTH - 1) / TILE_WIDTH;

    for ( int t = 0; t < numTiles; t++){
        int A_col = t * TILE_WIDTH + tx;
        int B_row = t * TILE_WIDTH + ty;

        Ads[ty][tx] = ( row < A_rows && A_col < A_cols ) ? A[row * A_cols + A_col] : 0.0f;
        Bds[ty][tx] = ( B_row < A_cols && col < B_cols ) ? B[B_row * B_cols + col] : 0.0f;

        __syncthreads();

        for ( int k = 0; k < TILE_WIDTH; k++){
            value += Ads[ty][k] * Bds[k][tx];
        }
        __syncthreads();
    }
    if ( row < A_rows && col < B_cols ){
        C[row * B_cols + col] = value;
    }
}

void launchMatMulTiled( const float* A, 
                             const float* B, 
                                   float* C, 
                                   int A_rows, 
                                   int A_cols, 
                                   int B_cols )
{
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize( (B_cols + TILE_WIDTH - 1) / TILE_WIDTH, 
                   (A_rows + TILE_WIDTH - 1) / TILE_WIDTH );

    matMulTiledKernel<<<gridSize, blockSize>>>( A, B, C, A_rows, A_cols, B_cols );
}