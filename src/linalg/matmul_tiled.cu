#include "linalg/matmul.h"
#include<cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void matMulSharedTiledKernel( const float* A, 
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

__global__ void matMulSharedRegisterTiledKernel( const float* A, 
                                      const float* B, 
                                            float* C, 
                                            int A_rows, 
                                            int A_cols, 
                                            int B_cols )
{
    __shared__ float Ads[2 * TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bds[TILE_WIDTH][TILE_WIDTH * 2];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row0 = blockIdx.y * ( 2 * TILE_WIDTH ) + ty;
    int row1 = row0 + TILE_WIDTH;
    int col0 = blockIdx.x * ( 2 * TILE_WIDTH ) + tx;
    int col1 = col0 + TILE_WIDTH;

    float c00 = 0.0f, c01 = 0.0f, c10 = 0.0f, c11 = 0.0f;
    int numTiles = ((A_cols + TILE_WIDTH - 1) / TILE_WIDTH);
    for(int t = 0; t < numTiles; t++){
        int tileCol = t * TILE_WIDTH + tx;
        int tileRow = t * TILE_WIDTH + ty;  
        Ads[ty][tx] = row0 < A_rows && tileCol < A_cols ? A[row0 * A_cols + tileCol] : 0.0f;
        Ads[ty + TILE_WIDTH][tx] = row1 < A_rows && tileCol < A_cols ? A[row1 * A_cols + tileCol] : 0.0f;
        Bds[ty][tx] = tileRow < A_cols && col0 < B_cols ? B[tileRow * B_cols + col0] : 0.0f;
        Bds[ty][tx + TILE_WIDTH] = tileRow < A_cols && col1 < B_cols ? B[tileRow * B_cols + col1] : 0.0f;
        __syncthreads();

        for ( int k = 0; k < TILE_WIDTH; k++){
            c00 += Ads[ty][k] * Bds[k][tx];
            c01 += Ads[ty][k] * Bds[k][tx + TILE_WIDTH];
            c10 += Ads[ty + TILE_WIDTH][k] * Bds[k][tx];
            c11 += Ads[ty + TILE_WIDTH][k] * Bds[k][tx + TILE_WIDTH];
        }
        __syncthreads();
    }
    if ( row0 < A_rows && col0 < B_cols ){
        C[row0 * B_cols + col0] = c00;
    }
    if ( row0 < A_rows && col1 < B_cols ){
        C[row0 * B_cols + col1] = c01;
    }
    if ( row1 < A_rows && col0 < B_cols ){
        C[row1 * B_cols + col0] = c10;
    }
    if ( row1 < A_rows && col1 < B_cols ){
        C[row1 * B_cols + col1] = c11;
    }
}



void launchMatMulSharedTiled( const float* A, 
                              const float* B, 
                                    float* C, 
                                    int A_rows, 
                                    int A_cols, 
                                    int B_cols )
{
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize( (B_cols + TILE_WIDTH - 1) / TILE_WIDTH, 
                   (A_rows + TILE_WIDTH - 1) / TILE_WIDTH );

    matMulSharedTiledKernel<<<gridSize, blockSize>>>( A, B, C, A_rows, A_cols, B_cols );
}

void launchMatMulRegisterTiled( const float* A, 
                             const float* B, 
                                   float* C, 
                                   int A_rows, 
                                   int A_cols, 
                                   int B_cols )
{
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize( (B_cols + (2 * TILE_WIDTH) - 1) / (2 * TILE_WIDTH), 
                   (A_rows + (2 * TILE_WIDTH) - 1) / (2 * TILE_WIDTH) );

    matMulSharedRegisterTiledKernel<<<gridSize, blockSize>>>( A, B, C, A_rows, A_cols, B_cols );
}
