#include "linalg/transpose.h"
#include <cuda_runtime.h>

#define TILE 32

// ===========================================================================
// Naive transpose: one element per thread.
// Goal: reads of A coalesced, writes to B strided (the baseline problem).
// ===========================================================================
__global__ void transposeNaiveKernel(const float* A, float* B, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if ( row < rows && col < cols )
        B[col * rows + row] = A[row * cols + col];
}
// ===========================================================================
// Tiled transpose: stage the swap through __shared__ float tile[TILE][TILE].
// Coalesced load -> __syncthreads() -> coalesced store reading the tile
// transposed (swap blockIdx.x/y for the output coords). Will hit bank
// conflicts on the transposed shared read.
// ===========================================================================
__global__ void transposeTiledKernel(const float* A, float* B, int rows, int cols)
{
    __shared__ float tile[TILE][TILE];
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    // Load matrix into the tile
    if ( row < rows && col < cols )
        tile[ty][tx] = A[row * cols + col];

    __syncthreads();
    int out_col = blockIdx.y * blockDim.y + tx;
    int out_row = blockIdx.x * blockDim.x + ty;

    if( out_row < cols && out_col < rows)
        B[out_row * rows + out_col] = tile[tx][ty];
}

// ===========================================================================
// Padded transpose: same as tiled, but __shared__ float tile[TILE][TILE + 1]
// to make the transposed shared read conflict-free.
// ===========================================================================
__global__ void transposePaddedKernel(const float* A, float* B, int rows, int cols)
{
    __shared__ float tile[TILE][TILE + 1];
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    if( row < rows && col < cols )
        tile[ty][tx] = A[row * cols + col];

    __syncthreads();

    int out_col = blockIdx.y * blockDim.y + tx;
    int out_row = blockIdx.x * blockDim.x + ty;

    if( out_row < cols && out_col < rows)
        B[out_row * rows + out_col] = tile[tx][ty];
}

// ---- launchers (wired; you implement the kernels above) --------------------
static dim3 gridFor(int rows, int cols)
{
    return dim3((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);
}

void launchTransposeNaive(const float* d_A, float* d_B, int rows, int cols)
{
    dim3 block(TILE, TILE);
    transposeNaiveKernel<<<gridFor(rows, cols), block>>>(d_A, d_B, rows, cols);
}

void launchTransposeTiled(const float* d_A, float* d_B, int rows, int cols)
{
    dim3 block(TILE, TILE);
    transposeTiledKernel<<<gridFor(rows, cols), block>>>(d_A, d_B, rows, cols);
}

void launchTransposePadded(const float* d_A, float* d_B, int rows, int cols)
{
    dim3 block(TILE, TILE);
    transposePaddedKernel<<<gridFor(rows, cols), block>>>(d_A, d_B, rows, cols);
}
