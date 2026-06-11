#include "linalg/dotprod.h"
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__inline__ __device__ float warpReduceSum(float val){
    for( int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__inline__ __device__ float blockReduceSum( float val ){
    __shared__ float warpSums[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    val = warpReduceSum( val );

    if ( lane == 0 ) warpSums[wid] = val;
    __syncthreads();

    int numWarps = ( blockDim.x + 31 ) / 32;
    val = ( threadIdx.x < numWarps ) ? warpSums[lane] : 0.0f;
    if( wid == 0) val = warpReduceSum( val );
    return val;
}

__global__ void dotProdKernel(const float* A, 
                              const float* B, 
                                float* C,
                                int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    float sum = 0.0f;
    for ( int i = idx; i < N; i += stride ){
        sum += A[i] * B[i];
    }

    sum = blockReduceSum( sum );
    if( threadIdx.x == 0) atomicAdd(C, sum);

}

void launchDotProd(const float* d_A,
                    const float* d_B, 
                    float* d_C, 
                    int N)
{
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if ( gridSize > 1024 ) gridSize = 1024;

    cudaMemset(d_C, 0, sizeof(float));   // accumulator must start at 0 for atomicAdd
    dotProdKernel<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_C, N);
}