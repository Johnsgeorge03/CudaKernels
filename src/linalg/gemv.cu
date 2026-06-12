#include "linalg/gemv.h"
#include <cuda_runtime.h>

// One warp per row reduces 32 partial sums into one output. Same helper you
// wrote for dotprod — reused verbatim.
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Naive launch geometry: one thread per output row.
#define NAIVE_BLOCK 256

// Warp-per-row launch geometry: WARPS_PER_BLOCK rows handled per block.
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4
#define WARP_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)   // threads per block

// ----------------------------------------------------------------------------
// Strategy A: thread-per-row. Thread t computes y[t] by walking row t. Within a
// warp the loads are stride-N (uncoalesced) — this is the slow baseline.
// ----------------------------------------------------------------------------
__global__ void gemvNaiveKernel(const float* A, const float* x, float* y,
                                int M, int N) 
{
    int tx  = threadIdx.x;
    float sum = 0;
    int row = blockDim.x * blockIdx.x + tx;
    for( int i = 0; i < N; i++){
        if( row < M )
            sum += A[row * N + i] * x[i];
    }
    if( row < M )
        y[row] = sum;
}

// ----------------------------------------------------------------------------
// Strategy B: warp-per-row. A warp owns one row; lane l strides the columns by
// 32 (coalesced), then warpReduceSum collapses the 32 partials. Lane 0 writes.
// ----------------------------------------------------------------------------
__global__ void gemvWarpKernel(const float* A, const float* x, float* y,
                               int M, int N) 
{
    int tx = threadIdx.x;
    int wid = blockIdx.x * WARPS_PER_BLOCK + ( tx >> 5 );
    int lane = threadIdx.x % 32;
    float sum = 0;
    for ( int i = lane; i < N; i += 32)
        if( wid < M)
            sum += A[wid * N + i] * x[i];
    sum = warpReduceSum( sum );

    if( lane == 0 && wid < M ) y[wid] = sum;
}

// ----------------------------------------------------------------------------
// Strategy C: warp-per-row with x staged in shared memory (dynamic, N floats).
// Cooperatively load x into smem, __syncthreads(), then proceed like B but read
// x from smem.
// ----------------------------------------------------------------------------
__global__ void gemvWarpSharedXKernel(const float* A, const float* x, float* y,
                                      int M, int N) 
{
    extern __shared__ float xs[];   // N floats
    // TODO (you): cooperatively copy x[0..N) into xs using all blockDim.x
    // threads (grid-stride over N), __syncthreads(), then the warp-per-row
    // reduction reading xs[c] instead of x[c].
    int tx = threadIdx.x;
    for( int i = tx; i < N; i += blockDim.x )
        xs[i] = x[i];
    
    __syncthreads();
    int wid = blockIdx.x * WARPS_PER_BLOCK + ( threadIdx.x >> 5 );
    float sum = 0; 
    int lane = threadIdx.x % 32;
    for ( int i = lane; i < N; i += 32)
        if ( wid < M )
            sum += A[wid * N + i] * xs[i];
    sum = warpReduceSum(sum);

    if( lane == 0 && wid < M )y[wid] = sum;
}

// ----------------------------------------------------------------------------
// Launchers (infra — wired for you).
// ----------------------------------------------------------------------------
void launchGemvNaive(const float* d_A, const float* d_x, float* d_y,
                     int M, int N) {
    int grid = (M + NAIVE_BLOCK - 1) / NAIVE_BLOCK;
    gemvNaiveKernel<<<grid, NAIVE_BLOCK>>>(d_A, d_x, d_y, M, N);
}

void launchGemvWarp(const float* d_A, const float* d_x, float* d_y,
                    int M, int N) {
    int grid = (M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;  // rows per block
    gemvWarpKernel<<<grid, WARP_BLOCK>>>(d_A, d_x, d_y, M, N);
}

void launchGemvWarpSharedX(const float* d_A, const float* d_x, float* d_y,
                           int M, int N) {
    int grid = (M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    size_t smem = static_cast<size_t>(N) * sizeof(float);   // x cached in smem
    gemvWarpSharedXKernel<<<grid, WARP_BLOCK, smem>>>(d_A, d_x, d_y, M, N);
}
