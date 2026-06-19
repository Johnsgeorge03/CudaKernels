#include "scan/scan.h"
#include <cuda_runtime.h>

// ============================================================================
// Launch geometry
// ============================================================================
// Single-block Hillis-Steele: one element per thread, so capacity == threads.
#define SCAN_BLOCK 1024

// Blelloch processes two elements per thread, so a block of BLELLOCH_THREADS
// scans up to 2 * BLELLOCH_THREADS elements.
#define BLELLOCH_THREADS 512
#define BLELLOCH_CAP (2 * BLELLOCH_THREADS)   // 1024 elements per block

// Multi-block path: each block inclusive-scans SCAN_BLOCK elements.
#define ELEMENTS_PER_BLOCK SCAN_BLOCK

int scanNumBlocks(int N) {
    return (N + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
}

// ============================================================================
// Algorithm 1 — Hillis-Steele inclusive scan (single block, O(N log N) work)
// ============================================================================
// Plan: load x into shared memory, then for d = 1, 2, 4, ... < N do
//   temp = (tid >= d) ? s[tid] + s[tid - d] : s[tid];
//   __syncthreads(); s[tid] = temp; __syncthreads();
// The read-into-register-then-write (with syncs) avoids the in-place race where
// one thread overwrites s[tid] before its neighbour has read it. Write s[tid]
// back to out[tid] at the end. Inclusive: out[i] includes x[i].
__global__ void scanHillisSteeleKernel(const float* in, float* out, int N) {
    __shared__ float s[SCAN_BLOCK];
    int tid = threadIdx.x;
    if(tid < N)
        s[tid] = in[tid];
    __syncthreads();

    float temp = 0;
    for(int d = 1; d < N; d *= 2){
        temp = tid - d >= 0 ? s[tid] + s[tid - d] : s[tid];
        __syncthreads();
        s[tid] = temp;
        __syncthreads();
    }
    if( tid < N)
        out[tid] = s[tid];
}

// ============================================================================
// Algorithm 2 — Blelloch work-efficient EXCLUSIVE scan (single block, O(N))
// ============================================================================
// Two elements per thread. Up-sweep (reduce) builds partial sums up the tree;
// then clear the last element to 0 and down-sweep, swapping at each node:
//   t = s[left]; s[left] = s[right]; s[right] += t;
// Strides double (offset *= 2) on up-sweep and halve on down-sweep, with a
// __syncthreads() between levels. Result is an EXCLUSIVE scan. (Stretch goal:
// pad shared indices to dodge the bank conflicts these 2^d strides create.)
__global__ void scanBlellochKernel(const float* in, float* out, int N) {
    __shared__ float s[BLELLOCH_CAP];
    int tid = threadIdx.x;

    // TODO (you): cooperatively load two elements per thread into s[] (guard
    // against N < BLELLOCH_CAP by zero-filling the tail), run the up-sweep,
    // zero the last slot, run the down-sweep, then write s[] back to out[].
    (void)s; (void)tid; (void)in; (void)out; (void)N;
}

// ============================================================================
// Algorithm 3 — Multi-block inclusive scan (three passes)
// ============================================================================
// Pass 1: each block inclusive-scans its own ELEMENTS_PER_BLOCK chunk into out,
// and writes that chunk's TOTAL (the last scanned value) into blockSums[blockIdx].
// You can reuse the Hillis-Steele logic here; just index off a per-block base
// and emit the block total. Guard the tail block where base+tid >= N.
__global__ void scanBlockInclusiveKernel(const float* in, float* out,
                                         float* blockSums, int N) 
{
    __shared__ float s[ELEMENTS_PER_BLOCK];
    int tid  = threadIdx.x;
    int base = blockIdx.x * ELEMENTS_PER_BLOCK;

    s[tid] = ( base + tid < N ) ? in[base + tid] : 0.0f; // partial block padding
    __syncthreads();

    float temp = 0.0f;
    for( int d = 1; d < ELEMENTS_PER_BLOCK; d *= 2){
        temp = ( tid - d >= 0 ) ? s[tid] + s[tid - d] : s[tid];
        __syncthreads();
        s[tid] = temp;
        __syncthreads();
    }
    if( base + tid < N )
        out[base + tid] = s[tid];

    if( tid == 0 ) blockSums[blockIdx.x] = s[ELEMENTS_PER_BLOCK - 1];
}

// Pass 2: EXCLUSIVE-scan blockSums in place, so blockSums[b] becomes the sum of
// all chunks before block b — i.e. the offset block b must add to every element.
// One block; assumes numBlocks <= SCAN_BLOCK (see header note).
__global__ void scanBlockSumsKernel(float* blockSums, int numBlocks) {
    __shared__ float s[SCAN_BLOCK];
    int tid = threadIdx.x;

    s[tid] = ( tid < numBlocks ) ? blockSums[tid] : 0.0f;
    __syncthreads();

    float temp = 0.0f;
    for( int d = 1; d < ELEMENTS_PER_BLOCK; d *= 2 )
    {
        temp = (tid - d >= 0 )? s[tid] + s[tid - d] : s[tid];
        __syncthreads();
        s[tid] = temp;
        __syncthreads();
    }

    if ( tid < numBlocks )
        blockSums[tid] = ( tid == 0 ) ? 0.0f : s[tid - 1];

}

// Pass 3: broadcast each block's offset back onto its chunk: out[i] += offset.
// Block 0 adds 0, so this is a no-op for it but harmless to run.
__global__ void addBlockOffsetsKernel(float* out, const float* blockSums, int N) {
    int idx = blockIdx.x * ELEMENTS_PER_BLOCK + threadIdx.x;
    if ( idx < N )
        out[idx] += blockSums[blockIdx.x];
}

// ============================================================================
// Launchers (infra — wired for you)
// ============================================================================
void launchScanHillisSteele(const float* d_in, float* d_out, int N) {
    scanHillisSteeleKernel<<<1, SCAN_BLOCK>>>(d_in, d_out, N);
}

void launchScanBlelloch(const float* d_in, float* d_out, int N) {
    scanBlellochKernel<<<1, BLELLOCH_THREADS>>>(d_in, d_out, N);
}

void launchScanFull(const float* d_in, float* d_out, float* d_blockSums, int N) {
    int numBlocks = scanNumBlocks(N);

    // Pass 1: scan each chunk, collect per-block totals.
    scanBlockInclusiveKernel<<<numBlocks, ELEMENTS_PER_BLOCK>>>(
        d_in, d_out, d_blockSums, N);

    // Pass 2: scan the totals into per-block offsets (single block).
    scanBlockSumsKernel<<<1, SCAN_BLOCK>>>(d_blockSums, numBlocks);

    // Pass 3: add each block's offset back onto its chunk.
    addBlockOffsetsKernel<<<numBlocks, ELEMENTS_PER_BLOCK>>>(
        d_out, d_blockSums, N);
}
