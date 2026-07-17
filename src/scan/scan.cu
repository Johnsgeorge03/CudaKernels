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

// Bank-conflict avoidance. Shared memory has 32 banks; word i lives in bank
// i % 32. The scan's 2^d strides make consecutive active threads collide — at
// offset 16 the stride is 32, so a whole warp hits ONE bank (32-way serialized).
// Fix: insert one pad word every 32 real words, so logical index i lives at
// i + (i >> 5). That nudges each 32-word run into a different starting bank,
// turning a colliding stride into a coprime one (same idea as tile[32][33]).
#define LOG_NUM_BANKS 5                               // 32 banks
#define CONFLICT_FREE_OFFSET(i) ((i) >> LOG_NUM_BANKS)
#define PADDED_CAP (ELEMENTS_PER_BLOCK + (ELEMENTS_PER_BLOCK >> LOG_NUM_BANKS))

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
    int n = BLELLOCH_CAP;

    // Load elements with zero padding
    s[2*tid] = (2*tid < N ) ? in[2*tid] : 0.0f;
    s[2*tid + 1] = ( 2*tid + 1 < N) ? in[2*tid + 1] : 0.0f;

    // Reduce
    int offset = 1;
    for( int d = n >> 1; d > 0; d >>= 1){
        __syncthreads();
        if( tid < d){
            int l = offset*(2*tid + 1) - 1;
            int r = offset*(2*tid + 2) - 1;
            s[r] += s[l];
        }
        offset <<= 1;
    }

    if( tid == 0 ) s[n - 1] = 0.0f;

    // Down-sweep: active threads do the swap-and-add.
    for (int d = 1; d < n; d <<= 1) {
        offset >>= 1;                    // stride halves each level
        __syncthreads();
        if (tid < d) {
            int l = offset*(2*tid + 1) - 1;
            int r = offset*(2*tid + 2) - 1;
            float t = s[l];
            s[l] = s[r];
            s[r] += t;
        }
    }
    __syncthreads();

    // Write back, guarded against the padded tail.
    if (2*tid     < N) out[2*tid]     = s[2*tid];
    if (2*tid + 1 < N) out[2*tid + 1] = s[2*tid + 1];
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
// Optimization — Blelloch (work-efficient) per-block scan for pass 1
// ============================================================================
// Drop-in replacement for scanBlockInclusiveKernel: same contract (inclusive
// scan of each ELEMENTS_PER_BLOCK chunk into out, chunk total -> blockSums[bid]),
// but O(N) up/down-sweep instead of O(N log N) Hillis-Steele. Two wrinkles vs
// your single-block scanBlellochKernel:
//   (1) Blelloch is EXCLUSIVE, but the multi-block result must stay INCLUSIVE
//       (pass 3 assumes it). Convert at the end: out[i] = exclusive[i] + in[base+i].
//   (2) The chunk TOTAL is s[n-1] right AFTER the up-sweep, BEFORE you clear it.
//       Grab it into blockSums[blockIdx.x] at that moment (exclusive scan would
//       otherwise discard the total).
// Launched with BLELLOCH_THREADS threads (2 elements/thread), numBlocks blocks.
__global__ void scanBlockBlellochKernel(const float* in, float* out,
                                        float* blockSums, int N) {
    __shared__ float s[ELEMENTS_PER_BLOCK];
    int tid  = threadIdx.x;
    int n    = ELEMENTS_PER_BLOCK;
    int base = blockIdx.x * ELEMENTS_PER_BLOCK;

    // 1. Two elements per thread (512 threads cover 1024). Keep them in
    //    registers (a, b) so the exclusive->inclusive fix-up in step 5 doesn't
    //    re-read global memory. Zero-pad the tail past N.
    int gA = base + 2 * tid;
    int gB = base + 2 * tid + 1;
    float a = (gA < N) ? in[gA] : 0.0f;
    float b = (gB < N) ? in[gB] : 0.0f;
    s[2 * tid]     = a;
    s[2 * tid + 1] = b;

    // 2. Up-sweep (reduce): identical to the single-block kernel.
    int offset = 1;
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int l = offset * (2 * tid + 1) - 1;
            int r = offset * (2 * tid + 2) - 1;
            s[r] += s[l];
        }
        offset <<= 1;
    }

    // 3. Right now s[n-1] holds the whole-chunk total. The clear is about to
    //    wipe it, so stash it into blockSums FIRST (same thread does both).
    if (tid == 0) {
        blockSums[blockIdx.x] = s[n - 1];
        s[n - 1] = 0.0f;
    }

    // 4. Down-sweep -> exclusive scan in s[].
    for (int d = 1; d < n; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int l = offset * (2 * tid + 1) - 1;
            int r = offset * (2 * tid + 2) - 1;
            float t = s[l];
            s[l] = s[r];
            s[r] += t;
        }
    }
    __syncthreads();

    // 5. The framework expects INCLUSIVE per-block output, but the down-sweep
    //    produced EXCLUSIVE. Add each element back: inclusive = exclusive + x.
    if (gA < N) out[gA] = s[2 * tid]     + a;
    if (gB < N) out[gB] = s[2 * tid + 1] + b;
}

// ============================================================================
// Optimization — Blelloch pass 1 with bank-conflict-free padded indexing
// ============================================================================
// Identical algorithm to scanBlockBlellochKernel; the ONLY change is that every
// shared-memory index i is remapped to i + CONFLICT_FREE_OFFSET(i). The load
// layout (2*tid, 2*tid+1) is deliberately left alone so this A/B isolates the
// bank-conflict variable and nothing else.
__global__ void scanBlockBlellochPaddedKernel(const float* in, float* out,
                                              float* blockSums, int N) {
    __shared__ float s[PADDED_CAP];
    int tid  = threadIdx.x;
    int n    = ELEMENTS_PER_BLOCK;
    int base = blockIdx.x * ELEMENTS_PER_BLOCK;

    // 1. Load two elements per thread into their PADDED shared slots.
    int gA = base + 2 * tid;
    int gB = base + 2 * tid + 1;
    float a = (gA < N) ? in[gA] : 0.0f;
    float b = (gB < N) ? in[gB] : 0.0f;
    int iA = 2 * tid;
    int iB = 2 * tid + 1;
    s[iA + CONFLICT_FREE_OFFSET(iA)] = a;
    s[iB + CONFLICT_FREE_OFFSET(iB)] = b;

    // 2. Up-sweep — same tree walk, padded addresses.
    int offset = 1;
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int l = offset * (2 * tid + 1) - 1;
            int r = offset * (2 * tid + 2) - 1;
            l += CONFLICT_FREE_OFFSET(l);
            r += CONFLICT_FREE_OFFSET(r);
            s[r] += s[l];
        }
        offset <<= 1;
    }

    // 3. Stash the chunk total before the clear wipes it.
    if (tid == 0) {
        int last = (n - 1) + CONFLICT_FREE_OFFSET(n - 1);
        blockSums[blockIdx.x] = s[last];
        s[last] = 0.0f;
    }

    // 4. Down-sweep -> exclusive scan.
    for (int d = 1; d < n; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int l = offset * (2 * tid + 1) - 1;
            int r = offset * (2 * tid + 2) - 1;
            l += CONFLICT_FREE_OFFSET(l);
            r += CONFLICT_FREE_OFFSET(r);
            float t = s[l];
            s[l] = s[r];
            s[r] += t;
        }
    }
    __syncthreads();

    // 5. Exclusive -> inclusive on write-back.
    if (gA < N) out[gA] = s[iA + CONFLICT_FREE_OFFSET(iA)] + a;
    if (gB < N) out[gB] = s[iB + CONFLICT_FREE_OFFSET(iB)] + b;
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

void launchScanFullBlelloch(const float* d_in, float* d_out, float* d_blockSums,
                            int N) {
    int numBlocks = scanNumBlocks(N);

    // Pass 1: work-efficient Blelloch per-block scan (the swap under test).
    scanBlockBlellochKernel<<<numBlocks, BLELLOCH_THREADS>>>(
        d_in, d_out, d_blockSums, N);

    // Passes 2 and 3 are identical to launchScanFull — reused verbatim.
    scanBlockSumsKernel<<<1, SCAN_BLOCK>>>(d_blockSums, numBlocks);
    addBlockOffsetsKernel<<<numBlocks, ELEMENTS_PER_BLOCK>>>(
        d_out, d_blockSums, N);
}

void launchScanFullBlellochPadded(const float* d_in, float* d_out,
                                  float* d_blockSums, int N) {
    int numBlocks = scanNumBlocks(N);

    // Pass 1: Blelloch with bank-conflict-free padded shared indexing.
    scanBlockBlellochPaddedKernel<<<numBlocks, BLELLOCH_THREADS>>>(
        d_in, d_out, d_blockSums, N);

    scanBlockSumsKernel<<<1, SCAN_BLOCK>>>(d_blockSums, numBlocks);
    addBlockOffsetsKernel<<<numBlocks, ELEMENTS_PER_BLOCK>>>(
        d_out, d_blockSums, N);
}
