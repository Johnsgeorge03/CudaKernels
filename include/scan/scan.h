#pragma once

// Prefix sum (scan). Given input x[0..N), produce a running aggregate:
//   inclusive: out[i] = x[0] + x[1] + ... + x[i]
//   exclusive: out[i] = x[0] + x[1] + ... + x[i-1]   (out[0] = 0)
//
// Unlike matmul/transpose/gemv, out[i] depends on the whole prefix before it,
// so this is NOT embarrassingly parallel. The algorithms below trade extra
// work and/or extra passes to recover parallelism from a serial dependency.

// ---- Single-block scans (require N <= one block's capacity) ----------------

// Hillis-Steele inclusive scan. In step d every element adds the one 2^d to its
// left; log2(N) steps, but O(N log N) total work (work-INefficient). N must fit
// in a single block (<= SCAN_BLOCK elements). Watch the in-place read/write race.
void launchScanHillisSteele(const float* d_in, float* d_out, int N);

// Blelloch (work-efficient) EXCLUSIVE scan: up-sweep then down-sweep over a
// balanced tree, O(N) work. Two elements per thread, so N <= 2 * BLELLOCH_THREADS.
// The 2^d strides invite shared-memory bank conflicts (cf. transpose).
void launchScanBlelloch(const float* d_in, float* d_out, int N);

// ---- Multi-block inclusive scan (arbitrary N) ------------------------------
// Three passes coordinate across blocks through a scratch array of per-block
// totals. d_blockSums must hold at least scanNumBlocks(N) floats. This variant
// assumes the block-sums themselves fit in ONE block scan, i.e.
// scanNumBlocks(N) <= SCAN_BLOCK  (true for N up to ~1M here); larger N would
// need the block-sums scan to recurse.
void launchScanFull(const float* d_in, float* d_out, float* d_blockSums, int N);

// Same three-pass structure as launchScanFull, but pass 1 uses the work-efficient
// Blelloch block scan (O(N)) in place of Hillis-Steele (O(N log N)). Passes 2 and 3
// are shared verbatim. Benchmark this against launchScanFull to see whether the
// work saving survives Blelloch's extra barriers and bank conflicts on this GPU.
void launchScanFullBlelloch(const float* d_in, float* d_out, float* d_blockSums, int N);

// As launchScanFullBlelloch, but pass 1 remaps every shared index i to
// i + (i >> 5), inserting a pad word every 32 to break the bank conflicts the
// 2^d strides cause (up to 32-way). Only variable changed vs launchScanFullBlelloch.
void launchScanFullBlellochPadded(const float* d_in, float* d_out,
                                  float* d_blockSums, int N);

// Number of blocks (== required length of the d_blockSums scratch array) that
// launchScanFull will use for a given N. Defined in scan.cu next to the geometry.
int scanNumBlocks(int N);
