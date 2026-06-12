#pragma once

// y = A * x   (general matrix-vector multiply)
// A is M x N (row-major), x is length N, y is length M.
// y[r] = sum_c A[r * N + c] * x[c]

// Thread-per-row baseline. Each thread walks an entire row alone, so within a
// warp the column accesses are stride-N => uncoalesced. Slow on purpose.
void launchGemvNaive(const float* d_A, const float* d_x, float* d_y,
                     int M, int N);

// Warp-per-row. A whole warp owns one row and strides across the columns by 32,
// so each step is one coalesced 128-byte transaction; a warpReduceSum collapses
// the 32 partials into one output element. This is the fast mapping.
void launchGemvWarp(const float* d_A, const float* d_x, float* d_y,
                    int M, int N);

// Warp-per-row, but x is staged into shared memory once per block and reused by
// every row the block owns. Tests whether caching x beats relying on L2.
void launchGemvWarpSharedX(const float* d_A, const float* d_x, float* d_y,
                           int M, int N);
