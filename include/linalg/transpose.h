#pragma once

// B = A^T
// A is rows x cols (row-major), B is cols x rows (row-major).
// B[c * rows + r] = A[r * cols + c]

// Coalesced reads, strided (uncoalesced) writes. Baseline.
void launchTransposeNaive(const float* d_A, float* d_B, int rows, int cols);

// Stage the transpose through shared memory: both global accesses
// coalesced, but the transposed shared read hits bank conflicts.
void launchTransposeTiled(const float* d_A, float* d_B, int rows, int cols);

// Tiled, with the shared tile padded to [TILE][TILE+1] to remove the
// bank conflicts.
void launchTransposePadded(const float* d_A, float* d_B, int rows, int cols);
