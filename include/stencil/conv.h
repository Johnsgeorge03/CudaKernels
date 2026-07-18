#pragma once

// 1D/2D convolution (stencil). Every output is a weighted sum of its
// neighborhood:
//   1D: out[i]    = sum_{j=-r..r}          in[i+j]        * mask[j+r]
//   2D: out[y][x] = sum_{dy,dx = -r..r}    in[y+dy][x+dx] * mask[dy+r][dx+r]
// Out-of-range neighbors count as zero (zero padding). Strictly this is
// cross-correlation — true convolution flips the mask — but for the symmetric
// masks used in filtering they coincide, and GPU folklore says "convolution".
//
// Two new techniques vs the kernels so far:
//   * __constant__ memory for the mask — every lane of a warp reads the SAME
//     mask element at the same time, and the constant cache broadcasts one
//     fetch to all 32 lanes.
//   * Halo (ghost-cell) loading — a block computing outputs [B, B+T) needs
//     inputs [B-r, B+T+r); the 2r extra elements are the halo it shares with
//     its neighbor blocks.

// Masks are (2*radius + 1) wide; radius is a runtime parameter bounded by:
constexpr int CONV_MAX_RADIUS = 8;   // 1D mask <= 17 floats, 2D <= 17x17

// Upload a mask into constant memory. Must be called before the *Const and
// *Tiled launchers (the naive variant takes its mask as a normal pointer).
// 1D: h_mask holds 2*radius+1 floats. 2D: (2*radius+1)^2 floats, row-major.
void setConv1dMask(const float* h_mask, int radius);
void setConv2dMask(const float* h_mask, int radius);

// ---- 1D, length n ----------------------------------------------------------

// A: one thread per output; input AND mask read straight from global memory.
// Baseline — all reuse is left to L1/L2.
void launchConv1dNaive(const float* d_in, const float* d_mask, float* d_out,
                       int n, int radius);

// B: as A, but the mask lives in constant memory (setConv1dMask first).
// A/B vs naive isolates the constant-cache broadcast win.
void launchConv1dConst(const float* d_in, float* d_out, int n, int radius);

// C: constant mask + the block's input range (tile + halo) staged in shared
// memory. A/B vs B isolates the smem-vs-L1/L2 input-reuse question.
// Prediction to test: for contiguous 1D data the caches may already capture
// the reuse and this could be a wash (cf. the gemv shared-x negative result).
void launchConv1dTiled(const float* d_in, float* d_out, int n, int radius);

// F: as C, but the radius is a compile-time template parameter, so the tap
// loop can fully unroll (no loop bookkeeping, all smem loads in flight at
// once). Supported radii are dispatched in the launcher (1, 2, 4, 8); other
// values fall back to the generic tiled kernel.
void launchConv1dTiledUnroll(const float* d_in, float* d_out, int n,
                             int radius);

// ---- 2D, row-major H x W ---------------------------------------------------

// D: one thread per output pixel, constant mask, global input reads. Each
// input pixel is needed by (2r+1)^2 outputs, and neighboring rows are W*4
// bytes apart — much harder for the caches than the 1D case.
void launchConv2dConst(const float* d_in, float* d_out, int H, int W,
                       int radius);

// E: constant mask + a (TILE+2r) x (TILE+2r) input tile staged in shared
// memory (output tile plus a halo ring around it).
void launchConv2dTiled(const float* d_in, float* d_out, int H, int W,
                       int radius);
