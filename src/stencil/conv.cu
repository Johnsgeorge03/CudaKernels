#include "core/cuda_utils.h"
#include "stencil/conv.h"

#include <cuda_runtime.h>

// ----------------------------------------------------------------------------
// Constant memory for the masks. __constant__ variables live at file scope in
// device memory (64 KB total per device) and are read through a small per-SM
// constant cache whose superpower is BROADCAST: when all 32 lanes of a warp
// read the same address — exactly the mask-loop pattern, where the index
// depends on the loop counter but not on threadIdx — one cache access feeds
// the whole warp. The compiler can even fold the read into the FMA as a
// c[bank][offset] operand: no separate load instruction, no register held.
// (Anti-pattern for another day: indexing __constant__ by threadIdx makes the
// lanes' addresses diverge and the accesses serialize.)
// ----------------------------------------------------------------------------
#define MASK_W_MAX (2 * CONV_MAX_RADIUS + 1)

__constant__ float c_mask1d[MASK_W_MAX];
__constant__ float c_mask2d[MASK_W_MAX * MASK_W_MAX];

void setConv1dMask(const float* h_mask, int radius) {
    CUDA_CHECK(cudaMemcpyToSymbol(c_mask1d, h_mask,
                                  (2 * radius + 1) * sizeof(float)));
}

void setConv2dMask(const float* h_mask, int radius) {
    int w = 2 * radius + 1;
    CUDA_CHECK(cudaMemcpyToSymbol(c_mask2d, h_mask, w * w * sizeof(float)));
}

// Launch geometry: one output element per thread.
#define CONV1D_BLOCK 256
#define CONV2D_TILE  16     // 16x16 = 256 threads per block
#define CONV2D_TILE_X 32
#define CONV2D_TILE_Y 8

// ----------------------------------------------------------------------------
// A: naive 1D. One thread per output element; mask read from global memory.
// ----------------------------------------------------------------------------
__global__ void conv1dNaiveKernel(const float* in, const float* mask,
                                  float* out, int n, int r)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // No barries so do early return. If __syncthreads() this could cause problems.
    if (i >= n) 
        return;
    
    float sum = 0;
    for(int j = -r; j <= r; j++){
        if ( i + j < n && i + j >= 0 )
            sum += mask[j + r] * in[i + j];
    }
    out[i] = sum;
}

// ----------------------------------------------------------------------------
// B: as A, but the mask is read from c_mask1d (constant memory).
// ----------------------------------------------------------------------------
__global__ void conv1dConstKernel(const float* in, float* out, int n, int r)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n )
        return;
    
    float sum = 0;
    for (int j = -r; j <= r; j++ ){
        if( i + j >= 0 && i + j < n )
            sum += c_mask1d[j + r] * in[i + j];
    }
    out[i] = sum;
}

// ----------------------------------------------------------------------------
// C: tiled 1D. The block stages its whole input range — blockDim.x outputs
// plus r on each side, i.e. blockDim.x + 2r floats — into dynamic shared
// memory, then computes entirely from smem. Zero padding is baked into the
// tile at load time, so the compute loop needs no bounds checks on the input.
//
// Load pattern: the tile's slot s holds global element
//     g = blockIdx.x * blockDim.x - r + s.
// Let thread t fill slots t, t + blockDim.x, ... until the tile is covered
// (at most 2 loads per thread while r <= blockDim.x / 2); out-of-range g
// gets 0.0f. __syncthreads() before any thread reads the tile.
// ----------------------------------------------------------------------------
__global__ void conv1dTiledKernel(const float* in, float* out, int n, int r)
{
    extern __shared__ float tile[];   // blockDim.x + 2r floats
    int tileSize = blockDim.x + 2 * r;
    int blockStart = blockDim.x * blockIdx.x;

    // Load to shared memory
    for( int s = threadIdx.x; s < tileSize; s += blockDim.x ){
        if ( blockStart - r + s >= 0 && blockStart - r + s < n )
            tile[s] = in[blockStart - r + s];
        else
            tile[s] = 0.0f;
    }
    __syncthreads();
    // Compute
    int idx = r + threadIdx.x;
    float sum = 0.0f;
    for( int j = -r; j <= r; j++ ){
        sum += c_mask1d[j + r] * tile[idx + j];
    }
    
    if ( blockStart + threadIdx.x < n )
        out[blockStart + threadIdx.x] = sum;
}

// ----------------------------------------------------------------------------
// F: tiled 1D with a COMPILE-TIME radius. `template <int R>` makes R a value
// the compiler knows while generating code — the compiler stamps out one
// concrete kernel per R used (see the launcher's switch), and inside each one
// R behaves like a literal number in the source. That lets `#pragma unroll`
// flatten the tap loop: the loop counter, compare, and branch vanish, and the
// 2R+1 shared-memory loads become independent instructions the hardware can
// overlap instead of issuing one per loop lap.
//
// Body: your conv1dTiledKernel verbatim, with every `r` becoming `R` (note:
// no runtime r parameter) and `#pragma unroll` on the line before the tap loop.
// ----------------------------------------------------------------------------
template <int R>
__global__ void conv1dTiledUnrollKernel(const float* in, float* out, int n)
{
    extern __shared__ float tile[];   // blockDim.x + 2R floats
    int tileSize = blockDim.x + 2 * R;
    int blockStart = blockDim.x * blockIdx.x;
    // Load the shared memory
    for(int s = threadIdx.x; s < tileSize; s += blockDim.x ) {
        if ( blockStart - R + s >= 0 && blockStart - R + s < n )
            tile[s] = in[blockStart - R + s];
        else
            tile[s] = 0.0f;
    }
    __syncthreads();

    // Compute
    float sum = 0.0f;
    #pragma unroll
    for( int j = -R; j <= R; j++ ){
        sum += c_mask1d[j + R] * tile[R + threadIdx.x + j];
    }
    if ( blockStart + threadIdx.x < n )
        out[blockStart + threadIdx.x] = sum;
}

// ----------------------------------------------------------------------------
// D: naive-ish 2D. One thread per output pixel (2D grid of 2D blocks), mask
// from c_mask2d, input straight from global memory. Row-major: pixel (y, x)
// is in[y * W + x].
// ----------------------------------------------------------------------------
__global__ void conv2dConstKernel(const float* __restrict__ in, float* __restrict__ out,
                                  int H, int W, int r)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    float sum = 0.0f;
    for ( int dy = -r; dy <= r; dy++){
        for (int dx = -r; dx <= r; dx++){
            int idin = (row + dy) * W + ( col + dx);
            if ( row + dy < H && row + dy >= 0 && 
                 col + dx < W && col + dx >= 0 )
                sum += in[idin] * c_mask2d[(dy + r)*(2*r + 1) + (dx + r)];
        }
    }

    if( row < H  && col < W )
        out[row * W + col] = sum;
}

// ----------------------------------------------------------------------------
// E: tiled 2D. Dynamic shared tile of side (CONV2D_TILE + 2r) — the block's
// output tile plus a halo ring. r = 4 -> 24x24 floats = 2.3 KB per block, so
// unlike gemv shared-x this does NOT dent occupancy.
//
// Load pattern (flatten-and-stride): treat the tile as a flat array of
// side*side slots and let the TILE*TILE threads stride over it:
//     for (s = tid; s < side*side; s += blockDim.x*blockDim.y)
// Slot s is tile row s/side, col s%side, which is global pixel
//     (blockIdx.y*TILE - r + s/side,  blockIdx.x*TILE - r + s%side);
// out-of-image slots get 0.0f. Consecutive s -> consecutive columns, so the
// loads coalesce along rows. __syncthreads(), then thread (ty, tx) computes
// its pixel from tile rows ty..ty+2r, cols tx..tx+2r.
//
// CLASSIC BUG: the tile's row stride is side = TILE + 2r, not TILE. Indexing
// with the wrong stride produces plausible-looking but wrong output — this is
// what the small odd-size correctness cases are for.
// ----------------------------------------------------------------------------
__global__ void conv2dTiledKernel(const float* in, float* out,
                                  int H, int W, int r)
{
    extern __shared__ float tile[];   // (CONV2D_TILE + 2r)^2 floats

    // TODO (you): flatten-and-stride load incl. halo + zero padding; barrier;
    // accumulate over the (2r+1)^2 window from the tile; write out[y*W + x]
    // if inside the image.
    // Load to shared memory
    int tid = threadIdx.y * CONV2D_TILE_X + threadIdx.x;
    int side_c = CONV2D_TILE_X + 2 * r;
    int side_r = CONV2D_TILE_Y + 2 * r;
    int row = 0, col = 0;
    for( int s = tid; s < side_c * side_r; s += blockDim.x * blockDim.y ){
        row = blockIdx.y*CONV2D_TILE_Y - r + s/side_c;
        col = blockIdx.x*CONV2D_TILE_X - r + s%side_c;
        if( row >= 0 && row < H && col >= 0 && col < W )
            tile[s] = in[row * W + col];
        else
            tile[s] = 0.0f;
    }
    __syncthreads();

    // Compute
    float sum = 0.0f;
    for ( int dy = -r; dy <= r; dy++){
        for (int dx = -r; dx <= r; dx++){
            int idx = (threadIdx.y + r + dy) * side_c + ( threadIdx.x + + r + dx);
            sum += tile[idx] * c_mask2d[(dy + r)*(2*r + 1) + (dx + r)];
        }
    }

    row = blockDim.y * blockIdx.y + threadIdx.y;
    col = blockDim.x * blockIdx.x + threadIdx.x;

    if( row < H  && col < W )
        out[row * W + col] = sum;
}

// ----------------------------------------------------------------------------
// F2: tiled 2D with a COMPILE-TIME radius — the 2D twin of kernel F. Body:
// your conv2dTiledKernel verbatim, every `r` becoming `R` (no runtime r
// parameter), `#pragma unroll` on the line before EACH tap loop. With R known
// at compile time the 81 taps flatten into straight-line code: the mask index
// (dy+R)*(2R+1)+(dx+R) is a constant per tap, so each weight rides inside its
// FFMA as a c[bank][offset] operand — the LDC instruction disappears entirely
// (see the 1D unrolled SASS: `FFMA R4, R5, c[0x3][0x4], R4`).
// ----------------------------------------------------------------------------
template <int R>
__global__ void conv2dTiledUnrollKernel(const float* __restrict__ in,
                                        float* __restrict__ out, int H, int W)
{
    extern __shared__ float tile[];   // (TILE_X + 2R) * (TILE_Y + 2R) floats

    // conv2dTiledKernel with r -> R and both tap loops unrolled.
    int tid = threadIdx.y * CONV2D_TILE_X + threadIdx.x;
    int side_c = CONV2D_TILE_X + 2 * R;
    int side_r = CONV2D_TILE_Y + 2 * R;
    int row = 0, col = 0;
    for( int s = tid; s < side_c * side_r; s += blockDim.x * blockDim.y ){
        row = blockIdx.y*CONV2D_TILE_Y - R + s/side_c;
        col = blockIdx.x*CONV2D_TILE_X - R + s%side_c;
        if( row >= 0 && row < H && col >= 0 && col < W )
            tile[s] = in[row * W + col];
        else
            tile[s] = 0.0f;
    }
    __syncthreads();

    // Compute
    float sum = 0.0f;
    #pragma unroll
    for ( int dy = -R; dy <= R; dy++){
        #pragma unroll
        for (int dx = -R; dx <= R; dx++){
            int idx = (threadIdx.y + R + dy) * side_c + ( threadIdx.x + R + dx);
            sum += tile[idx] * c_mask2d[(dy + R)*(2*R + 1) + (dx + R)];
        }
    }

    row = blockDim.y * blockIdx.y + threadIdx.y;
    col = blockDim.x * blockIdx.x + threadIdx.x;

    if( row < H  && col < W )
        out[row * W + col] = sum;
}

// ----------------------------------------------------------------------------
// Launchers (infra — wired for you).
// ----------------------------------------------------------------------------
void launchConv1dNaive(const float* d_in, const float* d_mask, float* d_out,
                       int n, int r) {
    int grid = (n + CONV1D_BLOCK - 1) / CONV1D_BLOCK;
    conv1dNaiveKernel<<<grid, CONV1D_BLOCK>>>(d_in, d_mask, d_out, n, r);
}

void launchConv1dConst(const float* d_in, float* d_out, int n, int r) {
    int grid = (n + CONV1D_BLOCK - 1) / CONV1D_BLOCK;
    conv1dConstKernel<<<grid, CONV1D_BLOCK>>>(d_in, d_out, n, r);
}

void launchConv1dTiled(const float* d_in, float* d_out, int n, int r) {
    int grid = (n + CONV1D_BLOCK - 1) / CONV1D_BLOCK;
    size_t smem = (CONV1D_BLOCK + 2 * r) * sizeof(float);
    conv1dTiledKernel<<<grid, CONV1D_BLOCK, smem>>>(d_in, d_out, n, r);
}

// A template needs its value at compile time, but r arrives at runtime — so
// we enumerate the radii we care about and let a switch pick the matching
// pre-compiled kernel. (Real libraries do exactly this: specialize the common
// cases, keep a generic fallback for the rest.)
void launchConv1dTiledUnroll(const float* d_in, float* d_out, int n, int r) {
    int grid = (n + CONV1D_BLOCK - 1) / CONV1D_BLOCK;
    size_t smem = (CONV1D_BLOCK + 2 * r) * sizeof(float);
    switch (r) {
    case 1: conv1dTiledUnrollKernel<1><<<grid, CONV1D_BLOCK, smem>>>(d_in, d_out, n); break;
    case 2: conv1dTiledUnrollKernel<2><<<grid, CONV1D_BLOCK, smem>>>(d_in, d_out, n); break;
    case 4: conv1dTiledUnrollKernel<4><<<grid, CONV1D_BLOCK, smem>>>(d_in, d_out, n); break;
    case 8: conv1dTiledUnrollKernel<8><<<grid, CONV1D_BLOCK, smem>>>(d_in, d_out, n); break;
    default: conv1dTiledKernel<<<grid, CONV1D_BLOCK, smem>>>(d_in, d_out, n, r); break;
    }
}

void launchConv2dConst(const float* d_in, float* d_out, int H, int W, int r) {
    dim3 block(CONV2D_TILE, CONV2D_TILE);
    dim3 grid((W + CONV2D_TILE - 1) / CONV2D_TILE,
              (H + CONV2D_TILE - 1) / CONV2D_TILE);
    conv2dConstKernel<<<grid, block>>>(d_in, d_out, H, W, r);
}

void launchConv2dTiled(const float* d_in, float* d_out, int H, int W, int r) {
    dim3 block(CONV2D_TILE_X, CONV2D_TILE_Y);
    dim3 grid((W + CONV2D_TILE_X - 1) / CONV2D_TILE_X,
              (H + CONV2D_TILE_Y - 1) / CONV2D_TILE_Y);
    size_t side_c = CONV2D_TILE_X + 2 * r;
    size_t side_r = CONV2D_TILE_Y + 2 * r;
    conv2dTiledKernel<<<grid, block, side_c * side_r * sizeof(float)>>>(
        d_in, d_out, H, W, r);
}

// Same enumerate-and-dispatch pattern as launchConv1dTiledUnroll: the template
// needs R at compile time, r arrives at runtime, so a switch picks the
// pre-compiled kernel and anything else falls back to the generic tiled one.
void launchConv2dTiledUnroll(const float* d_in, float* d_out, int H, int W,
                             int r) {
    dim3 block(CONV2D_TILE_X, CONV2D_TILE_Y);
    dim3 grid((W + CONV2D_TILE_X - 1) / CONV2D_TILE_X,
              (H + CONV2D_TILE_Y - 1) / CONV2D_TILE_Y);
    size_t smem = (CONV2D_TILE_X + 2 * r) * (CONV2D_TILE_Y + 2 * r)
                  * sizeof(float);
    switch (r) {
    case 1: conv2dTiledUnrollKernel<1><<<grid, block, smem>>>(d_in, d_out, H, W); break;
    case 2: conv2dTiledUnrollKernel<2><<<grid, block, smem>>>(d_in, d_out, H, W); break;
    case 4: conv2dTiledUnrollKernel<4><<<grid, block, smem>>>(d_in, d_out, H, W); break;
    case 8: conv2dTiledUnrollKernel<8><<<grid, block, smem>>>(d_in, d_out, H, W); break;
    default: conv2dTiledKernel<<<grid, block, smem>>>(d_in, d_out, H, W, r); break;
    }
}
