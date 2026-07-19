#include "core/cuda_utils.h"
#include "stencil/conv.h"
#include "test_utils.h"

#include <cstdio>
#include <vector>

// Double-precision reference; zero padding outside the H x W image.
static void conv2dCPU(const float* in, const float* mask, float* out,
                      int H, int W, int r)
{
    const int K = 2 * r + 1;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            double acc = 0.0;
            for (int dy = -r; dy <= r; ++dy) {
                int yy = y + dy;
                if (yy < 0 || yy >= H) continue;
                for (int dx = -r; dx <= r; ++dx) {
                    int xx = x + dx;
                    if (xx < 0 || xx >= W) continue;
                    acc += static_cast<double>(in[yy * W + xx]) *
                           mask[(dy + r) * K + (dx + r)];
                }
            }
            out[y * W + x] = static_cast<float>(acc);
        }
    }
}

// Separable 2D triangle mask: outer product of the normalized 1D triangle
// with itself, so the 2D weights also sum to 1.
static std::vector<float> makeTriangleMask2d(int r)
{
    const int K = 2 * r + 1;
    std::vector<float> m1(K);
    float sum = 0.0f;
    for (int j = 0; j < K; ++j) {
        int d = (j < r) ? (r - j) : (j - r);
        m1[j] = static_cast<float>(r + 1 - d);
        sum += m1[j];
    }
    for (float& v : m1) v /= sum;

    std::vector<float> m(K * K);
    for (int y = 0; y < K; ++y)
        for (int x = 0; x < K; ++x)
            m[y * K + x] = m1[y] * m1[x];
    // The outer product is symmetric under both transpose and 180° rotation,
    // so a kernel that indexes the mask transposed (or flipped) would still
    // pass. Perturb one corner: (0, K-1) maps to (K-1, 0) under either
    // symmetry, so those bugs now produce mismatches well above tolerance.
    m[0 * K + (K - 1)] += 0.25f;
    return m;
}

using Conv2dFn = void (*)(const float*, float*, int, int, int);

struct Variant {
    const char* name;
    Conv2dFn fn;
};

static const Variant kVariants[] = {
    { "conv2d_const",  launchConv2dConst },
    { "conv2d_tiled",  launchConv2dTiled },
    { "conv2d_tiledU", launchConv2dTiledUnroll },
};

// Signed, bounded values in [-1.5, 1.5].
static void fill(std::vector<float>& v)
{
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.25f;
}

// Runs correctness (and optionally perf) for both variants at one size.
static bool runCase(int H, int W, int r, bool perf)
{
    const int n = H * W;
    std::vector<float> h_in(n), h_ref(n), h_out(n);
    fill(h_in);
    std::vector<float> h_mask = makeTriangleMask2d(r);
    conv2dCPU(h_in.data(), h_mask.data(), h_ref.data(), H, W, r);
    setConv2dMask(h_mask.data(), r);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));

    bool allOk = true;
    for (const Variant& v : kVariants) {
        // Poison the output (0xFF bytes are NaNs) so a kernel that writes
        // nothing fails loudly instead of passing on stale data.
        CUDA_CHECK(cudaMemset(d_out, 0xFF, n * sizeof(float)));
        v.fn(d_in, d_out, H, W, r);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, n * sizeof(float),
                              cudaMemcpyDeviceToHost));

        bool ok = compareResults(h_out.data(), h_ref.data(), n);
        std::printf("%-13s (%4d x %4d, r = %d): %s\n",
                    v.name, H, W, r, ok ? "PASSED" : "FAILED");
        allOk &= ok;

        if (perf) {
            double ms = benchmarkMs([&] { v.fn(d_in, d_out, H, W, r); });
            // At r = 4 this kernel sits at the roofline ridge: ideal traffic
            // 8 bytes/pixel AND 2*(2r+1)^2 = 162 flops/pixel, so both GB/s
            // (vs 192) and GFLOP/s (vs 3900) are worth watching.
            const double K = 2.0 * r + 1.0;
            reportPerf(v.name, ms,
                       2.0 * n * sizeof(float),   // ideal: in once + out once
                       2.0 * K * K * n);          // one FMA per tap
        }
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return allOk;
}

int main()
{
    bool allPassed = true;

    // Odd sizes exercise all four edges and partial tiles in both dimensions;
    // r = 1 first because halo bugs are much easier to reason about at r = 1.
    allPassed &= runCase(333, 479, 1, /*perf=*/false);
    allPassed &= runCase(333, 479, 4, /*perf=*/false);

    // Perf size: 4096 x 4096 = 16.7M pixels, 64 MB in + 64 MB out.
    allPassed &= runCase(4096, 4096, 4, /*perf=*/true);

    std::printf("Conv2d: %s\n", allPassed ? "ALL PASSED" : "FAILURES");
    return allPassed ? 0 : 1;
}
