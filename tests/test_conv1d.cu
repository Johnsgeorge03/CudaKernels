#include "core/cuda_utils.h"
#include "stencil/conv.h"
#include "test_utils.h"

#include <cstdio>
#include <cstring>
#include <vector>

// Double-precision reference with explicit zero padding at the ends.
static void conv1dCPU(const float* in, const float* mask, float* out,
                      int n, int r)
{
    for (int i = 0; i < n; ++i) {
        double acc = 0.0;
        for (int j = -r; j <= r; ++j) {
            int k = i + j;
            if (k >= 0 && k < n)
                acc += static_cast<double>(in[k]) * mask[j + r];
        }
        out[i] = static_cast<float>(acc);
    }
}

// Normalized triangle mask (weights sum to 1), so outputs stay the same
// magnitude as inputs and the default compare tolerance applies.
static std::vector<float> makeTriangleMask(int r)
{
    std::vector<float> m(2 * r + 1);
    float sum = 0.0f;
    for (int j = 0; j <= 2 * r; ++j) {
        int d = (j < r) ? (r - j) : (j - r);
        m[j] = static_cast<float>(r + 1 - d);
        sum += m[j];
    }
    for (float& v : m) v /= sum;
    return m;
}

// All three variants behind one signature so the test can loop over them
// (captureless lambdas decay to plain function pointers; the const/tiled
// variants ignore the d_mask argument and read constant memory instead).
using Conv1dFn = void (*)(const float*, const float*, float*, int, int);

struct Variant {
    const char* name;
    Conv1dFn fn;
};

static const Variant kVariants[] = {
    { "conv1d_naive",
      [](const float* in, const float* mask, float* out, int n, int r) {
          launchConv1dNaive(in, mask, out, n, r);
      } },
    { "conv1d_const",
      [](const float* in, const float*, float* out, int n, int r) {
          launchConv1dConst(in, out, n, r);
      } },
    { "conv1d_tiled",
      [](const float* in, const float*, float* out, int n, int r) {
          launchConv1dTiled(in, out, n, r);
      } },
    { "conv1d_tiled_ur",
      [](const float* in, const float*, float* out, int n, int r) {
          launchConv1dTiledUnroll(in, out, n, r);
      } },
};

// Signed, bounded values in [-1.5, 1.5].
static void fill(std::vector<float>& v)
{
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.25f;
}

int main(int argc, char** argv)
{
    // --sweep: extra diagnostic benchmarks across mask radii (not part of the
    // default run, so ctest and the perf-plot script never see these lines).
    const bool sweep = argc > 1 && std::strcmp(argv[1], "--sweep") == 0;

    const int N_small = 100003;    // odd: exercises both edges + partial last block
    const int N_big   = 1 << 24;   // 16.7M floats; 64 MB in + 64 MB out
    const int R_perf  = 4;         // 9-tap mask for the perf runs

    bool allPassed = true;

    // ---- Correctness at small odd N, easy (r = 1) and full (r = 4) masks ---
    for (int r : { 1, 4 }) {
        std::vector<float> h_in(N_small), h_ref(N_small), h_out(N_small);
        fill(h_in);
        std::vector<float> h_mask = makeTriangleMask(r);
        conv1dCPU(h_in.data(), h_mask.data(), h_ref.data(), N_small, r);
        setConv1dMask(h_mask.data(), r);

        float *d_in, *d_mask, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in,   N_small * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_mask, (2 * r + 1) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out,  N_small * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N_small * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mask, h_mask.data(),
                              (2 * r + 1) * sizeof(float),
                              cudaMemcpyHostToDevice));

        for (const Variant& v : kVariants) {
            // Poison the output (0xFF bytes are NaNs) so a kernel that
            // writes nothing fails loudly instead of passing on stale data.
            CUDA_CHECK(cudaMemset(d_out, 0xFF, N_small * sizeof(float)));
            v.fn(d_in, d_mask, d_out, N_small, r);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_out.data(), d_out,
                                  N_small * sizeof(float),
                                  cudaMemcpyDeviceToHost));

            bool ok = compareResults(h_out.data(), h_ref.data(), N_small);
            std::printf("%-13s (N = %8d, r = %d): %s\n",
                        v.name, N_small, r, ok ? "PASSED" : "FAILED");
            allPassed &= ok;
        }

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_mask));
        CUDA_CHECK(cudaFree(d_out));
    }

    // ---- Correctness + perf at N_big ----------------------------------------
    {
        std::vector<float> h_in(N_big), h_ref(N_big), h_out(N_big);
        fill(h_in);
        std::vector<float> h_mask = makeTriangleMask(R_perf);
        conv1dCPU(h_in.data(), h_mask.data(), h_ref.data(), N_big, R_perf);
        setConv1dMask(h_mask.data(), R_perf);

        float *d_in, *d_mask, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in,   N_big * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_mask, (2 * R_perf + 1) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out,  N_big * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N_big * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mask, h_mask.data(),
                              (2 * R_perf + 1) * sizeof(float),
                              cudaMemcpyHostToDevice));

        for (const Variant& v : kVariants) {
            CUDA_CHECK(cudaMemset(d_out, 0xFF, N_big * sizeof(float)));
            v.fn(d_in, d_mask, d_out, N_big, R_perf);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N_big * sizeof(float),
                                  cudaMemcpyDeviceToHost));

            bool ok = compareResults(h_out.data(), h_ref.data(), N_big);
            std::printf("%-13s (N = %8d, r = %d): %s\n",
                        v.name, N_big, R_perf, ok ? "PASSED" : "FAILED");
            allPassed &= ok;

            double ms = benchmarkMs([&] {
                v.fn(d_in, d_mask, d_out, N_big, R_perf);
            });
            // Ideal DRAM traffic: read each input once, write each output
            // once. Mask reads and halo re-reads are exactly what the kernel
            // is supposed to keep OUT of DRAM, so they don't count — hitting
            // ~150+ GB/s here means the reuse machinery worked.
            reportPerf(v.name, ms, 2.0 * N_big * sizeof(float), 0.0);
        }

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_mask));
        CUDA_CHECK(cudaFree(d_out));
    }

    // ---- Diagnostic radius sweep (opt-in) ----------------------------------
    // The ideal DRAM traffic (read n, write n) is INDEPENDENT of r, so a
    // memory-bound kernel would show ~flat time as r grows. Time growing
    // linearly with r means per-tap instruction/latency cost dominates and
    // DRAM bandwidth is not the limiter.
    if (sweep) {
        std::vector<float> h_in(N_big);
        fill(h_in);

        float *d_in, *d_mask, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in,   N_big * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_mask, (2 * CONV_MAX_RADIUS + 1) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out,  N_big * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N_big * sizeof(float),
                              cudaMemcpyHostToDevice));

        for (int r : { 1, 2, 4, 8 }) {
            std::vector<float> h_mask = makeTriangleMask(r);
            setConv1dMask(h_mask.data(), r);
            CUDA_CHECK(cudaMemcpy(d_mask, h_mask.data(),
                                  (2 * r + 1) * sizeof(float),
                                  cudaMemcpyHostToDevice));

            for (const Variant& v : kVariants) {
                double ms = benchmarkMs([&] {
                    v.fn(d_in, d_mask, d_out, N_big, r);
                });
                char name[64];
                std::snprintf(name, sizeof(name), "%s_r%d", v.name, r);
                reportPerf(name, ms, 2.0 * N_big * sizeof(float), 0.0);
            }
        }

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_mask));
        CUDA_CHECK(cudaFree(d_out));
    }

    std::printf("Conv1d: %s\n", allPassed ? "ALL PASSED" : "FAILURES");
    return allPassed ? 0 : 1;
}
