#include "core/cuda_utils.h"
#include "scan/scan.h"
#include "test_utils.h"

#include <cstdio>
#include <vector>

// Double-precision CPU references so the float GPU results have a clean target.
static void scanInclusiveCPU(const float* in, float* out, int N) {
    double acc = 0.0;
    for (int i = 0; i < N; ++i) { acc += in[i]; out[i] = static_cast<float>(acc); }
}

static void scanExclusiveCPU(const float* in, float* out, int N) {
    double acc = 0.0;
    for (int i = 0; i < N; ++i) { out[i] = static_cast<float>(acc); acc += in[i]; }
}

int main() {
    // Single-block tests use exactly one block's capacity; the full multi-block
    // test uses a large N that is still a multiple of the chunk size.
    const int N_small = 1024;        // fits one Hillis-Steele / Blelloch block
    const int N_big   = 1 << 20;     // ~1.05M; 1024 blocks of 1024 elements

    bool allPassed = true;

    // Bounded values keep the running sum well-conditioned for fp32.
    auto fill = [](std::vector<float>& v) {
        for (size_t i = 0; i < v.size(); ++i)
            v[i] = static_cast<float>((i % 7) + 1) * 0.5f;
    };

    // ---- Single-block: Hillis-Steele (inclusive) ---------------------------
    {
        std::vector<float> h_in(N_small), h_ref(N_small), h_out(N_small);
        fill(h_in);
        scanInclusiveCPU(h_in.data(), h_ref.data(), N_small);

        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in,  N_small * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, N_small * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N_small * sizeof(float),
                              cudaMemcpyHostToDevice));

        launchScanHillisSteele(d_in, d_out, N_small);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N_small * sizeof(float),
                              cudaMemcpyDeviceToHost));

        bool ok = compareResults(h_out.data(), h_ref.data(), N_small, 1e-2f);
        std::printf("Hillis-Steele inclusive (N = %d): %s\n",
                    N_small, ok ? "PASSED" : "FAILED");
        allPassed &= ok;
        CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));
    }

    // ---- Single-block: Blelloch (exclusive) --------------------------------
    {
        std::vector<float> h_in(N_small), h_ref(N_small), h_out(N_small);
        fill(h_in);
        scanExclusiveCPU(h_in.data(), h_ref.data(), N_small);

        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in,  N_small * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, N_small * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N_small * sizeof(float),
                              cudaMemcpyHostToDevice));

        launchScanBlelloch(d_in, d_out, N_small);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N_small * sizeof(float),
                              cudaMemcpyDeviceToHost));

        bool ok = compareResults(h_out.data(), h_ref.data(), N_small, 1e-2f);
        std::printf("Blelloch exclusive      (N = %d): %s\n",
                    N_small, ok ? "PASSED" : "FAILED");
        allPassed &= ok;
        CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));
    }

    // ---- Multi-block: full inclusive scan + perf ---------------------------
    {
        std::vector<float> h_in(N_big), h_ref(N_big), h_out(N_big);
        fill(h_in);
        scanInclusiveCPU(h_in.data(), h_ref.data(), N_big);

        float *d_in, *d_out, *d_blockSums;
        CUDA_CHECK(cudaMalloc(&d_in,  N_big * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, N_big * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_blockSums, scanNumBlocks(N_big) * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N_big * sizeof(float),
                              cudaMemcpyHostToDevice));

        launchScanFull(d_in, d_out, d_blockSums, N_big);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N_big * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // Larger running sums here, so allow a looser absolute tolerance.
        bool ok = compareResults(h_out.data(), h_ref.data(), N_big, 1.0f);
        std::printf("Full multi-block scan   (N = %d): %s\n",
                    N_big, ok ? "PASSED" : "FAILED");
        allPassed &= ok;

        // Memory-bound: reads N and writes N (block-sums traffic is negligible).
        double ms = benchmarkMs([&] {
            launchScanFull(d_in, d_out, d_blockSums, N_big);
        });
        reportPerf("scan_full", ms, 2.0 * N_big * sizeof(float), 0.0);

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaFree(d_blockSums));
    }

    std::printf("Scan: %s\n", allPassed ? "ALL PASSED" : "FAILURES");
    return allPassed ? 0 : 1;
}
