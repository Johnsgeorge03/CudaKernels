# CUDA Kernels

A collection of optimized CUDA kernel implementations for numerical operations, with correctness tests and benchmarks.

## Kernels

- **Matrix Multiplication** (`src/linalg/matmul_*.cu`)
  - Naive implementation
  - Shared-memory tiled implementation
  - Register-tiled implementation (2x2 outputs per thread)
- **Dot Product** (`src/linalg/dotprod.cu`)
  - Grid-stride loads, warp-shuffle reduction, block reduction, atomic finish

## Requirements

- **CUDA Toolkit** (v12.0+) — https://developer.nvidia.com/cuda-downloads
- **MSVC** (Windows, Visual Studio 2022 Build Tools) or **GCC** (Linux)
- **CMake** (v3.18+)

## Building

```powershell
cmake -S . -B build
cmake --build build --config Release
```

The GPU architecture defaults to `61` (GTX 1000 series). For a different GPU, pass it at configure time:

```powershell
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86   # RTX 3000s
# 61 = GTX 1000s, 75 = RTX 2000s, 86 = RTX 3000s, 89 = RTX 4000s
# CMake 3.24+: -DCMAKE_CUDA_ARCHITECTURES=native to auto-detect
```

## Testing

Each test verifies the kernels against a CPU reference, then benchmarks them (median-of-samples timing, reporting GB/s for memory-bound kernels and GFLOP/s for compute-bound ones).

```powershell
ctest --test-dir build -C Release --verbose

# Or run directly
./build/Release/test_matmul.exe
./build/Release/test_dotprod.exe
```

## Visualizing Performance

`scripts/plot_perf.py` runs the test binaries, parses their benchmark output, and renders bar charts (one for compute-bound kernels in GFLOP/s, one for memory-bound kernels in GB/s). Requires Python 3.9+ with matplotlib.

```powershell
python scripts/plot_perf.py --peak-bw 192 --peak-flops 3900   # writes perf.png
```

`--peak-bw` / `--peak-flops` are your GPU's spec-sheet ceilings and draw the hardware limit on each chart; omit them if you just want the bars. New kernels show up automatically as long as their test prints through `reportPerf` in `tests/test_utils.h`.

## Adding a New Kernel

1. Add `include/<domain>/<kernel>.h` declaring the launch function(s).
2. Add `src/<domain>/<kernel>.cu` and list it in the `add_library` block in `CMakeLists.txt`.
3. Add `tests/test_<kernel>.cu` (use the helpers in `tests/test_utils.h`) and register it with one line: `add_kernel_test(test_<kernel>)`.

## Project Structure

```
├── include/
│   ├── core/
│   │   └── cuda_utils.h         # CUDA_CHECK error handling
│   └── linalg/
│       ├── matmul.h
│       └── dotprod.h
├── src/
│   └── linalg/
│       ├── matmul_naive.cu
│       ├── matmul_tiled.cu      # shared-memory tiled + register tiled
│       └── dotprod.cu
├── tests/
│   ├── test_utils.h             # shared correctness + benchmark helpers
│   ├── test_matmul.cu
│   └── test_dotprod.cu
├── scripts/
│   └── plot_perf.py             # benchmark visualization (matplotlib)
└── CMakeLists.txt
```

## Performance Notes (measured)

![Kernel throughput](docs/perf.png)

GTX 1060 Max-Q, sm_61, ~192 GB/s peak DRAM bandwidth. Laptop part: clocks drift with thermals, so timings use the median of several samples. To regenerate the chart above after a benchmark run: `python scripts/plot_perf.py --peak-bw 192 --peak-flops 3900 --out docs/perf.png`.

- **Dot product** (N = 16.7M floats): ~130–138 GB/s effective bandwidth (~70% of spec peak). The kernel is memory-bound; the reduction itself (warp shuffles + one `__syncthreads()`) is not the bottleneck.
  - `float4` vectorized loads **regressed ~5%** — coalesced scalar loads already saturate transaction width on Pascal, and the wider loads reduced loads-in-flight.
  - 4x loop unrolling was **neutral** (within run-to-run noise).
  - Conclusion: the scalar grid-stride kernel is kept as canonical; remaining gap to peak is the practical DRAM ceiling plus thermal throttling, not kernel code.

## Troubleshooting

- **"Unsupported gpu architecture"** — your CUDA version doesn't support the requested arch; change `CMAKE_CUDA_ARCHITECTURES`.
- **"no kernel image is available for execution"** — binary built for a different arch than your GPU; check compute capability (`nvidia-smi --query-gpu=compute_cap --format=csv`) and reconfigure.
- **CUDA not found at configure time** — ensure `nvcc` is on `PATH` or set `CUDA_PATH` to the toolkit root.

## License

MIT
