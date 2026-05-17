# CUDA Kernels

A collection of optimized CUDA kernel implementations for linear algebra operations.

## Features

- **Matrix Multiplication Kernels**
  - Naive implementation (`matmul_naive.cu`)
  - Tiled implementation with shared memory optimization (`matmul_tiled.cu`)

## Requirements

- **CUDA Toolkit** (v12.0+)
  - Download: https://developer.nvidia.com/cuda-downloads
  - For older GPUs (pre-Turing), use CUDA 12.x or earlier
  
- **MSVC** (Windows) or **GCC** (Linux)
  - Windows: Visual Studio 2022 Build Tools (free)
  - Linux: `sudo apt install build-essential`

- **CMake** (v3.18+)
  - Download: https://cmake.org/download/

## GPU Compute Capability Reference

Adjust `CUDA_ARCH` in `CMakeLists.txt` for your GPU:

- `sm_61` - GeForce GTX 1060, 1070, 1080, Quadro P5000/P6000
- `sm_75` - GeForce RTX 2060/2070/2080, Quadro RTX 4000/5000/6000
- `sm_86` - GeForce RTX 3060/3070/3080/3090
- `sm_89` - GeForce RTX 4070/4080/4090

## Building

### Windows (with MSVC)

```powershell
# Set CUDA_PATH if not already in PATH
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"

# Create build directory
mkdir build
cd build

# Configure and build
cmake ..
cmake --build . --config Release
```

### Linux (with GCC)

```bash
# Install dependencies (if needed)
sudo apt install build-essential cmake nvidia-cuda-toolkit

# Create build directory
mkdir build
cd build

# Configure and build
cmake ..
cmake --build . --config Release
```

## Testing

Run the test suite:

```powershell
# Windows
cd build
ctest -C Release --verbose

# Or run directly
./Release/test_matmul.exe
```

```bash
# Linux
cd build
ctest --verbose
./test_matmul
```

## Project Structure

```
├── include/
│   ├── core/
│   │   └── cuda_utils.h         # CUDA error handling utilities
│   └── linalg/
│       └── matmul.h             # Matrix multiplication kernels
├── src/
│   └── linalg/
│       ├── matmul_naive.cu      # Naive kernel implementation
│       └── matmul_tiled.cu      # Tiled kernel implementation
├── tests/
│   └── test_matmul.cu           # Test suite
├── benchmarks/                   # (Empty, add benchmarks here)
├── examples/                     # (Empty, add examples here)
└── CMakeLists.txt
```

## Building with Options

```powershell
# Build with benchmarks
cmake .. -DBUILD_BENCHMARKS=ON

# Build with examples
cmake .. -DBUILD_EXAMPLES=ON

# Build everything
cmake .. -DBUILD_BENCHMARKS=ON -DBUILD_EXAMPLES=ON
```

## Troubleshooting

### "Unsupported gpu architecture" error
- Your GPU or CUDA version doesn't support the specified compute capability
- Update `CUDA_ARCH` in `CMakeLists.txt` to match your GPU
- Or install a compatible CUDA version

### "no kernel image is available for execution"
- The compiled binary doesn't support your GPU's architecture
- Check your GPU's compute capability (use `nvidia-smi`)
- Update `CUDA_ARCH` and rebuild

### "CUDA toolkit not found"
- Set `CUDA_PATH` environment variable:
  ```powershell
  $env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
  ```
- Or add CUDA to your system PATH

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit pull requests.
