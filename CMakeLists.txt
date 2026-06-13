cmake_minimum_required(VERSION 3.18)

# Adjust for your GPU: 61 (GTX 1000s), 75 (RTX 2000s), 86 (RTX 3000s),
# 89 (RTX 4000s). With CMake 3.24+ you can pass
# -DCMAKE_CUDA_ARCHITECTURES=native to auto-detect.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 61)
endif()

project(CudaKernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ============================================================================
# Kernel library — add new kernel .cu files here
# ============================================================================
add_library(cuda_kernels STATIC
    src/linalg/matmul_naive.cu
    src/linalg/matmul_tiled.cu
    src/linalg/dotprod.cu
    src/linalg/transpose.cu
    src/linalg/gemv.cu
    src/scan/scan.cu
)
target_include_directories(cuda_kernels PUBLIC ${PROJECT_SOURCE_DIR}/include)

# ============================================================================
# Tests — one .cu file per test in tests/, registered with CTest
# ============================================================================
enable_testing()

function(add_kernel_test name)
    add_executable(${name} tests/${name}.cu)
    target_link_libraries(${name} PRIVATE cuda_kernels)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

add_kernel_test(test_matmul)
add_kernel_test(test_dotprod)
add_kernel_test(test_transpose)
add_kernel_test(test_gemv)
add_kernel_test(test_scan)

message(STATUS "CUDA Kernels Build Configuration:")
message(STATUS "  CUDA compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
