cmake_minimum_required(VERSION 3.18)
project(CudaKernels)

# Compiler settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

cmake_minimum_required(VERSION 3.18)
project(CudaKernels)

# Compiler settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA settings - use environment variable or find in standard locations
if(DEFINED ENV{CUDA_PATH})
    set(CUDA_ROOT "$ENV{CUDA_PATH}")
else()
    # Try standard installation paths
    if(EXISTS "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8")
        set(CUDA_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8")
    elseif(EXISTS "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4")
        set(CUDA_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4")
    else()
        message(FATAL_ERROR "CUDA toolkit not found. Please set CUDA_PATH environment variable.")
    endif()
endif()

set(NVCC_EXECUTABLE "${CUDA_ROOT}/bin/nvcc.exe")
set(CUDA_ARCH "sm_61")  # Adjust for your GPU: sm_61 (GTX 1060), sm_75 (RTX 2000s), sm_86 (RTX 3000s), sm_89 (RTX 4000s)

# Include and library directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${CUDA_ROOT}/include
)

link_directories(${CUDA_ROOT}/lib/x64)

# Function to compile CUDA files
function(add_cuda_executable target_name cuda_sources)
    set(cuda_obj_files)
    
    foreach(cuda_source ${cuda_sources})
        get_filename_component(cuda_name ${cuda_source} NAME_WE)
        set(cuda_obj "${CMAKE_CURRENT_BINARY_DIR}/${cuda_name}.obj")
        
        add_custom_command(
            OUTPUT ${cuda_obj}
            COMMAND ${NVCC_EXECUTABLE} -c ${cuda_source} -o ${cuda_obj} 
                    -I${PROJECT_SOURCE_DIR}/include
                    -I${CUDA_ROOT}/include
                    -arch=${CUDA_ARCH}
                    -std=c++17
            DEPENDS ${cuda_source}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            VERBATIM
        )
        list(APPEND cuda_obj_files ${cuda_obj})
    endforeach()
    
    add_executable(${target_name} ${cuda_obj_files})
    target_link_libraries(${target_name} ${CUDA_ROOT}/lib/x64/cudart.lib)
    target_include_directories(${target_name} PRIVATE ${PROJECT_SOURCE_DIR}/include ${CUDA_ROOT}/include)
endfunction()

# ============================================================================
# Build Library - Compile CUDA files to object files
# ============================================================================
set(LINALG_SOURCES
    src/linalg/matmul_naive.cu
    src/linalg/matmul_tiled.cu
)

set(LINALG_OBJS)
foreach(cuda_source ${LINALG_SOURCES})
    get_filename_component(cuda_name ${cuda_source} NAME_WE)
    set(cuda_obj "${CMAKE_CURRENT_BINARY_DIR}/${cuda_name}.obj")
    
    add_custom_command(
        OUTPUT ${cuda_obj}
        COMMAND ${NVCC_EXECUTABLE} -c ${cuda_source} -o ${cuda_obj}
                -I${PROJECT_SOURCE_DIR}/include
                -I${CUDA_ROOT}/include
                -arch=${CUDA_ARCH}
                -std=c++17
        DEPENDS ${cuda_source}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        VERBATIM
    )
    list(APPEND LINALG_OBJS ${cuda_obj})
endforeach()

add_library(cuda_kernels STATIC ${LINALG_OBJS})
set_target_properties(cuda_kernels PROPERTIES LINKER_LANGUAGE CXX)

# ============================================================================
# Tests
# ============================================================================
enable_testing()

set(TEST_OBJ "${CMAKE_CURRENT_BINARY_DIR}/test_matmul.obj")

add_custom_command(
    OUTPUT ${TEST_OBJ}
    COMMAND ${NVCC_EXECUTABLE} -c tests/test_matmul.cu -o ${TEST_OBJ}
            -I${PROJECT_SOURCE_DIR}/include
            -I${CUDA_ROOT}/include
            -arch=${CUDA_ARCH}
            -std=c++17
    DEPENDS tests/test_matmul.cu
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    VERBATIM
)

add_executable(test_matmul ${TEST_OBJ})
target_link_libraries(test_matmul cuda_kernels "${CUDA_ROOT}/lib/x64/cudart.lib")
target_include_directories(test_matmul PRIVATE ${PROJECT_SOURCE_DIR}/include ${CUDA_ROOT}/include)
set_target_properties(test_matmul PROPERTIES LINKER_LANGUAGE CXX)
add_test(NAME MatMulTest COMMAND test_matmul)

# ============================================================================
# Build options
# ============================================================================
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
option(BUILD_EXAMPLES "Build examples" OFF)

if(BUILD_BENCHMARKS)
    # add_executable(bench_matmul benchmarks/bench_matmul.cu)
    # target_link_libraries(bench_matmul cuda_kernels)
endif()

if(BUILD_EXAMPLES)
    # add_executable(example_matmul examples/example_matmul.cu)
    # target_link_libraries(example_matmul cuda_kernels)
endif()

message(STATUS "CUDA Kernels Build Configuration:")
message(STATUS "  CUDA Toolkit: ${CUDA_ROOT}")
message(STATUS "  NVCC: ${NVCC_EXECUTABLE}")
message(STATUS "  Architecture: ${CUDA_ARCH}")
message(STATUS "  Build Benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "  Build Examples: ${BUILD_EXAMPLES}")
