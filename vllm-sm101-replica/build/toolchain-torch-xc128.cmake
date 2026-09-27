# NOTE: /path/to/* are placeholders — set them to your own workspace
#       (original host-internal paths removed for publication).
# torch 交叉工具链 — 基于 llama.cpp 实证版 toolchain-thor-cuda128.cmake, 增加 Python/BLAS/CUDA12.8 全库
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
# 交叉构建期宿主工具(如 sleef/mkdisp)经 qemu 运行
set(CMAKE_CROSSCOMPILING_EMULATOR "/usr/bin/qemu-aarch64-static;-L;/path/to/xbuild/noble-sysroot/root")

set(P /path/to/xbuild)
set(SR ${P}/noble-sysroot/root)
set(SBSA ${P}/sbsa-linux)
set(SBSA_T ${P}/sbsa-linux/targets/sbsa-linux)

set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc-14)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++-14)
set(CMAKE_CUDA_COMPILER ${SBSA}/bin/nvcc)
set(CMAKE_CUDA_HOST_COMPILER aarch64-linux-gnu-g++-14)
set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES ${SBSA}/include)
set(CMAKE_CUDA_LIBRARIES ${SBSA}/lib/libcudart.so)
set(CUDA_CUDART_ROOT ${SBSA})
set(CUDA_CUDART_INCLUDE_DIR ${SBSA}/include)
set(CUDA_CUDART_LIBRARY ${SBSA}/lib/libcudart.so)
set(CUDA_TOOLKIT_ROOT_DIR ${P}/sbsa-linux)
set(CUDAToolkit_ROOT ${P}/sbsa-linux)
set(CUDAToolkit_VERSION 12.8.93)
set(CMAKE_IGNORE_PATH /usr/local/cuda /usr/local/cuda-13 /usr/local/cuda-13.0)
set(CMAKE_PREFIX_PATH ${P}/sbsa-linux)

set(CMAKE_SYSROOT ${SR})
set(CMAKE_FIND_ROOT_PATH ${SR} ${SBSA} ${P}/sbsa-linux)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_LIBRARY_PATH ${SBSA}/lib ${SR}/usr/lib/aarch64-linux-gnu ${SR}/usr/lib/aarch64-linux-gnu/blas ${SR}/usr/lib/aarch64-linux-gnu/lapack)
set(CMAKE_INCLUDE_PATH ${SBSA}/include)

# Python: host 解释器跑构建脚本, include/lib 用 sysroot arm64
set(Python3_EXECUTABLE /usr/bin/python3.12)
set(Python3_INCLUDE_DIR ${SR}/usr/include/python3.12)
set(Python3_LIBRARY ${SR}/usr/lib/aarch64-linux-gnu/libpython3.12.so)
set(PYTHON_EXECUTABLE /usr/bin/python3.12)
set(PYTHON_INCLUDE_DIR ${SR}/usr/include/python3.12)
set(PYTHON_LIBRARY ${SR}/usr/lib/aarch64-linux-gnu/libpython3.12.so)
set(PYTHON_LIBRARIES ${SR}/usr/lib/aarch64-linux-gnu/libpython3.12.so)

# BLAS/LAPACK (sysroot reference netlib)
set(BLAS_LIBRARIES ${SR}/usr/lib/aarch64-linux-gnu/blas/libblas.so)
set(LAPACK_LIBRARIES ${SR}/usr/lib/aarch64-linux-gnu/lapack/liblapack.so)
set(LAPACK_INCLUDE_DIR ${SR}/usr/include)

set(CMAKE_C_FLAGS_INIT "--sysroot=${SR} -isystem ${SR}/usr/include -isystem ${SR}/usr/include/aarch64-linux-gnu")
set(CMAKE_CXX_FLAGS_INIT "--sysroot=${SR} -isystem ${SR}/usr/include -isystem ${SR}/usr/include/aarch64-linux-gnu")
set(CMAKE_CUDA_FLAGS_INIT "")
set(CMAKE_EXE_LINKER_FLAGS_INIT "-L${SBSA}/lib -Wl,--allow-shlib-undefined")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "-L${SBSA}/lib -Wl,--allow-shlib-undefined")
