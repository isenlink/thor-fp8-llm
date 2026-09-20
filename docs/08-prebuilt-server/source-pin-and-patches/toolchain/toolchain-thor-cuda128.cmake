# Thor (aarch64) 交叉编译工具链 — CUDA 12.8 / sm_101a
# 用法: cmake -DCMAKE_TOOLCHAIN_FILE=toolchain-thor-cuda128.cmake -B build -S <llama.cpp 源码目录>
# 先把 THOR_XBUILD 指到你的目录（该目录下要有 noble-sysroot/root 与 sbsa-linux/）
set(THOR_XBUILD "/path/to/xbuild" CACHE PATH "内含 aarch64 sysroot 与 CUDA sbsa")
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(P ${THOR_XBUILD})
set(SR ${P}/noble-sysroot/root)
set(SBSA ${P}/sbsa-linux)
set(SBSA_T ${P}/sbsa-linux/targets/sbsa-linux)

set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc-14)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++-14)
set(CMAKE_CUDA_COMPILER ${P}/nvcc-xc128.sh)
set(CMAKE_CUDA_HOST_COMPILER aarch64-linux-gnu-g++-14)
set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES ${SBSA}/include)
set(CMAKE_CUDA_LIBRARIES ${SBSA}/lib/libcudart.so)
set(CUDA_CUDART_ROOT ${SBSA})
set(CUDA_CUDART_INCLUDE_DIR ${SBSA}/include)
set(CUDA_CUDART_LIBRARY ${SBSA}/lib/libcudart.so)
set(CUDA_TOOLKIT_ROOT_DIR ${P}/sbsa-linux)
set(CUDAToolkit_VERSION 12.8.93)
set(CMAKE_IGNORE_PATH /usr/local/cuda /usr/local/cuda-13 /usr/local/cuda-13.0)
set(CMAKE_PREFIX_PATH ${P}/sbsa-linux)

set(CMAKE_SYSROOT ${SR})
set(CMAKE_FIND_ROOT_PATH ${SR} ${SBSA} ${P}/sbsa-linux)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_LIBRARY_PATH ${SBSA}/lib)
set(CMAKE_INCLUDE_PATH ${SBSA}/include)

set(CMAKE_C_FLAGS_INIT "--sysroot=${SR} -isystem ${SR}/usr/include -isystem ${SR}/usr/include/aarch64-linux-gnu")
set(CMAKE_CXX_FLAGS_INIT "--sysroot=${SR} -isystem ${SR}/usr/include -isystem ${SR}/usr/include/aarch64-linux-gnu")
set(CMAKE_CUDA_FLAGS_INIT "")
set(CMAKE_EXE_LINKER_FLAGS_INIT "-L${SBSA}/lib -Wl,--allow-shlib-undefined -lcudart -lcublas -lcublasLt -lcuda")
