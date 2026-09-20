#!/bin/bash
# Thor (aarch64, sm_101a) 交叉编译 llama.cpp — CUDA 12.8 版
# 用法: THOR_XBUILD=/path/to/xbuild ./build-thor-cuda128.sh <源码目录> [build目录] [-j N]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
SRC=${1:?用法: build-thor-cuda128.sh <源码目录> [build目录] [-j N]}
BUILD=${2:-build-thor-cuda128}
JOBS=${3:--j 64}
cmake -DCMAKE_TOOLCHAIN_FILE="$here/toolchain-thor-cuda128.cmake" -B "$BUILD" -S "$SRC" \
  -DTHOR_XBUILD="${THOR_XBUILD:-/path/to/xbuild}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=101a \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_CUDA=ON -DGGML_CPU=ON -DGGML_CPU_AARCH64=ON
cmake --build "$BUILD" $JOBS
echo "产物: $BUILD/bin/ ($(ls "$BUILD/bin" | wc -l) 个)"
