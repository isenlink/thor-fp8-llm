#!/bin/bash
# Thor (aarch64, sm_101a) 交叉编译脚本 — CUDA 12.8 版（现网 DriveOS 7.0.3 板专用）
# 产物在现板可直接跑（驱动报告 CUDA 12.8.90）

# 用法: ./build-thor-cuda128.sh <源码目录> [build目录] [-j N]
set -e
P=/data3/works/thor-driveos/xbuild
SRC=${1:?用法: build-thor-cuda128.sh <源码目录> [build目录]}
BUILD=${2:-$P/build-cuda128-$(basename $SRC)}
JOBS=${3:--j 96}

cmake -DCMAKE_TOOLCHAIN_FILE=$P/toolchain-thor-cuda128.cmake -B $BUILD -S $SRC \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=101a \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_CUDA=ON -DGGML_CPU=ON -DGGML_CPU_AARCH64=ON
cmake --build $BUILD $JOBS
echo "产物: $BUILD/bin/ ($(ls $BUILD/bin/ | wc -l) 个)"
