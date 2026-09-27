#!/bin/bash
# NOTE: /path/to/* are placeholders — set them to your own workspace.
#       Original host-internal paths were removed for publication.
# torch 2.9.0 交叉编译 sm_101 全量构建 (x86_64 host (96 cores) → Thor aarch64)
set -uxo pipefail
cd /path/to/work/pytorch-src
source /path/to/work/xvenv/bin/activate

export CROSS_SYSROOT=/path/to/xbuild/noble-sysroot/root
export CMAKE_TOOLCHAIN_FILE=/path/to/xbuild/toolchain-torch-xc128.cmake
export CUDACXX=/path/to/xbuild/sbsa-linux/bin/nvcc
export CUDAHOSTCXX=aarch64-linux-gnu-g++-14
export TORCH_CUDA_ARCH_LIST="10.1"
export PYTORCH_BUILD_VERSION=2.9.0 PYTORCH_BUILD_NUMBER=0
export MAX_JOBS=16
export USE_CUDA=1 USE_CUDNN=0 USE_CUFILE=0 USE_CUDSS=0 BUILD_TEST=0 USE_ROCM=0 USE_MPI=0 USE_ITT=0 USE_RPC=0
export USE_XNNPACK=0 USE_QNNPACK=0
export CC=aarch64-linux-gnu-gcc-14 CXX=aarch64-linux-gnu-g++-14
export LDSHARED="aarch64-linux-gnu-g++-14 --sysroot=/path/to/xbuild/noble-sysroot/root -shared"
export LDFLAGS="-L/path/to/xbuild/sbsa-linux/lib -L/path/to/work/pytorch-src/torch/lib --sysroot=/path/to/xbuild/noble-sysroot/root"
export CFLAGS="-I/path/to/xbuild/noble-sysroot/root/usr/include/python3.12 -I/path/to/xbuild/noble-sysroot/root/usr/include -isystem /path/to/xbuild/noble-sysroot/root/usr/include/aarch64-linux-gnu"
export CPPFLAGS="$CFLAGS"
export USE_PRIORITIZED_TEXT_FOR_LD=0
export SLEEF_TARGET_EXEC_USE_QEMU=1
export CMAKE_CROSSCOMPILING_EMULATOR="/usr/bin/qemu-aarch64-static;-L;/path/to/xbuild/noble-sysroot/root"

echo "=== CROSS BUILD START $(date +%T) ==="
python3 setup.py bdist_wheel > /path/to/work/cross-build.log 2>&1
rc=$?
echo "=== CROSS BUILD END rc=$rc $(date +%T) ==="
ls -la dist/ || true
