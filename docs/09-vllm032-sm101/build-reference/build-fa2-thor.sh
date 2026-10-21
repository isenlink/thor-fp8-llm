#!/usr/bin/env bash
set -euo pipefail
ROOT=/opt/thor-cc/vllm032-host
X=/opt/thor-cc/xbuild
W=/opt/thor-cc/thorpush
T13=$W/torch213-src/pytorch-2.13.0/build/lib.linux-x86_64-cpython-312/torch
export TORCH_CUDA_ARCH_LIST=10.1
export TORCH_INSTALL_PREFIX="$T13"
export LD_LIBRARY_PATH="$X/sbsa-linux/lib:$X/noble-sysroot/root/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
"$W/vvenv/bin/cmake" -S "$ROOT/optimization-20261020/fa2-src" -B "$ROOT/optimization-20261020/fa2-build" -G Ninja \
 -DCMAKE_TOOLCHAIN_FILE="$X/toolchain-torch-xc128.cmake" \
 -DCMAKE_CUDA_COMPILER="$X/sbsa-linux/bin/nvcc" \
 -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCHS=10.1 -DNVCC_THREADS=2 \
 -DVLLM_PYTHON_EXECUTABLE="$W/vvenv/bin/python" \
 -DPython_EXECUTABLE="$W/vvenv/bin/python" \
 -DPython_INCLUDE_DIR="$X/noble-sysroot/root/usr/include/python3.12" \
 -DPython_LIBRARY="$X/noble-sysroot/root/usr/lib/aarch64-linux-gnu/libpython3.12.so" \
 -DTorch_DIR="$T13/share/cmake/Torch" -DCaffe2_DIR="$T13/share/cmake/Caffe2"
"$W/vvenv/bin/cmake" --build "$ROOT/optimization-20261020/fa2-build" --target _vllm_fa2_C -j 8
