#!/usr/bin/env bash
set -euo pipefail

X=/opt/thor-cc/xbuild
W=/opt/thor-cc/thorpush
ROOT=/opt/thor-cc/vllm032-host
SRC=$ROOT/vllm-src
T13=$W/torch213-src/pytorch-2.13.0/build/lib.linux-x86_64-cpython-312/torch
DIST=$ROOT/dist

test "$(git -C "$SRC" rev-parse HEAD)" = 5df63a7203cdb9328d824a27b682e77c7b9928be
test -f "$T13/share/cmake/Torch/TorchConfig.cmake"
test -x "$X/sbsa-linux/bin/nvcc"
aarch64-linux-gnu-g++-14 --version | grep -q '14\.'

mkdir -p "$DIST"
export SETUPTOOLS_SCM_PRETEND_VERSION=0.32.0
export MAX_JOBS=14
export NVCC_THREADS=2
export VLLM_TARGET_DEVICE=cuda
export TORCH_CUDA_ARCH_LIST=10.1
export VLLM_MAIN_CUDA_VERSION=12.8
export GIT_ALLOW_PROTOCOL=file:https
export VLLM_CUTLASS_SRC_DIR=$W/.deps/cutlass-4.7.1
export QUTLASS_SRC_DIR=$W/.deps/qutlass-e74319e3405ce6d71965732880f5dc1f52371f64
export FLASH_MLA_SRC_DIR=$W/.deps/FlashMLA-0eee43b12f034b657133cf2afca6a72ebb6efccf
export DEEPGEMM_SRC_DIR=$SRC/.deps/deepgemm-src
export DEEPSELECT_SRC_DIR=$SRC/.deps/deepselect-src
export FMHA_SM100_SRC_DIR=$SRC/.deps/fmha_sm100-src
export FLASH_KDA_SRC_DIR=$ROOT/deps/FlashKDA-17a037d
export VLLM_FLASH_ATTN_SRC_DIR=$SRC/.deps/vllm-flash-attn-src
export TML_FA4_SRC_DIR=$W/.deps/tml-fa4-75765e76a9c2c012c1f6ecd64577eb646eb4d303
export TRITON_KERNELS_SRC_DIR=$W/.deps/triton-kernels-v351/python/triton_kernels/triton_kernels
export CMAKE_ARGS="-DCMAKE_TOOLCHAIN_FILE=$X/toolchain-torch-xc128.cmake -DCMAKE_CUDA_COMPILER=$X/sbsa-linux/bin/nvcc -DPython_INCLUDE_DIR=$X/noble-sysroot/root/usr/include/python3.12 -DPython_LIBRARY=$X/noble-sysroot/root/usr/lib/aarch64-linux-gnu/libpython3.12.so -DTorch_DIR=$T13/share/cmake/Torch -DCaffe2_DIR=$T13/share/cmake/Caffe2"
export TORCH_INSTALL_PREFIX=$T13
export LD_LIBRARY_PATH=$X/sbsa-linux/lib:$X/noble-sysroot/root/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}

cd "$SRC"
exec "$W/vvenv/bin/python" -m pip wheel . --no-deps --no-build-isolation -w "$DIST" -v
