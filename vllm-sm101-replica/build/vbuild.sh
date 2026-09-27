#!/bin/bash
# NOTE: /path/to/* are placeholders — set them to your own workspace.
#       Original host-internal paths were removed for publication.
# vLLM 0.11.2 交叉编译 (x86_64 host_64 -> Thor aarch64 sm_101)
# 重建版 2026-09-25：工作区迁移 /path/to/work（/tmp 版因重启丢失）
set -e
X=/path/to/xbuild
W=/path/to/work
export SETUPTOOLS_SCM_PRETEND_VERSION=0.11.2  # GitHub tarball 无 .git，scm 取不到版本
export MAX_JOBS=16
export NVCC_THREADS=2
export VLLM_TARGET_DEVICE=cuda
export TORCH_CUDA_ARCH_LIST="10.1"
export VLLM_CUTLASS_SRC_DIR=$W/.deps/cutlass-4.2.1
export QUTLASS_SRC_DIR=$W/.deps/qutlass-830d2c4537c7396e14a02a46fbddd18b5d107c65
export FLASH_MLA_SRC_DIR=$W/.deps/FlashMLA-46d64a8ebef03fa50b4ae74937276a5c940e3f95
export VLLM_FLASH_ATTN_SRC_DIR=$W/.deps/flash-attention-58e0626a692f09241182582659e3bf8f16472659
export CMAKE_ARGS="-DCMAKE_TOOLCHAIN_FILE=$X/toolchain-torch-xc128.cmake -DCMAKE_CUDA_COMPILER=$X/sbsa-linux/bin/nvcc -DPython_INCLUDE_DIR=$X/noble-sysroot/root/usr/include/python3.12 -DPython_LIBRARY=$X/noble-sysroot/root/usr/lib/aarch64-linux-gnu/libpython3.12.so"
export LD_LIBRARY_PATH=$X/sbsa-linux/lib:$X/noble-sysroot/root/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH

cd $W/vllm-src
# 注意：勿在 vbuild.sh 内 rm -rf build —— 会丢增量；仅在被污染时手动清
exec $W/vvenv/bin/python -m pip wheel . --no-deps --no-build-isolation -w $W/dist -v
