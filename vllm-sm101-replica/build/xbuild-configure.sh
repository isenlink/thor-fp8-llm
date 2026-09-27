#!/bin/bash
# NOTE: /path/to/* are placeholders — set them to your own workspace.
#       Original host-internal paths were removed for publication.
# torch 2.9.0 交叉编译 sm_101 (x86_64 host → Thor aarch64, CUDA 12.8)
set -uxo pipefail
cd /path/to/work/pytorch-src
source /path/to/work/xvenv/bin/activate 2>/dev/null || {
  python3.12 -m venv /path/to/work/xvenv && source /path/to/work/xvenv/bin/activate
  pip install -q --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
  pip install -q -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 | tail -1
  pip install -q cmake ninja -i https://pypi.tuna.tsinghua.edu.cn/simple
}

export CROSS_SYSROOT=/path/to/xbuild/noble-sysroot/root
export CMAKE_TOOLCHAIN_FILE=/path/to/xbuild/toolchain-torch-xc128.cmake
export CUDACXX=/path/to/xbuild/nvcc-xc128.sh
export CUDAHOSTCXX=aarch64-linux-gnu-g++-14
export TORCH_CUDA_ARCH_LIST="10.1"
export PYTORCH_BUILD_VERSION=2.9.0 PYTORCH_BUILD_NUMBER=0
export MAX_JOBS=32
export USE_CUDNN=0 BUILD_TEST=0 USE_ROCM=0 USE_MPI=0
export USE_ITT=0  # ittapi 有 host 二进制坑, 关闭
export CMAKE_ONLY=1

echo "=== CROSS CONFIGURE START $(date +%T) ==="
python3 setup.py bdist_wheel --cmake-only > /path/to/work/cross-cfg.log 2>&1
echo "=== CONFIGURE rc=$? $(date +%T) ==="
grep -aE "CMake Error|Could NOT find|error:" /path/to/work/cross-cfg.log | head -10
