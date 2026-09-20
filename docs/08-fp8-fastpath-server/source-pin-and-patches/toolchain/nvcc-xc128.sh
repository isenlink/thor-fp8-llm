#!/bin/bash
# aarch64 nvcc (CUDA 12.8 sbsa-linux) 包装：在 x86 主机上经 qemu 静态模拟运行这个 aarch64 nvcc
# 要点：-target-dir sbsa-linux / -allow-unsupported-compiler / QEMU_LD_PREFIX 指向 sysroot
set -euo pipefail
THOR_XBUILD=${THOR_XBUILD:-/path/to/xbuild}
: "${THOR_XBUILD:?set THOR_XBUILD to the dir holding noble-sysroot/ and sbsa-linux/}"
SR=$THOR_XBUILD/noble-sysroot/root
CUDA=$THOR_XBUILD/sbsa-linux
CC=${CC_HOST:-aarch64-linux-gnu-g++-14}
export QEMU_LD_PREFIX=$SR
exec qemu-aarch64-static -L "$SR" "$CUDA/bin/nvcc" \
  -target-dir sbsa-linux \
  -ccbin "$CC" \
  -allow-unsupported-compiler \
  -I"$CUDA/targets/sbsa-linux/include" \
  -Xcompiler "--sysroot=$SR" \
  -L"$CUDA/lib" \
  "$@"
