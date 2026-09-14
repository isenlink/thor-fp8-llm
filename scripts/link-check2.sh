#!/bin/bash
# Author: AI assistant
set -euo pipefail
task_dir=$HOME/work/thor-ai-assistant
build_dir=$HOME/work/thor-driveos/xbuild/build-f8shim
tool_dir=$HOME/work/thor-driveos/xbuild
aarch64-linux-gnu-g++-14 --sysroot="$tool_dir/noble-sysroot/root" -O2 -std=c++17 \
  "$task_dir/b3-correctness2.cpp" "$task_dir/mmvq.cu.o" "$task_dir/ggml-cuda.cu.o" \
  -I"$tool_dir/llama-cpp-latest/ggml/include" -Wl,--start-group \
  "$build_dir/ggml/src/libggml.a" "$build_dir/ggml/src/libggml-base.a" \
  "$build_dir/ggml/src/libggml-cpu.a" "$build_dir/ggml/src/ggml-cuda/libggml-cuda.a" \
  -Wl,--end-group -L"$tool_dir/sbsa-linux/lib" -Wl,--allow-shlib-undefined \
  -lcudart -lcublas -lcublasLt -lcuda -lgomp -lpthread -ldl -o "$task_dir/b3-correctness2"
