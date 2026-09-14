#!/bin/bash
# Author: AI assistant
set -euo pipefail
cd $HOME/work/thor-driveos/xbuild/build-f8shim/tools/server
task_dir=$HOME/work/thor-ai-assistant
tool_dir=$HOME/work/thor-driveos/xbuild
aarch64-linux-gnu-g++-14 --sysroot="$tool_dir/noble-sysroot/root" -O3 \
  CMakeFiles/llama-server.dir/main.cpp.o "$task_dir/mmvq.cu.o" \
  -o "$task_dir/llama-server-ai-assistant-b3fixed" \
  libllama-server-impl.a libserver-context.a ../../common/libllama-common.a \
  ../../common/libllama-common-base.a ../mtmd/libmtmd.a ../../src/libllama.a \
  ../../ggml/src/libggml.a -ldl ../../ggml/src/libggml-cpu.a \
  ../../ggml/src/ggml-cuda/libggml-cuda.a ../../ggml/src/libggml-base.a \
  -L"$tool_dir/sbsa-linux/lib" -Wl,--allow-shlib-undefined \
  -lgomp -lpthread -lm -lcudart -lcublas -lcublasLt -lculibos -lcuda \
  ../../vendor/hash/libvendor-hash.a ../ui/libllama-ui.a ../../vendor/cpp-httplib/libcpp-httplib.a
