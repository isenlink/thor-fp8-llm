# Author: AI assistant
include $HOME/work/thor-driveos/xbuild/build-f8shim/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/flags.make

$HOME/work/thor-ai-assistant/mmvq.cu.o:
	cd $HOME/work/thor-driveos/xbuild/build-f8shim/ggml/src/ggml-cuda && /usr/local/cuda-12.8/bin/nvcc -allow-unsupported-compiler -I$HOME/work/thor-driveos/xbuild/sbsa-linux/include -Xcompiler --sysroot=$HOME/work/thor-driveos/xbuild/noble-sysroot/root -forward-unknown-to-host-compiler -ccbin=aarch64-linux-gnu-g++-14 $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c $HOME/work/thor-driveos/xbuild/llama-cpp-latest/ggml/src/ggml-cuda/mmvq.cu -o $@
