# 源码 pin（随包二进制 = `bin/llama-server-aarch64-sm101a-nvfp4` 的来源）

| 项 | 值 |
|---|---|
| 上游仓库 | `https://github.com/ggml-org/llama.cpp` |
| **commit（pin）** | `72797e89198ab564fd0e6baa54ab196e8dd1d884` |
| commit 时间/标题 | 2026-09-10 10:06:07 +0300 · `vulkan : add command-buffer debug labels for GPU profilers (#28101)` |
| 检出 | `git clone https://github.com/ggml-org/llama.cpp && git -C llama.cpp checkout 72797e89198ab564fd0e6baa54ab196e8dd1d884`（用任意可达镜像/代理均可，我们当年走的是代理前缀） |
| 建议源码路径 | `/home/dev/llama-cpp-build/xbuild/llama-cpp-latest` —— 这是**件里实际嵌的路径**（件里可见 `…/xbuild/llama-cpp-latest/ggml/src/ggml-cuda/mmvq.cu` 这类字符串）；<br>换路径不影响功能，但 sha256 必不同（件 not stripped，DWARF 带绝对路径） |
| 我们的改动 | `patches/01-as-built/`（确定在件里）+ `patches/02-in-binary-post-modified/`（标记命中但为超集）；判定证据见 `patches/PATCH-FIDELITY.tsv` |
| 构建后改动（**不含**） | `patches/POST-BUILD-DRIFT.tsv` |

## 这一 pin 已经自带的东西（不需要我们补）

起草器路径在这个 pin 的上游代码里就有：`common/arg.cpp` / `common/speculative.cpp` / `src/llama-arch.*` 里能搜到 `dflash` 相关符号，
即 `--spec-type draft-dflash` 由**上游**提供；我们额外做的是 NVFP4/FP8 权重侧快路与量化/转换链（见补丁集）。

## 我们改了什么（`patches/01-as-built/`，逐个文件）

| 文件 | 改了哪块 | 为什么 |
|---|---|---|
| `ggml/include/ggml.h` | 新增类型/接口声明 | NVFP4/FP8 快路的开关与类型 |
| `ggml/src/ggml-common.h` | block 布局/记录宽度 | NVFP4 block-scale（每 64-K 记录 36 B）的寻址前提 |
| `ggml/src/ggml.c` | 张量/类型注册 | 同上 |
| `ggml/src/ggml-quants.{c,h}` | 量化/反量化 | 转换链与 CPU 侧参考 |
| `ggml/src/ggml-cuda/vecdotq.cuh` | 点积/反量化 | GPU 侧参考路径 |
| `ggml/src/ggml-cuda/dequantize.cuh` | 反量化 | 同上 |
| `ggml/src/ggml-cuda/convert.cu` | 格式转换 | NVFP4/FP8 互转 |
| `ggml/src/ggml-cuda/fp8-cublaslt.cu` | **新增**：cuBLASLt FP8 快路（env 门控） | 权重侧快路的一条腿（未命中时回落） |
| `ggml/src/ggml-cuda/fp8-mma.cuh` | **新增**：FP8 MMA 原语 | 同上 |
| `ggml/src/ggml-cuda/fp8-mmq.cu` | **新增**：FP8 MMQ 原型 | 同上 |
| `ggml/src/ggml-cuda/fp8-mmq-production.cu` | **新增**：FP8 MMQ 生产接线 | 同上 |
| `convert_hf_to_gguf.py`、`conversion/base.py`、`gguf-py/gguf/{constants,quants}.py` | 转换链 | 从公开 NVFP4 权重转到 GGUF |

新增的 `.cu` 会被 `ggml/src/ggml-cuda/CMakeLists.txt` 的 `file(GLOB "*.cu")` 自动收进构建，
**不需要**改 CMake（这一点已验证：随包二进制里带着 `fp8-cublaslt.cu` 的字符串）。
