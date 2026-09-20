# 官方件 / 第三方件的名字与链接

本包**不再分发**下面这些件（体积与许可各自解决），只登记名字与出处：

| 名称 | 是什么 | 链接 | 备注 |
|---|---|---|---|
| `llama.cpp` | 上游推理引擎源码 | `https://github.com/ggml-org/llama.cpp` | 用 pin `72797e89198ab564fd0e6baa54ab196e8dd1d884`（见 `SOURCE-PIN.md`） |
| 该 pin 的提交页 | 直接看这一版 | `https://github.com/ggml-org/llama.cpp/commit/72797e89198ab564fd0e6baa54ab196e8dd1d884` | 2026-09-10 |
| CUDA Toolkit | 工具链（12.8.x；我们用 nvcc **12.8.93**） | 归档页 `https://developer.nvidia.com/cuda-toolkit-archive` | Linux → aarch64-sbsa 分支 |
| CUDA sbsa 包源（Debian/Ubuntu arm64） | `apt` 用的 NVIDIA sbsa 软件源 | `https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/` | 板上装 runtime 也走这个源 |
| `Qwen3.8-27B` NVFP4 权重 | 我们 target 权重的来源（NVFP4 量化，多分片 safetensors） | `https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4`（国内镜像把域名换成 `hf-mirror.com`） | 转换脚本 = 本包 `patches/01-as-built/` 里的 `convert_hf_to_gguf.py` + `conversion/` + `gguf-py/` |
| DFlash2 起草器 | 外挂 draft 模型（块扩散），`arch=dflash`、**81 张量** | 上游 PR：`https://github.com/ggml-org/llama.cpp/pull/27342` | 这个 PR 的说明里有 checkpoint 的出处；本目录旁边的 `thor-prebuilt-2026-09-19/optional-drafter/` 也放了一档 0.52 GiB 的现成件 |
| 社区实测库 `qwen38-mtp` | 40 人 / 53 配置的社区实测汇总（草稿深度、KV、上下文） | `https://github.com/sudoingX/qwen38-mtp` | 我们 9-08 那次提速的关键参考 |
| FFmpeg / whiper 等 | 与本包无关（视频线用） | — | 不列 |

## 判断"是不是官方件"的简单规则

- **本包 `patches/` 里的东西 = 我们自己的改动**（基于上面 pin 的 diff），不是上游；
- 上表里带 `github.com/ggml-org/`、`developer.nvidia.com`、`huggingface.co` 的 = 官方/上游件；
- 任何**权重文件**（`.gguf` / `.safetensors`）都不在本包里。
