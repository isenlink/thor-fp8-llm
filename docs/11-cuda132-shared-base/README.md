# 11 · CUDA 13.2 共用底座（forward-compat 驱动 + runtime + llama.cpp sm_110a）

> **方法来源致谢**：CUDA 13 forward-compat 在 Thor 上的方法来自网友
> **[souyou233 (EnkiMa)](https://github.com/souyou233)** 的公开工作——
> [Minimax-H3-on-thor-u](https://github.com/souyou233/Minimax-H3-on-thor-u)（H3 视频 +
> CUDA 13 前向兼容完整复盘）与 [Qwen-Image-on-thor-u](https://github.com/souyou233/Qwen-Image-on-thor-u)
> （Qwen-Image 2.1，共用同一 cu130 运行时）。本目录把同一方法用于 llama.cpp 线：
> CUDA 13.2 原生 `sm_110a` 编译 + 13.2 runtime + 13.2 compat 驱动。

## 核心思路（一句话）

把 NVIDIA 官方 CUDA 13 forward-compat `libcuda.so.1` + JIT 伴侣库放在
`LD_LIBRARY_PATH` 最前面（系统 `libnvrm_*` 保留），让 CUDA 13 工具链/运行时在
CUDA 12 时代的内核驱动上跑起来。souyou233 用它解开 `comfy_kitchen` 的
`cuda_version < (13,)` 版本门（H3/image 线，原生 int8/fp8，1.9× 提速）；
本线用它跑 llama.cpp 的 13.2 原生 `sm_110a` 编译产物。

**LLM / H3 / Image / vLLM 四条部署线共用本底座的 compat + runtime**，各自只带自己的
模型与启动脚本——这是多线对齐的关键。vLLM 0.32 生产部署线实录见
[vllm032-line/vllm032-cu13-deployment.md](vllm032-line/vllm032-cu13-deployment.md)。

> **H3 线现状（2026-10-22）**：CUDA 13 前向兼容已成功复现（Thor02，
> ComfyUI 0.38 + torch 2.13+cu130 + comfy_kitchen 0.2.39，端口 8189）——8 步生产
> LoRA 同口径 289.4 s/段 vs sd.cpp 1049 s/段 = **3.63×**，20 段整批 19/20 过门禁。

## 下载（ModelScope）

**共用底座**：<https://www.modelscope.cn/models/navyyang/thor-cuda132-shared>

| 件 | 说明 | 大小 |
|---|---|---|
| `compat/compat-orin-13.2.tar.gz` | CUDA 13.2 forward-compat 驱动库（`libcuda.so.1.1` + 12 个 JIT 伴侣） | 134 MB |
| `runtime/cu132-runtime.tar` | CUDA 13.2.86 runtime + toolkit（cudart/cublas/cublasLt/nvrtc + include + nvcc 工具链） | 1.17 GB |
| `bin/llama-server-cu132-sm110a` | llama.cpp 服务端，13.2 原生 `sm_110a`（144 个 cubin），DFlash2 + mmproj | 81 MB |
| `src/llama-cpp-thor-t4ares-workingtree-20260928.tar.gz` | llama.cpp 工作树快照（基线 `72797e89` + T4/DFlash 补丁） | 38 MB |
| `scripts/` | 交叉编译脚本与工具链（build/toolchain/nvcc 包装/最小探针） | — |
| `run-8080-cu132.sh` / `run-8082-cu132.sh` | 8080（27B+DFlash2）/ 8082（VL-8B）启动脚本（端口幂等） | — |
| `systemd/` | 两个服务 unit 母本（oneshot + RemainAfterExit，`active (exited)` 正常） | — |
| `rollback-to-cu128.sh` | 一键回滚 CUDA 12.8 生产 | — |
| `docs/` + `evidence/` | 板端实测文档 + 验收证据 | — |

下载后先 `sha256sum -c SHA256SUMS.txt` 再使用。

**模型权重**（不随底座分发）：
- 8080 三件（RadixArk-F8attn-v2 + DFlash2-BF16 + mmproj-HauhauCS）：
  **ModelScope** <https://www.modelscope.cn/models/navyyang/qwen38-27b-gguf-t4ares-thor01>
  （Apache 2.0）；上游 HF：[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)、
  [incoai/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF)、
  [HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF](https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF)
- 8082 两件（Qwen3-VL-8B-Instruct-Q6_K + mmproj-F16）：
  **HF** [unsloth/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen3-VL-8B-Instruct-GGUF)
  （基座 [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)，Apache 2.0）

## 板端部署（快速开始）

```bash
# 1) 底座落持久层（/zeekr_data 是持久层；/home、/etc 是 overlay 会丢）
cd /zeekr_data/ai_workspace/thor01-cu132-20261022
tar xzf compat/compat-orin-13.2.tar.gz -C compat/
tar xf runtime/cu132-runtime.tar -C runtime/
ln -sf ../../compat/compat_orin/libcuda.so.1.1 runtime/lib/libcuda.so

# 2) 模型就位（见上方链接），启动
bash run-8080-cu132.sh   # 27B + DFlash2 + mmproj，端口 8080
bash run-8082-cu132.sh   # Qwen3-VL-8B，端口 8082（等 8080 就绪后再起）

# 3) 验证（板端无 curl，用 python3）
python3 -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=5).read().decode())"
```

生产环境用 `systemd/` unit 母本（拷 `/etc/systemd/system/` + `daemon-reload` +
`enable` + `start`），unit 母本必须放持久层并由掉电恢复脚本从持久母本重装
（底座仓 `docs/PERSISTENCE-20261022.md` 有完整核对）。

## 实测数据（Thor01，2026-10-22）

- 27B（RadixArk + DFlash2 + mmproj）加载 20.7 s，systemd 启动 21 s；
  96-token 请求 decode 30.19 t/s，DFlash2 接受率 74.5%（79/106）。
- VL-8B 加载 6.1 s；纯文本 decode 21.8 t/s；64×64 图片题正确返回。
- 期间 dmesg 无 Xid、无 OOM、无 timeout。
- 大页池 25600×2MB（50 GB）；8082 加载后 HugePages_Free 可低至几十页（正常）。

## 边界与已知问题（必读）

1. **T4-MMQ 快路未生效**：13.2 compat 把 Thor 报成 `cc=11.0`（1100），
   T4 守卫只接受 `10.1a`（1010），因此本二进制走通用 CUDA 路径，
   速度**不是** T4 优化后的速度。恢复 T4 的两条路（13.0.88 编译 sm_101a +
   13.2 runtime，或扩展 T4 守卫到 1100）见底座仓
   `docs/THOR01-CU132-RESULT-AND-THREE-STACK-20261022.md` §6。
2. **上游 issue 27763 风险仍在**：社区报告 13.2 编译的 SM110 cubin 在
   >8 层 offload 时输出确定性垃圾（13.0.88 正常、13.3 修复）。本线 27B/VL
   实测输出正常，但那是单模型少量 prompt 的质量门槛，**每次换模型/换构建
   必须做真实输出验收，不能只看 `/health`**。
3. **生产优先 13.0.88 编译 + 13.2 runtime**；13.2 原生编译作为实验分支。
4. 与旧 CUDA 12.8 生产（T4/ARES 内核，200K decode 22 t/s）不是严格 A/B，
   表面速度差异不能归因于 CUDA 版本（prompt/输出 token 数不同）。

## 许可

- NVIDIA CUDA 13.2 compat 驱动库与 runtime：NVIDIA 专有软件，按
  [CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/index.html) 二进制再分发
  （EULA 允许），原样打包未修改。
- llama.cpp：[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) MIT +
  生产 T4/DFlash 补丁（工作树内可对照）。
- 模型：全部 Apache 2.0 基座（见上方链接）。
- 溯源链：底座仓 `NOTICE.md`。
