# tqLLM (TurboQuant) 交叉编译移植 Thor + Qwen3.8-27B-TQ_L 200K 长上下文实测（2026-11-09）

> 测试机：Thor 主机（统一配置，可互换，测试时端口仅 22/53 无其他负载）
> 引擎：turbo-tan/tqllm `main @ d7d9f7d`（v1.4.4 线，2026-11-08 push）
> 执行：AI02-Z（HPZ8 上的 ZCode）；本文数字全部对回服务端 timings 原始 JSON
> 上游反馈：已向上游提 PR [turbo-tan/tqllm#32](https://github.com/turbo-tan/tqllm/pull/32)（docs/BENCHMARKS.md 加 Thor 小节）

## 1. 一句话结论

tqLLM 官方不支持 Thor（仅 RTX 30/40/50、GB10/DGX Spark、Mac），但 x86 主机交叉编译
（sm_110a，**零源码改动**）后上板一次成功：Qwen3.8-27B-TQ_L 在 200K 上下文
（prompt 实填 197,913 tok）下 decode **6.28 tok/s（无草稿）→ 15.50 tok/s
（DFlash2 草稿器，2.47×）**，prefill 350 tok/s，带/不带草稿器输出逐字一致（greedy exact）。

## 2. 交叉编译（x86 主机，零源码改动）

- 工具链：nvcc 13.1（/usr/local/cuda-13）+ aarch64-linux-gnu-g++-14（Ubuntu 14.2.0）
- 关键参数：`CMAKE_CUDA_ARCHITECTURES=110a`（板上 GPU 实测 CC 10.1，CUDA 13 口径即 sm_110a）
- 工具链文件要点（CMake toolchain）：

  ```cmake
  set(CMAKE_SYSTEM_NAME Linux)
  set(CMAKE_SYSTEM_PROCESSOR aarch64)
  set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc-14)
  set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++-14)
  set(CMAKE_CUDA_COMPILER /usr/local/cuda-13/bin/nvcc)
  set(CMAKE_CUDA_HOST_COMPILER aarch64-linux-gnu-g++-14)
  set(CUDAToolkit_ROOT /usr/local/cuda-13/targets/aarch64-linux)
  set(CMAKE_CUDA_ARCHITECTURES 110a)
  set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
  set(CMAKE_FIND_ROOT_PATH /usr/local/cuda-13/targets/aarch64-linux)
  ```

  注：cuda-13 工具链自带 targets/aarch64-linux 树，但缺 cusolver/nvrtc/nvJitLink/
  libcudart_static.a 与 libcuda.so stub——从 NV 官方 arm64 镜像
  `nvcr.io/nvidia/cuda:13.1.1-{runtime,devel}` 提取补齐（与 nvcc 13.1 同代）。
- 产物：tqserve 38.9MB，ELF aarch64，单 cubin sm_110a
- 依赖 NEEDED：libcudart.so.13 / libcuda.so.1 / libcusolver.so.12 / libcublas.so.13 /
  libcublasLt.so.13（libcusparse.so.12 经 cusolver 传递依赖，板载需一并带）

## 3. 板上运行时三件套

| 部件 | 来源 | 说明 |
|---|---|---|
| compat_orin 13.2 驱动 | CUDA 13 全栈存档（模型库） | `usr/local/cuda-13.2/compat_orin/`，纯 NVIDIA 驱动库 |
| CUDA 13.2 运行库 | 环境卷存档 + docker 补 cusolver/cusparse | cublas 13.4.1.3、nvrtc/nvJitLink 13.2.86、cudart 13.2.86 |
| tqserve + 模型 | 交叉编译产物 + 模型库 | 见下 |

⚠️ **关键坑位（必读）**：repro-kit 环境卷里的 `native-driver/` compat 包（自带 libc/ld-linux）
在板上 cuInit 即段错误，**不可用**；必须用 `compat_orin` 包（不含 libc，靠板系统 glibc 2.39）。
与 cu132 共用底座先例（LD_LIBRARY_PATH=compat_orin:runtime/lib）一致。

### 模型与权重（sha256 已核验）

- 主模型：`Qwen3.8-27B-TQ_L.gguf`（14,721,469,184 B），来源 HF `YTan2000/Qwen3.8-27B-TQ`
- 草稿器：`drafters/Qwen3.8-27B-DFlash2-TQ3.gguf`（785,487,488 B）
- 草稿词表：`mtp-vocab/atx_65536.txt`（65,536 深度草稿词表）
- TQ 格式说明：TQ_L 为 tqLLM 自研 TurboQuant 格式（TQ3/TQ4 变体），KV 缓存走引擎默认
  （TQ 文件配套 tq3 KV，未显式传 `--kv`）

### 启动命令（板上路径为部署操作目标）

```sh
R=/zeekr_map/tqllm-test-20261109
C=$R/usr/local/cuda-13.2/compat_orin
setsid nohup env LD_LIBRARY_PATH=$C:$R/cuda/lib \
  $R/bin/tqserve --model $R/models/Qwen3.8-27B-TQ_L.gguf \
  -md $R/models/drafters/Qwen3.8-27B-DFlash2-TQ3.gguf \
  --draft-vocab $R/models/mtp-vocab/atx_65536.txt \
  --host 0.0.0.0 --port 8086 --context 200000 \
  </dev/null >$R/serve.log 2>&1 &
```

- 200K 上下文直接被接受（无 fit 降档）；常驻 RAM 峰值 ~53.5G/60G（50G 大页池口径）
- 健康检查 `GET /health`；监控页 `http://<板IP>:8086/monitor`
- ⚠️ ssh 起后台进程必须 `setsid nohup … </dev/null`，否则随 ssh 会话退出被杀（板上老坑）

## 4. 测试结果

配置口径：贪心（temperature=0）；引擎 commit d7d9f7d；同一会话内串行；数值为服务端
`timings` 对象口径（prompt_ms/predicted_ms）；长上下文 prompt 为固定英文百科式文本重复
（8K 基准同源），completion=205 tok 固定；单请求。

### 4.1 长上下文 200K（本次目标，TQ_L）

| 指标 | 无草稿 | DFlash2 草稿器 |
|---|---:|---:|
| prompt tokens | 197,913（≈满 200K 的 99.1%） | 同左（逐字节同 prompt） |
| prefill | **350.8 tok/s**（564 s） | **350.2 tok/s**（565 s，草稿器不参与 prefill） |
| **decode** | **6.28 tok/s**（32.5 s / 205 tok） | **15.50 tok/s**（13.2 s / 205 tok） |
| 加速比 | 1×（基线） | **2.47×** |
| 草稿接受率 | — | 139/462 = **30.1%** |
| 输出质量 | LONGOK + 正确摘要 | 与无草稿逐字一致（exact） |

解读：
- 无草稿 200K decode（6.28）较短上下文（12.1 @2K）近乎腰斩——长 KV 的每 token 注意力
  开销所致，与内存带宽模型一致；草稿器把有效带宽利用拉回（15.5 tok/s ≈ 273 GB/s
  理论上限的 84%）
- 草稿接受率 30.1% 低于短上下文 35-40%（上游 BENCHMARKS 口径），加速比相应从 3.2×
  降到 2.5×——长上下文下摘要类任务草稿命中率略降，属预期行为
- prefill 350 tok/s 时 TTFT ≈ 9.4 分钟（197.9K prompt）；GPU 占用 99%，温度峰值 52°C，稳定

### 4.2 短上下文参照（同引擎同日，2026-11-09 上午，3 次取中位）

| prompt | prefill | decode（无草稿） | decode（DFlash2） |
|---:|---:|---:|---:|
| 2K（1,374 tok） | 780 tok/s | 12.1 tok/s | 38.6 tok/s（3.2×） |
| 8K（5,109 tok） | 920 tok/s | 11.9 tok/s | 39.3 tok/s（3.3×） |

- exact 验证：固定中文提问（雪为什么是白的，max_tokens=300）带/不带草稿器输出**逐字一致**
- 中文推理正确性冒烟：自述模型身份正确、7^6=117,649 计算正确

### 4.3 测试期间板上状态

- 测试机当时无其他负载（端口仅 22/53，无在跑业务）；测试全程未 reboot（板上铁律遵守）
- RAM 峰值 53.5G/60G（tegrastats 口径），lfb 最低 5×4MB；/zeekr_map 占用 19G
  （引擎+库+模型），余 38G

## 5. 已知坑位（复现部署必读）

1. **compat 包选择**：`compat_orin`（无 libc）可用；`native-driver`（自带 libc）段错误
   ——cuInit 层就崩
2. **hf-mirror 断流**：大文件（14.7G）下载两次在 4.8G 处挂死；`hf` 续传认不出旧分片；
   解法 `curl -C - --speed-limit 20480 --speed-time 45` 断点续传 + sha256 终验
   （本机千兆约 11 MB/s）
3. **板上工具**：无 curl、无 rsync；scp 可用；后台进程配方 `setsid nohup … </dev/null`
4. **词表/草稿器路径**：tqserve 要求 `-md` 与 `--draft-vocab` 路径精确，文件拍平放置会报
   `bad draft vocab file`
5. **200K 直接可用**：`--context 200000` 无需调大页池（50G 池内装下 197.9K prompt + 205 输出）

## 6. 与本仓既有线的关系

- 本线 = **第三条引擎线**（llama.cpp 线 / vLLM 线 / tqLLM 线），2026-11-09 一次性移植验证，
  未作生产部署（测试后服务已停）
- 同模型（Qwen3.8-27B）跨引擎 200K 档对照（不同量化/不同引擎，仅供量级参考，不可直比）：
  - llama.cpp 线（NVFP4 + DFlash2）：200K decode 22 t/s（T4/ARES 定案版，见 README）
  - vLLM 0.32 cu13 线（NVFP4 + DFlash2）：≥200K 档见 docs/11 vllm032-line
  - tqLLM 线（TQ_L + DFlash2）：200K decode 15.50 t/s（本文）
- tqLLM 的 TQ 量化格式与 llama.cpp GGUF / vLLM safetensors 互不通用，模型权重需单独转换
