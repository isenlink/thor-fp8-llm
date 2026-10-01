# 09 · Thor01 T4/ARES 生产部署（Qwen3.8-27B · DFlash2 · 262K）

> One-liner: production deployment package for Qwen3.8-27B (NVFP4-FFN + F8-attention) with
> DFlash2 speculative decoding and multimodal (mmproj) on NVIDIA DRIVE Thor (sm_101a):
> 262144 context, 200K decode measured 21.9–22.5 t/s, short-prompt 34.5–34.8 t/s.

> 本目录 = 2026-10-01 定稿的生产部署包（AI 助手 Codex 整理、双 AI 复核）。**预编译二进制不在本仓**
> （体积原因），发布在 ModelScope，链接与校验和见下。

## 预编译二进制下载

**ModelScope**（含二进制 + 源码快照 + 测试数据 + 全件 SHA256SUMS）：
🔗 https://www.modelscope.cn/models/navyyang/thor01-qwen38-27b-dflash2-t4ares-deploy

| 文件 | 大小 | sha256（前 8 位） |
|---|---|---|
| `bin/llama-server-m2c-t4` | 79,029,552 B | `2066e681…` |
| `src/llama-cpp-thor-t4ares-workingtree-20260928.tar.gz` | 38,257,130 B | `2ef20ca5…` |

（全 13 件校验和清单 `SHA256SUMS.txt` 在 ModelScope 仓内；下载后务必 `sha256sum -c` 核验。）

## 运行环境

- NVIDIA DRIVE Thor（p3960 / Tegra264 / sm_101a），DriveOS 7.0.3，CUDA 12.8（驱动报 12.8.90）
- 统一内存 58 GB；GPU 大页池 **23552 页（46 GiB）**——KV f16 @262K ≈ 18.3 GB 连同权重在池内
- 池必须冷启动分配（改 `vm.nr_hugepages` 后重启一次；在线加页会碎片化，见
  [05-system-tuning/hugepage-pool.md](../05-system-tuning/hugepage-pool.md)）
- 板端目录约定：模型放 `/brand_data/ai_workspace/models/`；运行时 `/tmp/c58t4/m2c/`
  （tmpfs 重启丢失，二进制需重传或改放持久分区）

## 模型清单（3 件，sha256 已核验）

| 文件 | 来源 | sha256 |
|---|---|---|
| `RadixArk-F8attn-v2.gguf`（主模型，NVFP4 FFN + F8 注意力投影，qwen35 arch） | **内部量化件，未公开发布**；基座 [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)。转换链见 [03-model-conversion](../03-model-conversion/README.md) | `d15dac91…2e7e9` |
| `Qwen3.8-27B-DFlash2-BF16.gguf`（投机草稿，1.9B） | [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) | `26d47ca2…736bc` |
| `mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf`（多模态投影） | HuggingFace 作者 HauhauCS | `5681b690…2dd142` |

> 复现必须拿到第一件（内部件）；后两件可按链接自行下载并核对 sha。

## 优化方式（相对上游 llama.cpp 的改动）

源码 = llama.cpp base `72797e89`（2026-09-10）+ `source/workingtree-diff.patch`（约 104 KB，
26 个文件）。与推理速度直接相关的点：

- **T4 MMQ 内核**（`ggml/src/ggml-cuda/t4_mmq*.cuh`）：tcgen05 NVFP4 GEMV，生产参数
  `T4_CPS=7 T4_STAGES=2`；**ARES**：A 矩阵常驻 smem（`ABD=8 cpr=20 ACAP=68`），
  per-stage A 搬运在编译期删除
- CUDA graph 优化 `GGML_CUDA_GRAPH_OPT=1`；`GGML_MMVQ_MAX=2`；`GGML_NVFP4_WIDE_MAX=0`
- 投机解码 `--spec-type draft-dflash --spec-draft-n-max 7`（草稿 GGUF `dflash.block_size=8`，
  n_max 已在硬帽）
- 运行时环境变量全集见 `source/deploy.sh`（板上 start/stop/restart/status 管理脚本）

量化谱系：主模型为内部 NVFP4 混合量化（F8 注意力投影 + NVFP4 FFN 权重），草稿为官方 BF16。

> 谱系说明：本目录是 `tcgen05` 分支那条"T4 实验线"的**定案发布版**（精度已定论、
> 与生产二进制逐字一致）；该实验分支的过程记录保留不动。

## 源码构建

`source/` 内容：`workingtree-diff.patch` + `base-commit.txt`（自行打补丁）、
完整工作树快照（ModelScope 仓内，推荐直接用）、`toolchain-thor-cuda128-sbsa.cmake`、
`build-thor-cuda128.sh`、`deploy.sh`（板端管理脚本）。

```bash
# x86_64 主机交叉编译（需 aarch64-linux-gnu-gcc-14 + CUDA 12.8 sbsa 工具包）
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 72797e89198ab564fd0e6baa54ab196e8dd1d884
git apply /path/to/workingtree-diff.patch
cmake -DCMAKE_TOOLCHAIN_FILE=toolchain-thor-cuda128-sbsa.cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=101a \
  -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DGGML_CPU=ON -DGGML_CPU_AARCH64=ON
cmake --build build -j
# 产物 build/bin/llama-server 应与 ModelScope 包内 bin/llama-server-m2c-t4 一致
# （sha256 2066e681…；注意 not-stripped DWARF 内嵌源码绝对路径，换构建路径 sha 必不同——
#   判同一性比对 .text 段或用官方 verify 思路，见 08 目录 BUILD.md 三条验收标准）
```

## 部署步骤（板端）

```bash
# 1) 模型拷贝到持久分区固定路径
/brand_data/ai_workspace/models/RadixArk-F8attn-v2.gguf
/brand_data/ai_workspace/ai02/models/Qwen3.8-27B-DFlash2-BF16.gguf
/brand_data/ai_workspace/models/mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf
# 2) 二进制与脚本
mkdir -p /tmp/c58t4/m2c && cp bin/llama-server-m2c-t4 /tmp/c58t4/m2c/ && chmod +x /tmp/c58t4/m2c/llama-server-m2c-t4
# 3) 启动（脚本内含完整环境变量与启动参数；start 拒绝覆盖已在跑的 8080）
./bin/deploy.sh start     # status / stop / restart 同参
# 4) 验证
curl --noproxy '*' http://<板IP>:8080/health        # {"status":"ok"}
curl --noproxy '*' http://<板IP>:8080/v1/models     # n_ctx=262144, ftype=NVFP4, 含 multimodal
```

启动日志应出现（缺一即配置未生效）：
`T4-MMQ: 环配置覆盖 T4_CPS=7 T4_STAGES=2`、`T4-MMQ: A 常驻核上线 ABD=8 …`、
`adding speculative implementation 'draft-dflash'`、`loaded multimodal model`。

等效启动命令（`deploy.sh` 内原样）：

```bash
env T4_MMQ=1 T4_CPS=7 T4_STAGES=2 T4_ARES=1 \
    GGML_CUDA_GRAPH_OPT=1 GGML_MMVQ_MAX=2 GGML_NVFP4_WIDE_MAX=0 LLAMA_SPEC_TIMING=1 \
/tmp/c58t4/m2c/llama-server-m2c-t4 \
  -m /brand_data/ai_workspace/models/RadixArk-F8attn-v2.gguf \
  -md /brand_data/ai_workspace/ai02/models/Qwen3.8-27B-DFlash2-BF16.gguf \
  --mmproj /brand_data/ai_workspace/models/mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf \
  --mmproj-offload --alias qwen3.8-27b --reasoning-effort medium \
  -ngl 99 -c 262144 -fa on --cache-type-k f16 --cache-type-v f16 \
  --parallel 1 --port 8080 --host 0.0.0.0 \
  --spec-type draft-dflash --spec-draft-n-max 7
```

## 测试方式

`test/bench8091.py <target_tokens> <trial名>`：构造确定性的 ~target 长度 prompt，
经 `/completion` 测一轮（默认 n_predict=384），输出服务端 timings
（prompt/decode 分离计时 + draft acceptance）。

```bash
python3 test/bench8091.py 199980 mytrial   # 200K 口径（脚本内地址 127.0.0.1:8080，板上跑）
```

判读：`timings.predicted_per_second` 即 decode t/s；`prompt_per_second` 为 prefill 速度；
服务端日志 `draft acceptance = … mean len = …` 给投机接受率。

## 实测数据（原始 jsonl 在 `test/`）

短提示（2026-10-01，板端在线服务直测）：`prompt 1995 tok`、`n_predict 384`、预热后连续 3 次。
decode **34.51 / 34.76 / 34.77 t/s**（最快 **34.7718**），prefill 约 `220.5 t/s`，draft acceptance
`309/517 = 59.8%`。证据：`test/short2000-codex.jsonl`。不作为 200K 口径成绩。

200K（同 binary、同模型、同配对 prompt 口径）：

| 日期 | 场景 | decode t/s | prefill t/s | 备注 |
|---|---|---:|---:|---|
| 2026-09-28 | 冻结验收 p1p200k-r1 | **22.4638** | ~133 | `test/dflash-…-p1p200k-r1.bench.jsonl` |
| 2026-09-30 | 生产抽测 c60spot200k | **21.8846** | 134.3 | acc 0.5103，`test/c60spot200k.jsonl` |
| 2026-10-01 | 复测 spec-n7p0 | **22.1094** | ~133 | mean_len 4.45 / step 201.3 ms，`test/spec-n7p0-200k.bench.jsonl` |

- 200K 首 token（prefill）约 25 min（~134 t/s）；满 262K prompt 首 token ~32 min，
  decode 推算 ~20-21 t/s（KV 读取随 ctx 线性放大，未实测，标注为推算）。
- 服务质量：200K 长文任务 replacement_chars=0，输出正常。

## 已知边界

200K decode 30 t/s 在当前模型+草稿+硬件下暂无可证路径：attention 已贴字节地板
（全注意力仅 17/65 层，KV 读 13.9 GB/step）、q8 KV 实测变贵、投机 acceptance 侧
n_max/p_min 均无杠杆；**保障读数 = 200K 21.9-22.5 t/s**。

---

[整理者注] 入仓处理清单：① 板上真实数据分区路径改代称 `/brand_data/`（原路径含品牌字样）；
② 内网 IP / 主机代号 / 账号名未出现在原稿，未涉及；③ 二进制与工作树快照不入 git，
改发布于 ModelScope 并附 sha256；④ 其余技术参数、命令、实测数字 100% 保留。
