# NVIDIA DRIVE Thor (Tegra264) LLM Deployment Notes / 车规域控本地大模型部署实战笔记

> 民间实测记录：在 NVIDIA DRIVE Thor 域控板（p3960-0010 / Tegra264, sm_101a, DriveOS 7.0.3, CUDA 12.8）上
> 从零部署 llama.cpp GPU 推理并调优至 NVFP4 量化路线的全过程。
>
> Grassroots field notes: deploying and optimizing llama.cpp GPU inference on an
> NVIDIA DRIVE Thor automotive domain controller, from cross-compilation to
> NVFP4 quantization, all numbers measured on real hardware.
>
> [English version / 英文版本](README.en.md)

## Why this repo exists / 为什么有这个仓库

DRIVE Thor 的民间本地 LLM 部署资料几乎为零：官方只提供 DriveOS SDK 视角，
社区讨论停留在"能不能跑"。我们在 2026-09 用约一周时间从零摸索走通了全链路，
踩过的坑、失败实验、实测数据都完整记录在这里，希望能省下下一个人的摸索时间。

> **🔧 遇到问题请先看 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)** —
> 按症状索引的踩坑速查表，收录 30 条实际踩过的坑（含英文报错原文，便于搜索命中），
> 覆盖交叉编译、模型转换、推理调优、系统/挂载四大类。

## Hardware / 硬件

| 项目 | 规格 |
|------|------|
| SoC | NVIDIA Tegra264 (DRIVE Thor, Blackwell 架构) |
| GPU | compute capability 10.1 (sm_101a), 14 SM @ 1530MHz, L2 24MiB |
| GPU 可见显存 | 20 GiB（统一内存分区，物理内存 58 GiB，差额为设备树级内核 carveout） |
| 系统 | DriveOS 7.0.3, aarch64, CUDA 12.8 运行库齐全但无任何编译器 |

## Key results / 关键实测结果

| 项目 | 结果 |
|------|------|
| llama.cpp 交叉编译 | 0.4.0-dev 全套 91 个二进制，含 CUDA sm_101a 后端 ✅ |
| Qwen3-4B Q4_K_M 基准 | pp128 1516 tok/s / tg64 40.2 tok/s（ngl=99 全 GPU） |
| Qwen3.8-27B NVFP4 | decode **26.48 tok/s**（超社区 25.89 目标），accuracy 85.9% |
| 128K 长上下文 decode | 16.4±0.7 tok/s（生产基线，MTP K7 + F8 attn + NVFP4 MLP） |
| B3 kernel 优化（standalone） | NVFP4 MMVQ gemv 144→207 GB/s（+44%），根因 = y 向量重复 L2 读取（见 04 目录） |
| B3 定案（perj 修复 + MTP K12 p0.5） | 2K decode **31.02 tok/s**（+20.6%）、128K decode **19.61 tok/s**（+16.2%），输出与基线逐字一致（见 04 目录 b3-final-results） |
| 9-13 后续优化复盘 | B4/B5 verify kernel 路线、vocab crop、draft F8 均经配对 A/B 否决；19.61 tok/s 维持生产最优（见 04 目录 takeover / draft-levers / incidents） |
| 9-16 DFlash2 结论修订 | **撤回**"DFlash2 已淘汰"：当时"acc 崩"的真因是目标模型权重被重编码损坏，非 DFlash2 本身；配对重测（6 类 × 5 题）DFlash2 **全面快于内置 MTP**（全类中位 +8.1%、代码类 +28.3%），且草稿量化到 560MB 后**输出逐字一致**（见 04 目录 `dflash2-revalidation-2026-09-16.md`） |
| 9-16 投机草稿配方 | 无置信门控时：内置 MTP 甜点 **n_max=2-3**、DFlash2 甜点 **n_max=5**（n4-n7 全在 ±2% 平台区；硬上限 = draft 头 block size 8）；外挂 MTP 与内置**完全等价**（20.31 vs 20.32，白占 1.37G）；**`p-min` 门控各档全部低于无门控最优**（不要设）；**`dspark` 与 `dflash` 是同一实现**（无额外收益）；收益**强依赖内容类型**（见 04 目录 `speculative-drafting-recipes`） |
| 9-16 KV 预算与 256K | 65 层中**只有 16 层全注意力**（其余线性注意力）⇒ KV 仅 **64 KB/token**，256K f16 只 16.8 GB，**实测跑通**；TTFT 30K→256K = 67 s→25.6 min，解码仅 −28%；**同一长文后续提问 TTFT 5.9 s**（前缀缓存）；**KV 类型消融：q8_0 省 6.5GB 但解码 −34~36%**（acceptance 不变 ⇒ 纯反量化开销），**256K 定档 f16**（见 05 目录 `kv-budget-and-256k`） |
| 9-16 运行期内存增长 **已定案** | 每**新提示词** +626 MB ⇒ 约 12 条后 OOM：根因 = **主机侧 prompt cache 默认上限 8 GiB > 本机可用内存（46 GiB 大页池后只剩 ~7.5 GB）**；三臂剂量-反应坐实（`--cache-ram` 0/512/默认 → 平台 +0.7G/+1.2G/线性至 OOM）；**修复 = `--cache-ram 512`**（见 05 目录 `runtime-memory-growth`） |
| GPU 大页池 | 20G → 42G（后续扩至 46G）并固化；⚠️ **只能扩不能缩**（缩池会导致 `unable to allocate CUDA0 buffer`，A/B 实测见 05 目录） |
| 满载温度 | 72–74°C（被动散热，稳定） |

## Repo structure / 目录结构（整理中）

```
TROUBLESHOOTING.md        踩坑速查（症状索引，含报错原文 ← 优先看这个）
docs/
  01-hardware-recon/      板端环境摸底：显存真相、carveout、tmpfs、存储布局
  02-cross-compile/       x86 主机交叉编译 aarch64 + sm_101a 全套工具链
  03-model-conversion/    FP8 → GGUF 转换三层障碍（架构名分发/分片命名/numpy ABI）+ 模型文件台账
  04-nvfp4-optimization/  NVFP4 量化路线实验记录（含失败实验 MTP K7、B3 kernel 级优化决策链、microbench 拆解、正确性事故修复链、定案成绩、后续否决路线与 GPU 死锁事故纪律、多板并行测试准备、**投机草稿配方（深度甜点/混合量化/内容类型依赖）**、**DFlash2 结论撤销**）
  05-system-tuning/       GPU 大页池扩容与固化（**含"只能扩不能缩"的反证**）、overlay 持久化方法论、断电自愈架构、温度管理、**KV 预算与 256K 实测**、**运行期内存增长调查（已定案：prompt cache 上限 > 可用内存）**、**llama-server 部署配置速查（Q4_K_M MoE 实例）**、**MTP 投机解码启用（Qwen3.5-MoE 实测）**
  06-benchmarks/          各阶段基准数据与复现命令（含 200K 基准台账）、**基准方法论（三个口径陷阱 + 配对设计）**
scripts/                  板端/主机实用脚本（串口探测、GPU 池检查、B3 kernel microbench 全家桶）
  tcgen05-nvfp4-gemv/     **代码级整理包**：sm_101a 上 tcgen05 NVFP4 GEMV 快路内核（−49 ms/步）+ 四道机械测量纪律工具（安灯/臂闸门/数据可信度/启停预算）+ 无需板卡即可跑的判据 demo
```

## Status / 状态

🚧 内容整理中（源文档脱敏与重组进行时）。已转 public，敏感信息已按全历史审查记录复核（见仓库审查留档）。

## License

MIT — 见 [LICENSE](LICENSE)。
