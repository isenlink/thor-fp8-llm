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
| GPU 大页池 | 20G → 42G（后续扩至 46G）并固化 |
| 满载温度 | 72–74°C（被动散热，稳定） |

## Repo structure / 目录结构（整理中）

```
TROUBLESHOOTING.md        踩坑速查（症状索引，含报错原文 ← 优先看这个）
docs/
  01-hardware-recon/      板端环境摸底：显存真相、carveout、tmpfs、存储布局
  02-cross-compile/       x86 主机交叉编译 aarch64 + sm_101a 全套工具链
  03-model-conversion/    FP8 → GGUF 转换三层障碍（架构名分发/分片命名/numpy ABI）+ 模型文件台账
  04-nvfp4-optimization/  NVFP4 量化路线实验记录（含失败实验 MTP K7、B3 kernel 级优化决策链、microbench 拆解、正确性事故修复链与定案成绩）
  05-system-tuning/       GPU 大页池扩容与固化、overlay 持久化方法论、温度管理
  06-benchmarks/          各阶段基准数据与复现命令（含 200K 基准台账）
scripts/                  板端/主机实用脚本（串口探测、GPU 池检查、B3 kernel microbench 全家桶）
```

## Status / 状态

🚧 内容整理中（源文档脱敏与重组进行时）。private 阶段 = 内容审校期，转 public 前会完成敏感信息复核。

## License

MIT（待定）
