# tcgen05 技术调研

> One-liner: Why tcgen05 (5th-gen Tensor Core PTX instructions) works on
> Thor's sm_101a but NOT on consumer RTX 50 series (sm_120), and the roadmap
> for using it to double llama.cpp prefill on Thor.
>
> 日期：2026-09-08（本仓库版：脱敏整理）

## 一句话结论

tcgen05 = Blackwell 数据中心级芯片（sm_100a/sm_101a）的第五代 Tensor Core PTX 指令族。**Thor（sm_101a）完全支持**。llama.cpp 官方 master 只在 Flash-Attention 的 fattn-mma 部分用了它的前身模式，通用 MMA 路径未用 tcgen05——这就是社区"prefill ×2"魔改的空间来源。

## 硬件支持范围（关键澄清）

- ✅ **sm_100a / sm_101a**（数据中心 Blackwell：B200/GB200，**Thor/Tegra264 sm_101a**）—— tcgen05 完整支持
- ❌ **sm_120**（消费级 Blackwell：RTX 50 系）—— **不支持** tcgen05！用回 mma.sm_120 变体（Modular issue #5707 明确报错："tcgen05 instructions only applicable on sm_100a, sm_101a"）
- 这解释了为什么社区叫它"Blackwell 专属魔改"——RTX 5090 反而不能用，Thor 因为是 sm_101a 反而可以

## 指令族全景（PTX ISA 8.7+ / CUDA 12.8 已支持）

```
tcgen05.alloc / dealloc / relinquish_alloc_permit  # TMEM 分配管理
tcgen05.ld / st                                     # TMEM ↔ 寄存器/SMEM
tcgen05.cp                                          # 异步数据搬运进 TMEM
tcgen05.mma / mma.sp / mma.ws                       # MMA 本体（.sp=稀疏 .ws=weight stationary）
tcgen05.wait / fence / commit / shift               # 同步原语
```

## 与前代对比（为什么快 2-4 倍）

| 代际 | 指令 | A/B 存放 | 累加器 C/D | 发射者 |
|---|---|---|---|---|
| Ampere sm80 | mma | 寄存器 | 寄存器 | 全 warp 集体 |
| Hopper sm90 | wgmma | SMEM | 寄存器 | warpgroup 集体 |
| **Blackwell sm100** | **tcgen05.mma** | **SMEM（A 也可 TMEM）** | **TMEM（张量内存）** | **单线程！** |

三大架构红利：
1. **TMEM（Tensor Memory）**：每 SM 256KB 专用累加器内存，从寄存器堆解放出来
2. **单线程发射 MMA**：一个线程代表整个 CTA 发指令（前代要 128 线程协同），延迟恒定 ~11 cycles
3. **CTA-pair**：相邻两个 CTA 联合做一个 MMA（cta_group::2），tile 翻倍无需额外同步

## 学习资源（按可操作性排序）

1. **"tcgen05 for dummies"（gau-nernst）** ⭐ 最佳入门：纯 CUDA C++ + 内联 PTX，从零到 98% CuBLAS 速度（1506 TFLOPS 基线），9 个版本迭代全记录，代码在 github.com/gau-nernst/learn-cuda（02e_matmul_sm100）。B200 上跑通——**sm_101a 同代同指令集，可直接借鉴**
2. **CUTLASS Blackwell 文档**（docs.nvidia.com/cutlass）— 官方 UMMA 抽象，工业级实现
3. **Colfax Research 教程**（research.colfax-intl.com）— tcgen05 GEMM kernel 手把手
4. **PTX ISA 文档** — 指令权威定义
5. **MLC.ai 现代GPU编程教材** — block-scaled MMA（NVFP4 缩放因子如何进 TMEM）讲解

## llama.cpp 现状与魔改空间

- 现有 fattn-mma（Flash-Attention MMA 路径）已经是 tile 化 tensor core 写法，但用 mma（Ampere 风格）而非 tcgen05
- 通用 GEMM（llama 的 decode/prefill 主力）用的是朴素 CUDA core + mma 混合——**tcgen05 化 = 社区 prefill ×2 的来源**
- NVFP4 关键：block-scaled tcgen05.mma 原生支持 .kind::f8f6f4 + UE4M3/UE8M0 scale factor 从 TMEM 读——**NVFP4 全速需要 block-scaled 路径**

## 自研路线评估

1. **最小可行**：把 ggml-cuda 的 mul_mat 主路径（dequant + mma）换成 tcgen05 block-scaled 版——参考 gau-nernst v1-v3 迭代 + CUTLASS SM100 block-scaled 示例
2. **工程量**：入门学习 2-3 天 + kernel 原型 3-5 天 + ggml 集成调优 1-2 周
3. **风险**：Thor 只有 14 SM（B200 有 148），tile 尺寸/并行度策略要重调；TMEM 每 SM 256KB 固定
4. **预期收益**：prefill ×2（社区实测），decode 依赖显存带宽，tcgen05 对 decode 提升有限——decode 大幅提升主要靠 MTP/dflash2 投机解码放大有效吞吐

## 行动建议

- 短期：clone gau-nernst 的 learn-cuda 仓库，在 Thor 上跑通他的 matmul 各版本作 sm_101a 可行性验证
- 中期：ggml 的 fattn-mma 已有现成 mma pipeline 结构，参照改造成 tcgen05 路径
- 解码端收益主力还是 MTP 参数调优（开箱 +30%）+ dflash2 对比测试

**实测佐证**：我们自写 tcgen05 BF16 matmul kernel（v5）在 Thor 上实测 **60.4/60.2 TFLOPS 可复现**（对照 B200 同 kernel 1302 TFLOPS，Thor 约为其 1/21.5，符合 SM 数与频率比例）——见 [硬件档案](../01-hardware-recon/hardware-archive.md)。
