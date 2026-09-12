# GPT 评审简报 — B3 流水线拆解结论（R4.5 → R5 输入）

日期：2026-09-12
项目：Thor T5000 (sm_101a) Qwen3.8-27B decode 优化
当前成绩：128K decode 16.4±0.7 t/s；目标 35 t/s（网友 31 t/s 可信）

## 0. 本轮撤回/修正的旧结论（显式标注）

1. **撤回**「36B NVFP4 block 非 32B sector 对齐导致 coalescing 差」——microbench2 实测：真实 stride-36 映射(250 GB/s) 比朴素连续映射(127) 还快，假设推翻。
2. **撤回**「dp4a 指令是瓶颈」——microbench3 实测 dp4a 成本 ≈ 0（255→258 GB/s）。
3. **撤回**「寄存器压力/occupancy 是 221→144 gap 主因」——ptxas -v 实测 M=1 实例化 56 寄存器零 spill，occupancy 75%。
4. **修正** B1 结论「Thor NVFP4 走 256 线程 MMQ」——dispatch 代码核实：M=1 decode 走 **MMVQ 路径**（32 线程/1 warp，`should_use_mmvq` 优先于 `should_use_mmq`）。256 线程 MMQ 是 prefill/大 batch 路径。

## 1. 已确认事实（全部有实测/源码依据）

### 1.1 三级带宽 gap（M=1, ffn_gate K=5120 N=17408, 50MB 权重）

| 层级 | GB/s | 来源 |
|---|---|---|
| 全模型 per-call | 95 | R32 op_prof（488µs/46MB） |
| standalone 单算子（hot/cold） | 144 / 140 | b3-standalone（ggml backend 同路径） |
| 裸 kernel（精确复刻映射，无 y） | 221 | b3-microbench3 |
| 纯 load 上限（同映射） | 255 | b3-microbench3 |
| 硬件 DRAM 上限 | 273 | FLASH_ATTN 实测 100% |

### 1.2 流水线拆解（microbench3，精确复刻真实映射 grid(17408) block(32)）

| 变体 | GB/s | 增量成本 |
|---|---|---|
| 纯 load | 255 | — |
| +dp4a | 258 | **0（dp4a 免费）** |
| +LUT（6×byte_perm 查表） | 221 | **17%（LUT 解码）** |

### 1.3 上限实验（GPT R4 建议，已做）

Q8_0（预展开无 NVFP4 解包）同形状：214-260 GB/s vs NVFP4 140-172 → 差距在 NVFP4 解码路径，不在 dp4a。

### 1.4 缓存排除

cold（每次迭代 flush 24MB L2）只比 hot 慢 3-12% → 95 vs 144 的 gap 不是 L2 驻留差异。

### 1.5 新发现：y 向量重复 L2 读取（microbench4 待验证）

真实 kernel（mmvq.cu:696 + vecdotq.cuh:349-355）：每个行 block 除读权重外，还要读**整段 y 输入向量**（Q8_1, 5120B）。
- y 仅 5KB，全驻 L2
- 但 17408 个 block 各读一遍 → **y 的 L2 流量 ≈ 89MB/次调用 = 权重 DRAM 流量(50MB) 的 1.8 倍**
- 假设：221→144 gap 的主因（microbench3 用常量代替 y，漏了这笔账）
- 若实锤，优化方向：**y 进 smem 复用**（每 block 32 线程共享一份 y，L2 流量降 32 倍）——比 cp.async 双缓冲更小更直接的改动

### 1.6 形状效应

ffn_down(K=17408) 比 ffn_gate(K=5120) 快 35%（同字节数）→ K 维越长，y 复用率越高（y 相对权重越小），与 y 重复读取假设一致。

## 2. 当前瓶颈排序（按证据强度）

1. **y 向量重复 L2 读取**（新头号嫌疑，待 microbench4 验证）
2. **LUT 解码 6×byte_perm**（已实锤 17%）
3. **全模型访存交织**（144→95 的 gap，多算子并发 + L2 被 KV 冲刷，难单独优化）

## 3. 已排除的路径

- native FP4 MMA（tcgen05）：sm_101a 硬件不存在（B1 实锤）
- 社区 #28514 TMA patch：不能直接移植（同上）
- L2 预取门控扩展（R35）：中性，非突破口
- decode/verify 分裂 dispatch（R34）：2K 打平
- 36B 非对齐 coalescing：microbench2 推翻
- dp4a 指令成本：microbench3 排除
- 寄存器压力：ptxas -v 排除

## 4. 请评审的问题

1. **y 进 smem 复用**方案：M=1 时 y 对所有行相同，每 block 把 5KB y 读进 smem 共享，L2 流量 89MB→2.8MB。但 smem 5KB/block × 并发 block 数是否挤占 occupancy？（当前 56 寄存器 + 384B smem，加 5KB 后 smem 成瓶颈：228KB/SM ÷ 5.4KB ≈ 42 block，仍够 36 warp 上限？）
2. 若 y 复用实锤，**是否值得同时做 LUT 优化**（17%）？寄存器 LUT（把 16 值表放寄存器，用 select 链替代 byte_perm）在 DRAM-bound 下收益是否还存在？
3. **144→95 的全模型 gap**：多算子访存交织，有没有已知的 Thor/Tegra 上 gemv 与 attention 并发的优化经验？
4. 验收线维持：standalone 144→200+（≈BF16 gemv 平齐）再进 llama.cpp 全模型验证。

## 5. 环境备注

- 板：Thor T5000，14 SM，L2 24MB，SMclk 1530MHz，LPDDR5X 273GB/s
- 09-12 12:43 GPU 进程卡死事件已定案：microbench4 代码 bug（非对齐 int 读）拖死 nvgpu 通道，重启后恢复；教训已落盘
- 交叉编译：qemu + aarch64 g++-14 + CUDA 12.8，单文件 mmvq.cu 全量编译 ~37 分钟

## 6. 09-12 下午进展：kernel 已改 + 速度达标 + 正确性失败（当前阻塞）

**已实施**：y 预读到寄存器跨行复用 + rows_per_block 1→8（仅 NVFP4 + M=1 路径，3 处改动，设计见 B3-Y-REUSE-KERNEL-DESIGN.md）。

**速度（standalone，改前→改后）**：ffn_gate 144.1→207.1（+44%）、ffn_down 194.9→225.7（+16%）、lm_head 156.4→235.6（+50%）；Q8_0 对照不变。

**正确性失败（A/B 对照，同 prompt temperature=0）**：
- 基线+MTP ✅ 完美 / 基线+无MTP ✅ 完美（两次逐字一致，确定性确认）
- 改后+无MTP ⚠️ 大体通顺但结尾破碎（"I'm a 24-year-old male... I common cold symptoms"）
- 改后+MTP ❌ 重复循环（"I'm a student student student..."），acceptance 0.36→0.048 波动

**结论**：kernel 改动引入细微数值 bug；M=1 核心大体对（y 映射数学已逐行核对）；MTP 把小 bug 放大。

**bug 嫌疑排序**：A. R=8 多行归约/写回路径（NVFP4 首次激活 rows_per_block>1，最可疑）；B. y preload ds/scale 细节；C. standalone F32 输入与生产 Q8_1 路径不一致（待查）。

**请评审**：
1. R=8 归约路径（mmvq.cu:766-836）在 rows_per_block>1 时是否有已知坑？（此前 NVFP4 只跑过 R=1）
2. 定位策略是否合理：standalone 加 CPU 参考（dequantize_row_nvfp4 + 点积）对比 → rows_per_block 改 env var 一次编译测 R=1/R=8 二分？
3. 若 bug 在 R=8 归约，是否值得保留 y 复用但回退 R=1（microbench5 显示 R=1+y 复用仍有收益：y 流量 94.7MB→11.8MB 需要 R=8 才成立，R=1 时 y 复用本身无收益——请确认 R=1 时 y 预读是否还有意义）？

**详细数据与踩坑**：handoff-2026-09-12-b3.md（交接文档，含 microbench3/4/5 全表、A/B 对照、10 条踩坑、下一步优先级）
