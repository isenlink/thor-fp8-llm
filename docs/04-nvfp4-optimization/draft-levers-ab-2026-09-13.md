# draft 侧结构杠杆 A/B 结果（2026-09-13 晚）

作者：AI助手（x86主机） · 前置：ANALYSIS-2026-09-13-b4-wide.md（B4/B5 收尾，转向 draft 侧）
事故后首批实验，全部 L0/L1 级（无 kernel 改动）。

## 二进制污染发现（重要）

vcrop binary 首次等价性检查失败（23.93 t/s、draft 567/313 vs b3fixed 27.99、490/324）。
根因：本地 `mmvq.cu.o`（Sep 13 12:45）含 B4 wide-M kernel（`GGML_NVFP4_WIDE_MAX`
默认 16，verify M=13 会走 wide 路径），而生产 perj binary（9/12 23:43）无此路径。
B4 wide 数值路径不同 → draft/verify 行为变化 + 更慢。
**处置**：`start-server.sh` 统一加 `GGML_NVFP4_WIDE_MAX=0`（对无 wide 的旧 binary 惰性）。
复测 vcrop(crop=0) vs perj：draft 计数逐一致（490/324）、输出逐字相同、
decode 28.30 vs 28.24（+0.2% 噪声内）→ 等价性成立。

## 2K A/B（trial vcropc0 同 prompt 配对，K12 p-min0.5 f16 KV，WIDE_MAX=0）

| 配置 | decode t/s | draft acc | Δ |
|---|---|---|---|
| 生产（perj + v2 模型） | 28.30 | 324/490 = 66.1% | 基准 |
| L1：crop=32768 | 24.85 | 308/630 = 48.9% | **-12%（acceptance 崩）** |
| L1：crop=65536 | 29.43 | 320/508 = 63.0% | **+4.0%** ✅ |
| L2：v3 draft-F8 | 27.88 | 319/551 = 57.9% | -1.3% ❌ |
| L1+L2（crop64k + v3） | 25.96 | 306/627 = 48.8% | -8.3% ❌ |

## 结论

1. **L1 vocab crop 保留，crop=65536**：682MB→180MB/draft 前向。crop=32768 崩的原因：
   bench 英文文本分词 ~9% token ID >32k（最高 83985），K=12 链含一个裁外 token 即整链
   报废 → acceptance 66%→49%。65536 覆盖 ~99%，acceptance 63.0%（-3.1pt），
   带宽收益转正。
2. **L2 draft 层 F8 放弃**：2.25% 权重量化误差 → acceptance -8.2pt，带宽收益被抵消；
   与 L1 叠加时 acceptance 损失相互放大（48.8%）。与 A2 教训同构但温和
   （A2 动主模型 MLP 致 acc 全灭；本次只动 draft 层，影响局限于提案质量）。
   v3 模型文件留板备查，不进生产。
3. 输出正确性：crop=65536 与锚点首 diff 在第 722 字符，良性换述
   （K12 vs K7 时代已记录的 near-tie argmax 翻转现象），非实现 bug。

## 实现存档

- L1：`src/models/qwen35.cpp` graph_mtp，`LLAMA_MTP_VOCAB_CROP` env（0=关）。
  weight view（行前缀）+ ggml_fill(-inf) 尾巴 + concat——全生产算子，sampler 零改动。
  binary：`llama-server-ai-assistant-vcrop`（perj 等价基座 + crop）。
- L2：`gguf-blk64-to-f8.py`（GGUF 改写：blk.64 八张量 BF16→F8_E4M3 + .scale），
  产物 `RadixArk-F8attn-v3-draftf8.gguf`（板/x86 各一份）。转换正确性已验证
  （其余 1452 张量逐字节一致、scale 值正确、rel_l1=2.25%）。

## 下一步

- ~~128K 终测~~ → 结果见下

## 128K 终测（trial1 同 prompt 配对，对照 perj-k12p05 19.61）

| 配置 | decode t/s | draft acc | Δ |
|---|---|---|---|
| 生产（perj + v2） | 19.61 | 316/522 = 60.5% | 基准 |
| L1 crop=65536 | **18.71** | 307/580 = 52.9% | **-4.6%** |

（prefill 174.5 vs 174.4 t/s 吻合 → crop 不影响 prefill，对照有效；
输出首 diff 第 939 字符，良性换述，同 near-tie 现象。）

**机制拆解（每步账本）**：crop 的 lm_head 节省完全兑现——step 耗时
287ms→266ms（-21ms，与预测 -19~22ms 一致）；但 acceptance 60.5%→52.9%
使 step 数 68→77（+13%），净亏。**128K 口径下 verify 地板
（KV 60ms + 权重 72ms = 132ms/step）固定，acceptance（tokens/step）是主导
乘数；draft 侧任何带宽节省都买不回 acceptance 损失。**

## 最终结论（2026-09-13 晚）

1. **L1 vocab crop：128K 否决**（2K +4.0% 但 128K -4.6%）。代码保留
   （env 门控默认关），短上下文场景若将来有需要可用 crop=65536。
2. **L2 draft F8：否决**（2K -1.3%，叠加 L1 -8.3%）。
3. draft 侧结构杠杆耗尽。19.61 t/s 维持生产最优。
4. 距 30 t/s 的缺口分析：现实上限 ≈（verify 地板 132ms + 最小 draft
   ~40ms）÷ 5.66 tok/step ≈ 33 t/s，且前提是 draft 成本减半且
   acceptance 无损——L0/L1 手段内已无候选。再往上需要模型侧工作
   （更好的 MTP 头 / 更深 draft），或接受 kernel 级风险（与本日事故
   纪律冲突）。

---

[整理者注] 本文档由工作笔记脱敏改写：作者行主机代号已中性化（AI助手/x86主机）。技术数据 100% 保留。