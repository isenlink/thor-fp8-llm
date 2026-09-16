# DFlash2 block-diffusion drafting revalidated — our 9/13 "rejected" verdict is retracted

**English summary**: On 2026-09-13 we closed the DFlash2 block-diffusion speculative-drafting route after it looked slower with collapsed acceptance. A paired re-measurement on 2026-09-16 shows the opposite: with a numerically healthy target model, DFlash2 is the fastest drafter we have measured on this platform. The earlier failure came from the *target model*, whose weights were damaged by an unrelated requantization experiment — not from DFlash2.

# DFlash2 块扩散起草：结论撤销与新证据（2026-09-16）

> 作者：AI助手（x86主机）
> 触发原因：接管另一块同型板的实验时，回收了 20+ 份历史日志，并用统一口径重跑配对基准
> 影响文档：`b3-final-results-2026-09-13.md`（§2.1、§4）、`06-benchmarks/community-field-results.md`、
> `01-hardware-recon/hardware-archive.md`（3 处）

---

## 0. 本轮撤回的旧结论（显式标注）

| 旧结论（出处） | 处理 | 依据 |
|---|---|---|
| 「DFlash2 块扩散起草：两个目标模型全慢，acc 崩」→ 路线淘汰（`b3-final-results` §4） | **撤回** | 2026-09-16 配对实测：DFlash2 全面快于内置 MTP（全类中位 +8.1%，代码类 +28.3%） |
| 「A5-2 DFlash2 blk8 17.18，比 A5-1 慢 21.83%，淘汰」（`community-field-results`） | **加注**：该读数来自数值受损的目标模型，不能用于判定 DFlash2 本身 | 同上 |
| 「dflash2 树形投机：llama.cpp 暂不支持，已搁置」（`hardware-archive` 3 处） | **撤回** | llama.cpp 已原生提供 `--spec-type draft-dflash`（块扩散）与 `--spec-type draft-dspark`；本平台实测跑通 |

## 1. 为什么当时判错（归因复盘，比数字本身值钱）

那次判定的实验**同时改了两个变量**：

1. 把目标模型的权重**重新编码**（NVFP4 → F8）——该实验已被独立证伪：**acceptance 71.4% → 0.68%，decode −61%**，
   根因是权重重编码误差破坏主/草稿数值一致性（见 `a1-decision-2026-09-11.md` / A2 记录）；
2. 同时换了草稿方案（DFlash2）。

观察到的"acceptance 崩到 12-32%"被归因给了草稿，实际是**目标模型的 logits 已经损坏**。
⇒ **教训（已在别处写入纪律，这里再记一次）**：
- **一次只动一个变量**，否则"哪个变量导致失败"无法判定；
- **草稿类方案的验收必须绑定"数值健康的目标模型"**——先验证目标模型自身 acceptance 正常，再评草稿；
- 跨文档引用旧结论时，要连"当时的实验前提"一起引，否则会传播错误归因。

## 2. 新证据（配对基准：同 target / 同 ctx / 同提示词集）

**设计**：同一目标模型（27B NVFP4，MTP 版）、ctx 16384、greedy（temp 0）、max_tokens 256、
**6 类内容 × 5 条不同题**（共 25 题），每请求取服务端解码口径。

| 内容类别（n=5） | DFlash2 草稿（560MB，n_max=5） | 内置 MTP（n_max=3） | DFlash2 相对 |
|---|---:|---:|---:|
| 代码·Python | **29.37** | 22.89 | **+28.3%** |
| 英文·概念解释 | **22.55** | 19.99 | +12.8% |
| 中文·营销文案 | **23.22** | 20.97 | +10.7% |
| 中文·原理推理 | **21.53** | 19.71 | +9.2% |
| 英文·清单问答 | **20.76** | 19.38 | +7.1% |
| 中文·散文写作 | 16.13 | **18.61** | **−13.3%** |
| **全类中位** | **21.61** | 19.99 | **+8.1%** |

### 2.1 收益与内容类型强相关（重要）

同一配置在**代码类 29.4 t/s** 与**中文散文类 16.1 t/s** 之间差 **1.8 倍**。
⇒ 报"最高 t/s"必须绑定内容类型；只报单一数字会误导（我们早期就被此坑过一次）。

### 2.2 深度甜点与硬上限

- 草稿深度扫描：**n_max = 5 最优**（n_max 3 → 明显偏低；6/7 与 5 持平或略降）；
- **硬上限 = draft 头的训练块大小（本模型为 8）**：请求 `n_max=9` 时服务端日志明确
  `requested draft size (n_max=9) exceeds the trained block size 8 -- clamping to 8`，
  ⇒ 扫深度前先看 `block_size`，超过它没有意义。

### 2.3 草稿量化不改变输出（等价性验证）

用同一提示词、同 temp 0、同 max_tokens，分别在 **560MB 的混合量化草稿**与 **1.14GB 的 Q4_K_M 草稿**下生成：

| 比对 | 结果 |
|---|---|
| 提示词数 | 30 |
| **逐字完全相同** | **30 / 30** |
| 不同 | 0 |

⇒ 投机解码本身无损（被拒草稿由目标模型重算），**把草稿压到 560MB 对输出零影响**，同时省内存、还更快。
这比 perplexity 对照更直接：逐字比对是等价性证明，ppl 只给统计相似度。

## 3. 复现要点

- 启动参数（相对基线只加草稿相关项）：
  `-md <dflash2-draft>.gguf --spec-type draft-dflash --spec-draft-n-max 5`
  （配合 `-fa on`、f16 KV；目标模型必须自测 acceptance 健康后再评比）
- 验收三步：① 短补全输出通顺（数值没崩）→ ② `grep 'draft acceptance'` 与基线同档对照 → ③ 才看 t/s
- 深度消融前先读 draft 头元数据里的 `block_size`

## 4. 仍然成立的旧结论（不要一起撤销）

- `q8_0` KV 在长上下文的双重惩罚（反量化开销 + 草稿质量）——未受影响；
- "配方迁移必须核对模型+栈双重前提"——**本轮恰恰是这条的正面案例**；
- 长上下文下 K 过深反而拖累——DFlash2 的甜点同样落在中部（n_max 5），机制一致。

---

## [整理者注] 处理清单

- 品牌/车型/渠道类词：无（本文只描述 DRIVE Thor 平台与 llama.cpp 行为）
- 内网 IP / MAC / 序列号 / 账号名 / 凭据 / ssh 命令：已移除
- 人名与 AI 助手名：统一为"AI助手"
- 主机代号：统一为"x86主机"
- 板载数据分区路径：统一写作 `/brand_data/`（代称说明见仓库根 README）
- 内部项目代号：以模型系列名指代
