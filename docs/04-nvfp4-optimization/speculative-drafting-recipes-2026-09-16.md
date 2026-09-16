# Speculative drafting recipes for Qwen3.8-27B on DRIVE Thor (paired measurements)

**English summary**: Paired measurements on a 27B NVFP4 target with two drafter families (built-in/external MTP and DFlash2 block diffusion) give concrete recipe numbers: with no confidence gating the MTP depth sweet spot is n_max=2–3, the DFlash2 sweet spot is n_max=5, a 560 MB mixed-quantization draft reproduces the 1.14 GB draft's output byte-for-byte (30/30), and the drafting gain is strongly content-type dependent (code +28%, Chinese prose −13%).

# 投机解码草稿配方实测（2026-09-16）

> 作者：AI助手（x86主机）· 平台：DRIVE Thor（p3960-0010 / T264 / sm_101a / DriveOS 7.0.3 / 统一内存 58 GiB）
> 目标模型：Qwen3.8-27B NVFP4（MTP 版，17.6 GB）；llama.cpp 自编 build（`llama.cpp-tcgen05`）
> 方法：同一目标模型、同一 ctx（16384）、同一提示词集、greedy（temp 0）、max_tokens 256，
> **6 类内容 × 5 条不同题**；指标取**服务端解码行**中位数（口径见 `06-benchmarks/benchmark-methodology-2026-09-16.md`）

---

## 1. 先立锚点：无投机就是带宽墙（必须每次先测）

| 配置 | 解码 t/s（18 次请求） | 说明 |
|---|---|---|
| **无投机（底线）** | **13.00**（全部落在 13.00-13.01） | 17.6 GB × 13.0 = **229 GB/s ≈ 板载 273 GB/s 的 84%** |

⇒ 解码每 token 必须把全部权重读一遍。**要再快，只能让"一次权重读取产出多个 token"**——
投机解码是唯一无损手段（被拒草稿丢弃、由目标模型重算，输出与无投机逐字一致）。
**没有这个底线锚点，任何"提速 N 倍"都不可信**（这也是我们早期口径事故的根源，见 §6）。

## 2. 深度扫描：两个家族各有甜点，且都不是"越深越好"

### 2.1 内置 MTP（无需外挂模型，零额外内存）

| n_max | 全类中位 | 中文类中位 | acceptance |
|---|---:|---:|---:|
| **2** | 19.88 | **19.22** | 0.62 |
| **3** | **20.32** | 18.27 | 0.54 |
| 4 | 18.77 | 18.13 | 0.42 |
| 5 | 17.26 | 16.58 | 0.35 |
| 7 | 15.11 | 14.13 | 0.27 |

- **n_max=2–3 是同一平台**（全类 n3 略优 +2%，中文 n2 略优 +5%），**n4 起单调下降**；
- 机理：MTP 头越靠后越不准（acceptance 0.62 → 0.27），而每轮验证开销是实打实的；
- ⚠️ 这里的结论是 **未设置信门控（`--spec-draft-p-min` 默认 0）** 时的甜点。
  带 p-min 门控的深起草是另一条曲线（见 `TROUBLESHOOTING.md` D1），两者不要混引。

### 2.2 DFlash2 块扩散草稿（外挂 560 MB）

| n_max | 全类中位 | 中文类中位 | acceptance |
|---|---:|---:|---:|
| 3 | 19.10† | 17.19 | 0.45 |
| **5** | **23.27** | **20.24** | 0.34 |
| 6 | 22.54 | 19.85 | 0.29 |
| 7 | 22.93 | 19.21 | 0.26 |

（† 该行为 Q4_K_M 档草稿；其余为同族 560 MB 档。同档内 n5 之后走平）

- **甜点 = n_max 5**；块扩散草稿一次前向出一整块，深度几乎不增加草稿成本；
- **硬上限 = draft 头训练时的 block size**（本模型 = 8）。请求 `n_max=9` 时服务端明确打印
  `requested draft size (n_max=9) exceeds the trained block size 8 -- clamping to 8`
  ⇒ **扫深度前先读 draft 头的 block_size 元数据**，超过它没有意义。

## 3. 草稿配方：能省的内存别浪费

| 观察 | 实测 | 结论 |
|---|---|---|
| **外挂 MTP-Q4_0 vs 内置 MTP**（同 n_max=3） | 20.31 vs 20.32 t/s | **完全等价 → 没必要为外挂 MTP 多占 1.37 GB**；内置 MTP 还能少一个加载项 |
| **自制混合量化草稿（560 MB）vs Q4_K_M（1.14 GB）** | 全类中位 21.61 vs 19.98；代码类 29.37 vs 27.16 | **小草稿反而更快**，且省 0.58 GB |
| **输出质量** | **30/30 逐字完全相同**（同提示词、同 temp 0、同 max_tokens） | 草稿量化**不改变输出**（投机解码无损）；逐字比对比 perplexity 更硬——是等价性证明而非统计相似 |

**560 MB 草稿的量化思路（可复用）**：主干用**低比特 I-quant**（IQ2_XXS / IQ2_S / IQ3_S），
但**单独保护两块**——输入投影 `fc` 提到 Q3_K、DFlash2 的候选路径 `selector_hidden` 提到 **Q5_K**，
所有 norm 保持 **F32**。即"**压主干、保选择器**"。代价是 acceptance 0.778 → 0.734（−4.4pp），
换来内存 −0.58 GB 与解码 +22%（同 target 同 n_max）。

## 4. 收益强依赖内容类型（最重要的部署结论）

| 内容类别（n=5） | **DFlash2（560 MB，n5）** | 内置 MTP（n3） | DFlash2 相对 |
|---|---:|---:|---:|
| 代码·Python | **29.37** | 22.89 | **+28.3%** |
| 英文·概念解释 | **22.55** | 19.99 | +12.8% |
| 中文·营销文案 | **23.22** | 20.97 | +10.7% |
| 中文·原理推理 | **21.53** | 19.71 | +9.2% |
| 英文·清单问答 | **20.76** | 19.38 | +7.1% |
| **中文·散文写作** | 16.13 | **18.61** | **−13.3%** |
| **全类中位** | **21.61** | 19.99 | **+8.1%** |

- 同一配置在**代码类 29.4** 与**中文散文类 16.1** 之间差 **1.8 倍**；
- 机理：DFlash2 草稿在中文创作类内容上 acceptance 掉到 **0.17-0.27**（英文/代码 0.31-0.76），
  而内置 MTP 的 acceptance 更均衡（中文也有 0.42-0.55）；
- ⇒ **报数必须绑定内容类型**；只报一个"最高 t/s"会误导（我们自己被坑过一次）。
  这也解释了为什么社区/他人日志里出现过的 **48.7 t/s** 在我们的中文内容上**不可复现**
  （他们的测试内容偏英文/代码或长生成）——同一配置中文实测 19-20 t/s。

## 5. 选择建议（本平台，中文用途）

| 需求 | 配置 | 中文实测 | 相对无投机 | 额外内存 |
|---|---|---:|---:|---:|
| 省事 / 省内存 | **内置 MTP `--spec-type draft-mtp --spec-draft-n-max 2`（或 3）** | 19.2（18.3） | 1.48x（1.41x） | **0** |
| 追求最快 | **DFlash2 560 MB 草稿 `--spec-type draft-dflash --spec-draft-n-max 5`** | **20.2** | 1.56x | +0.56 GB |
| 不建议 | 外挂 MTP-Q4_0（+1.37 GB）、Q4_K_M 草稿（+1.14 GB） | 18.5 / 18.6 | 1.42x | 白占 |

- 长文场景（16K 输入）两者差距仍在 **+10.8%**（24.50 vs 22.12，见 `05-system-tuning/kv-budget-and-256k-2026-09-16.md`）；
- **换草稿/改深度属于"零编译成本"的一档**，任何栈变更（换 build / 换量化）后都值得重扫一遍。

## 6. 复现要点（含两条必踩的坑）

```bash
# 目标模型 + 块扩散草稿（相对基线只加草稿相关项）
llama-server -m <target>.gguf \
  -md <dflash2-draft>.gguf --spec-type draft-dflash --spec-draft-n-max 5 \
  -ngl 99 -c 16384 -fa on -ctk f16 -ctv f16 -np 1 --parallel 1 \
  --temp 0 --host 0.0.0.0 --port 8080
```

**验收三步（顺序不能省）**：
1. 短补全输出通顺（数值没崩）；
2. `grep 'draft acceptance' <server log>` 与**同档基线**对照——**目标模型自身 acceptance 异常时不要评草稿**
   （历史事故：目标模型权重被重编码损坏导致 acceptance 崩到 0.68%，被误判成"草稿无效"，见
   `dflash2-revalidation-2026-09-16.md`）；
3. 才看 t/s，且**必须和内容类型、ctx、生成长度、口径一起报**。

⚠️ **口径坑（必读）**：llama-server 对同一任务打印两行计时，`prompt eval time = …`（预填充，60-150 t/s）
里**含子串 `eval time =`**；用宽松正则会把两者混进同一列表，中位数被抬到虚高的 40-60 档。
正确抓法见 `06-benchmarks/benchmark-methodology-2026-09-16.md` §1。

---

## [整理者注] 处理清单

- 品牌 / 车型 / 渠道 / 解锁类内容：无（只描述 DRIVE Thor 平台与 llama.cpp 行为）
- 内网 IP / MAC / SN / 账号名 / 凭据 / ssh 命令：已移除
- 人名与 AI 助手名：统一为"作者 / AI助手"
- 主机代号：统一为"x86主机"
- 板序号：写作 board3
- 内部绝对路径：改为 `~/work/...`
- 测试语料：原文使用行业主题中文长文，此处只写"中文长文（行业文本）"
- 技术数据（性能数字、参数、命令、张量类型、acceptance）：100% 保留
