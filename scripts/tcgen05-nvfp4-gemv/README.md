# DRIVE Thor 本地推理优化 · 代码级工作整理包（2026-09-17）

> **Share-ready code bundle: local LLM decode optimization on NVIDIA DRIVE Thor / sm_101a.**
> 状态：**只整理，未推送**。是否发布、发到哪、用什么身份发，由项目负责人决定。
> 配套文档包：`SHARE-2026-09-16-tcgen05-sm101a-findings.md`（ISA 层发现）。本包只收**代码**。

---

## 0. 一句话

在一块 NVIDIA DRIVE Thor（sm_101a，14 SM）上，把 Qwen3.8-27B NVFP4 + 起草器的 decode 步耗
按权重侧快路（tcgen05 NVFP4 MMQ 环）砍掉 **−49 ms/步（−19~−21%，与 ctx 无关）**；
同时留下一套**可机械核的实验纪律工具**（判据要能判别、夹具要有来源、报错要停线、每次断电前先写死下两臂）。
目标 **200K KV decode 30 t/s 未达成**（还差 54 ms），且已证「腿回到 260 GB/s 量级」这条路没有证据支持——
这条负结论和正收益一样值钱，见 §3。

## 1. 板卡与目标（事实层，非推测）

- 平台：NVIDIA DRIVE Thor（p3960-0010 / Tegra264），**sm_101a（CC 10.1）**，14 SM，统一内存 47 GB 量级。
- 模型侧：Qwen3.8-27B，NVFP4（block-scale，每 64-K 记录 36 B，**不是 16 B 对齐**）+ 起草器（DFlash2 口径）。
- 目标：200K KV decode **30 t/s**（⇔ 步耗 ≤150.3 ms，对应 mean_len 4.51）。
- 口径纪律：**判据只看步耗 ms**（`ms_per_step = 1000 × mean_len / tps`）。t/s 是 acceptance 依赖量，
  同一份件同一 trial 也能在 19.33–22.18 t/s 之间浮动，跨臂比 t/s 等于自欺（`results/20260916-116`）。
- 同板参照引擎（外部包，`results/20260915-62`）：8K 41.84 / 128K 27.39 / 200K 24.09 t/s。

## 2. 主结果：量化收益（全部 V1_VERIFIED，逐条可查工件）

| 口径 | 关闭快路 | 开启快路 | Δ | 出处 |
|---|---|---|---|---|
| 128K 步耗 | 235.861 ms | **186.038 ms** | **−49.82 ms（−21.1%）** | `results/20260916-116` · `results/20260917-126` |
| 200K 步耗 | 254.120 ms | **204.842 ms**（复跑 204.306，Δ−0.26%） | **−49.28 ms（−19.4%）** | `results/20260917-124`（同 boot 校准 off 臂 254.583） |
| 权重拷贝腿速率 | 85.4 GB/s | **136.5 GB/s** | **+60%** | `results/20260916-113` · `results/20260917-123` |

- **两档 ctx 的收益几乎相同（−49.8 / −49.3 ms，差 0.5 ms）** ⇒ 这是**权重侧**成本，与 KV 长度无关。
  这条原本只是模型假设，现由实测直接验证（`results/20260917-126 §2.1`）。
- 打臂前用 8K 代理预测 205.8 ms，实测 204.842 ms（Δ−0.5%）⇒ **小形状代理可用于预测大形状**。
- **未达成**：128K 30 t/s 需 ≤158.7 ms（差 27–34 ms）；200K 需 ≤150.3 ms（差 54–71 ms）。
- 端到端 t/s（DFlash2 口径）：128K cold 22.178 / 200K cold 19.698（该读数的件与臂见 `STATUS-2026-09-17-eod.md §0`）。

## 3. 负结果黑名单（省别人的时间——每一条都是实测，不是推测）

| 假设 | 实测 | 结论 | 出处 |
|---|---|---|---|
| 拷贝粒度 4 B/次 → 16 B/次 | 原型 257.69 → 226.85 GB/s（**只值 −12%**），而缺口是 1.89× | **排除**：粒度解释不了腿的慢 | `results/20260917-127` |
| 原型慢是因为喂的是理想化取数 | 真实 GGML 权重切片（生产寻址）227.75 → 215.28 GB/s（**−5.5%**；copy 臂 −16.6%） | **排除**：内存侧三假设（粒度/寻址/prefetch·padding）全关 | `results/20260917-128` |
| ARES（A 段常驻 smem） | 102.1 vs 基线 100.9 GB/s（**+1.2%**） | **关闭**（命中预注册门 `<110 关线`） | `results/20260917-118` |
| 环的等待组深度 `<1>`→`<0>` | ratio 1.032 ≤ 1.3，两臂 D dump 逐字节相同 | **否决** | `results/20260917-121` |
| MMQ prefetch / M-padding | 逐项为负 | **否决** | `ANALYSIS-2026-09-13-mmq-prefetch.md` |
| KV 量化 q8_0 @256K | 省 6.5 GB，但 decode 掉 1/3 | **不划算** | 公开仓库 `docs/05-system-tuning/kv-budget-and-256k` |
| 「acceptance 拉满就能过 30 t/s」 | 满接受上界仍 <30 t/s | **否决** | `ANALYSIS-2026-09-14-synth-acceptance.md` |

**关键定性结论**：T4 腿 136.5 GB/s 与隔离原型 257 GB/s 之间差 **1.88×**，而粒度只值 −12%、生产寻址只值 −5.5%，
⇒ **缺口在消费者/结构侧，不在访存侧**；「把腿优化回 260 GB/s」这条路**没有证据支持**，本线已收摊（`STATUS-2026-09-17-eod.md §6`）。

## 4. 这包真正的可复用资产：四道机械纪律（A/B/C/D）

> 由来：一夜之间 8 处小错，根因全部是同一句话——**把「读一遍觉得对」当成了验证**。
> 后来把「验证」做成四个能被机器执行的步骤，此后同类错误为零。

| 机制 | 防的是什么 | 工具 |
|---|---|---|
| **A 安灯** | 「等 X 之后再看」把板子等死 | `board/t4gate.sh`（`andon` 只读取证 + 停线牌 `evidence/.halt`；停线牌在架时一切 preflight 机械 BLOCKED；只有现场断电重启 + 全绿才能摘牌） |
| **B 每臂闸门** | 臂开了才发现判据不可满足 / 件不对 | `board/arm_gate.sh`（pre/post 洁净检查 + 残留进程取证）、`host/verdict.py`（判据先跑 `--replay`，**好/坏样本判定必须不同**才准上板） |
| **C 数据可信度** | 把 wedge 前 boot 的读数当承重证据 | `evidence/data-credibility.tsv` 五档标记；`verdict.py --cred-out` 只写 V1/V4，**由判据决定，不由叙述决定** |
| **D 启停预算** | 断电重启一次白烧一个 boot | `t4gate.sh budget` + `evidence/arm-queue.tsv`：**每次请求断电前必须先把下一 boot 的两臂与判据写死** |

配套的**夹具来源审计**（`host/check_fixture_provenance.py`）：pass 夹具必须是**真臂读数原件**（或由原件改**被测列**派生），
手敲键名的「自洽假夹具」判 FAIL——因为键名写错时，假夹具会让 `REPLAY=OK` 照样通过（真实踩过，`results/20260917-126 §3`）。

## 5. 目录

```
01-tcgen05-nvfp4-gemv/   内核线：sm_101a 上 tcgen05 块缩放 FP4 GEMV（环 + 生产接线）
  src/                   *.cu / *.cuh / *.h（内核、宿主驱动、布局单一来源、接缝）
  host/                  接入判据真值表单测、canonical 布局重放单测
02-measurement-discipline/  纪律工具（可独立复用，与板卡型号无关）
  host/                  verdict.py · check_fixture_provenance.py · gguf_rowstride.py · step_metrics.py …
  board/                 arm_gate.sh · t4gate.sh · rb2_mem_{arm,driver}.sh
03-judgement-demo/       判据示范（**无需板卡即可跑通**，见 §7）
  fixtures/              *.spec（判据）+ *.metrics（真臂读数派生的好/坏夹具）+ PROVENANCE.tsv
  tests/                 接缝回归（沙盒 + 假宿主）
04-build-and-run/        构建 / 上传 / 臂编排（Tier 2：可复跑，含宿主环境假设）
MANIFEST.tsv             文件清单：group | file | tier | sha256_16 | lines | 说明
```

`MANIFEST.tsv` 的 `tier`：**T1 = 判断可复用、值得精读**；**T2 = 有用但与本机环境耦合**（构建/编排脚本）。

## 6. 复现门槛（不知道这些会卡住）

- **工具链**：nvcc **12.8.93**（`sm_101a` 才认 `tcgen05` 的 7 操作数形式）；CUTLASS 4.8.0 头（831 文件，
  丢了可从归档包一键恢复）；宿主侧用 **Driver API + dlopen**（不依赖运行时链接）。
- **参数不许从文档取**：GGUF 的行步长/记录宽度一律**从 GGUF 头直读**（`gguf_rowstride.py`），
  文档只做导航，冲突以工件为准。
- **板端 `/tmp` 每次断电清空** ⇒ 复跑前必须重传驱动件（`upload_drivers.sh` 内置缺件检查，缺即 ABORT）。
- **同一事实只在一处维护**：结果文档只写一行结论 + 指向 `evidence/` 原件，禁止复制数字。
- **GPU 实验有致死风险**：核函数级改动可能把 nvgpu 通道打死（无自愈，只能现场断电）。
  本包不含上板编排的时间戳/板名细节，请按自己的板卡补齐，并先落实 A/B/C/D 四门再开臂。

## 7. 本地自检（不需要板卡，30 秒）

```bash
cd 03-judgement-demo
# ① 同一条真臂读数，两条互斥判据必须给出相反判定
python3 ../02-measurement-discipline/host/verdict.py --spec fixtures/rb2_mem_h0.spec --metrics fixtures/rb2_mem_prod_high.metrics   # VERDICT=PASS
python3 ../02-measurement-discipline/host/verdict.py --spec fixtures/rb2_mem_h1.spec --metrics fixtures/rb2_mem_prod_high.metrics   # VERDICT=FAIL
# ② 判据判别力预检（好样本 1 个 / 坏样本 2 个）
python3 ../02-measurement-discipline/host/verdict.py --spec fixtures/rb2_mem_h0.spec \
        --replay fixtures/rb2_mem_prod_high.metrics fixtures/rb2_mem_prod_low.metrics fixtures/rb2_mem_prod_dm.metrics
# 期望输出：REPLAY=OK 判别力确认（FAIL=2 / PASS=1）
```

`bash -n` 对包内全部 `*.sh` 通过；`verdict.py` 退出码约定 `0=PASS 1=FAIL 2=SPEC_DEFECT 3=用法错`。

## 8. 脱敏说明与已知缺口

**规则来源**：内部维护规则 §1.3（内容红线 + 脱敏表）+ 本仓既有发布约定。**机械执行、逐条 grep 复核**，不靠自报。

已处理：主机代号 / 助手代号 / 板序号 → 中性名；本地绝对路径 → 占位符（`/path/to/project`、`~/work/…`）；
账号名 → `user`；**凭据读取行与 `sshpass` 调用整体删除**；内网 IP → `${BOARD_ADDR}` 占位符；内部项目代号 → 中性词；
品牌 / 车型 / 渠道类词汇（零命中）。

发布前建议按内部维护规则 §二 的词表在包根目录再跑一遍扫描，命中类别为：
① 品牌/车型/渠道词；② 内网地址与账号名；③ 主机与助手代号、本地绝对路径；④ 凭据类命令。
**词表本身不在此罗列**（复核清单不得反向泄漏被脱敏的原始值）。

**已知缺口（有意为之）**：

- 正文里的 `results/…` `evidence/…` 是**内部工件编号**，用途是溯源与交叉核对（每条数字都能被追到原件）；
  对外读者拿不到原件，忽略这些编号不影响阅读与复现；
- 本包是**子集**，不含 `evidence/` 原始日志、模型权重、以及带时间戳/板名的上板编排脚本；
- 脚本里的相对路径（如 `t4/m2c/...`）沿用原项目布局，复现需按自己的目录树调整；
- 占位符 `${BOARD_ADDR}` / `${BOARD_SSH}` / `/path/to/project` / `~/work/thor-driveos` 需自行赋值；
- 涉及第三方社区包的**身份指纹**（包 sha、作者线索）一律不出现。
