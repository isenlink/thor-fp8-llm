# production-wire 税项定位（rb2 诊断系列 · 2026-09-20/21）

> **整理者注**：本文由协作 AI 的实验记录（results/20260920-141 ~ 20260921-145）整理而成，
> 数据为协作方实测入档，本仓未复测。原始证据目录与内部协调文档未入仓。
> 文中 `boot-0920a` / `boot-0921a` / `boot-0921b` / `boot-0921c` 为启动会话代称
> （一次冷启动 = 一个 boot，每个 boot 有 2 个 GPU 实验臂预算，同 boot anchor 配对是判据纪律的一部分）。

## 一句话结论

**生产腿 `136.5 GB/s` 相对隔离 rb2 原型的缺口，主税项定位到 A ring/SFA 生产接线**：
`prodsf 203.53 → proda 144.21 GB/s（−29.1%）`，`proda` 已距生产腿仅约 +5.6%。

## 税项瀑布（逐项排除链）

| 诊断臂 | 剥离/接入的内容 | main 读数 | 相对成本 | 结论 |
|---|---|---:|---:|---|
| anchor（gran1 全消费者原型） | — | 226.55~227.75 | 基线 | 历史锚点稳定 |
| `bonly` | 只搬 B（保留 cp.async/mbarrier/寻址，跳过消费者） | **259.67** | B 环本身可回 260 | **V1_VERIFIED**：B 生产/搬运环不是瓶颈 |
| `prodaddr` | 真实 GGML 权重切片寻址 | 215.28 | −5.5% | 量级排除（历史 `results/20260917-128`） |
| `nod` | 跳过 D `tmem_ld` 回读 + global store | 218.99 | ≈0（+0.05%） | **排除** D-store 主税（V2 探索读数，见可信度） |
| `prodsf` | 接入生产 B sf 流 + SFB `tmem_st` | 203.53 | −10.0% | 有实成本但非主缺口（V2） |
| `proda` | 再接入 A ring（ABD=16）+ SFA 动态写入 | **144.21** | **−29.1%** | **主税项定位：A ring/SFA 生产接线**（V2） |
| 生产腿 | 全部生产语义 | **136.5** | — | `proda` 已接近，仅高 +5.6% |

## 各臂关键事实

### rb2-bonly（V1_VERIFIED）

- 判决：`HCONS_CONFIRMED`——`bonly 259.67 ≥ 240` 冻结门，两臂 `GATE4=OK`。
- B 侧 `cp.async` 请求、mbarrier 环和真实寻址都不是主瓶颈；
  `bonly → anchor` 的 ~14.6% 是原型消费者成本，真正要解释的是生产腿 vs `prodaddr` 的 1.58×。
- 内存侧三假设至此全部关闭：粒度（−12%）、生产寻址（−5.5%）、B-only（B 环可到 260）。

### rb2-nod（V2 探索读数）

- 本 boot anchor `218.88` 低于冻结阈值 `>=220` 约 0.5%，按机械规则不升 V1；
  `nod 218.99` 与 anchor 差仅 `+0.11 GB/s`——跳过 D readback/store 不释放吞吐。
- `NO_MMA` 无需单独上板：`copy` 子臂（`257.14/257.22`）已覆盖，MMA 开关几乎不改变生产寻址原型。

### rb2-prodsf（V2 性能判别，`d_check=0`）

- `prodsf = prodaddr + 生产 B sf 流 + SFB tmem_st`：相对同 boot anchor −10.0%，
  相对历史 `prodaddr` −5.5%——不是 1.58× 级别。
- copy 子臂对 sf/额外 smem 更敏感（195.58，较 anchor copy −24.1%），但 main 仍 >200 GB/s。

### rb2-proda（V2 性能判别，`d_check=0`）

- `proda = prodaddr + B sf/SFB + A ring(ABD=16) + SFA 动态写入`。
- 构建门：`PRODA_EFFECTIVE LDGSTS=81 > prodsf=72`、`UTCOMMA=124` 一致、
  `PRODA_BUDGET=FIT abd=16 total_smem=182336 ≤ limit=232448`、selftest `pass=111 fail=0`。
- 判决行：`RB2_PRODA_RESULT=MEASURED anchor_main_gbps=227.52 proda_main_gbps=144.21`。
- **含义边界**：`proda` 是"成本定位"——A ring/SFA 生产接线量级很重；
  **不是**完整生产 TMEM 列分区的数值等价证明。

## 方法论（判据纪律）

- **同 boot anchor 配对**：每臂与同 boot 的 anchor 臂对比，消除 boot 间漂移。
- **GATE4 机械门**：anchor ≥220 冻结阈值；跌破只记 V2，不做 V1 判决（nod 即实例）。
- **清洁门**：HugePages `23552/23552`、ERR=0、GRF=0、DSTATE 空、端口无残留。
- **臂预算**：每 boot 2 臂，用满必须冷启动换新 boot；不做重复臂。

## 下一步（不再重复的诊断臂）

以下方向已收敛/排除，不要重复：NO_MMA / NO_D_STORE / B-only / 粒度 / prodaddr / B sf / A ring 诊断。

1. `prod-epi-guard`：接入生产 `epi_rdy/epi_done + m_real + row_scale`，量化剩余 ~5-6% 缺口。
2. A ring/SFA 减税设计：保留生产语义，把 A ring/SFA 成本移出关键路径或降低同步/搬运税。

## 关联

- 生产腿读数与内存侧排除链见 `03-reference/negative-results.md`（"这条线已收摊"条的延续——本系列把"消费者/结构侧"具体化到了 A ring/SFA）。
- 权重拷贝腿速率沿革见 `scripts/tcgen05-nvfp4-gemv/README.md`（85.4 → 136.5 GB/s）。
