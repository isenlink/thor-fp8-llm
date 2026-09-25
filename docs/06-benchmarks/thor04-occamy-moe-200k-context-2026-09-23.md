# Thor04 — Occamy-1.0（qwen35moe MoE）200K KV 长上下文实测

> **线路定位**：本文档属 **Q4_K_M MoE 实测线**（与仓库名指向的 FP8/NVFP4 主线不同题，
> 曾在独立分支上评审，2026-09-25 经维护者确认并入 `main`）。
> 同族部署配置见 `docs/05-system-tuning/llama-server-config.md`、
> 同族 MoE 吞吐见本目录 `thor03-qwen36-35b-a3b-moe-2026-09-15.md`。

> **One-liner**: A Q4_K_M MoE (~35B total / ~3B active, 21.17 GB) runs with
> **`-c 200000` (n_ctx_slot = 200192)** on a 14-SM DRIVE Thor board: a 35,651-token
> prompt is accepted where the previous 32768 build would reject it, prefill holds
> ~694–705 tok/s, short-context decode stays at 50.4 tok/s (unchanged from the 32K
> baseline) and drops only to 43.9 tok/s at 34K context. Pairing the 27B's DFlash2
> draft with this MoE target **crashes** (`GGML_ASSERT(ggml_can_repeat)` during
> speculative init) — a compatible drafter is required.

**日期**：2026-09-23
**板卡**：Thor04（DRIVE Thor / sm_101 / 14 SM / 统一内存 58 GiB，其中 46 GiB 大页池；板卡序列号与内网地址略）
**模型**：Occamy-1.0，GGUF `Q4_K_M`，**21,166,757,696 bytes（21.17 GB）**，llama.cpp 架构名 `qwen35moe`
（MoE，~35B 总参 / ~3B 激活）
**引擎**：本机交叉编译的 llama.cpp `tcgen05` 构建（与该板生产服务同一二进制）
**前置**：同板生产服务（Qwen3.8-27B NVFP4-Q8 + DFlash2 草稿，端口 8080）测试期间停用，测完恢复
（这是硬约束：两个服务同跑会因显存冲突互崩）

---

## 一、结论速览

| 问题 | 结论 |
|---|---|
| 200K KV 真的能用吗 | **能**。`-c 200000` 起服务，日志 `n_ctx_slot = 200192`；实发 **35,651 token** 的 prompt 被完整接收（旧 32768 基线装不下），prefill 693.9 tok/s |
| 换 200K 后短上下文变慢吗 | **没有**。decode 50.35 / 49.92 tok/s，与 32K 基线 50.11 tok/s 持平 ⇒ **容量扩 6.1×，短上下文速度不变** |
| 长上下文代价多大 | 34,122 token 上下文时 decode **43.88 tok/s**（比短上下文 −12.5%），prefill 仍 696.5 tok/s |
| prefill 随上下文变慢吗 | 基本不变：~700 token 584 tok/s、~4K 705、~34K 694–696（短 prompt 的 584 是固定开销占比高，非长文衰减） |
| 能挂 27B 的 DFlash2 草稿提速吗 | **不能，会崩**（见 §四负面结果）：形状不兼容，投机初始化 `GGML_ASSERT(ggml_can_repeat)` 失败、进程退出。需要 `qwen35moe` 兼容的草稿模型 |

---

## 二、32K 基线 vs 200K 配置

| 配置 | 上下文 | prefill | decode | 备注 |
|---|---|---|---|---|
| 32K 基线（`-c 32768`，当日早些时候） | 32,768 | 44.95 t/s ※ | **50.11 t/s** | ※ 该轮 prompt 更长，prefill 口径与下三行不同，仅作参考 |
| 200K：测试 A（短 prompt ~700 tok + 512 decode） | 200,192 | 584.4 t/s | **50.35 t/s** | 墙钟 12.1 s |
| 200K：测试 B（~4K prompt + 256 decode） | 200,192 | 705.3 t/s | **49.92 t/s** | 墙钟 8.5 s |
| 200K：测试 C（35,651 token prompt） | 200,192 | 693.9 t/s | —（只出 1 个 token 即收尾） | **证明 >32768 的 prompt 可用** |
| 200K：测试 C2（34,122 token + 256 decode） | 200,192 | 696.5 t/s | **43.88 t/s** | 墙钟 52.2 s |

启动参数（无投机版，实测生效）：

```bash
llama-server \
  -m /brand_map/aispace/occamy/occamy-1.0-Q4_K_M.gguf \
  -ngl 99 -c 200000 -np 1 --parallel 1 \
  -fa on -ctk f16 -ctv f16 \
  -b 2048 -ub 512 -t 12 \
  --cache-ram 4096 \
  --host 0.0.0.0 --port 8081 \
  --alias occamy-35b-thor04 \
  --temp 0 --seed 1234 --jinja -n 8192
```

启动日志关键行：`load_model: initializing, n_slots = 1, n_ctx_slot = 200192, kv_unified = 'false'`
→ 模型加载约 17.6 s 即 `model loaded` / `listening`。

**KV 占用【推算】**：本族 MoE 的 f16 KV 约 80 KB/token（同族 256K = 20 GB，见
`thor03-qwen36-35b-a3b-moe-2026-09-15.md` 的 KV 表）⇒ 200K ≈ **16 GB**，
加上 21.17 GB 权重，在 46 GiB 大页池内可容纳。该推算未单独实测内存读数，标【推算】。

---

## 三、测试方法（可复现）

1. 停掉同板生产服务（GPU 显存互斥），**停之前先记录其原始状态**（模型/参数/端口/PID）
2. 用上述参数拉起 Occamy 服务（8081，与生产 8080 错开）
3. 从主机直接打 HTTP（板端无 `curl` 时用主机侧客户端），`temperature 0`、`seed 1234`
4. 三档 prompt：短（~700 tok）、~4K、~34K（英文 filler 文本拼接，token 数按 usage 回读校准）
5. 记录服务端 `timings`（`prompt_per_second` / `predicted_per_second`）与墙钟
6. 测完停测试服务、拉起生产服务，`/health` 回 `ok` 并核对启动参数与停机前逐字一致

> 提示：思维链模型在 `max_tokens` 偏小时会把 token 花在 reasoning 上导致
> `content` 为空（见 `thor04-multi-slot-concurrency-kv-reuse-2026-09-16.md` §3.2），
> 测 decode 时要么给足 `max_tokens`，要么用 `--temp 0` + 明确续写指令。

---

## 四、负面结果：DFlash2 草稿与本模型不兼容（不要重复踩）

给 Occamy 挂同板现成的 27B DFlash2 草稿（`-md <draft>.gguf --spec-type draft-dflash -ngld 99 --spec-draft-n-max 5`）：

```
common_speculative_init_result: loading draft model '.../Qwen3.8-27B-DFlash2-Q2_K_S-MIX.gguf'
ggml.c:2263: GGML_ASSERT(ggml_can_repeat(b, a)) failed
→ 进程崩溃（PROC_DEAD）
```

**判定**：草稿模型与目标模型的形状（MoE vs dense 的重复关系）对不上，投机初始化阶段直接 assert。
**结论**：要给 `qwen35moe` 上投机，必须先有一个**同架构兼容的草稿模型**；现有 27B 的 DFlash2 草稿不可复用。
（对照：同一草稿配 27B dense 目标模型是稳定可用的，见
`thor03-dflash2-speculative-matrix-2026-09-15.md`。）

---

## 五、运维脚本与留档

- 启动脚本：`/brand_map/aispace/occamy/start_server.sh`（含用法三步与本篇的实测结论注释）
- 原 32768 版备份：同目录 `start_server.sh.bak-32768-20260923`
- 测试日志：同目录 `logs/server.log`
- 生产恢复验收：`/health` → `{"status":"ok"}` + `/v1/models` 返回生产模型别名 + 进程命令行与停机前一致

---

## 整理者注（脱敏与出处）

[整理者注] 本文档由内部实测会话记录整理而成，**全部性能数字为实测原值未改动**
（prefill/decode/墙钟/token 数、文件字节数、日志原文、assert 报错原文）。

脱敏项（只列类别，不列原值）：板卡序列号、内网 IP 地址略去；
板载分区路径（原路径名含车辆品牌字样）改 **代称**：
- **`/brand_map/`**（新增代称）≈ 原「地图分区」：DriveOS 板上另一独立非易失分区，约 59G，
  与 `/brand_data/` 互不隶属，模型文件放在这一侧以避开数据分区的空间压力
- **`/brand_data/`**（仓库既有代称）≈ 板载数据分区，约 105G，板载 vblkdev，与只读根分区独立
读者在自己板卡上执行 `ls /` 即可看到真实分区名。

`Thor04` 为内部板位编号，不含序列号/地址。本篇数据我方自测，未复测第三方数据。
