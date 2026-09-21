# NVIDIA DRIVE Thor · 预编译 llama.cpp 服务端（sm_101a · NVFP4 + 起草器）

**这包是给你直接跑的二进制，不用编译。** 板上实测：**128K 上下文 22.18 t/s、200K 上下文 19.70 t/s**（纯解码、冷启动、DFlash2 起草器口径，见 §3）；**这两档读数已用本包自己的二进制复测过（分别 22.19 / 19.60 t/s），精度 150 题全对（见 §3/§7）**。
我们自己的部署路径跑出来是 22.18 / 19.70；如果你按同一套思路装完只有 9 t/s，先看 §6 的排障清单——大概率是起草器或环境变量漏了一环。

- 件：`01-binary/llama-server-aarch64-sm101a-nvfp4`（74.7 MiB，ELF64 aarch64，内含 `sm_101a` cubin）
- 启动：`02-launcher/start-server.sh`（把 `MODEL` / `DRAFT` 两个变量指到你的 gguf 即可）
- 校验：`01-binary/SHA256SUMS.txt`

---

## 1. 包里有什么

```
01-binary/llama-server-aarch64-sm101a-nvfp4   预编译服务端（含 DFlash2 起草器支持）
01-binary/SHA256SUMS.txt                      校验和
02-launcher/start-server.sh                   生产启动脚本（改 2 个变量就能跑）
02-launcher/stop-server.sh                    干净停服
02-launcher/prepare-host.sh                   GPU 内存池/大页准备（开机后跑一次）
03-reference/prod-measurements.md             实测读数与口径（含 prefill）
03-reference/troubleshoot-9tps.md             9 t/s 排障清单（10 条，带判据）
03-reference/negative-results.md              负结果黑名单（别人踩过的坑，已实测否决）
03-reference/accuracy-selfcheck.md            长上下文精度自检清单（8 方向）
MANIFEST.tsv                                  文件清单 + sha256
```

## 2. 先确认这三条（不满足就别往下走）

1. **平台**：NVIDIA DRIVE Thor（`sm_101a`，14 SM），aarch64 Linux。件里只有 `sm_101a` 的 cubin，别的卡跑不了。
2. **运行库**（板端 BSP 自带；`ldd` 能看到就行）：`libcudart.so.12`、`libcublas.so.12`、`libcublasLt.so.12`、`libcuda.so.1`、`libgomp.so.1`、`libstdc++.so.6`。
   自检：`ldd ./01-binary/llama-server-aarch64-sm101a-nvfp4 | grep -c 'not found'` → 必须是 `0`。
3. **GPU 内存池**：Thor 的"显存"是从统一内存里划的 **2 MiB 大页池**。我们跑 **46 GiB 池（23552 页）**，并用平台自带的 `gpu-carveout.sh -g 40` 划 40 GiB 给 GPU。
   自检：`grep -E 'HugePages_Total|Hugepagesize' /proc/meminfo` → `HugePages_Total: 23552`。
   **池必须冷启动分配**（在线加页会碎片化，加到 32 GiB 就上不去了）；改完 `vm.nr_hugepages` 要重启一次。

## 3. 实测读数（口径纪律：看 **ms/步**，别看 t/s）

`ms_per_step = 1000 × mean_len / t/s`，`mean_len` = 每步起草器实际贡献的 token 数（= 1 + 接受数）。
t/s 是"接受率依赖量"，同一份件同一 trial 也能在 19.33–22.18 t/s 之间浮动，**跨臂比 t/s 等于自欺**。

| 上下文 | 模式 | t/s | mean_len | 接受率 | **ms/步** | prefill |
|---|---|---|---|---|---|---|
| 2K | warm | 25.76 | 6.41 | 0.622 | **248.8 ms** | 回退排练正控臂 `c55bak2k`（2K 口径，非 8K） |
| **128K** | cold | **22.178** | 5.2 | 0.604 | **234.5 ms** | 127,964 tok / 730.0 s（175.3 t/s） |
| 128K | warm | 22.192 | 5.2 | 0.604 | 234.3 ms | — |
| **200K** | cold | **19.698** | 5.0 | 0.573 | **253.8 ms** | 199,980 tok / 1366.2 s（146.4 t/s） |

- 目标口径是 200K 上下文 **30 t/s**（⇔ 步耗 ≤150.3 ms）；**本件没到**，差 54 ms（≈21%）。**本件是"能用的最好一件"，不是"最快的一件"。**
- 同板另一套路线（tcgen05 内核，非本包）参照：8K 41.8 / 128K 27.4 / 200K 24.1 t/s——想再往上走要靠内核级改动，不在本包范围。
- **本包二进制已用自己跑过一遍（2026-09-19）**：128K decode **45.07 ms/token = 22.19 t/s**、200K **51.02 ms/token = 19.60 t/s**，与上表生产读数差 **−0.07% / +0.50%**；同一轮 150 题 × 8 方向精度 **128K 与 200K 各 150/150、0 空回答**（§7）。
- 我们没有把这些数字做成"承诺"：你的量化件、模型结构、池大小、温度都会改读数。**先跑 8K 冒烟，再按 §3 口径自己测一遍**。

## 4. 启动：生产配置（逐参数）

```bash
# ① 推荐：DFlash2 起草器（需要起草器文件，见 §5）
export MODEL=/path/to/target-nvfp4.gguf     # 你的 NVFP4 量化件（我们用的是 Qwen3.8-27B 级 NVFP4）
export DRAFT=/path/to/dflash2-draft.gguf
export SPEC=dflash2 CTX=131072 SPEC_N=7     # CTX: 128K=131072 / 200K=204800；SPEC_N: code·JSON=7，中文散文=3
bash 02-launcher/start-server.sh

# ② 没有起草器文件时的回退：用 target GGUF 自带的 MTP 头（零额外下载，见 §5.2）
MODEL=$MODEL SPEC=mtp CTX=131072 bash 02-launcher/start-server.sh

# ③ 对照臂：只跑 target（看起草器到底值多少）
MODEL=$MODEL SPEC=none CTX=8192 bash 02-launcher/start-server.sh
```

实际命令行（`start-server.sh` 生成的就是这几句之一）：

```
# SPEC=dflash2（推荐）
<binary> -m $MODEL -ngl 99 -c $CTX -fa on --cache-type-k f16 --cache-type-v f16 \
         --parallel 1 --port 8080 -md $DRAFT --spec-type draft-dflash --spec-draft-n-max $SPEC_N
# SPEC=mtp（不要 -md；用 target 文件里自带的 MTP 头）
<binary> -m $MODEL -ngl 99 -c $CTX -fa on --cache-type-k f16 --cache-type-v f16 \
         --parallel 1 --port 8080 --spec-type draft-mtp --spec-draft-n-max 12 --spec-draft-p-min 0.5
# 环境变量（脚本已内置）：
#   GGML_CUDA_GRAPH_OPT=1  GGML_MMVQ_MAX=2  GGML_NVFP4_WIDE_MAX=0
#   注意：本件**只读前两个**（字符串级核对过）；第三个是给后续实验件留的门，设了无害
```

三条路线实测（同一件、同一板、f16 KV）：

| 路线 | 128K | 200K | 备注 |
|---|---|---|---|
| **DFlash2**（`SPEC=dflash2`） | **22.178 t/s**（步耗 234.5 ms） | **19.698 t/s**（253.8 ms） | 本包推荐口径 |
| MTP 头回退（`SPEC=mtp`） | **19.61 t/s** | ~14.68 t/s（p-min 0.7 口径） | 零额外下载；200K 掉到 15 以下 |
| 只跑 target（`SPEC=none`） | ~12.9 t/s（8K，社区件同板实测 12.85） | **6.85 t/s（本机 200K 实测）** | 对照用，别拿它上生产 |

逐条为什么：

| 参数 | 作用 | 少了的后果 |
|---|---|---|
| `-ngl 99` | 所有权重层上 GPU | 部分层落 CPU ⇒ 掉到个位数 t/s |
| `-fa on` | FlashAttention | 关掉后长上下文 decode 明显变慢，且 KV 吃更多内存 |
| `--cache-type-k/v f16` | KV 保持 f16 | 换 `q8_0` 省 ~6.5 GiB，但 decode 掉约 1/3（实测，不划算） |
| `--parallel 1` | 单会话 | 多并行会把池榨干并拖慢单流 |
| `-md … --spec-type draft-dflash --spec-draft-n-max 7` | 起草器（**收益最大的一环**） | 没有它：8K 量级 ~12.9 t/s（社区件同板 12.85），本机长上下文更低（**200K 去 spec 实测 6.85 t/s**） |
| `GGML_CUDA_GRAPH_OPT=1` | CUDA Graph 复用 | 每步多出 host 侧开销 |
| `GGML_MMVQ_MAX=2` | 本件的 NVFP4 权重侧快路选型（**这件真正会读**） | 快路选错 ⇒ 权重拷贝腿变慢（实测差 ~50 ms/步） |
| `GGML_NVFP4_WIDE_MAX=0` | 给**后续实验件**留的门；本件的二进制里**没有**这个字符串 | 设了无害、不设也无害；别当提速开关 |

## 5. 起草器是必须的（DFlash2）

本件把 **DFlash2 起草器（llama.cpp PR #27342 形态）编进去了**，判据是它能接受 `--spec-type draft-dflash` 且**不报 "wrong number of tensors"**：

```bash
grep -c draft-dflash <(strings 01-binary/llama-server-aarch64-sm101a-nvfp4)   # >0 即本件支持
```

- DFlash2 checkpoint 与 target 的量化**解耦**：它从 target 的固定层（6/20/34/48/62）读 hidden states，所以**一个起草器文件驱动所有 NVFP4 target**。
- 体积档位：Q2_K_S **0.52 GiB**（推荐，实测比大件更快）/ Q4_K_M 1.06 GiB / Q8_0 2.06 GiB / BF16 3.60 GiB（我们生产用这档）。**Q2_K_S 是最优档不是妥协**：它接受率低一点，但自己那趟前向便宜得多，净值更赚。
- DFlash2 vs DFlash1 的自辨判据：**张量数 81 vs 58**（同一个 `--spec-type draft-dflash`）。文件对不对不用猜，读 GGUF 头即可：
  `python3 -c "import struct,sys;print(struct.unpack('<4sIQQ',open(sys.argv[1],'rb').read(24)))" <draft.gguf>`
  → 第 3 个数 = 张量数：**81 = DFlash2（本件可用）**、58 = DFlash1、19 = MTP 侧车（不是这个用法，见 §5.2）。
- 常见踩坑：社区里的 `…-FastMTP-32K.gguf` 是 `arch=qwen35`、19 张量的 **MTP 侧车**，喂给 `--spec-type draft-dflash` 会被拒；它对应 §5.2 那条路线（且 FastMTP 需要专门编译开关，本件的 `SPEC=mtp` 用的是 target 自带 MTP 头，不需要这个文件）。
- 深度 `--spec-draft-n-max`：**code/JSON 用 7，中文散文用 3**；散文的接受率会掉到 0.12–0.16，深度 7 时起草几乎白干，默认 5 是折中而非最优。
- ⚠️ **本包不含起草器权重文件**（权重体积与分发许可自己解决）。如果你手上没有 DFlash2 起草器，说一声，我们单独放最小那档（0.52 GiB）。

## 5.2 没有起草器文件怎么办：MTP 头回退（零额外下载）

`--spec-type draft-mtp` 用的是 **target GGUF 里自带的 MTP 头**，**不需要 `-md`**、不需要额外文件：

```
<binary> -m $MODEL -ngl 99 -c 131072 -fa on --cache-type-k f16 --cache-type-v f16 \
         --parallel 1 --port 8080 --spec-type draft-mtp --spec-draft-n-max 12 --spec-draft-p-min 0.5
```
实测（同件同板）：**128K 19.61 t/s**、200K ~14.68 t/s（后者 p-min 旋钮用 0.7）。
⇒ 只要 128K，这条就够（>15 t/s）；**要 200K 且 ≥15 t/s，只能上 DFlash2**。

## 6. 「装完只有 9 t/s」排障清单

按顺序查，每条都给了判据；**先查 1–4 再查 5**（这四条占绝大多数）。

| # | 检查 | 判据（一条命令） | 期望 |
|---|---|---|---|
| 1 | 起草器到底开没开 | `grep -c draft-dflash <(strings <binary>)`；服务日志里搜 `draft acceptance` | >0；接受率 > 0.5 |
| 2 | `-md` 是否真的生效 | 日志里搜 `spec`/`draft`，或看 `n_draft` 非 0 | 生效 |
| 3 | 环境变量 | `tr '\0' '\n' < /proc/<pid>/environ \| grep -E 'GGML_'` | 至少有 `GGML_CUDA_GRAPH_OPT=1` 与 `GGML_MMVQ_MAX=2`（本件只读这两个） |
| 4 | 全层上 GPU / FA / KV | `tr '\0' '\n' < /proc/<pid>/environ` 无用，直接看启动日志 `n_gpu_layers`、`flash_attn`、`cache type` | 99 / on / f16 |
| 5 | 大页池够不够 | `grep HugePages_Total /proc/meminfo` | 23552（46 GiB）；低于 42 GiB 时 200K 会掉 CPU 或失败 |
| 6 | 大页是否冷启动 | `HugePages_Free` 是否 ≈ Total | 冷启动后全绿 |
| 7 | carveout 是否配 | `gpu-carveout.sh -g 40`（或你的平台对应命令） | 已执行 |
| 8 | 起草器文件对不对 | 启动日志搜 `wrong number of tensors`；或读 GGUF 头第 3 个数（§5） | 81 = DFlash2；19 = MTP 侧车，不能用这个玩法 |
| 9 | target 量化件 | 是否 NVFP4（block-scale，每 64-K 记录 36 B） | 是；别的量化档走不到这份读数 |
| 10 | 是否多进程抢卡 | `pgrep -af llama-server` | 只有一个 |

补充：**别用 `pkill -f llama-server` 这种宽匹配**清进程（会误杀别的服务）；用 `02-launcher/stop-server.sh`。

## 6.1 9 t/s 是哪一档（对照已公开的配方数字）

公开配方里的数字**全部挂在同一个自编件 + 一个外挂起草器上**，少任何一环就会掉到 9–13 t/s 这一档：

| 场景 | 参考读数 | 来源 |
|---|---|---|
| 无投机（纯 target 单流，16K ctx） | **13.00 t/s**（= 229 GB/s，板载带宽上限的 84%） | 公开配方 §1 底线锚点 |
| 投机深度给太深（3 → 7） | 11.42 → **7.24 t/s** | 公开 TROUBLESHOOTING |
| MTP 头，n_max = 2 / 3 | 19.88 / **20.32 t/s** | 公开配方 §2.1 |
| DFlash2，n_max = 5（560 MB 起草器） | **23.27 t/s** | 公开配方 §2.2 |
| 200K + 完全去掉起草器 | **6.85 t/s**（本机实测） | 本包 §3 |
| 128K + 本包件 + DFlash2 | **22.178 t/s** | 本包 §3 |

⇒ **9 t/s 落在"起草器没生效"那一档**（比 16K 的 13.00 还低，只比 200K 去 spec 的 6.85 高）。先照 §6 的 1–4 条查，再怀疑硬件。

还有一条容易被忽略的事实：**公开仓库只发文档 + 一个 NVFP4/MMVQ 补丁**，86 个文件里没有任何可执行件、没有模型权重、没有 llama.cpp 源码树、也没有 release 下载链接。所以按它复现的人必然是"自己编译 + 自备起草器"，跟出这些数字的件不是同一个二进制。这就是"要不要配合你们的编译件"的答案：

- **件**：本包二进制 = 那份自编件的对外版（含 drafter 路径 + NVFP4 权重侧快路 + 三条 env 门）；
- **起草器**：见 `optional-drafter/`（0.52 GiB 档），或任何 **81 张量**的 DFlash2 件；
- **参数 + 池**：见 §4 与 §6；另外长跑别忘了 `--cache-ram 512`（公开文档里的运行期内存增长定案：不设它约 12 条新提示词后 OOM）。

## 7. 精度：怎么在 30 分钟里自证（不要只信速度）

**结论先给（2026-09-19 实测：150 题 × 8 方向，`temperature=0`、`seed` 固定、预算 512）**

| 请求形态 | 32K 上下文 | 200K 上下文 |
|---|---|---|
| **chat（`/v1/chat/completions`，= 生产前端用法）** | **150/150 全对，0 空答** | **150/150 全对，0 空答** |
| raw-completion（`/completion`，`doc + "\nAnswer:"`） | 83/150（**该臂预算不足，已被 chat 口径重跑取代**） | 112/150；**空回答 37/150 = 24.7%** |

- **空回答是题面形态的产物，不是本件缺陷**：同一份文档、同一上下文、同一份二进制，改用 chat 接口后空答从 37 例降到 **0**，且全部答对。
  证据：去掉起草器后空答仍是同一批 37 题（逐题一致）⇒ 与 `-md`/起草器无关；换成 chat 形态后消失。
  **所以你用聊天前端 / OpenAI 兼容接口时不会碰到它；只有自己拼 raw prompt 才可能碰到。**
- **raw 形态里已给出答案的题：112/113 = 99.1% 正确。**
- 另有两条同口径实测，请一起记住：① **换配置会改答案**（去掉起草器会翻掉 1 题 letter 计数）；
  ② 32K 与 200K 的 chat 输出**逐字一致 148/150**，另 2 题只差 JSON 里的空格（判分都对）。

自测请按下面 8 个方向（我们用的同一套方向），**用 chat 接口**：
1. 算术运算 2. JSON 格式跟随 3. 列表计数约束 4. 标识符转录
5. 字母计数 6. 数字排序 7. 数字串转录 8. 中英混排转录

口径要求：**`temperature=0`、`seed` 固定、回答预算 ≥512 token、32K 与 128K/200K 各跑一遍**。
判据：**应该 100% 正确、0 空回答**；若你用 raw `/completion` 拼题面并在长上下文看到空回答，先换 chat 接口再判断是不是本件的问题。

## 8. 负结果黑名单（别人已实测否决，别再花时间）

| 假设 | 实测 | 结论 |
|---|---|---|
| 权重拷贝粒度 4 B → 16 B | 257.7 → 226.9 GB/s（只值 −12%），而缺口是 1.89× | 排除（粒度解释不了） |
| 慢是因为喂了理想化取数 | 真实 GGML 权重切片只差 −5.5% | 排除（内存侧三假设全关） |
| MMQ prefetch / M-padding | 逐项为负 | 否决 |
| 环等待组深度 `<1>`→`<0>` | 两臂 D dump 逐字节相同（ratio 1.032） | 否决 |
| KV 量化 q8_0 @256K | 省 6.5 GiB，decode 掉 1/3 | 不划算 |
| "接受率拉满就能过 30 t/s" | 满接受上界仍 <30 t/s | 否决 |
| A 段常驻 smem | 102.1 vs 100.9 GB/s（+1.2%） | 关闭（低于预注册门） |

定性结论：`136.5 GB/s`（生产）与 `257 GB/s`（隔离原型）之间差 1.88×，而粒度只值 −12%、寻址只值 −5.5% ⇒ **缺口在消费者/结构侧，不在访存侧**。

**后续定位（2026-09-21）**：rb2 诊断系列已把"消费者/结构侧"具体化——`prodsf 203.53 → proda 144.21 GB/s（−29.1%）`，`proda` 已距生产腿仅 +5.6%，**主税项 = A ring/SFA 生产接线**；D-store ≈0、B sf/SFB ≈−10%、B 环本身可回 260。详见 [`03-reference/production-wire-attribution-2026-09-21.md`](03-reference/production-wire-attribution-2026-09-21.md)。

## 9. 已知限制与诚实声明

- **本件是我们内部分支的编译产物**，基于 llama.cpp 上游 `72797e89`（2026-09-10）加本地补丁集（NVFP4/MMQ 权重侧快路、`speculative.cpp` 起草路径、server 侧若干改动），**不保证与上游 HEAD 兼容**。
- 我们**没有**把更快的实验件放进本包：有一条把 128K 步耗压到 186.0 ms（25.75 t/s）的路线，但它的精度结论**未定论**（长上下文下 2 题退化），按"宁可慢也不能错"的原则没发。
- 二进制里 325 处编译期源码路径字符串已做**等长替换**脱敏（内部项目名/构建路径 → 中性路径占位）。替换后已验证：文件长度不变、ELF 头/程序头/段头/动态段逐字节不变、**反向替换可逐字节还原出原文件**（⇒ 除这些字符串外，两者完全相同）。**替换后本件 sha256 与原内部件不同**，以 `01-binary/SHA256SUMS.txt` 为准。脱敏映射表本身不在包内列出（列出来等于反向泄漏）。
- 不含任何凭据、内网地址、账号名、主机代号、内部资产名；不含模型权重；不含我们内部文档编号（需要追溯的话，问我们）。
  包内文本文件 + 二进制字符串都过了一遍黑名单扫描（内部项目/品牌/账号/内网/资产名），命中为 0。
- 200K 大页池 46 GiB + 权重 16 GiB 级 + f16 KV，48 GB 统一内存是**紧的**：先 128K 跑通再上 200K。

## 10. 校验

```bash
sha256sum -c 01-binary/SHA256SUMS.txt
```
