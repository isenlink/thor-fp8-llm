# Runtime memory growth solved: the host prompt cache exceeds available RAM

**English summary**: llama-server's resident memory grew by ~626 MB for every *novel* prompt until the process was OOM-killed around the 12th prompt. The cause is the host-side prompt cache (`--cache-ram`, default **8192 MiB**): every new prompt stores a state snapshot in host RAM, and each snapshot carries the slot's context checkpoints — one checkpoint measures 149.8 MiB even for a 42-token prompt. Because this board's 46 GiB hugepage pool leaves only ~7.3 GB of movable RAM, the process dies long before the cache reaches its own 8 GiB limit. Fixed by `--cache-ram 512`. Three earlier hypotheses (hugepage pool, CUDA graph capture, `--ctx-checkpoints`) were each excluded by A/B experiments.

# 运行期内存增长：**根因定案 = 主机侧 prompt cache 超过可用内存**（2026-09-16）

> 平台：DRIVE Thor（p3960-0010 / T264 / sm_101a / DriveOS 7.0.3，统一内存 58 GiB，46 GiB 大页池）
> 栈：llama.cpp 自编 build + Qwen3.8-27B NVFP4（17.6 GB）
> 结论：**已定位并修复**；本文保留完整的"现象 → 排除 → 定案 → 验证"链，供他人复用这套排查方法

---

## 1. 症状

单个 server 实例连续处理**不同**提示词（30 题内容矩阵）时，稳定在第 12-13 个请求被内核 OOM 杀掉：

```
llama-server invoked oom-killer: gfp_mask=0x140cca(GFP_HIGHUSER_MOVABLE|__GFP_COMP)
oom-kill: global_oom ... Out of memory: Killed process (llama-server)
        anon-rss:7937488kB, file-rss:1337828kB
```

客户端先看到 `Remote end closed connection without response`，其后全部 `Connection refused`
（**server 已死，不是网络问题**）。

## 2. 第一步：把"增长绑定什么"测出来

| 实验 | 结果 | 结论 |
|---|---|---|
| 同一提示词连续 6 次（256 token） | RSS 1367 → 1895 → 2372 → **2372 → 2372 → 2372** | 重复同一提示词**不增长** |
| 同一提示词 6 次（64 token） | 全程平坦 | 与**生成长度无关** |
| 换新提示词 | 每次抬升 | 增长绑定"**新的、不同的提示词**" |

⇒ 不是"每请求泄漏"，也不是"每 token 泄漏"，而是**每个新提示词一条**。这也解释了为什么早期用
**6 个固定提示词 × 3 次重复**的 harness 从未触发（只有 6 条新上下文）。

## 3. 第二步：三个顺理成章的假设，全部被自己的 A/B 证伪

| # | 假设 | 实验 | 结果 |
|---|---|---|---|
| 1 | 46 GiB 大页池把常规内存挤空 | 把池缩到 4 GiB / 20 GiB 后加载 | ❌ **池是必需品**：4 GiB 报 `unable to allocate CUDA0 buffer`，20 GiB 加载中途被杀，46 GiB 才 ready |
| 2 | CUDA graph 按 shape 缓存累积 | `GGML_CUDA_DISABLE_GRAPHS=1` 重跑同一曲线 | ❌ 曲线与默认**逐点一致**（+623 vs +626 MB/提示词） |
| 3 | 上下文检查点（`--ctx-checkpoints`）累积 | `--ctx-checkpoints 0` 重跑 | ❌ 曲线**逐点一致**（当时默认值也被误记为 8，实为 32） |

> 方法论：**"现象 + 一个看似自洽的机制"不等于定案**。上面每个假设都做过实验，
> 也因此把范围收窄到"某条**主机侧**、按提示词记账的缓存/状态"。

## 4. 第三步（定案）：读源码找"主机侧按提示词记账的结构"，然后做剂量-反应实验

源码里 server 有两个**互相独立**的主机侧状态仓库：

| 机制 | 参数 | 默认 | 单个条目 |
|---|---|---|---|
| **prompt cache**（`server_prompt_cache`） | `--cache-ram N`（MiB） | **8192** | 一次"提示词状态"快照（含该 slot 的**上下文检查点**） |
| context checkpoints（每 slot） | `--ctx-checkpoints N` / `--checkpoint-min-step` | 32 / 8192 | 一个检查点（`std::vector<uint8_t>`，**主机堆内存**） |

**剂量-反应实验（决定性）**：同一模型、同 ctx 16384、同 15 个**不同**提示词，只改 `--cache-ram`：

| `--cache-ram` | 加载后 | 请求 3 | 6 | 9 | 12 | 15 | 表现 |
|---|---:|---:|---:|---:|---:|---:|---|
| **0**（关） | 1365 MB | 2044 | 2047 | 2054 | 2054 | **2054** | **+689 MB 后完全平坦** ✅ |
| **512** | 1368 MB | 2515 | 2518 | 2525 | 2525 | **2525** | **+1157 MB 后完全平坦** ✅ |
| **默认（8192）** | 1367 MB | 3137 | ↑ | ↑ | ↑ | ❌ **第 13 个请求 OOM** | **线性增长** ❌ |

**平台高度随 flag 值变化**（0 → 512 相差约 0.5 GB）⇒ 吃内存的就是这个缓存；
默认档则一路涨到设备装不下为止。

**默认档的完整增长曲线**（同一批 15 个不同提示词，作为复现对照）：

| 新提示词序号 | 加载后 | 3 | 6 | 9 | 12 | 13 |
|---|---:|---:|---:|---:|---:|---:|
| RSS | 1367 MB | 3137 MB | 5004 MB | 6875 MB | **8691 MB** | ❌ OOM |
| `MemAvailable` | 7273 MB | 5562 MB | 3703 MB | 1825 MB | **109 MB** | — |

⇒ 线性 **+626 MB / 新提示词**；第 12 个请求时可用内存只剩 109 MB，第 13 个被 OOM 杀掉。

**server 自己给出的两个关键数字**（`-v` 可见）：

```
load_model: prompt cache is disabled - use `--cache-ram N` to enable it
slot create_check: created context checkpoint 1 of 32 (pos_min = 41, pos_max = 41,
                   n_tokens = 42, size = 149.791 MiB)
```

⇒ **一个只有 42 token 的提示词，其上下文检查点就有 149.8 MiB**——检查点大小按**全量状态**算（含
混合线性注意力层的循环状态），**与提示词实际长度无关**。这就是"每新提示词 ~0.6 GB"的来源。

## 5. 为什么默认值在这块板上必然 OOM（机制）

```
58 GiB 总内存
├─ 46 GiB 大页池（CUDA 分配路径必需，见 §3 假设 1）
└─ ~7.5 GiB 常规/可移动内存（内核 + tmpfs + 服务）
        └─ 默认 prompt cache 上限 = 8192 MiB = 8 GiB  ← 比剩下的还多
```

**缓存自身逻辑没问题**（超限会跳过、会淘汰最旧条目），问题在于**默认上限 8 GiB > 本机可用的
~7.5 GB**：缓存还没涨到自己的天花板，进程就先被 OOM 杀了。

⇒ **这是一条"默认值和平台不匹配"的缺陷**，不是内存泄漏。

## 6. 修复

| 方案 | 效果 | 代价 |
|---|---|---|
| **`--cache-ram 512`（推荐）** | 内存平台稳定（+1.2 GB 封顶），**前缀缓存仍然工作** | 无（512 MiB 足够覆盖典型复用窗口） |
| `--cache-ram 0` | 最省（+0.7 GB 封顶） | **关闭前缀缓存** ⇒ 破坏"一次载入、反复提问"用法（见 `kv-budget-and-256k-2026-09-16.md`），并连带关闭 `--cache-idle-slots` |
| 每 10 个提示词重启实例（旧规避） | 有效但笨 | 浪费装载时间；**已被上面的 flag 取代** |

## 7. 验证

用修好的配置跑**原来必崩的那个规模**：**30 个不同提示词**、**单 server 实例**、`--cache-ram 512`、
**不再分批重启**：

| 进度 | RSS | MemAvailable |
|---|---:|---:|
| 加载后 | 1369 MB | 7277 MB |
| 第 5 题 | 2524 MB | 6179 MB |
| 第 10 题 | 2531 MB | 6171 MB |
| 第 15 题 | 2532 MB | 6164 MB |
| 第 20 题 | 2532 MB | 6152 MB |
| **第 30 题（跑完）** | **2532 MB** | **6151 MB** |

⇒ **30/30 全部完成、无 OOM，RSS 在 +1.16 GB 处封顶后完全平坦**。
对照组（同一批提示词、默认 `--cache-ram`）：第 13 个请求被 OOM 杀掉。

## 8. 复现脚本

```bash
# 逐请求采样：RSS / 大页池空闲 / MemAvailable
cat > /tmp/probe.sh <<'EOF'
#!/bin/bash
p=$(pgrep -x llama-server | head -1)
[ -n "$p" ] && awk '/VmRSS/{print $2}' /proc/$p/status || echo 0
awk '/HugePages_Free/{print $2}' /proc/meminfo
awk '/MemAvailable/{print $2}' /proc/meminfo
EOF
```
```
# 三臂对照：只改 --cache-ram，其余全同
llama-server ... --cache-ram 0     # 期望：平坦
llama-server ... --cache-ram 512   # 期望：平台在 ~1 GB 增量
llama-server ... (默认)            # 期望：线性增长至 OOM
```
判据：**平台高度随 `--cache-ram` 变化** ⇒ 命中本问题；若平台高度与 flag 无关，则另有来源。
排查顺序固定：①先用"重复同一提示词"分离 per-request / per-new-prompt ②再逐个 A/B 排除
大页池 / CUDA graph / checkpoints ③再读源码找"主机侧按提示词记账的结构" ④最后用**剂量-反应**
钉死（只改一个容量参数，看平台是否随它移动）。

---

## [整理者注] 处理清单

- 品牌 / 车型 / 渠道 / 解锁类内容：无
- 内网 IP / MAC / SN / 账号名 / 凭据 / ssh 命令：已移除
- 人名与 AI 助手名：统一为"作者 / AI助手"
- 主机代号：统一为"x86主机"；板序号写作 board3
- 板载分区路径：写作 `/brand_data/`
- 技术数据（内存曲线、内核报错原文、flag 名称与默认值、检查点大小、实验命令）：100% 保留
