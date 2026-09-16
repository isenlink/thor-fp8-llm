# Runtime memory growth: ~626 MB per new prompt, OOM after ~12 distinct prompts

**English summary**: On this stack llama-server's resident memory grows by ~626 MB for every *new* (novel) prompt and is never released, so a sweep with ~25 distinct prompts gets OOM-killed around prompt 12. Repeating the same prompt does not grow memory. We ruled out three hypotheses with A/B experiments (hugepage pool, CUDA graph capture, context checkpoints); the root cause is still unidentified, and the working mitigation is to restart the server every ~10 distinct prompts.

# 运行期内存增长：每"新提示词"约 626 MB 不释放（2026-09-16）

> 平台：DRIVE Thor（p3960-0010 / T264 / sm_101a / DriveOS 7.0.3，统一内存 58 GiB，46 GiB 大页池）
> 栈：llama.cpp 自编 build（`llama.cpp-tcgen05`）+ Qwen3.8-27B NVFP4（17.6 GB）
> 状态：**根因未定**，已排除 3 个假设，给出可用规避方案与复现方法

---

## 1. 症状

用一个 server 实例连续跑**不同**提示词（内容矩阵，25 条不同题）时，稳定在第 12-13 个请求被 OOM 杀掉：

```
llama-server invoked oom-killer: gfp_mask=0x140cca(GFP_HIGHUSER_MOVABLE|__GFP_COMP)
oom-kill: global_oom ... Out of memory: Killed process (llama-server)
        anon-rss:7937488kB, file-rss:1337828kB
```

客户端侧表现为 `Remote end closed connection without response`，其后全部 `Connection refused`
（**server 已死，不是网络问题**）。

## 2. 取证：逐请求内存曲线（无任何草稿模型，ctx 16384）

| 请求序号 | RSS (MB) | 大页池空闲(页) | MemAvailable (MB) |
|---:|---:|---:|---:|
| （加载后） | 1367 | 14631 | 7304 |
| 1 | 1884 | 14582 | 6845 |
| 3 | 3134 | 14582 | 5597 |
| 5 | 4379 | 14582 | 4350 |
| 8 | 6247 | 14582 | 2466 |
| 10 | 7494 | 14582 | 1225 |
| 12 | 8706 | 14582 | 102 |
| 13 | ❌ OOM | | |

**两条关键事实**：
1. **线性增长 ≈ 626 MB / 新提示词**（1367 → 8706，9 个请求 +7.3 GB）；
2. **大页池占用完全不变（14582 页恒定）** ⇒ 增长在**常规 / 可移动内存侧**，
   与模型权重、KV 的池分配**无关**。

## 3. 判别实验：增长绑定什么

同一 server 实例做三段测试：

| 段 | 操作 | RSS 结果 | 结论 |
|---|---|---|---|
| A | **同一**提示词 × 6 次（256 token） | 1367 → 1895 → **2372 → 2372 → 2372 → 2372** | 重复同一提示词**不增长** |
| B | 同一提示词 × 6 次（64 token） | 2376 → 2379 → …（**平坦**） | 与**生成长度无关** |
| C | **换新**提示词 × 4 次（64 token） | 2529 → 2379 → 2379 → 2379 | 新提示词会抬一点，短生成时几乎不涨 |

⇒ **增长绑定"新的、不同的提示词/上下文"**，不是请求次数，也不是生成 token 数。这也解释了历史现象：
- 早期配对基准用 **6 个固定提示词 × 3 次重复** ⇒ 只有 6 个新上下文（增长 ~3.8 GB，
  在 7.3 GB 余量内）→ **234 次请求无事故**；
- 本轮内容矩阵 **25 条不同提示词** → **第 12 条撞墙**。

## 4. 已排除的假设（每个都有 A/B 实验，不是推理）

| # | 假设 | 实验 | 结果 |
|---|---|---|---|
| 1 | 46 GB 大页池抢走常规内存导致 OOM | 把池缩到 4 GB / 20 GB 后加载 | ❌ **证伪**：4 GB 报 `unable to allocate CUDA0 buffer`；20 GB 加载中途被杀；恢复 46 GB 立即可用 ⇒ **池是必需的，不能缩** |
| 2 | CUDA graph 按 shape 缓存累积 | `GGML_CUDA_DISABLE_GRAPHS=1` 重跑同曲线 | ❌ **无效**：曲线与默认**逐点一致**（仍 ~626 MB/新提示词） |
| 3 | 上下文检查点（`--ctx-checkpoints`，默认 8）累积 | `--ctx-checkpoints 0` 重跑 | ❌ **无效**：曲线同样逐点一致 |

**尚未验证的方向**（留给后续 / 上游）：提示词缓存与 LCP 槽状态、`--kv-unified` 行为、
ggml CUDA 后端按"新图形状"缓存 workspace 的策略、`--cache-reuse` 相关路径。

⚠️ **上游复现前提**：本板 llama.cpp 是**自编 build**（含自定义 kernel），
报上游前需在**标准 build** 上验证是否能复现。

## 5. 规避方案（已用于多轮测试，稳定）

**分批重启**：每个 server 实例处理**不超过 10 条不同提示词**（重复同一提示词可无限次），到量重启。
（按 626 MB/条、常规内存余量 ~7.3 GB 计算，10 条约用 6 GB，留 2 倍余量。）

配套纪律（已固化到 benchmark 流程，见 `06-benchmarks/benchmark-methodology-2026-09-16.md` §5）：
- 启动前 `pkill -9 -x llama-server` 确认无残留（避免两实例并发加载导致 OOM）；
- 启动失败自动重试 3 次并打印 server 日志尾部，避免"静默失败"；
- 长跑期间每 5 分钟看一次进度与内存水位。

## 6. 复现脚本（本仓库脚本目录风格，放板端）

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
# 判别型探测（同一提示词重复 / 换新提示词 / 不同生成长度）
for i in $(seq 1 12); do <发一次请求>; echo "$i $(bash /tmp/probe.sh | tr '\n' ' ')"; done
```

**判据**：RSS 随"不同提示词数"线性上升 ⇒ 命中本问题；RSS 随"请求次数"上升 ⇒ 是另一种泄漏。

---

## [整理者注] 处理清单

- 品牌 / 车型 / 渠道 / 解锁类内容：无
- 内网 IP / MAC / SN / 账号名 / 凭据 / ssh 命令：已移除
- 人名与 AI 助手名：统一为"作者 / AI助手"
- 主机代号：统一为"x86主机"；板序号写作 board3
- 板载分区路径：写作 `/brand_data/`
- 技术数据（内存曲线、内核报错原文、实验命令、阈值）：100% 保留
- 未定项显式标注"根因未定"，不写成已解决
