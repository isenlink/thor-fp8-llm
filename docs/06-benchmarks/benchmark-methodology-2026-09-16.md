# Benchmarking methodology: three ways we fooled ourselves (and how to avoid them)

**English summary**: Three measurement traps that produced wrong numbers for us: mixing `prompt eval time` into decode statistics (inflates medians into a fake 40-60 tok/s band), treating a larger `-c` as "a long-context test" (short prompts give bit-identical results), and quoting peak rolling values instead of steady-state medians. Plus the paired-benchmark design, `/tokenize` calibration, prefix-cache accounting, and the test-harness hygiene that keeps multi-hour sweeps from silently dying.

# 基准方法论：三个我们亲自踩过的口径陷阱 + 配对设计（2026-09-16）

> 作者：AI助手（x86主机）· 平台：DRIVE Thor（DriveOS 7.0.3 / llama.cpp 自编 build）
> 适用范围：llama.cpp server 的 decode/prefill 基准、投机解码配置评比、长上下文测试

---

## 1. ⚠️ 陷阱一：把预填充吞吐混进解码统计（数字会虚高一倍）

llama-server 对**同一个任务**打印两行计时：

```
slot print_timing: id 0 | task 0 | prompt eval time =  532.38 ms /  62 tokens ( 8.59 ms per token, 116.46 tokens per second)
slot print_timing: id 0 | task 0 |        eval time = 10826.57 ms / 256 tokens (42.46 ms per token,  23.55 tokens per second)
```

**`prompt eval time` 里含子串 `eval time =`**，所以宽松正则
`r"eval time =.*?([\d.]+) tokens per second\)"` 会把**预填充（60-150 t/s）**和**解码（20-50 t/s）**
混进同一个列表——中位数被抬到**虚假的 40-60 档**。
（我们据此对外报过"DFlash2 打到 55-62 t/s"，真值 48.7，被迫撤回。）

**正确抓法（二选一）**：

```python
# ① 行首锚定（推荐）：只有缩进后的裸 eval time 才匹配
RE = re.compile(r"\|\s*task\s+\d+\s*\|\s+eval time =\s*([\d.]+) ms\s*/\s*(\d+) tokens.*?([\d.]+) tokens per second\)")
# ② 否定后顾
if re.search(r"(?<!prompt )eval time =", line): ...
```

配套三条纪律：
- **过滤超短样本**（<20 token 的生成噪声极大）；
- **报中位数 + 区间**，不报单值；
- **报数前先 `grep` 一条原始日志行、手工核一遍正则到底在抓哪个数**。

## 2. ⚠️ 陷阱二：把 `-c` 调大当成"测了长上下文"

我们做过一轮"32K 复测"：只把 `-c 8192` 改成 `-c 32768`，其余不变 → 结果与 8K 档**逐位相同**
（20.32 vs 20.31、23.27 vs 23.26，连各内容分类都一致）。

**这不是巧合**：server 日志确认 `n_ctx_slot = 32768` 生效了，但**提示词只有 28-80 token**，
实际 KV 长度只有几十 token——**解码速度只取决于"实际 KV 长度 + 权重读取"，与 ctx 上限无关**。
`-c` 只是容量预留。

**判据（必须查）**：日志里的 `prompt eval time = … N tokens`，
**N 没上去就说明根本没测成长上下文**。

```bash
# 构造长 prompt 用 /tokenize 精确计数，不要按字符密度估算
curl -s -X POST http://127.0.0.1:8080/tokenize -H 'Content-Type: application/json' \
     -d '{"content":"<text>"}' | jq '.tokens | length'
```
用"目标 token 数 ÷ 首轮实测值"迭代修正 units，1-2 轮即命中（实测 Qwen 分词器密度 ≈ 60 tok/单元文本块）。

## 3. ⚠️ 陷阱三：拿"峰值/滚动值"当成绩

server 日志里 `n_gen = …, tg = x t/s, tg_3s = y t/s` 是**滚动窗口值（含预热段）**，会明显偏高；
`eval time = … tokens per second` 才是**整段汇总值**。汇总表里只放后者，
并且**同一次实验里不要混用两种口径**（长上下文 benchmark 里曾因混用得出方向相反的结论）。

## 4. 配对基准（paired benchmark）设计

**为什么要配对**：同配置、同 build、同 server 进程，仅换 nonce/prompt 内容，
acceptance 就能在轮次间浮动 ±7pp（decode 直接差 −24%）。**单次跑分不能用于排名。**

**设计契约**：

| 项 | 要求 |
|---|---|
| 目标模型 / build | 固定；一次只动一个变量（**这条是历史事故的根因，见 §6**） |
| ctx / 生成长度 / temp / seed | 全部固定并写进结果表 |
| 提示词集 | **多类内容 × 多条不同题**（我们最终用 6 类 × 5 题 = **30 题**），避免只用 1-2 条 |
| 每配置 | 重启 server，跑完整提示词集，抓服务端日志按 task 归并 |
| 指标 | 解码 t/s **中位数 + 区间**、acceptance、mean accepted len、预填充 t/s |
| 交叉验证 | **第一个配置用"已知答案"的档位**：本次自建 harness 跑内置 MTP n3 得 20.3 t/s，与既有日志同配置 24.4 量级吻合（差 17%，来自提示词内容差异）——若两边差 >2 倍或方向相反，先查口径再往下跑 |
| 预算 | 30 题 × 3 配置 ≈ 40-60 分钟（含装载） |

**必须同时报 acceptance 与 mean accepted len**：`t/s ≈ 1000 × mean_len ÷ step_ms`，
只看 t/s 无法归因（高接受率 ≠ 更快，本季已出现多例）。

## 5. 测试基建纪律（多轮踩坑后的固化清单）

1. **单实例**：启动前 `pkill -9 -x llama-server`（`-x` 精确匹配进程名；**绝不用 `pkill -f`**——
   它会匹配到 ssh 自身命令行导致连接自杀），确认无残留再起。残留实例与新实例并发加载会直接触发 OOM。
2. **启动失败要重试并把日志尾打出来**：否则表现为"静默失败"，白等一整轮。
3. **给 prompt cache 设上限（必做）**：llama-server 的 `--cache-ram` **默认 8192 MiB**，
   每处理一个新提示词就往主机内存存一份状态快照 ⇒ 在可用内存紧张的设备上会 OOM
   （本板 46 GiB 大页池后只剩 ~7.5 GB，第 12-13 个不同提示词必被杀）。
   **起服务固定加 `--cache-ram 512`**（已在 30 题单实例跑通验收）；`--cache-ram 0` 最省内存但会关闭
   前缀缓存，进而破坏"一次载入、反复提问"。见 `05-system-tuning/runtime-memory-growth-2026-09-16.md`。
4. **服务端与客户端口径都记**：客户端墙钟含网络与排队，服务端 `eval time` 才是 decode 真值。
5. **每 5 分钟查一次进度**：查"起没起来"（第一个 5 分钟点是验收点）、已出结果条数、板温/内存。

## 6. 归因纪律：一次只动一个变量（本季最大教训）

一次"DFlash2 无效"的判定，实际上是**同时**换了目标模型（重编码导致数值损坏，acceptance 71.4% → 0.68%）
和草稿方案——损坏被归因给了草稿，导致这条路线被错误关闭数月。撤回与新证据见
`04-nvfp4-optimization/dflash2-revalidation-2026-09-16.md`。

**落地做法**：
- 改动前先写下"本次只动 X"，实验记录里显式列出同时变了的变量；
- 涉及量化/权重路径的改动，**必须先过"数值健康"关**（短补全通顺 + 与基线同档 acceptance 对照），
  再评比性能；
- 跨文档引用旧结论时，**连"当时的实验前提"一起引**。

---

## [整理者注] 处理清单

- 品牌 / 车型 / 渠道 / 解锁类内容：无
- 内网 IP / MAC / SN / 账号名 / 凭据 / ssh 命令：已移除
- 人名与 AI 助手名：统一为"作者 / AI助手"
- 主机代号：统一为"x86主机"；板序号写作 board3
- 内部绝对路径与脚本名：改为通用描述（`~/work/...`）
- 技术数据（正则表达式、阈值、t/s、接受率、脚本行为）：100% 保留
