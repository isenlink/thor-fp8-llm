# 10-01 定稿增量 — Thor04 生产 262K 配置（在 9/27 复现包基础上）

> **One-liner**: production-final config update (2026-10-01, rev.2 2026-10-02, rev.3 2026-10-02) on top of the
> 9/27 reproduction package: 262144 context / 2 concurrent / MTP+1, 52G hugepage pool,
> prefix-caching proven non-functional for the GDN architecture (engine forces it off),
> 150-question acceptance passed with 0 empty answers.
> **rev.2 (2026-10-02)**: MTP speculative depth finalized to **1 — depth ≥2 is PROHIBITED**:
> A/B/C test proved num_speculative_tokens 2 or 3 crash the engine (GPU illegal write)
> under concurrent load; n=1 verified 30/30 + speed 17.3-17.7 tok/s (faster than n=3).
> **rev.3 (2026-10-02)**: OpenAI Function Calling / tool-call support added:
> `--enable-auto-tool-choice --tool-call-parser qwen3_xml`. **Parser must be `qwen3_xml`**
> (Qwen3-family native XML format) — `hermes` parser fails to parse Qwen3 XML tool calls
> (7 logged errors, raw XML leaks into message content). Requests without `tools` are
> unaffected; verified structured `tool_calls` returned for both `auto` and `required`.

> **关系**：`vllm-sm101-replica/`（本目录其余部分）= 9/27 复现包（框架怎么装、怎么跑通）；
> 本文档 = **2026-10-01 Thor04 生产定稿配置**（部署参数最终值 + 实测定案）。
> 两份一起看 = 完整复现路径。

## 定稿启动命令（Thor04 生产，vLLM 0.11.2 + torch 2.9.0）

```bash
venv/bin/vllm serve <模型目录> \
  --served-model-name qwen3.8-27b \
  --max-model-len 262144 \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.90 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --reasoning-parser deepseek_r1 --enforce-eager \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --port 8080
```

> **tool-call 参数（rev.3 新增）**：`--enable-auto-tool-choice --tool-call-parser qwen3_xml`
> 提供 OpenAI Function Calling 支持。**parser 必须选 `qwen3_xml`**——Qwen3 系模型输出的工具调用
> 是 Qwen3 原生 XML 格式（`<tool_call><function=…><parameter=…>`），社区常见的 `hermes` parser
> 只认 JSON 格式，实测解析失败（日志 `hermes_tool_parser.py Error in extracting tool call`），
> XML 原文漏进 `message.content`。不带 `tools` 字段的请求完全不受影响（普通对话路径零变化）。

**环境前提**（52G 大页池档，与 9/27 包 docs/02-BOARD-PREP.md 流程同源）：

| 项 | 定稿值 | 说明 |
|---|---|---|
| 大页池 | **26624 页 = 52 GiB**（`vm.nr_hugepages`，固化四处，冷启动分配） | 池太小 CUDA OOM、太大宿主 OOM；52G 是本负载实测档 |
| 模型 | Qwen3.8-27B NVFP4（compressed-tensors：attention/linear_attn/lm_head = fp8 int8 激活，MLP = nvfp4 g16） | 本体 22.2G + MTP draft 810M |
| max_position_embeddings | 262144 | tokenizer model_max_length 同值 |

## 内存账（52G 池实测）

| 项 | 值 |
|---|---|
| 模型加载 | 22.23 GiB（24.5 s，含 MTP draft 4.5 s） |
| KV cache | **24.02 GiB = 383,368 tokens**（vLLM PagedAttention，GDN 混合架构） |
| 262K 满长单请求并发余量 | **1.40x**（即 KV 池够 1.4 个 262K 满长请求） |
| `--max-model-len 262144` 语义 | 单请求允许最长 = 配置上限；多请求按需调度，超池会排队 |

## prefix caching 定案（2026-10-01，重要）

**`--enable-prefix-caching` 对本架构（Qwen3-Next hybrid GDN/线性注意力）不生效——写了也白写**：

1. 启动日志：`Hybrid or mamba-based model detected without support for prefix caching: disabling.`
2. 源码链（vLLM 0.11.2）：
   - `vllm/engine/arg_utils.py:1847` — hybrid 模型默认 `default_prefix_caching=False`
   - `vllm/model_executor/models/config.py:294`（`MambaModelConfig.verify_and_update_config`）— 静默改回 False
   - `vllm/model_executor/models/qwen3_next.py:1228` — 对应 assert
3. 根因：GDN（gated delta net）是 SSM 线性注意力，状态不可跨请求安全复用
4. 运行期指标恒为 `Prefix cache hit rate: 0.0%`

**实践含义**：每次请求全量 prefill；多轮对话只有会话内 chunked prefill（默认开），
无跨请求缓存。200K 口径 prefill 吞吐 ~20,000 t/s（≈100 s/次）。

## 验收数据（2026-10-01，150 题精度套件）

| 项 | vLLM-P3（本包） | llama.cpp 同板（对照） |
|---|---|---|
| 空答 | **0/150** | 32/150（21.3%，200K 前缀缓存命中场景缺陷） |
| 总分 | **119/150** | 118/150 |
| 单流速度 | ~14 t/s（MTP+1，MAL 1.78） | 短文 12.6 t/s |
| 200K 双发 | ✅ 存活 | 引擎缺陷（空答） |
| prefill | **~47 倍快**（9,500 vs 200 t/s） | — |

选型结论：**长上下文/高吞吐 prefill 场景选 vLLM（本包）；短结构化输出场景 llama.cpp 更快**。

## 与 9/27 复现包的差异

| 项 | 9/27 包（`vllm-sm101-replica/` 其余部分） | 10-01 定稿（本文档） |
|---|---|---|
| 池 | 54G | **52G**（26624 页） |
| max-model-len | 200K | **262144** |
| max-num-seqs | 未强调 | **2**（262K 长对话并发平衡点） |
| MTP | 1.7-2.0 MAL 口径 | **num_speculative_tokens=1**（实测 MAL 1.88-1.96、接受率 0.885+、17.3-17.7 t/s）。**🔴 10-02 定案：深度 ≥2 禁用**——n=3 在并发下 3 次崩溃（含 2 次生产自然复现）、n=2 压测复现，均为 GPU 野写（`MMU FAULT invalid pde` → CUDA illegal memory access → EngineDead）；n=1 对照 90/90 + 生产 30/30 全过。曾试 n=3 的 MAL 2.76 读数虽真实，但**并发不安全，勿再启用**（vLLM 0.11.2 + qwen3_next 混合架构，崩溃在请求起步步进，与 KV/池/温度无关） |
| enforce-eager | 可选 | **定稿保留**（排障期配置；去掉可上 CUDA Graph 但需先验坑 9） |
| prefix caching | 有 flag | **明确不生效**（见上节，flag 无需写） |

## 快速复现 checklist（增量部分）

1. [ ] 按 `00-QUICKSTART.md` 装好 venv + wheel（本包 `binaries/`）
2. [ ] 大页池 = **26624 页（52G）**，固化 + 冷启动重启（回读 `/proc/meminfo` 验证）
3. [ ] 模型目录就位（本体 + `model_mtp.safetensors` + tokenizer）
4. [ ] 用本文档"定稿启动命令"拉起，等日志 `Application startup complete`
5. [ ] 验收四条日志：`Model loading took 22.23 GiB` / `Available KV cache memory: 24.02 GiB` /
      `GPU KV cache size` 38 万级 / `Maximum concurrency for 262,144 tokens: 1.40x`
6. [ ] 冒烟 `/v1/chat/completions` 200

---

**变更记录**：2026-10-01 首版（依据 Thor04 生产定稿通报 + 当日两轮串口实测 + 150 题验收）。
**rev.2 2026-10-02**：MTP 深度定案 = 1，**≥2 禁用**。三臂实验（n=3 崩×3 / n=2 崩 / n=1 对照 90+30 全过，
n=1 速度 17.3-17.7 t/s 快于 n=3）——vLLM 0.11.2 + qwen3_next 上 `num_speculative_tokens≥2`
遇并发请求即 GPU 野写崩溃（dmesg `MMU FAULT invalid pde, virt write` → EngineDead），
与 KV 占用/池/温度无关；板上生产脚本已回 n=1 并验收。定案报告见仓库外
`thor-work/thor04-mtp3-crash-abtest-20261002.md`（内部档案）。
**rev.3 2026-10-02**：加 tool-call 支持（`--enable-auto-tool-choice --tool-call-parser qwen3_xml`，
agent 接入 Function Calling 必备）。实测：第一版 `hermes` parser 对 Qwen3 XML 工具格式解析失败
（7 条报错、XML 漏进正文，用户侧表现为回复质量异常），换 `qwen3_xml` 后 `auto`/`required` 均正确
返回结构化 `tool_calls`，普通对话与质量冒烟零变化，health 200。
