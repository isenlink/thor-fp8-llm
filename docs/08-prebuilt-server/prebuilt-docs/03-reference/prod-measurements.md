# 实测读数与口径（本件的来源）

平台：NVIDIA DRIVE Thor / aarch64 / `sm_101a` / 14 SM，统一内存 47 GB 量级。
配置：target = Qwen3.8-27B 级 NVFP4（block-scale，每 64-K 记录 36 B）；起草器 = DFlash2 BF16；
`-ngl 99 -fa on --cache-type-k/v f16 --parallel 1`；env `GGML_CUDA_GRAPH_OPT=1 GGML_MMVQ_MAX=2 GGML_NVFP4_WIDE_MAX=0`；
大页池 23552 页（46 GiB）+ carveout 40 GiB；`temperature=0`，每 trial 生成 256 token。

## 口径（先看这条）

```
ms_per_step = 1000 × mean_len / tps        # mean_len = 1 + 接受 token 数
```
t/s 是**接受率依赖量**：同一份件同一 trial 也会在 19.33–22.18 t/s 之间浮动。
**判据只看 ms/步**，跨臂比 t/s 等于自欺。

## 全点表（冷/热启动）

| 上下文 | 模式 | t/s | mean_len | 接受率 | **ms/步** | 起草/步 | 接受/步 |
|---|---|---|---|---|---|---|---|
| 8K | warm | 25.758 | 6.41 | 0.622 | **248.8** | — | — |
| **128K** | **cold** | **22.178** | 5.2 | 0.604 | **234.47** | 6.93 | 4.18 |
| 128K | warm | 22.192 | 5.2 | 0.604 | 234.32 | 6.93 | 4.18 |
| **200K** | **cold** | **19.698** | 5.0 | 0.573 | **253.83** | 6.95 | 3.98 |

原始账本行（便于逐位复现）：

```
SEG trial=128k mode=cold tps=22.178065 mean_len=5.2 acc_rate=0.60411 ms_per_step=234.466 draft_n=341 draft_acc=206 prompt_n=127964
SEG trial=200k mode=cold tps=19.698450 mean_len=5.0 acc_rate=0.57303 ms_per_step=253.827 draft_n=356 draft_acc=204 prompt_n=199980
```

## prefill（长上下文预填，很多人会低估这段）

| 上下文 | prompt tokens | 耗时 | 速率 |
|---|---|---|---|
| 128K | 127,964 | 729.98 s | **175.3 t/s** |
| 200K | 199,980 | 1366.2 s | **146.4 t/s** |
| 200K（另一独立套件，197,850 tok） | 197,850 | 1461.5 s | 135.4 t/s |

⇒ 200K 首屏约 **23 分钟**。要更快就用 prompt cache / 更短的初始上下文，别指望 decode 侧的优化救首屏。

## 目标与差距（诚实）

- 200K 目标 30 t/s ⇔ 步耗 ≤150.3 ms；本件 253.8 ms，**差约 54 ms（≈21%）**。
- 同板另一套路线（tcgen05 内核，非本包）参照：8K 41.84 / 128K 27.39 / 200K 24.09 t/s。
- **仪器臂不算战绩**：用 `--spec-synth-len` 把接受率人为拉满得到的 128K 34.80 / 200K 29.32 t/s 是上界探针，不是可达性能。

## 口径偏移的坑（我们踩过）

同一份件、同一 ctx，只要 `mean_len` 变了，t/s 就会变；把两次不同 `mean_len` 的 trial 拿来比 t/s，
能"证明"出 15% 的假收益。任何优化结论都必须落回 **ms/步**。
