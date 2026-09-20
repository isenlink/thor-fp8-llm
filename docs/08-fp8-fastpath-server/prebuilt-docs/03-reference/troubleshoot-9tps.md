# 「按同一套思路装完只有 9 t/s」排障清单

结论先行：**9 t/s 量级几乎一定是"起草器没生效"或"环境变量没生效"**，不是编译问题。
本件就是为省掉编译这一步：把件换成本包二进制，再按下面 1–10 逐条核对。

## 一条命令看全局

```bash
BIN=01-binary/llama-server-aarch64-sm101a-nvfp4
pid=$(pgrep -f "$(basename $BIN)" | head -1)
echo "--- 件 ---";   sha256sum $BIN
echo "--- 环境 ---"; tr '\0' '\n' < /proc/$pid/environ | grep -E 'GGML_|LLAMA_'
echo "--- 命令行 ---"; tr '\0' ' ' < /proc/$pid/cmdline; echo
echo "--- 池 ---";    grep -E 'HugePages_Total|HugePages_Free' /proc/meminfo
echo "--- 进程 ---";  pgrep -af llama-server | wc -l
```

## 逐条

| # | 检查 | 判据 | 不满足怎么办 |
|---|---|---|---|
| 1 | 件里有没有起草器支持 | `strings $BIN \| grep -c draft-dflash` > 0 | 用本包件（别的 build 没编进 PR #27342 会直接报 "wrong number of tensors"） |
| 2 | `-md` 是否给了 | `cmdline` 里有 `-md <draft.gguf>` | 补上；没有起草器 = ~12.9 t/s（8K，社区件同板 12.85）/ **6.85 t/s（本机 200K 实测）** |
| 3 | `--spec-type draft-dflash` | `cmdline` 里有 | 不要写成 `draft-mtp`（那是另一条路线，读数不同） |
| 4 | 三个 GGML 变量 | `GGML_CUDA_GRAPH_OPT=1`、`GGML_MMVQ_MAX=2`、`GGML_NVFP4_WIDE_MAX=0` | 用 `02-launcher/start-server.sh` 起，别手工拼 |
| 5 | 全层上 GPU | 日志里 `n_gpu_layers` 是 99（或 `offloaded 层数`） | `-ngl 99`；部分层落 CPU 直接掉到个位数 |
| 6 | FlashAttention / KV | 日志里 `flash_attn = 1`、`cache type = f16` | `-fa on --cache-type-k f16 --cache-type-v f16` |
| 7 | 大页池 | `HugePages_Total ≥ 21504`（42 GiB），我们跑 23552（46 GiB） | `02-launcher/prepare-host.sh`，然后**重启**（冷启动才全量） |
| 8 | carveout | 已执行平台自带的 `gpu-carveout.sh -g 40` | 见 `prepare-host.sh` |
| 9 | 起草器文件对不对 | 日志无 `wrong number of tensors` | DFlash2 = 81 张量；DFlash1 = 58；件里带 PR 才能吃 81 |
| 10 | 有没有多进程抢卡 | `pgrep -af llama-server` 只有 1 个 | 用 `02-launcher/stop-server.sh` 清（不要宽匹配 pkill） |

## 读数对不对（自测口径）

```bash
curl -s localhost:8080/completion -H 'Content-Type: application/json' \
  -d '{"prompt":"用一句话解释什么是 KV cache","n_predict":256,"temperature":0,"seed":123}'
# 然后从服务日志取这两个数：
#   eval time = ... ms / 256 tokens ( x ms per token )
#   draft acceptance rate = ...
```
- 目标：**128K 上下文 ms/步 235 ms 量级、接受率 0.55–0.62**；
- 若接受率 < 0.3：起草器不匹配（换成 DFlash2 的 Q2_K_S/BF16），或题目是**中文散文**（散文本来接受率低，用 `SPEC_N=3`）；
- 若接受率正常但步耗 > 300 ms：查 5–8 条（层数/FA/池）。

## 常见误判

- **别用 `-c 262144` 硬上**：200K 以上的 f16 KV 会把 46 GiB 池吃穿，表现是启动失败或掉 CPU，看起来"变慢"。
- **别开 `--parallel > 1`**：并发吃池，单流速率会掉。
- **别用 KV q8_0 省内存**：省 6.5 GiB 但 decode 掉约 1/3（实测）。
