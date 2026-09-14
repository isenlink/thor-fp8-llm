# Thor qwen3.8-27B 128K+ decode 重新分析（OpenAI rerun）

目标：KV >= 128K 时 decode >= 30 t/s。当前生产候选仍为
`llama-server-ai-assistant-perj` + `RadixArk-F8attn-v2.gguf` + f16 KV + MTP K12 p-min 0.5。

## 现状

| 配置 | 2K decode | 128K decode | 备注 |
|---|---:|---:|---|
| baseline K7 p0.6 | 25.72 | 16.88 | perj 前 |
| perj K7 p0.6 | 28.38 | 18.19 | NVFP4 y 复用根修 |
| perj K12 p0.5 | 31.02 | 19.61 | 当前生产候选 |
| gatepre K12 p0.5 | 31.05 | 19.73 | fused gate 也复用 y；收益约噪声级 |

128K 距 30 t/s 仍差 1.52x；短上下文 30+ 不计入目标。

## 本轮新证据

### R30/R32 128K op profile

已把旧 R30/R32 脚本更新为当前生产候选参数：perj、K12/p0.5、f16 KV，并去掉宽泛 `pkill -f` 风险。

R30 graph-on 结果：

- prompt 123924 tok，prefill 172.2 t/s
- decode 14.92 t/s（`GGML_OP_PROF=1` 有明显开销，不作性能排名）
- draft acceptance 301/625，mean len 4.96
- decode 尾段账本：FLASH_ATTN_EXT 27.6%；NVFP4 MMVQ 合计约 41.6%；F8 GEMM 约 14.2%；小算子分散

R32 eager 结果：

- prompt 123922 tok，prefill 172.5 t/s
- decode 14.97 t/s（profile/eager 口径）
- draft acceptance 301/609，mean len 5.12
- decode 尾段账本：FLASH_ATTN_EXT 31.5%；NVFP4 MMVQ 合计约 40.7%；F8 GEMM 约 11.5%

环境：R30/R32 GPU 利用率均值约 94-96%，最高 GPU/Tj 温度约 72C，没有热墙或明显 CPU 空洞证据。

### NVFP4 LUT 算术化解码

新增 `scripts/b3-microbench6.cu`，比较现有 LUT 解码与固定 E2M1 算术/SWAR 解码。

结果（Thor，300 iters）：

- correctness mismatches=0
- LUT: 224.0 GB/s
- arithmetic: 202.0 GB/s
- load-only: 253.0 GB/s

结论：这个算术化实现正确但慢约 10%，关闭为主线候选；不要直接移入生产 kernel。

### gatepre fused gate y 复用

发现 perj 只让主 `tmp` 路径使用 `vec_dot_nvfp4_q8_1_preload`，融合 FFN 的 `tmp_gate` 仍走普通 vecdot。实现 gatepre：当 `type == GGML_TYPE_NVFP4 && rows_per_cuda_block > 1` 时，gate 分支也使用同一份 `y_pre/y_ds`。

验证：

- `b3-correctness-gatepre`: column_failures=0
- `b3-correctness2-gatepre`: cpu_failures=0
- 2K `trial1`: perj 31.0175，gatepre 31.0469，输出逐字一致，draft 332/478 一致
- 128K `trial1`: perj 19.6149，gatepre 19.7273，输出逐字一致，draft 316/522 一致

结论：gatepre 安全但收益约 +0.6%，不足以改变路线；可保留为小补丁，不应作为 30 t/s 主线。

## 重新判断

1. 参数扫（K、p-min、MMVQ_MAX）已经接近局部最优；同一配置不同 prompt 会造成很大速度差，后续必须固定 trial 才能排名。
2. 128K decode 的两个最大块是 NVFP4 MMVQ 与 FlashAttention/KV。前者仍有软件优化空间，后者接近物理带宽边界。
3. 单点 kernel 小优化的空间正在变小：算术解码失败，gatepre 仅噪声级；继续只抠 NVFP4 vecdot 很难给 1.5x。
4. 真正接近 30 t/s 需要结构性减少工作量：减少 draft 链重复 forward/lm_head，或减少长 KV attention 字节。q8_0 KV 已证伪，所以 KV 路线不能简单换 q8_0。

## 下一步优先级

P0：做 draft 链/lm_head 结构拆解。R30/R32 显示 `MUL_MAT:NVFP4:HUGE-N` 虽调用少但单次约 7 ms，MTP 深链中 lm_head/采样/top-k 对 step 成本敏感。下一步应量化每个 accepted token 对应的 draft forward 次数、lm_head 次数和 verify batch 形状。

P1：保留 gatepre，但只在后续其他改动一起打包验证；单独收益不足以长期占用实验时间。

P1：针对 FlashAttention 做“不可优化边界”量化：记录 64K/96K/128K 同配置 decode t/s 与 FLASH_ATTN_EXT avg_us，拟合 KV 长度斜率。如果线性项占比过高，30 t/s 只能靠减少 KV 读取字节或提升 acceptance 摊薄。

P2：小算子融合只作为补充。R32 里 CPY/CONCAT/MUL/RMS/ROPE/SET_ROWS 等总和不小，但分散且每项微秒级，单独收益很难超过几个百分点。

## 当前板端状态

- 已恢复 `llama-server-ai-assistant-perj`，K12/p0.5/f16 KV，端口 8080。
- 新实验产物在 `/brand_data/ai_workspace/ai-assistant/`：`llama-server-ai-assistant-gatepre`、`b3-correctness-gatepre`、`b3-correctness2-gatepre`。
- 本地证据目录新增：
  - `evidence/r30-perj-k12p05/`
  - `evidence/r32-perj-k12p05/`
  - `evidence/microbench/b3-microbench6-nvfp4-arith.log`
  - `evidence/gatepre/`

---

[整理者注] 本文档由工作笔记脱敏改写：板载数据分区路径改 `/brand_data/` 代称（DriveOS 板上该分区约 105G、板载 vblkdev、与只读根分区独立，原路径名含车辆品牌字样，为保持品牌中立统一写作 `/brand_data/`，读者在自己板卡上执行 `ls /` 即可看到真实分区名）。技术数据（性能数字、hash、参数、命令）100% 保留。