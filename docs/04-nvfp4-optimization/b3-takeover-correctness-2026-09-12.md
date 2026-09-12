# B3 接手与正确性定位

作者：AI助手

## 已完成

- 已读 HANDOFF-2026-09-12-B3.md、A1 §9.16、R5 §6 和源码树 AGENTS.md。
- SSH 可用；交接 PID 19344 的基线服务仍运行，未停止。实际参数为 q8_0 KV、无 MTP，不是 128K 最优 f16 KV + MTP 配置。
- 基线短请求复现原记录的 Daniel 输出。
- `ggml_cuda_mul_mat_vec_q` 要求 F32 src1，并调用 `quantize_row_q8_1_cuda`。standalone F32 输入不是另一条错误路径。
- 定位：B3 preload 两处条件只检查 NVFP4 和 rows_per_cuda_block > 1；M=2..8 原本也具有两行/block，因此错误进入 preload 分支。预读指针没有 j*stride_col_y，所有输出列都使用第 0 列输入。
- 板上已用旧库链接的 b3-columns-buggy 复现：M=1 PASS；M=2/3/4/8 的第 0 列 PASS，所有后续列 FAIL。例：M=2 第二列应 199.988，实际 99.9939。
- 修复只在上述两处添加 `ncols_dst == 1`，保留 M=1 的 R=8 优化。
- 原 mmvq.cu 保存在本目录 mmvq-before-ai-assistant.cu；未回滚其他助手改动。

## 验证中

- 交叉构建会话已启动，尚未取得修复后产物及测试结果。
- b3-columns.cpp 已扩展为有效有限权重、正负量化值、不同列输入和 CPU 反量化点积参考。输入块最大值构造为 127 的倍数，使 Q8_1 scale 精确可表示。
- 之后需要：修复后多列/CPU 检查、全模型基线输出对照、MTP 正确性，然后才是长上下文性能测试。

## 22:38 更新

- 用 x86 CUDA 12.8.93 nvcc + aarch64 g++14 + sbsa headers 直接交叉编译 MMVQ 成功，无需 qemu 运行 cicc/ptxas。`native-mmvq.mk` 保留原构建定义和 flags，产物单独存储。
- 修复后板上数值测试：column_failures=0、cpu_failures=0；修复前分别为 13 和 312。
- 修复后 server SHA256：9c9072803a092f18eab669ab025806ff3ca98123743a5bcfd54f135376160cc7，主机/板上相同。
- 基线 PID 19344 已按 exe 路径验证后 TERM 正常停止，大页全部释放；没有使用宽泛 pkill。
- 新 server 同 q8_0 KV、无 MTP 配置，三个固定请求的完整输出与基线逐字相同。生成速度约 10.17 vs 10.00 tok/s，仅是短请求观察，不能用作长上下文收益结论。
- 正在切换 f16 KV + MTP K7，继续验证。

## 配对短上下文验证

- f16 KV + MTP K7，固定 trial1，输入 1957 tokens，输出 384，cache_n=0。
- 基线：25.7208 tok/s，decode 14890.665 ms。
- 修复后：27.8837 tok/s，decode 13735.631 ms，约 +8.41%。
- 两者输出全文相同，draft_n=356、draft_n_accepted=316、mean len=6.27 均相同。
- 这是单次配对结果，尚不能外推为 128K 性能。128K 配对测试已启动，实际输入 123925 tokens。
- 原 qemu CMake 构建也已成功结束；实际本轮并未花满交接预估的 37 分钟。原 build-f8shim 产物已包含两行修复，独立 native 产物仍单独保留。

## 判断边界

已证明的是多列串用输入这一确定性错误，不是已证明所有数值问题都已修复。旧 standalone 用随机字节填充 F32 输入，可能包含 NaN/Inf；原速度数字需要使用有效输入复验。

本轮没有采用交接中 `pkill -9 -f llama-server` 的宽泛停止方式，也没有对外提交或发布。

## 2026-09-13 交接续跑（128K 配对完成）

- 基线 128K trial1B 已于 9-12 23:52 在板上完成（上一轮遗留），本轮拉回 evidence/baseline-128k-trial1B.jsonl。
- 基线 server（PID 21265，exe 路径已验证）TERM 停止，2 秒退出，大页 23552/23552 全部释放。
- 修复版 server（SHA256 9c907280… 与上文记录一致）同 f16 KV + MTP K7 配置启动，log 为板上 fixed-f16-mtp-128k.log，跑同 prompt（123925 tokens）配对。
- 结果（单次配对）：prompt 174.64 vs 174.34 t/s（+0.17%，持平）；decode 17.78 vs 16.88 t/s（+5.37%）；wall 731.1 vs 733.7 s。输出全文逐字相同。draft_n 378 vs 382、acceptance 0.762 vs 0.754 略有差异，但最终输出一致。
- 对比 2K 配对 decode +8.41%，128K 收益收窄至 +5.37%，符合 MMVQ 在 decode 占比随上下文变长下降的预期。
- 证据已同步本地 evidence/：baseline-128k-trial1B.jsonl、fixed-128k-trial1.jsonl。
- 修复版 server 仍在板上运行（PID 23843/23845，端口 8080），未停止。

## 待办 / 遗留

- 板上存在无文档的 perj 变体（llama-server-ai-assistant-perj、b3-correctness-perj、b3-correctness2-perj，9-12 23:43-23:44 构建，hash 与 b3fixed 不同），疑似另一会话的替代修复（可能保留 preload 分支改 per-column 指针），未经数值验证，使用前需先跑 b3-correctness 对照。
- 目前 128K 与 2K 均只有单次配对，如需置信度可补 trial2/trial3（基线需重启 server 互换）。

## 2026-09-13 perj 定案（跨会话续完）

perj = 9-12 23:43 构建的 per-column-j preload 修复（preload 移入 j 循环内，指针加 j*stride_col_y），是 B3 的根因修复，verify 路径（M=2..8）也获得 y 复用。当时会话中断未验证，本轮补完：

- 板上 hash 与本地 23:43 产物逐一对应（llama-server-ai-assistant-perj=3dafb50…，b3-correctness-perj=1cb7ea8…，b3-correctness2-perj=2c1ee40…）。注意命名：板上 b3fixed(9c90728…)=gated 修复；本地同名文件 23:43 后被 perj 覆盖。
- 正确性：b3-correctness-perj 全列 PASS（column_failures=0，batched==single 逐位一致）；b3-correctness2-perj cpu_failures=0（k=128/5120 × M=1/2/3/4/8）。
- 2K 配对（trial1 确定性 prompt）：decode baseline 25.72 → gated 27.88 → **perj 28.38 t/s**（对基线 +10.4%）；三者输出逐字相同，draft 316/356 全同。
- 128K 配对：decode 16.88 → 17.78 → **18.19 t/s**（对基线 +7.8%）；prompt 174.3-174.6 三方持平；输出逐字相同。
- perj 严格优于 gated 且数学上是根因修复 → **perj 定为新生产 binary**。
- 踩坑：llama-server 有两个进程，TERM 一个不够，须 ps 确认全灭再启动新 server（perj 首次启动撞 8080 端口失败即因此）。
- 踩坑：bench 早期 prefill 瞬时速率（~235 t/s）是 KV 尚小的暖区读数，不是真实提速；对比须用同进度点。

## 当前战线（目标 30 t/s）

| 口径 | 基线 | perj | 距 30 |
|---|---|---|---|
| 2K decode | 25.72 | 28.38 | +5.7% |
| 128K decode | 16.88 | 18.19 | +65% |

后续杠杆（按成本）：MMVQ_MAX=8 让 verify 走 perj MMVQ（零编译）→ K 值重扫（零编译）→ LUT 17% 算术化解码（kernel 工作）。

## 2026-09-13 MTP 参数重扫（perj 构建，2K 确定性 trial1）

perj kernel 改变了 verify/draft 成本结构，旧栈"K7 最优"结论不再适用。零编译扫描结果（decode t/s）：

| 配置 | decode | draft acc | 判定 |
|---|---|---|---|
| K7 p0.6（旧最优） | 28.38 | 316/356 | 基线 |
| K12 p0.6 | 30.41 | 328/451 | ✅ 突破 30 |
| K12 p0.5 | **31.02** | 332/478 | ✅ 当前最优 |
| K12 p0.4 | 30.36 | 334/529 | 过深略亏 |
| K16 p0.5 | 27.21 | 335/570 | ❌ 过深反噬 |

- MMVQ_MAX=8（verify 全走 perj MMVQ）：28.32 vs MMVQ_MAX=2 的 28.38 → 打平，维持 =2（与 R34 B0 结论一致）。
- K12/p05/p04 输出与 K7 参考非逐字同（verify batch 形状变化导致 near-tie argmax 翻转，首 diff 处为良性换述），文本连贯、任务行为一致；K7 族三配置（baseline/gated/perj）之间逐字相同。
- 铁律提醒：pkill -f 会连坐 ssh 会话本身（命令行含匹配串），杀 server 必须用板上脚本文件承载；pgrep 模式要对上 "./llama-server-ai-assistant-perj" 这种相对路径 cmdline。
- 128K 确认（K12 p0.5）进行中。

## 2026-09-13 最终状态（30 t/s 里程碑达成 @2K）

128K 确认完成：K12 p0.5 decode **19.61 t/s**（vs K7 perj 18.19 = +7.8%；vs 原基线 16.88 = +16.2%），mean len 5.94，prompt 174.38 持平，输出连贯。

| 配置 | 2K decode | 128K decode |
|---|---|---|
| 基线（b3 修复前）K7 p0.6 | 25.72 | 16.88 |
| gated K7 p0.6 | 27.88 | 17.78 |
| perj K7 p0.6 | 28.38 | 18.19 |
| **perj K12 p0.5（生产候选）** | **31.02** | **19.61** |
| 对基线累计 | **+20.6%** | **+16.2%** |

生产候选 = 板上 llama-server-ai-assistant-perj（SHA256 3dafb50…）+ GGML_CUDA_GRAPH_OPT=1 GGML_MMVQ_MAX=2 + f16 KV + MTP K12 p-min 0.5。证据全部在 evidence/（perj-k12p05-128k-trial1.jsonl 等）。
板上当前运行：perj K12 p0.5（log perj-k12-p05b.log，端口 8080）。

## 距离 128K 30 t/s 的差距（诚实评估）

- 128K 每 token 51ms；理论下限（权重 72ms + KV 60ms per step，len 5.94 摊薄）≈ 22ms/token ≈ 45 t/s。缺口 2.3× 在 kernel 效率与 MTP 执行开销。
- 剩余已识别杠杆：NVFP4 LUT 算术化解码（standalone 测过 LUT 成本 17%）；draft 链 12×~4-5ms；小算子 ~17%。均需 kernel/结构工作，单次收益预期个位数 %。
- 物理边界：128K f16 KV 16.5GB 每 verify step 全读（FLASH_ATTN 已 100% 带宽利用）；要 30+ 必须减少 KV 字节（MLA/更激进 KV 量化，q8_0 已证伪更慢）。

## ⚠️ 目标校准（2026-09-13 用户确认）

**正式目标：KV ≥ 128K 口径下 decode ≥ 30 t/s。** 2K 的 31.02 不计入达标。
参照锚点：网友同模型族已在 KV 200K 实现 31 t/s（另一 128K 口径报告 34.91 t/s），物理上可行。
当前位置：128K 19.61 t/s（perj + K12 p0.5）。差距 1.53×。

## 给下一会话的路线图（按优先级）

0. **先重测账本**：perj 落地后 R30/R32 的 128K step 拆账已过期。用 r30/r32 脚本在 perj binary 上重跑滚动 op_prof，确认 y 复用后 NVFP4 MLP 从 ~131ms/step 降到多少、新的时间大头是什么。不要凭旧账本选杠杆。
1. **NVFP4 LUT 算术化解码**：microbench3 实测 LUT（byte_perm 查表）成本 17%（221 vs 258 GB/s）。研究 E2M1→int8 的 LOP3/算术 SWAR 解码替代 get_int_from_table_16。standalone 先行（mb3/mb5 框架 1 分钟级迭代），收益为正再进全模型。门禁：b3-correctness 两个全 PASS + 配对逐字一致。
2. **draft 链开销**：每 step 12 次串行 draft forward，各含一次 lm_head（635MB 读取，y 复用后 ~2.7ms）+ draft 层权重。12×(4~5ms) ≈ 50-60ms/step，占 128K step（~300ms）约 1/5。候选：draft 共用/跳过冗余计算、draft lm_head 词汇裁剪（结构上相当于 top-k 投机的合法化，需评估对 acc 影响）。
3. **小算子 ~17%**：GDN 门控 BF16 N48（29280 次调用、launch 主导）等，融合或批量化。
4. **acceptance 上限**：128K 口径 acc 60-75%（mean len 5.94）由模型 draft 头质量决定，K/p-min 已扫完（K12 p0.5 最优）。再往上需要更好的 draft 头，非 kernel 工作。
5. **物理边界**：128K f16 KV = 16.5GB，每 verify step 全读 60ms（FLASH_ATTN 已 100% 带宽）。KV 减字节（MLA/架构改动）是最后手段，工程量大。
6. 每次 A/B 必须遵守：同 trial 确定性 prompt、同进度点对比 prefill、输出口径一致性检查、单 trial 不排名（重要结论 3 连测）。

## 切换前状态快照（2026-09-13 01:5x）

- 板上运行中：llama-server-ai-assistant-perj + K12 p0.5（log perj-k12-p05b.log，端口 8080，128K 配对已完成，服务空闲）
- 全部证据已同步本地 work/thor-ai-assistant/evidence/
- 文档已落盘：docs/13-阶段成果总结-2026-09-13.md（思路+操作手册）、docs/14-模型文件台账.md
- 源码工作区：~/work/thor-driveos/xbuild/llama-cpp-latest（git status 有 16 文件改动，mmvq.cu 当前 = perj 版，原版备份在 work/thor-ai-assistant/mmvq-before-ai-assistant.cu）
