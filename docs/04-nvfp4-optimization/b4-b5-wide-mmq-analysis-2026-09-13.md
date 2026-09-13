# B4：NVFP4 wide-M verify kernel（2026-09-13 下午）

作者：AI助手（x86主机） · 前置：ANALYSIS-2026-09-13-openai-rerun.md（P0 draft 链拆解）

## P0 结论：draft 链细分计时（LLAMA_SPEC_TIMING 插桩，spectime binary）

`common/speculative.cpp` 加了 `LLAMA_SPEC_TIMING=1` 插桩（decode 入队 / GPU sync 等待 /
CPU 采样 / embedding 取回 / 候选 五段拆分）。binary = perj mmvq + 插桩，
`native-spectime.mk` 构建（首次构建踩坑：必须复用 build-f8shim 的 .a，否则缺自定义
type 43 F8 注册，模型加载报 "invalid ggml type 43"）。

128K 口径（trial1，K12 p0.5，19.64 t/s 与 perj 一致 → 插桩开销可忽略）：

| 段 | 每 step | 占比 |
|---|---|---|
| draft 链合计 | 83.95 ms | ~28%（step ≈ 302ms） |
| 其中 GPU wait（draft 前向真实耗时） | 75.92 ms | 8.73 ms/次 × 8.7 次 |
| 其中 decode 入队 | 7.95 ms | |
| 其中 CPU 采样/embd/候选 | <0.1 ms | 噪声级，排除 |

2K 口径：draft 链 66ms，wait 62.25ms（7.2ms/次）。**CPU 采样全程可忽略，
此前"采样耗时"其实是异步执行的 GPU 前向在 sync 处归账。**

## draft 前向成本模型（b3-microbench7 实测，真实 ggml 路径，板上）

draft 层（blk.64）全部 BF16：attn_q 120MB + attn_o 60MB + ffn 3×170MB + eh_proj 100MB
≈ 840MB；lm_head 共享 output.weight NVFP4 5120×248320 = 682MB。

| op (M=1) | 实测 | 带宽 |
|---|---|---|
| lm_head NVFP4 5120×248320 | 3.04 ms | 235.6 GB/s（接近满载，perj kernel 已优化到位） |
| draft BF16 层合计 | ~3.0 ms | 200-237 GB/s（BF16 GEMV 已高效） |
| 合计+同步开销 | ~7.2 ms @2K | 与 spectime wait 吻合 |
| @128K | 8.73 ms | +1.4ms ≈ draft 层 KV 读（128K f16） |

→ draft 侧已接近带宽地板；剩余杠杆是结构性：draft 层 BF16→F8/NVFP4（省 ~2ms/次）
或 draft lm_head 词汇裁剪（682MB→~92MB @32k 词）。单项预期 5-8%。

## 关键发现：verify 侧 NVFP4 M>2 掉到 ~90 GB/s

| NVFP4 形状 | M=1 (MMVQ perj) | M=8 (MMVQ) | M=13 (MMQ dp4a) |
|---|---|---|---|
| ffn 5120×17408 | 208.6 GB/s | 82.0 | 90.4 GB/s |
| ffn 17408×5120 | 227.0 | 83.1 | 80.6 |
| lm_head 5120×248320 | 235.6 | 86.3 | 96.9 |

对照 F8（cuBLASLt）：M=1 与 M=13 均 ~227-232 GB/s —— 证明 M=13 带宽可达。

MMVQ 多列路径每列重读权重+重做 LUT 解码 → M=8 衰减到 82 GB/s；
MMQ（M=9..13，verify 实际路径）也仅 ~90。R30 账本 NVFP4 占 decode ~41.6%，
若 M=13 提到 ~220 GB/s，理论可省 ~60-70ms/step（302ms 口径）→ +25% 量级。

## B4 设计（mul_mat_vec_q_nvfp4_wide）

- 触发：NVFP4 && M∈[3,16] && 无 ids/fusion（env GGML_NVFP4_WIDE_MAX 可调，默认 16）
- 结构：每线程权重块读一次 → LUT 解码一次 → 寄存器复用 × M 列 dp4a；
  y（q8_1）按列从 L1/L2 重读（y 总量小）
- R=2 行/block、4 warps；M=3..16 模板实例化
- 估算 issue 预算：M=13 @220GB/s 约占发射能力 ~50%，可行

## 事故记录：int4 对齐崩溃拖死 GPU

- 首版 y 加载用 `*(const int4*)bq8->qs` —— block_q8_1 的 qs 偏移 4B（half2 ds），
  int4 需 16B 对齐 → GR fault → nvgpu 通道卡死（recovery blocked），
  已知-good binary 也跑不动 → 只能 reboot 板（~11:35）。
- 修复：y 改用 8×4B 标量 int 加载（与 perj preload 相同方式）。
- 教训（增补运维纪律）：**新 kernel 上板前先用静态检查审对齐；
  block_q8_1/block_nvfp4 的 qs 都不是 16B 对齐，禁用 int4 加载**。

## 2026-09-13 下午定论：B4 wide（dp4a 路线）证伪，转向 MMQ 配置调优

板上微基准（GGML_NVFP4_WIDE_MAX=0，b3-microbench7 扩展 M=16/24/32）：

| NVFP4 5120×17408 | M=1 | M=8 | M=13 | M=16 | M=24 | M=32 |
|---|---|---|---|---|---|---|
| GB/s | 208.7 | 82.2 | 90.2 | 90.3 | 62.9 | 57.9 |

关键证据链：
1. **M=13 与 M=16 相同（90 GB/s）** → 排除 tile 碎片化（16 已是整 tile）。
2. **Q8_0(90MB)/Q4_0(47MB)/NVFP4(47MB) 在 M=13 耗时同为 ~0.55ms** → MMQ
   在 Thor 上与权重字节数无关，纯延迟/发射受限。
3. dp4a SIMT 路线发射预算紧张：每 32B 权重块（64 值）每列需 16 条 dp4a，M=13
   → 208 条/32B；要到 230 GB/s 需 ~1500 G dp4a/s。wide kernel 实测最高 97 GB/s
   （≈630 G dp4a/s），距离达标还差 ~2.4×，调寄存器/流水补不上这个量级。
   → **B4 wide kernel 路线放弃**（正确性没问题，发射预算量级不够）。
4. MMQ 全部配置（含 Blackwell NVFP4 的 FP4 MMA 原生路径）都是 256 线程 +
   occupancy=1 → 每 SM 仅 8 warps，全局加载为同步 4B 拷贝（无 cp.async），
   延迟无法隐藏。

新方向（B5）：MMQ 配置实验 —— NVFP4 Blackwell CASE nthreads 256→512
（mmq-config-blackwell.cuh，一个常量），编译产物
mmq-instance-nvfp4-512t.cu.o + libggml-cuda-512t.a + b3-microbench7-512t。
验证标准：5120×17408 M=13 从 90 GB/s 显著提升（目标 >150）。

## 2026-09-13 傍晚：架构事实核查（关键修正）

1. **Thor = sm_101a，llama.cpp 的 BLACKWELL_MMA_AVAILABLE 要求 CC>=1200，
   Thor 不走原生 FP4 MMA**：NVFP4 MMQ 实际路径 = Ampere 配置
   （256 线程/occupancy=1/I=128/LAYOUT_NVFP4/K_vram=256）+ NVFP4 解码 q8
   + int8 MMA（vec_dot_q8_0_16_q8_1_mma）。之前 patch blackwell 配置表做的
   两个实验 binary（512t/o2）对 Thor 是空操作，实测逐位相同反而证明测量稳定。
2. **ptxas 探针（本地编译，未上板）**：
   - mma.sync kind::mxf4nvf4 block_scale → sm_101a 不支持（ptxas 拒绝）
   - mma.sync m16n8k64 s8 → 不支持
   - Thor 张量核菜单：fp16/bf16 m16n8k16、int8/fp8 m16n8k32，无 FP4 捷径。
3. **ccprobe 上板实测：14 SMs、CC 10.1、显存 47GB（统一内存）、L2 24MB**。
   - dp4a 按 32/SM/clk（1/4 率）×14×~1.5GHz ≈ 670G/s → M=13 上限 ~103 GB/s，
     wide kernel 实测 97 → dp4a 路线确实到顶，放弃无误。
   - int8 MMA：M=13 verify ffn 实测 0.555ms ↔ 2.57 T MAC/s ≈ 理论
     （14×256×1.5G≈5.4T）的 48%；cuBLASLt FP8 对照 55%。**verify 在 Thor 上
     是算力-bound（int8 MMA），不只是带宽问题**（带宽地板 0.2ms vs 0.55ms）。
4. 新假设：MMQ 每个 K-tile 同步加载（无 cp.async 双缓冲），occupancy=1 时
   加载延迟无别的 block 可掩盖 → 48% 效率的来源。
   实验 B5-ao2：Ampere NVFP4 配置 occupancy 1→2（这次改的是正确文件）。

## 2026-09-13 傍晚 2：K_vram 路线事故与定论

1. **K_vram 256→512/1024 的"3.4× 提速"是错误计算假象**：NVFP4 sram 布局
   硬编码 256（sram_stride 84 ints/行、x_df 偏移 64、load_tiles 线程映射、
   主循环 y 轮数 2 轮均写死），增大 K_vram 后 kernel 少算 3/4 的 y 轮次 →
   快但全错（b3-correctness2 M>=9 全挂，1200 处失败）。
2. 完整修复（5 处：K 感知 sram_stride host+device、load scale 偏移、
   threads_per_row、vec_dot x_df 偏移、主循环 y 轮数泛化）后门禁全过
   （22 组合 cpu_failures=0），**但修正后 K512 实测 84.7 GB/s < K256 的
   90.2 —— K_vram 路线无真实收益，放弃**。
3. **i64o2 配置（I=64 + occ=2）把 GPU 跑挂**：kernel hang，进程 D 态卡
   channel_free，只能物理断电。
   **更正（2026-09-13 晚，串口守护实据）**：error notifier 实际存在——
   14:56:59 `error notifier set to 13, ch 398 owned by b3-correctness2`，
   15:00:51 任务阻塞 >120s，与 #1/#2 签名完全一致。此前"无 notifier"
   记录有误（当时仅从板上进程态推断，未见串口）。
   该配置已拉黑（*.BADHANG）。教训：config 常量也有 hang 风险，
   一次只改一个、先过门禁再测性能。
4. 结论：verify NVFP4 ~90 GB/s 是 int8-MMA 路径在当前 MMQ 结构下的实际上限，
   配置空间（线程数/occupancy/I/K_vram）均已证伪。剩余理论空间
   （48%→90% MMA 效率）需要 cp.async 流水线级别的重写，风险/收益不划算。
   **B4/B5 整体收尾：verify 侧 kernel 优化到此为止，转向 draft 侧结构性杠杆
   （lm_head 词汇裁剪 ~22ms/step、draft 层 BF16→F8 ~17ms/step）。**

## 原待办（已被上述定论取代，存档）

1. 板重启后：跑 b3-correctness(-fixed) + b3-correctness2（须含 M=3,4,8 → wide 路径；
   建议扩展 M 覆盖到 13）
2. b3-microbench7 对比 wide 开关（GGML_NVFP4_WIDE_MAX=0/16）M=3..13 带宽
3. 收益为正 → 全模型门禁：2K/128K 配对逐字一致 + 性能（注意 M≤8 会从 perj 切到 wide，
   输出仍应逐字一致：同 q8_1 y、同 dp4a 语义，仅归约顺序变化 → 可能有尾数差，
   判定标准 = 文本级一致 + acceptance 不塌）

---

[整理者注] 本文档由工作笔记脱敏改写：作者行主机代号已中性化（AI助手/x86主机）。技术数据 100% 保留。