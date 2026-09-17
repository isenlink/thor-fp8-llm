// t4_smem_layout.h — T4 系列内核的**动态 smem 尺寸单一来源**（内核与宿主必须包含同一个头）
//
// 事故背景（INCIDENTS §八 / results/20260916-71）：cuLaunchKernel 的 sharedMemBytes 小于内核实际需求时
//   **没有任何运行时错误**，内核会静默越界写 smem（我们由此打挂过一次 GR 引擎，只能断电）。
// 用法：
//   内核：  #include "t4_smem_layout.h"；在入口第一件事做 T4_SMEM_REQUIRED <= T4_SMEM_HOST_REQ 的校验
//   宿主：  在 launch 前 assert(requested == required)，并把 requested 作为 sharedMemBytes 传入
#pragma once

// ---- 各内核的布局参数（改这里，不要改派生值）----
#define T4_V7_A_TILE     16384   // B7/B8 的 A（激活，常驻）
#define T4_V7_B_CHUNK     4096   // 一个 K=256 块（4×K64）
#define T4_V7_CPS            4   // 每个 stage 装几个块
#define T4_V7_STAGES         6   // stage 数
#define T4_V7_MBAR_BYTES   128   // full[] + empty[] + 收尾 mbar + taddr 槽

#define T4_V7_B_STAGE   (T4_V7_B_CHUNK * T4_V7_CPS)
#define T4_SMEM_REQUIRED (T4_V7_A_TILE + T4_V7_STAGES * T4_V7_B_STAGE + T4_V7_MBAR_BYTES)

// 宿主申请值：向上取整到 1 KB（并保证 >= REQUIRED）
#define T4_SMEM_HOST_REQ (((T4_SMEM_REQUIRED + 1023) / 1024) * 1024)

#if defined(__cplusplus)
static_assert(T4_SMEM_HOST_REQ >= T4_SMEM_REQUIRED, "host smem request must cover kernel requirement");
#endif

// ---- B9（cp.async warp specialization：8 个生产者 warp + 1 个消费者 warp）----
//  形状/环深与 B7 逐位相同（B7  = 16 KB A + 12 × 4 KB stage），只改「谁来搬 / 谁发 MMA」
#define T4_V9_A_TILE     16384   // 常驻 A（4 个 K=64 子块 × 4096 B）
#define T4_V9_B_STAGE     4096   // 一个 stage = 一个 K=256 块（4×K64）
#define T4_V9_STAGES        12   // 环深（= 12 × 4 KB = 48 KB）
#define T4_V9_NPROD        256   // 生产者线程数（= 8 warp × 32）
#define T4_V9_NTHR       (T4_V9_NPROD + 32)   // 288 = 9 warp（含 1 个消费者 warp）
#define T4_V9_MBAR_BYTES (8 * T4_V9_STAGES * 2 + 8 + 8)   // full[] + empty[] + 收尾 mbar + taddr 槽
#define T4_V9_SMEM_REQUIRED (T4_V9_A_TILE + T4_V9_STAGES * T4_V9_B_STAGE + T4_V9_MBAR_BYTES)
#define T4_V9_SMEM_HOST_REQ (((T4_V9_SMEM_REQUIRED + 1023) / 1024) * 1024)

#if defined(__cplusplus)
static_assert(T4_V9_SMEM_HOST_REQ >= T4_V9_SMEM_REQUIRED, "host smem request must cover B9 kernel requirement");
#endif

// ---- B9f（cp.async warp specialization，**放大握手单元**）----
//   动机（results/20260916-75）：B9c 的逐 4 KB 跨 warp mbarrier 握手实测 ~560 ns/块（B7 同板对照 181.7 GB/s，
//   B9c 仅 72 GB/s）⇒ 把一次握手的搬运单元由 4 KB 放大到 **32 KB（CPS=8 块）× 3 stage**，
//   握手次数按字节降到 1/8；环深 3 stage = 24 块，尾流按 F12 公式补 (STAGES-2) 轮。
#define T4_V9F_A_TILE    16384   // 常驻 A
#define T4_V9F_B_CHUNK    4096   // 一个 K=256 块
#define T4_V9F_CPS          8    // 每 stage 装 8 块 = 32 KB
#define T4_V9F_STAGES       3    // 环深（3 × 32 KB = 96 KB）
#define T4_V9F_NPROD      256
#define T4_V9F_NTHR     (T4_V9F_NPROD + 32)
#define T4_V9F_MBAR_BYTES (8 * T4_V9F_STAGES * 2 + 8 + 8)
#define T4_V9F_SMEM_REQUIRED (T4_V9F_A_TILE + T4_V9F_STAGES * T4_V9F_CPS * T4_V9F_B_CHUNK + T4_V9F_MBAR_BYTES)
#define T4_V9F_SMEM_HOST_REQ (((T4_V9F_SMEM_REQUIRED + 1023) / 1024) * 1024)

#if defined(__cplusplus)
static_assert(T4_V9F_SMEM_HOST_REQ >= T4_V9F_SMEM_REQUIRED, "host smem request must cover B9f kernel requirement");
#endif

// ---- B10/真实流（T4-4 原型：真实 NVFP4 字节 + 真实缩放 + 跨 chunk K 累加）----
//   chunk = 4 × 1024 B（K=64 canonical 板）+ 4 × 128 B（缩放字）= 4608 B
#define T4_V10R_A_TILE    16384
#define T4_V10R_B_CHUNK    4608   // qs 4096 + sf 512
#ifndef T4_V10R_CPS
#define T4_V10R_CPS           8   // 每 stage 8 块 = 36 864 B（20260916-79 扫臂定版：232.4 GB/s vs CPS=5 的 211.0）
#endif                            // 编译期可用 -DT4_V10R_CPS=N 单变量扫臂（20260916-78/79）
                                  // TMEM 预算（SFB 基址 48、SF_COLS_STEP=4）：48 + STAGES*CPS*16 ≤ 512 ⇒ CPS≤9 @S=3
#ifndef T4_V10R_STAGES
#define T4_V10R_STAGES        3   // 环深；编译期可用 -DT4_V10R_STAGES=N 单变量扫臂（20260916-79）
#endif
#define T4_V10R_NPROD       256
#define T4_V10R_NTHR       (T4_V10R_NPROD + 32)
#define T4_V10R_MBAR_BYTES (8 * T4_V10R_STAGES * 2 + 8 + 8 + 32)   // 20260916-83：+16 = epi_rdy/epi_done；84：再 +16 = 双缓冲按 D 缓冲分组
#define T4_V10R_SMEM_REQUIRED (T4_V10R_A_TILE + T4_V10R_STAGES * T4_V10R_CPS * T4_V10R_B_CHUNK + T4_V10R_MBAR_BYTES)
#define T4_V10R_SMEM_HOST_REQ (((T4_V10R_SMEM_REQUIRED + 1023) / 1024) * 1024)

#if defined(__cplusplus)
static_assert(T4_V10R_SMEM_HOST_REQ >= T4_V10R_SMEM_REQUIRED, "host smem request must cover kernel requirement");
#endif

// ---- v92（T4-A 接线第 2 步：**真实 A 按 chunk 流进环**）----
//   与 v10r 的唯一结构差别：A 不再常驻（不再有 A_TILE），而是和 B 一样按 (stage, chunk) 进环。
//   每个 A chunk = M*128 B（4 个 K=64 canonical 板）+ M*16 B（4 板 × M*4 缩放字）= 18432 B @M=128。
//   ⇒ 环总字节 = STAGES*CPS*(A_CHUNK + B_CHUNK)。M=128 的真实 A 是 B chunk 的 4 倍，
//     故 CPS 必须降（CPS=4/S=3 → 276 KB 装不下；CPS=3/S=3 → 207 456 B 可行）。
//   TMEM：SFA 与 SFB 都按 stage 分区（各 STAGES*CPS*NSUB*4 列），D 仍双缓冲 64 列。
// ★ 94（M1b）：A 的 smem 驻留按 MREAL 压缩 —— 每个 board(=K64 子块) 只保留 ceil(MREAL/8) 个 8 行组。
//   描述符仍按 canonical M=128 的 8 行组 swizzle 走 ⇒ **前 MREAL 行的偏移逐字节不变**；
//   第 ≥MREAL 行读到的越界字节只进被忽略的 D 填充行（矩阵乘按行独立），不进有效行。
//   MREAL=128 时新式退化为 18432（与旧值逐字节相同 ⇒ 零回归）。
#ifndef T4_V92_MREAL
#define T4_V92_MREAL       128
#endif
#define T4_V92_A_BD_ROWS ((((T4_V92_MREAL) + 7) / 8) * 8)
#define T4_V92_A_SUB     ((T4_V92_A_BD_ROWS) * 32)              // 一个 board 的 nib 字节
#define T4_V92_A_SF_BD   ((((T4_V92_MREAL) + 3) / 4) * 16)      // 一个 board 的 sf 字节
#define T4_V92_A_CHUNK   ((4 * (T4_V92_A_SUB)) + (4 * (T4_V92_A_SF_BD)))
#define T4_V92_B_CHUNK     4608
#ifndef T4_V92_CPS
#define T4_V92_CPS           3
#endif
#ifndef T4_V92_STAGES
#define T4_V92_STAGES        3
#endif
#define T4_V92_NPROD        256
#define T4_V92_NTHR        (T4_V92_NPROD + 32)
#define T4_V92_MBAR_BYTES  (8 * T4_V92_STAGES * 2 + 8 + 8 + 32)
// ★ 94b：占位填充（诊断用）。用途：把 smem 撑到「每 SM 只放得下 1 块」，从而在**不改其它变量**的前提下
//   隔离「布局改动」与「occ=2（TMEM 双驻留）」两个嫌疑。默认 0 = 无填充。
#ifndef T4_V92_SMEM_PAD
#define T4_V92_SMEM_PAD       0
#endif
#define T4_V92_SMEM_REQUIRED \
  (T4_V92_STAGES * T4_V92_CPS * (T4_V92_A_CHUNK + T4_V92_B_CHUNK) + T4_V92_MBAR_BYTES)
// ★ 94b：填充只加在**宿主申请值**上（内核的 OFF_* 与 REQUIRED 的等式断言保持不变；多申请的 smem 内核不碰）。
#define T4_V92_SMEM_HOST_REQ (((T4_V92_SMEM_REQUIRED + T4_V92_SMEM_PAD + 1023) / 1024) * 1024)

#if defined(__cplusplus)
static_assert(T4_V92_SMEM_HOST_REQ >= T4_V92_SMEM_REQUIRED, "host smem request must cover kernel requirement");
static_assert(T4_V92_SMEM_HOST_REQ <= 232448, "v92 ring must fit 227 KB dynamic smem");
#endif
