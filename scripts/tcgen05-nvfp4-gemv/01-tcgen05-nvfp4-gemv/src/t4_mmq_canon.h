// t4_mmq_canon.h — T4（自研 tcgen05 NVFP4 快路）生产接线的**布局单一来源**（纯 C++，无 CUDA / 无 ggml 依赖）
//
// 结论链（都有实测定论，勿凭直觉改）：
//   · 权重（B）**零预处理**：canonical 板的片段边界与生产 GGML NVFP4 字节逐字节同源（results/20260916-109，
//     48+40 块全等），且 16 B granule 内的 nibble 归属两侧一致 ⇒ 真实张量**纯拷贝**即可（results/20260916-112，
//     chunks=1..20 逐位同构）。故生产里**不做任何权重 relayout、零额外常驻内存**。
//   · 激活（A）由本布局的量化器产出：canonical 板 + **ggml nibble 约定**（二者成对，改一侧必错）。
//
//   A 板（覆盖 K=64）内字节偏移（M = 128 = tcgen05 的 MMA M，固定）：
//     off(m, sub, l) = (m%8)*16 + (sub/2)*128 + (m/8)*256 + (sub%2)*8 + l
//     sub ∈ [0,4)：板内第几个 16 元素子块；l ∈ [0,8)：
//       低半字节 = 元素 16*sub + l，高半字节 = 元素 16*sub + l + 8     ← ggml 约定
//   A 缩放：sf_off(m, board, sub) = board*(M*4) + m*4 + sub，单字节 = 该子块的 UE4M3 **原始码**
//     （这 4 个字节就是直接写进 TMEM SFA 的那个 32 位字）。
//   B 记录（GGML NVFP4）：每 64-K 一条 **36 B** = [4 B sf（4 个 UE4M3 子块缩放）][32 B nib]；
//     行字节步长 = (K/64)*36（可带补零，只要求 4 B 对齐）。
//
//   A 的环形驻留按 MREAL **压缩**（M1b）：每个板只保留前 ceil(MREAL/8) 个 8 行组 ⇒ 前 MREAL 行的
//     偏移逐字节不变（描述符仍按 M=128 的 8 行 swizzle 走），第 ≥MREAL 行落到被丢弃的 D 行。
#pragma once

#include <cstddef>
#include <cstdint>

// 纯宿主编译（单测）与 NVCC 编译共用同一个头 ⇒ constexpr 辅助函数需要随环境切换限定符
#if defined(__CUDACC__)
#define T4MMQ_HD __host__ __device__
#else
#define T4MMQ_HD
#endif

namespace t4mmq {

// ---- 固定几何（与 tcgen05 NVFP4 MMA / ggml NVFP4 块定义一致）----
constexpr int MMA_M   = 128;   // tcgen05 块缩放 FP4 的 MMA M（板内固定）
constexpr int MMA_N   = 32;    // 同上 N
constexpr int MMA_K   = 64;    // 一个 canonical 板覆盖的 K
constexpr int KT      = 256;   // 环里「一个 chunk」覆盖的 K
constexpr int NSUB    = KT / MMA_K;              // 4 个板 / chunk
constexpr int K_SUB   = 16;                      // 子块元素数
constexpr int B_REC   = 36;                      // GGML NVFP4：每 64-K 记录字节数（4 B sf + 32 B nib）

constexpr int A_BD_STRIDE = MMA_M * 32;          // A nib 板步长（全局缓冲固定按 M=128 排）
constexpr int A_SF_STRIDE = MMA_M * 4;           // A sf 板步长

constexpr int B_SUB   = MMA_N * MMA_K / 2;       // 1024：一个 K=64 的 B canonical 板
constexpr int QS_CHUNK = B_SUB * NSUB;           // 4096
constexpr int SF_SUB  = MMA_N * 4;               // 128：32 n × 4 B
constexpr int SF_CHUNK = SF_SUB * NSUB;          // 512
constexpr int B_CHUNK = QS_CHUNK + SF_CHUNK;     // 4608

// ---- A 板内寻址 ----
T4MMQ_HD constexpr inline int a_nib_off_in_board(int m, int sub, int l) {
    return (m % 8) * 16 + (sub / 2) * 128 + (m / 8) * 256 + (sub % 2) * 8 + l;
}
// 元素 k（板内 [0,64)）所在字节：低/高半字节由 (k%16) < 8 决定
T4MMQ_HD constexpr inline size_t a_nib_off(int m, int k64, int k) {
    return (size_t) k64 * A_BD_STRIDE + a_nib_off_in_board(m, k / K_SUB, (k % K_SUB) % 8);
}
T4MMQ_HD constexpr inline size_t a_sf_off(int m, int k64, int sub) {
    return (size_t) k64 * A_SF_STRIDE + (size_t) m * 4 + sub;
}

// ---- B（canonical 板）在 smem 里的落位（v11 已验证的行内映射）----
//   源：行主序真实张量，行内每 64-K 记录 36 B；u = 记录内第几个 4 B 单元（0..7，其中 0..3 = nib 前 16 B，
//   4..7 = 后 16 B），row = 板内第几行 [0,32)。
T4MMQ_HD constexpr inline int b_smem_off(int row, int u) {
    return (row % 8) * 16 + ((u >= 4) ? 128 : 0) + (row / 8) * 256 + (u & 3) * 4;
}
//   B 的 sf 区（每 chunk 512 B = 4 子步 × 32 行 × 4 B，行主序）
T4MMQ_HD constexpr inline int b_sf_off(int substep, int row) { return substep * SF_SUB + row * 4; }

// ---- MREAL 压缩（M1b）----
T4MMQ_HD constexpr inline int a_bd_rows(int mreal)     { return ((mreal + 7) / 8) * 8; }         // 8 行组对齐
T4MMQ_HD constexpr inline int a_sub_bytes(int mreal)   { return a_bd_rows(mreal) * 32; }         // 一个板的 nib 字节
T4MMQ_HD constexpr inline int a_sf_bd_bytes(int mreal) { return a_bd_rows(mreal) * 4; }          // 一个板的 sf 字节（= 4 B/行）
T4MMQ_HD constexpr inline int a_chunk_bytes(int mreal) { return NSUB * (a_sub_bytes(mreal) + a_sf_bd_bytes(mreal)); }

// ---- 环几何 ----
struct ring_cfg {
    int cps;      // 每个 stage 装几个 chunk
    int stages;   // 环深
};
T4MMQ_HD constexpr inline int ring_a_stage(int mreal, ring_cfg c) { return a_chunk_bytes(mreal) * c.cps; }
T4MMQ_HD constexpr inline int ring_b_stage(ring_cfg c)            { return B_CHUNK * c.cps; }
T4MMQ_HD constexpr inline int ring_mbar_bytes(ring_cfg c) {
    return 8 * c.stages * 2 + 8 + 8 + 16 + 16;   // full[]+empty[] / 收尾 mbar / taddr 槽 / epi_rdy+epi_done（各 2）
}
T4MMQ_HD constexpr inline int ring_smem_required(int mreal, ring_cfg c) {
    return c.stages * (ring_a_stage(mreal, c) + ring_b_stage(c)) + ring_mbar_bytes(c);
}
T4MMQ_HD constexpr inline int ring_smem_host_req(int mreal, ring_cfg c) {
    return ((ring_smem_required(mreal, c) + 1023) / 1024) * 1024;
}
// TMEM 列：D 双缓冲 64 列 + SFA/SFB 各按 stage 分区（每 (cp,s) 16 列 = 4 子步 × 4 lane 组）
T4MMQ_HD constexpr inline int ring_tmem_cols(ring_cfg c) { return 64 + 2 * c.stages * c.cps * NSUB * 4; }
T4MMQ_HD constexpr inline bool ring_tmem_fits(ring_cfg c) { return ring_tmem_cols(c) <= 512; }
T4MMQ_HD constexpr inline int ring_occ_limit(ring_cfg c) {
    // 每 SM 的 TMEM 列预算（512）与动态 smem 预算（227 KB）共同决定
    return 512 / ring_tmem_cols(c);
}

constexpr int SMEM_PER_SM_MAX = 227 * 1024;   // board1 实测 232448 B 上限，留 1 KB 余量

}  // namespace t4mmq
