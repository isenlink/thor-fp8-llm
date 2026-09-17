// t4_prodaddr_layout.h — R-B2「H-mem 生产寻址」判别臂的**参数单一来源**
//
// 事实来源（不是叙述）：model-nvfp4.gguf 头直读（GGUF tensor offset 差分 ⇒ 真实行步长）
//   type40 NVFP4  K=5120  N=17408  128 张量  6120 MiB  row_stride=2880 = 20*144（无补零）
//   type40 NVFP4  K=17408 N=5120    64 张量  3060 MiB  row_stride=9792 = 68*144（无补零）
//   ⇒ 主形状取 K=5120（占 NVFP4 字节 62%）：CPR = K/256 = 20，ROW_STRIDE = CPR*144 = 2880
//   （CARD §7 原写「行距 8704」= q8 KV smem 预算里的数，与本模型无关，已作废）
//
// 与生产布局的一致性**由编译期 static_assert 守**（t4_prodaddr_check.cpp 同时 include 本头与生产
//   t4_mmq_canon.h，逐 (row,u) 比对 b_smem_off 全表）⇒ 生产头改了这里会直接编译失败。
#pragma once

#ifndef T4_V9F_CPR
#define T4_V9F_CPR 20
#endif

// 纯宿主编译（一致性门）与 NVCC 编译共用同一个头 ⇒ 限定符随环境切换（同 t4_mmq_canon.h 写法）
#if defined(__CUDACC__)
#define T4PROD_HD __host__ __device__
#else
#define T4PROD_HD
#endif

namespace t4prod {
constexpr int REC         = 36;                       // 每 64-K 记录：4 B sf + 32 B nib
constexpr int NSUB        = 4;                        // 一个 chunk = 256 K = 4 个 K=64 板
constexpr int B_SUB       = 1024;                     // 一个 K=64 的 B canonical 板字节
constexpr int ROWS        = 32;                       // 板内行数（= MMA_N）
constexpr int CPR         = T4_V9F_CPR;               // 每个 n-tile 的 chunk 数（= K/256）
constexpr int CHUNK_ROWS  = 4 * REC;                  // 144：一个 chunk 在一行里占的字节（4 记录）
constexpr int ROW_STRIDE  = CPR * CHUNK_ROWS;         // 2880：真实行步长
constexpr int TILE_BYTES  = ROWS * ROW_STRIDE;        // 92160：一个 n-tile（32 行）的字节数

// canonical 板内 4 B 单元落位（与 t4mmq::b_smem_off 同一张表：row%8 → 16 B 行内块，u>=4 → 后 16 B）
T4PROD_HD constexpr inline int b_smem_off(int row, int u) {
  return (row % 8) * 16 + ((u >= 4) ? 128 : 0) + (row / 8) * 256 + (u & 3) * 4;
}
// 单元 (row,u) 在记录内的源字节偏移（记录 = [4 B sf][32 B nib]）
T4PROD_HD constexpr inline int src_off_in_rec(int u) { return 4 + 4 * u; }
}  // namespace t4prod
