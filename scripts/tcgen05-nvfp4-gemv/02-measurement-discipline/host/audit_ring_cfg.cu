// audit_ring_cfg.cu — 环配置的**上板前宿主几何审计**（零板端动作）
//
// 目的：把每个候选 (CPS,STAGES) 的两条硬预算（TMEM 列 NCOLS、动态 smem SMEM/SMEM_REQ）与
// 环内偏移一次性算清，并断言：① NCOLS ≤ 512（TADDR_NCOLS_MAX，内核 static_assert 同款）；
// ② SMEM_REQ ≤ 每 SM 上限（227 KB 级，本板 233472 B）；③ 每 chunk 的 A/B 字节与请求数上界。
// 用法（宿主，需要 nvcc 12.8：/usr/local/cuda/bin/nvcc）：
//   /usr/local/cuda/bin/nvcc -std=c++17 -I<prod>/ggml/src/ggml-cuda/t4 -I<prod>/ggml/src/ggml-cuda audit_ring_cfg.cu -o /tmp/audit_ring_cfg && /tmp/audit_ring_cfg
// 只做静态算术：不发射任何内核、不碰板卡。

#include "t4_mmq_ring.cuh"

#include <cstdio>

using namespace t4mmq;

constexpr int SM_PER_SM_LIMIT = 227 * 1024;   // 每 SM 可申请的动态 smem 上界（板端实测 232448 B/CTA 上限内的保守值）

template <int ABD, int CPS, int STAGES, bool PAD, int ACAP = 0>
static void row(const char * tag, const int nblk) {
    using G = RingGeom<ABD, CPS, STAGES, ACAP>;
    const int req = (PAD && G::SMEM_REQ < 118784) ? 118784 : G::SMEM_REQ;
    const int inflight = CPS * B_CHUNK;                       // 一个 stage 的 B 字节（在飞单元）
    const int a_chunk = a_chunk_bytes(ABD);
    std::printf("%-16s ABD=%-4d CPS=%d S=%d pad=%d ACAP=%-3d | NCOLS=%4u (budget 512, %s) | SMEM=%6d REQ=%6d | "
                "在飞 B=%5d B | A/chunk=%5d B | 每块 tile 的 stage 数 (%d chunks/tile)=%.2f\n",
                tag, ABD, CPS, STAGES, (int) PAD, ACAP, G::NCOLS,
                (G::NCOLS <= TADDR_NCOLS_MAX ? "OK " : "OVER"), G::SMEM, req, inflight, a_chunk,
                nblk, (double) nblk / (double) CPS);
    static_assert(G::NCOLS <= 512u, "TMEM 列预算超限");
    static_assert(G::SMEM <= SM_PER_SM_LIMIT, "smem 超每 SM 上限");
    static_assert(G::SMEM_REQ <= SM_PER_SM_LIMIT, "smem 申请值超每 SM 上限");
    if (PAD) {
        static_assert(118784 * 2 > 233472, "pad 值必须让两个 CTA 装不下（occ=1 硬保证）");
    }
         if (req >= 118784) std::printf("               ↳ occ=1 由 smem 强制\n");
    else if (G::NCOLS > 256u) std::printf("               ↳ ⚠ NCOLS>256 且未 pad：两个 CTA 同 SM 会卡死在 tcgen05.alloc\n");
    else                      std::printf("               ↳ occ 不强制（默认配置：256+256=512 恰好装得下）\n");
}

int main() {
    std::printf("== T4 环配置审计（gate/up: 20 chunks/tile；ffn_down: 68 chunks/tile；M≤8 ⇒ ABD=8）==\n");
    // 生产 decode 的主力档：M ≤ 8 ⇒ ABD=8（M=13 走 ABD=16，同式缩放即可）
    row<8, 3, 2, false>("现行(默认)", 20);
    row<8, 1, 2, false>("窄长流", 20);
    row<8, 4, 3, true>("候选 A", 20);
    row<8, 3, 4, true>("候选 B", 20);
    row<8, 7, 2, true>("候选 C", 20);
    row<8, 7, 2, true>("候选 C(down)", 68);
    row<8, 4, 3, true>("候选 A(down)", 68);
    std::printf("\n== A 常驻形态（T4-M2c-4b，ACAP=68；ABD=8：A 区 78.3 KB / ABD=16：156.7 KB）==\n");
    row<8, 3, 2, true, 68>("ARES 3/2", 20);
    row<8, 4, 3, true, 68>("ARES 4/3", 20);
    row<8, 3, 2, true, 68>("ARES 3/2(down)", 68);
    row<8, 4, 3, true, 68>("ARES 4/3(down)", 68);
    std::printf("\n== ABD=16（M=13 的 verify 档）抽查 ==\n");
    row<16, 3, 2, false>("现行(默认)", 20);
    row<16, 4, 3, true>("候选 A", 20);
    row<16, 7, 2, true>("候选 C", 20);
    row<16, 3, 2, true, 68>("ARES 3/2", 20);
    row<16, 4, 3, true, 68>("ARES 4/3", 20);
    std::printf("\n== 生产形状的每调用 A/B 字节（per n-tile）==\n");
    std::printf("gate/up : B=20×4608=%6d B  A=%6d B（%.1f%% of B）  n-tiles=544  → 每块 %.1f tiles\n",
                20 * B_CHUNK, 20 * a_chunk_bytes(8), 100.0 * 20 * a_chunk_bytes(8) / (20.0 * B_CHUNK), 544 / 14.0);
    std::printf("ffn_down: B=68×4608=%6d B  A=%6d B（%.1f%% of B）  n-tiles=160  → 每块 %.1f tiles\n",
                68 * B_CHUNK, 68 * a_chunk_bytes(8), 100.0 * 68 * a_chunk_bytes(8) / (68.0 * B_CHUNK), 160 / 14.0);
    return 0;
}
