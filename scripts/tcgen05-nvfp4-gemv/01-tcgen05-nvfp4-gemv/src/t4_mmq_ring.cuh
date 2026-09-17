// t4_mmq_ring.cuh — T4（tcgen05 NVFP4）生产主内环：**真实 GGML 权重 + 真实激活 + 持久块**
//
// 血统（逐段照抄已验证件，改动逐条列在下面）：
//   · 环结构 / warp 分工 / F12 尾流 / F14 握手 / D 双缓冲 epilogue：evidence/20260916-92（v92，CPS=3/S=3，
//     d_mismatch=0 于 780×13 臂）。
//   · B（权重）**纯拷贝** 4 B 粒度寻址：evidence/20260916-112（v11，真实张量零预处理，chunks=1..20 逐位同构）。
//   · 本文件的三处改动（都是接线必需，不是优化）：
//     ① **持久块**：一个块连续吃多个 n-tile（F19：固定开销每块付一次；grid=544 独立 n-tile 只有 53.5 GB/s）。
//        tile 边界 = 每 cpr(=K/256) 步一次 ⇒ epilogue 的 n-tile 计数器自然复用（v92 的 cpr 语义）。
//     ② D 落盘带 `m < M_real` 守卫 + 乘 `row_scale[m]`（F28），落进 [n][M_real] 的 N 主序缓冲（F21）。
//     ③ A 环按 MREAL **压缩**（M1b：ABD = ceil(M/8)*8），板内前缀与 M=128 全局布局逐字节同源 ⇒ 同一份 A
//        缓冲既服务小 M（tile 内只搬前缀）也服务大 M。
//
// 不变量（照抄 B9f/TECH §10，勿动）：arrive ∈ full[(n+2)%S]；生产侧跑 nstages+(S-2) 轮，尾部不搬数据。
#pragma once

#include "t4_mmq_canon.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace t4mmq {

constexpr int      NPROD = 256;                  // 生产者线程数（8 warp）
constexpr int      NTHR  = NPROD + 32;           // 288 = 9 warp（含 1 个消费者 warp）
constexpr uint32_t TADDR_POISON = 0xA5A5A5A5u;
constexpr uint32_t TADDR_NCOLS_MAX = 512u;
constexpr uint32_t IDESC_B5 = 0x08080480u;       // host 对拍定值（v5）
constexpr uint32_t LBO_BYTES = 128u, SBO_BYTES = 256u;
constexpr uint32_t BOUND = 200000000u;
// TMEM 列布局（F15：SF 基址按 4 列步进；SFA 与 SFB 各自按 stage 分区，F32：A 的 sf 列 = 基址 + lane 组号）
constexpr uint32_t COL_D0 = 0u, COL_D1 = 32u, COL_SFA_ST = 64u;
constexpr uint32_t SF_COLS_STEP = 4u;

// ---------------------------------------------------------------- 环几何
// ACAP > 0 ⇒ **A 常驻形态**（T4-M2c-4b）：A 不再随 stage 走环，而是在入口一次性把 `cpr` 个 chunk 的
// A（含 sf）搬进固定区 `OFF_A`，尺寸上限 `ACAP × A_CHUNK`（编译期钉住，运行期由调用方保证 cpr ≤ ACAP）。
// 动机（静态可证，见 tasks/T4-M2c-4b-a-resident.md §0）：A 的拷贝索引只与「tile 内 chunk 序号」有关 ⇒
// 现状下同一份 A 在每个 n-tile 被重搬一遍（每块 38.9 次），A 的指令/请求被放大 38.9 倍。
template <int ABD, int CPS, int STAGES, int ACAP = 0>
struct RingGeom {
    static constexpr bool ARES = (ACAP > 0);
    static constexpr int A_SUB     = a_sub_bytes(ABD);        // 一个板（K=64）的 nib 字节
    static constexpr int A_SF_BD   = a_sf_bd_bytes(ABD);      // 一个板的 sf 字节
    static constexpr int A_CHUNK   = NSUB * (A_SUB + A_SF_BD);
    static constexpr int A_STAGE   = A_CHUNK * CPS;
    static constexpr int B_STAGE   = B_CHUNK * CPS;
    static constexpr int MBAR_B    = 8 * STAGES * 2 + 8 + 8 + 16 + 16;
    static constexpr int A_REGION  = ARES ? (ACAP * A_CHUNK) : (STAGES * A_STAGE);
    static constexpr int SMEM      = A_REGION + STAGES * B_STAGE + MBAR_B;
    static constexpr int SMEM_REQ  = ((SMEM + 1023) / 1024) * 1024;
    static constexpr int OFF_A     = 0;
    static constexpr int OFF_B     = OFF_A + A_REGION;
    static constexpr int OFF_MFULL = OFF_B + STAGES * B_STAGE;
    static constexpr int OFF_MEMPT = OFF_MFULL + 8 * STAGES;
    static constexpr int OFF_MBAR  = OFF_MEMPT + 8 * STAGES;
    static constexpr int OFF_TADDR = OFF_MBAR + 8;
    static constexpr int OFF_EPI_RDY  = OFF_TADDR + 8;
    static constexpr int OFF_EPI_DONE = OFF_EPI_RDY + 16;
    static constexpr uint32_t SF_STAGE_COLS = (uint32_t)(CPS * NSUB) * SF_COLS_STEP;
    static constexpr uint32_t COL_SFB_ST = COL_SFA_ST + (uint32_t)STAGES * SF_STAGE_COLS;
    static constexpr uint32_t NCOLS = COL_SFB_ST + (uint32_t)STAGES * SF_STAGE_COLS;

    static_assert(OFF_EPI_DONE + 16 == SMEM, "偏移必须与 SMEM 等式一致");
    static_assert(NCOLS <= TADDR_NCOLS_MAX, "SF 段超出 TMEM 列预算");
    static_assert(A_SF_BD % 16 == 0 && A_SUB % 16 == 0, "A 环必须 16 B 粒度可搬");
    static_assert(SMEM_REQ <= 232448, "环超出每 SM 动态 smem 上限（227 KB）");
    static_assert(COL_SFA_ST >= COL_D1 + 32u, "SFA 不得压到 D 缓冲");
    T4MMQ_HD static constexpr uint32_t SFA_COL(int st, int cp, int s) {
        return COL_SFA_ST + (uint32_t)st * SF_STAGE_COLS + (uint32_t)(cp * NSUB + s) * SF_COLS_STEP;
    }
    T4MMQ_HD static constexpr uint32_t SFB_COL(int st, int cp, int s) {
        return COL_SFB_ST + (uint32_t)st * SF_STAGE_COLS + (uint32_t)(cp * NSUB + s) * SF_COLS_STEP;
    }
};

// ---------------------------------------------------------------- PTX 包装（与 v5/CUTLASS 同源）
__device__ __forceinline__ uint32_t cvta_smem(const void * p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void tmem_alloc(uint32_t dst, uint32_t ncols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(dst), "r"(ncols));
}
__device__ __forceinline__ void tmem_relinquish() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}
__device__ __forceinline__ void tmem_dealloc(uint32_t taddr, uint32_t ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols));
}
__device__ __forceinline__ void proxy_fence_shared() { asm volatile("fence.proxy.async.shared::cta;"); }
__device__ __forceinline__ void fence_before_thread_sync() { asm volatile("tcgen05.fence::before_thread_sync;"); }
__device__ __forceinline__ void fence_after_thread_sync() { asm volatile("tcgen05.fence::after_thread_sync;"); }
__device__ __forceinline__ void mbar_init(uint32_t mbar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(mbar), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(uint32_t mbar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(mbar));
}
__device__ __forceinline__ void commit_mbar(uint32_t mbar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" ::"r"(mbar));
}
__device__ __forceinline__ uint32_t mbar_wait_bounded_p(uint32_t mbar, uint32_t bound, uint32_t phase) {
    uint32_t ok = 0;
    asm volatile(
        "{\n\t.reg .pred p;\n\t.reg .u32 cnt;\n\tmov.u32 cnt, %2;\n"
        "L1_%=:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %3;\n\t"
        "@p bra L2_%=;\n\t"
        "sub.u32 cnt, cnt, 1;\n\t"
        "setp.ne.u32 p, cnt, 0;\n\t"
        "@p bra L1_%=;\n\t"
        "mov.u32 %0, 0;\n\t"
        "bra L3_%=;\n"
        "L2_%=:\n\t"
        "mov.u32 %0, 1;\n"
        "L3_%=:\n\t}"
        : "=r"(ok)
        : "r"(mbar), "r"(bound), "r"(phase));
    return ok;
}
__device__ __forceinline__ void tmem_ld_32x32b_x32(uint32_t taddr, uint32_t * r) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]),
          "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]),
          "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]),
          "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]),
          "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
        : "r"(taddr));
}
__device__ __forceinline__ void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;"); }
__device__ __forceinline__ void tmem_st_32x32b_x1(uint32_t taddr, uint32_t v) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};" ::"r"(taddr), "r"(v));
}
__device__ __forceinline__ void tmem_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;"); }
__device__ __forceinline__ void mma_mxf4nvf4(uint32_t d_tmem, uint64_t adesc, uint64_t bdesc, uint32_t idesc,
                                             uint32_t tsfa, uint32_t tsfb, uint32_t scale_c) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X "
        "[%0], %1, %2, %3, [%5], [%6], p;\n\t}"
        ::"r"(d_tmem), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(scale_c), "r"(tsfa), "r"(tsfb));
}
// smem 描述符：SWIZZLE_NONE / version=1 / LBO=8 / SBO=16（16 B 单位）—— 与 v5 的 CUTLASS 路径**逐位相同**
// （v5 的 DG_IDESC_CUT vs HAND 对拍过）。位域按 cute/arch/mma_sm100_desc.hpp 的 SmemDescriptor（非 SM107 路径）：
//   [0,14) start_address>>4 | [16,30) lbo>>4 | [32,46) sbo>>4 | [46,48) version | [49,52) base_offset
//   | [52] lbo_mode | [61,64) layout_type
__device__ __host__ __forceinline__ constexpr uint64_t sdesc_word(uint32_t smem_byte_addr, uint32_t lbo_bytes, uint32_t sbo_bytes) {
    return ((uint64_t) ((smem_byte_addr >> 4) & 0x3FFFu)) |
           ((uint64_t) ((lbo_bytes >> 4) & 0x3FFFu) << 16) |
           ((uint64_t) ((sbo_bytes >> 4) & 0x3FFFu) << 32) |
           ((uint64_t) 1u << 46) |                 // version = 1
           ((uint64_t) 0u << 49) |                 // base_offset = 0
           ((uint64_t) 0u << 52) |                 // lbo_mode = 0
           ((uint64_t) 0u << 61);                  // layout_type = SWIZZLE_NONE = 0
}
// elect.sync：v5 用 cute::elect_one_sync，生产件不带 cutlass ⇒ 自备同语义实现
__device__ __forceinline__ bool elect_one() {
    uint32_t pred = 0;
    asm volatile("{\n\t.reg .pred %%px;\n\telect.sync _|%%px, 0xFFFFFFFF;\n\tselp.b32 %0, 1, 0, %%px;\n\t}"
                 : "+r"(pred));
    return pred != 0;
}
__device__ __forceinline__ void cp_async16(uint32_t smem, const void * g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem), "l"(g));
}
__device__ __forceinline__ void cp_async4(uint32_t smem, const void * g) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(smem), "l"(g));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int NPG>
__device__ __forceinline__ void cp_wait_group() { asm volatile("cp.async.wait_group %0;" ::"n"(NPG)); }
// 生产者内部的 128 线程命名屏障（warp0..3；id=1 避开 __syncthreads 的 id=0）：跨 warp 的 smem 交接
__device__ __forceinline__ void bar_sync_sf()  { asm volatile("bar.sync 1, 128;" ::: "memory"); }
__device__ __forceinline__ void bar_sync_epi() { asm volatile("bar.sync 2, 128;" ::: "memory"); }

// ---------------------------------------------------------------- 主核
// gB      : 真实 GGML NVFP4 权重（行主序，行步长 row_stride 字节）
// gAn/gAs : canonical A（nib / sf），全局按 M=128 板步长排；本内核只搬 ABD 行的前缀
// gRs     : row_scale[M_real]（F28）
// Dtmp    : [n_global][M_real] 的 N 主序输出（F21）
// cpr     : 每个 n-tile 的 chunk 数（= K/256）
template <int ABD, int CPS, int STAGES, int ACAP = 0>
__global__ void __launch_bounds__(NTHR) t4_mmq_ring(
        const unsigned char * __restrict__ gB,
        const unsigned char * __restrict__ gAn,
        const unsigned char * __restrict__ gAs,
        const float * __restrict__ gRs,
        float * __restrict__ Dtmp,
        unsigned cpr,                     // chunk 数 / n-tile
        int ntiles, int tile_start, int tile_end,
        int m_real, size_t row_stride, unsigned * diag) {
    using G = RingGeom<ABD, CPS, STAGES, ACAP>;
    if (G::SMEM > G::SMEM_REQ) return;                    // 契约守护（事故 #5 后强制）
    if (G::ARES && (unsigned) ACAP < cpr) return;          // A 常驻区装不下本形状 ⇒ 不接管（调用方已判过）
    // 持久块分派：宿主按「每块一段连续 n-tile」发 grid（F19）。grid=1 ⇒ 区间原样（与 v92/v11 单块行为一致）。
    if ((int) gridDim.x > 1) {
        const int per   = (ntiles + (int) gridDim.x - 1) / (int) gridDim.x;
        const int b_st  = (int) blockIdx.x * per;
        tile_start = b_st;
        tile_end   = (b_st + per < ntiles) ? b_st + per : ntiles;
    }
    extern __shared__ __align__(1024) unsigned char smem[];
    const int t = threadIdx.x, warp = t / 32;
    unsigned char * smA = smem + G::OFF_A;
    unsigned char * smB = smem + G::OFF_B;
    const uint32_t mfull  = cvta_smem(smem + G::OFF_MFULL);
    const uint32_t membty = cvta_smem(smem + G::OFF_MEMPT);
    const uint32_t mbar   = cvta_smem(smem + G::OFF_MBAR);
    const uint32_t epi_rdy  = cvta_smem(smem + G::OFF_EPI_RDY);
    const uint32_t epi_done = cvta_smem(smem + G::OFF_EPI_DONE);
    unsigned char * slot = smem + G::OFF_TADDR;

    const int ntiles_blk = tile_end - tile_start;
    const unsigned G_steps = (unsigned) ntiles_blk * cpr;             // 本块的总步数
    const bool epi_on = (cpr != 0u);                                  // 每 cpr 步一个 n-tile 边界
    const unsigned nstages = (G_steps + (unsigned) CPS - 1u) / (unsigned) CPS;

    if (t == 0) {
        for (int i = 0; i < 8; ++i) diag[i] = 0u;
        *(volatile uint32_t *) slot = TADDR_POISON;
        mbar_init(mbar, 1u);
        for (int i = 0; i < 2; ++i) { mbar_init(epi_rdy + 8u * i, 1u); mbar_init(epi_done + 8u * i, 1u); }
        for (int i = 0; i < STAGES; ++i) {
            mbar_init(mfull + 8 * i, (uint32_t) NPROD);
            mbar_init(membty + 8 * i, 1u);
        }
    }
    __syncthreads();

    // TMEM 分配（warp0 全活跃收敛；F1）
    if (warp == 0) { tmem_alloc(cvta_smem(slot), G::NCOLS); tmem_relinquish(); }
    __syncthreads();
    const uint32_t taddr = *(volatile uint32_t *) slot;
    const bool taddr_bad = (taddr == TADDR_POISON) || ((taddr >> 16) != 0u) ||
                           ((taddr & 0xFFFFu) >= TADDR_NCOLS_MAX) || ((taddr & 31u) != 0u);
    if (t == 0) diag[0] = taddr_bad ? 0u : 1u;

    // ---- A 常驻（T4-M2c-4b）：入口一次性把 cpr 个 chunk 的 A（nib + sf）搬进 OFF_A ----
    // 搬法与生产者逐字节同源（16 B 粒度、同一压缩前缀布局），只是从「每 stage 一次」变成「每块一次」。
    if (G::ARES && !taddr_bad) {
        if (t < NPROD) {
            const unsigned per_chunk = (unsigned) (NSUB * (G::A_SUB / 16) + NSUB * (G::A_SF_BD / 16));
            const unsigned n_nib     = (unsigned) (NSUB * (G::A_SUB / 16));
            const unsigned total     = (unsigned) cpr * per_chunk;
            for (unsigned q = (unsigned) t; q < total; q += (unsigned) NPROD) {
                const unsigned cc = q / per_chunk, r = q % per_chunk;
                unsigned char * dA = smA + (size_t) cc * G::A_CHUNK;
                if (r < n_nib) {
                    const unsigned bdg = r / (unsigned) (G::A_SUB / 16), u = r % (unsigned) (G::A_SUB / 16);
                    cp_async16(cvta_smem(dA + (size_t) bdg * G::A_SUB + (unsigned) (u * 16u)),
                               gAn + (size_t) cc * (size_t) (NSUB * A_BD_STRIDE) +
                                     (size_t) bdg * A_BD_STRIDE + (unsigned) (u * 16u));
                } else {
                    const unsigned r2  = r - n_nib;
                    const unsigned bdg = r2 / (unsigned) (G::A_SF_BD / 16), u = r2 % (unsigned) (G::A_SF_BD / 16);
                    cp_async16(cvta_smem(dA + G::A_CHUNK - NSUB * G::A_SF_BD +
                                         (size_t) bdg * G::A_SF_BD + (unsigned) (u * 16u)),
                               gAs + (size_t) cc * (size_t) (NSUB * A_SF_STRIDE) +
                                     (size_t) bdg * A_SF_STRIDE + (unsigned) (u * 16u));
                }
            }
            cp_commit();
            cp_wait_group<0>();
            proxy_fence_shared();
            fence_before_thread_sync();
        }
        __syncthreads();
        fence_after_thread_sync();
    }

    // 描述符
    uint64_t ad[STAGES][CPS][NSUB], bd[STAGES][CPS][NSUB];
    if (!taddr_bad) {
        for (int st = 0; st < STAGES; ++st)
            for (int cp = 0; cp < CPS; ++cp)
                for (int s = 0; s < NSUB; ++s) {
                    ad[st][cp][s] = sdesc_word(cvta_smem(smA + st * G::A_STAGE + cp * G::A_CHUNK + s * G::A_SUB), LBO_BYTES, SBO_BYTES);
                    bd[st][cp][s] = sdesc_word(cvta_smem(smB + st * G::B_STAGE + cp * B_CHUNK + s * B_SUB), LBO_BYTES, SBO_BYTES);
                }
    }

    uint32_t ok_all = 1;
    if (!taddr_bad) {
        if (t >= NPROD) {
            // ---------------- 消费者（warp8）----------------
            unsigned fails = 0;
            for (unsigned j = 0; j < nstages; ++j) {
                const int st = (int) (j % (unsigned) STAGES);
                const unsigned round = j / (unsigned) STAGES;
                if (!mbar_wait_bounded_p(mfull + 8 * st, BOUND, round & 1u)) { fails++; break; }
                fence_after_thread_sync();          // 与生产者 fence::before_thread_sync 配对
                const unsigned c0 = j * (unsigned) CPS;
                const int used = (int) ((c0 + (unsigned) CPS <= G_steps) ? (unsigned) CPS : (G_steps - c0));
                bool cp_fail = false;
                if (elect_one()) {
                    // 递增计数器（不在内层做除法）：k = 已完成/进行中的 n-tile 序号，rem = 块内 chunk 偏移
                    unsigned k   = c0 / cpr;
                    unsigned rem = c0 % cpr;
                    for (int cp = 0; cp < used; ++cp) {
                        if (epi_on && rem == 0u && k >= 2u) {
                            const unsigned wb = k - 2u;   // 复用 D 缓冲前，上一次（k-2）的 epilogue 必须已完成
                            if (!mbar_wait_bounded_p(epi_done + 8u * (wb & 1u), BOUND, (wb >> 1) & 1u)) {
                                fails++; cp_fail = true; break;
                            }
                            fence_after_thread_sync();
                        }
                        const uint32_t dtm = taddr + (epi_on ? ((k & 1u) ? COL_D1 : COL_D0) : COL_D0);
                        // A 常驻时描述符按「tile 内 chunk 序号 rem」现算（不占寄存器数组）；否则用入口算好的 ad[][][]
                        const uint32_t abase = cvta_smem(smA) + (uint32_t) rem * (uint32_t) G::A_CHUNK;
                        for (int s = 0; s < NSUB; ++s) {
                            const uint64_t adesc = G::ARES
                                                       ? sdesc_word(abase + (uint32_t) (s * G::A_SUB), LBO_BYTES, SBO_BYTES)
                                                       : ad[st][cp][s];
                            mma_mxf4nvf4(dtm, adesc, bd[st][cp][s], IDESC_B5,
                                         taddr + G::SFA_COL(st, cp, s), taddr + G::SFB_COL(st, cp, s),
                                         (rem == 0u && s == 0) ? 0u : 1u);   // 跨 chunk 累加 / 每 n-tile 复位
                        }
                        if (epi_on && rem == (cpr - 1u)) commit_mbar(epi_rdy + 8u * (k & 1u));
                        if (++rem == cpr) { rem = 0u; ++k; }
                    }
                    commit_mbar(membty + 8 * st);
                }
                if (cp_fail) break;
            }
            if (elect_one()) {
                commit_mbar(mbar);
                ok_all = (mbar_wait_bounded_p(mbar, BOUND, 0u) && fails == 0u) ? 1u : 0u;
            } else {
                ok_all = (fails == 0u) ? 1u : 0u;
            }
        } else {
            // ---------------- 生产者（warp0..7）----------------
            unsigned fails = 0;
            unsigned next_b = 0u;                        // 下一个待办 tile 边界
            const unsigned n_iter = nstages + (unsigned) (STAGES - 2);
            // 纯拷贝寻址：记录内 4 B 单元 u_ = t&7、行 row_ = (t>>3)&31（v11 已验）
            const int u_ = t & 7, row_ = (t >> 3) & 31;
            for (unsigned n = 0; n < n_iter; ++n) {
                const int st = (int) (n % (unsigned) STAGES);
                const unsigned round = n / (unsigned) STAGES;
                if (n < nstages) {
                    if (round > 0 && !mbar_wait_bounded_p(membty + 8 * st, BOUND, (round - 1u) & 1u)) { fails++; break; }
                    const unsigned c0 = n * (unsigned) CPS;
                    const unsigned nblk = (c0 + (unsigned) CPS <= G_steps) ? (unsigned) CPS : (G_steps - c0);
                    unsigned char * dstB = smB + (size_t) st * G::B_STAGE;
                    unsigned char * dstA = smA + (size_t) st * G::A_STAGE;
                    // 本 stage 的起始 (tile, chunk)：一次除法/取模，stage 内用递增计数器
                    unsigned tl = c0 / cpr, cc = c0 % cpr;
                    for (unsigned k = 0; k < nblk; ++k) {
                        const unsigned char * b_tile = gB + (size_t) (tile_start + tl) * 32u * row_stride;
                        // ---- B qs：真实张量的 4 B 粒度纯拷贝（36 B 记录不是 16 B 对齐）
                        {
                            const unsigned char * s2 = b_tile + (size_t) row_ * row_stride +
                                                       (size_t) (4u * cc) * B_REC + 4u + 4u * (unsigned) u_;
                            unsigned char * d2 = dstB + (size_t) k * B_CHUNK + b_smem_off(row_, u_);
                            for (int r = 0; r < NSUB; ++r)
                                cp_async4(cvta_smem(d2 + (unsigned) r * B_SUB), s2 + (size_t) r * B_REC);
                        }
                        // ---- B sf：warp0 独占（1× 读取）+ 128 线程命名屏障交接
                        if (warp == 0) {
                            const unsigned char * rs = b_tile + (size_t) (t & 31) * row_stride + (size_t) (4u * cc) * B_REC;
                            for (int r = 0; r < NSUB; ++r)
                                cp_async4(cvta_smem(dstB + (size_t) k * B_CHUNK + QS_CHUNK + r * SF_SUB + (unsigned) (t & 31) * 4u),
                                          rs + (size_t) r * B_REC);
                        }
                        // ---- A 环：全局（M=128 板步长）→ 环内 ABD 前缀，按板搬（16 B 粒度）
                        //      A 常驻形态下这一段整体消失（入口已搬过，且每个 tile 的 A 完全相同）
                        if (!G::ARES) {
                            const unsigned char * sAn = gAn + (size_t) cc * (NSUB * A_BD_STRIDE);
                            const unsigned char * sAs = gAs + (size_t) cc * (NSUB * A_SF_STRIDE);
                            unsigned char * dA = dstA + (size_t) k * G::A_CHUNK;
                            for (int q = t; q < NSUB * (G::A_SUB / 16); q += NPROD) {
                                const int bdg = q / (G::A_SUB / 16), u = q % (G::A_SUB / 16);
                                cp_async16(cvta_smem(dA + (size_t) bdg * G::A_SUB + (unsigned) u * 16u),
                                           sAn + (size_t) bdg * A_BD_STRIDE + (unsigned) u * 16u);
                            }
                            for (int q = t; q < NSUB * (G::A_SF_BD / 16); q += NPROD) {
                                const int bdg = q / (G::A_SF_BD / 16), u = q % (G::A_SF_BD / 16);
                                cp_async16(cvta_smem(dA + G::A_CHUNK - NSUB * G::A_SF_BD + (size_t) bdg * G::A_SF_BD + (unsigned) u * 16u),
                                           sAs + (size_t) bdg * A_SF_STRIDE + (unsigned) u * 16u);
                            }
                        }
                        if (++cc == cpr) { cc = 0u; ++tl; }
                    }
                    cp_commit();
                }
                if (n >= (unsigned) (STAGES - 2)) {
                    if (n < nstages) cp_wait_group<STAGES - 2>();
                    else             cp_wait_group<0>();
                    const int st_done = (int) ((n + 2u) % (unsigned) STAGES);
                    // 把 stage st_done 的 SF 字搬进 TMEM：A 侧（SFA 段，列 = 基址 + lane 组号，F32）
                    // 与 B 侧（SFB 段）同处写；上一次用到该 stage 的 MMA 已提交（empty[] 已放行）
                    if (warp < 4) {
                        bar_sync_sf();
                        const int nn = t & 31;
                        const int mrow = (warp & 3) * 32 + nn;
                        const unsigned char * sfsB = smB + (size_t) st_done * G::B_STAGE + QS_CHUNK;
                        for (int cp = 0; cp < CPS; ++cp)
                            for (int s = 0; s < NSUB; ++s) {
                                const unsigned char * pw = sfsB + (size_t) cp * B_CHUNK + (size_t) s * SF_SUB + (size_t) nn * 4;
                                const uint32_t w32 = (uint32_t) pw[0] | ((uint32_t) pw[1] << 8) |
                                                     ((uint32_t) pw[2] << 16) | ((uint32_t) pw[3] << 24);
                                tmem_st_32x32b_x1(taddr + ((uint32_t) ((warp & 3) * 32) << 16) +
                                                      G::SFB_COL(st_done, cp, s), w32);
                                // A 的 sf：[4 板][每板 ABD 行 × 4 B]。环形态取本 stage 的槽；
                                // A 常驻形态取「本 stage 首 chunk + cp」在常驻区里的槽（对 cpr 取模 ⇒ 恒在界内）
                                const unsigned char * pwa;
                                if (G::ARES) {
                                    const unsigned c0_done = (n - (unsigned) (STAGES - 2)) * (unsigned) CPS;
                                    const unsigned cc_done = (unsigned) ((c0_done + (unsigned) cp) % cpr);
                                    pwa = smA + (size_t) cc_done * G::A_CHUNK + G::A_CHUNK - NSUB * G::A_SF_BD +
                                          (size_t) s * G::A_SF_BD + (size_t) mrow * 4;
                                } else {
                                    pwa = smA + (size_t) st_done * G::A_STAGE +
                                          (size_t) cp * G::A_CHUNK + G::A_CHUNK - NSUB * G::A_SF_BD +
                                          (size_t) s * G::A_SF_BD + (size_t) mrow * 4;
                                }
                                const uint32_t w32a = (uint32_t) pwa[0] | ((uint32_t) pwa[1] << 8) |
                                                      ((uint32_t) pwa[2] << 16) | ((uint32_t) pwa[3] << 24);
                                tmem_st_32x32b_x1(taddr + ((uint32_t) ((warp & 3) * 32) << 16) +
                                                      G::SFA_COL(st_done, cp, s) + (uint32_t) (warp & 3), w32a);
                            }
                        tmem_wait_st();
                        fence_before_thread_sync();
                    }
                    mbar_arrive(mfull + 8 * st_done);   // 全部 256 个生产者线程各到达一次（与 init 的 NPROD 配对）
                    // 若本轮的消费者 stage 收尾了一个 n-tile ⇒ warp0..3 读回 D（滞后 2 stage，见 results/84b）
                    if (epi_on && warp < 4) {
                        const int jc = (int) n - (STAGES - 2);
                        if (jc >= 0) {
                            const unsigned steps_done = ((unsigned) (jc + 1) < nstages)
                                                            ? (jc >= 1 ? (unsigned) (jc - 1) * (unsigned) CPS : 0u)
                                                            : G_steps;
                            const unsigned b_max = steps_done / cpr;
                            bool ok_b = true;
                            while (next_b < b_max) {
                                if (!mbar_wait_bounded_p(epi_rdy + 8u * (next_b & 1u), BOUND, (next_b >> 1) & 1u)) {
                                    fails++; ok_b = false; break;
                                }
                                const int tile = tile_start + (int) next_b;
                                const uint32_t dtm = taddr + ((next_b & 1u) ? COL_D1 : COL_D0);
                                uint32_t r[32];
                                for (int i = 0; i < 32; ++i) r[i] = 0u;
                                tmem_ld_32x32b_x32(dtm + ((uint32_t) ((warp & 3) * 32) << 16), r);
                                tmem_wait_ld();
                                // D 落盘：N 主序 [n][m]，乘 row_scale[m]（F28），m >= m_real 的填充行丢弃
                                const int mm = (warp & 3) * 32 + (t & 31);
                                if (mm < m_real) {
                                    const float rs = gRs ? gRs[mm] : 1.0f;
                                    float * drow = Dtmp + (size_t) tile * 32u * (size_t) m_real + (size_t) mm;
                                    for (int nn2 = 0; nn2 < MMA_N; ++nn2)
                                        drow[(size_t) nn2 * (size_t) m_real] = __uint_as_float(r[nn2]) * rs;
                                }
                                fence_before_thread_sync();
                                bar_sync_epi();
                                if (warp == 0 && (t & 31u) == 0u) mbar_arrive(epi_done + 8u * (next_b & 1u));
                                next_b++;
                            }
                            if (!ok_b) break;
                        }
                    }
                }
            }
            if (t == 0 && fails != 0u) diag[1] = 0xFFFFFFFFu;
        }
    }
    __syncthreads();
    if (t == 0) diag[2] = ok_all;
    __syncthreads();
    if (warp == 0) {
        const unsigned ok_u = __shfl_sync(0xFFFFFFFFu, taddr_bad ? 0u : 1u, 0);
        if (ok_u) tmem_dealloc(taddr, G::NCOLS);
    }
}

}  // namespace t4mmq
