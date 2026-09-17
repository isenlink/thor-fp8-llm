// c58_v9f_fp4_ws32k.cu — T4-3 · B9f：cp.async warp specialization，**握手单元放大到 32 KB**
// [R-B2 变体] 本副本加 `T4_V9F_GRAN4` 开关：只把 B 的拷贝粒度从 16 B 换成 4 B（地址不变）。默认 0 = 归档口径。
//
// 立项依据（board4 同板配对，results/20260916-75）：B9c（逐 4 KB 跨 warp mbarrier 握手）只有 72.15 GB/s，
//   而同板 B7（朴素 __syncthreads + cp.async，逐 4 KB）是 181.70 GB/s ⇒ 跨 warp 握手的固定开销
//   （256 次 arrive + 两侧唤醒）在 4 KB 粒度上约 560 ns/块，必须**按字节摊薄**。
// 本件只改一件事：一次握手搬运 **CPS=8 块（32 KB）**；其余（A/B canonical 布局、idesc、SF 落点、K=256 四子步、
//   数据发生器、warp 分工 8 生产者 warp + 1 消费者 warp、握手协议形态）与 B9c 逐位相同 ⇒ 单变量对照。
//   环深 3 stage（3 × 32 KB = 96 KB）+ 16 KB A = 112 KB 动态 smem（occ 仍为 1 块/SM，与 B9c 相同）。
//   F12 尾流公式（与 B9c 同形，只是单位由「块」变「stage」）：消费者第 j 个 stage 的 full[] 到齐通知
//   发生在生产者第 n = j + (STAGES-2) 轮 ⇒ 生产者跑 nstages + (STAGES-2) 轮，尾部不再搬数据。
// flags：bit0=1 发 MMA；bit1=2 写痕迹（mk[1]=生产轮数 / mk[2]=消费 stage 数）；bit2=4 每 warp 1 次 arrive
//   （full[] 计数 = 8 = 生产者 warp 数）而不是每线程 1 次（256）。计时臂只用 bit0。
#include "c58_v5_fp4_mma.cu"
#include "t4_smem_layout.h"   // 动态 smem 尺寸单一来源（事故 #5 后强制）
#include "t4_prodaddr_layout.h"   // H-mem 臂的生产寻址参数单一来源（CPR/ROW_STRIDE/b_smem_off）

namespace v9f {
using namespace v5;

constexpr int KT = 256;
constexpr int NSUB = KT / K;                 // 4
constexpr int A_SUB = 4096;                  // M*K/2（一个 K=64 子步）
constexpr int B_SUB = N * K / 2;             // 1024
constexpr int A_TILE = A_SUB * NSUB;         // 16384
constexpr int B_CHUNK = B_SUB * NSUB;        // 4096（一个 K=256 块）
constexpr int CPS = T4_V9F_CPS;              // 8 块/stage
constexpr int B_STAGE = B_CHUNK * CPS;       // 32768
constexpr int STAGES = T4_V9F_STAGES;        // 3
constexpr int NPROD = T4_V9F_NPROD;          // 256
constexpr int NTHR = T4_V9F_NTHR;            // 288
constexpr int OFF_A = 0;
constexpr int OFF_B = A_TILE;
constexpr int OFF_MFULL = OFF_B + STAGES * B_STAGE;
constexpr int OFF_MEMPTY = OFF_MFULL + 8 * STAGES;
constexpr int OFF_MBAR = OFF_MEMPTY + 8 * STAGES;
constexpr int OFF_TADDR = OFF_MBAR + 8;
constexpr uint32_t BOUND = 200000000u;       // mbarrier 有界自旋（超时→break，不无限挂）
constexpr uint32_t MK9F_STREAM = 9;
constexpr int NBW_WARP = NPROD / 32;         // 8：每 warp 1 次 arrive 时的 full[] 计数

static_assert(OFF_TADDR + 8 == (int)T4_V9F_SMEM_REQUIRED, "Offsets must match the single-source smem header");

__device__ __forceinline__ unsigned long long gtimer_ns() {
  unsigned long long v;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v));
  return v;
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
__device__ __forceinline__ void mbar_arrive(uint32_t mbar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(mbar));
}
// 痕迹槽（宿主映射内存，sys-scope release ⇒ 宿主可读）；slot>=1，mk[0] 留给 mark()
__device__ __forceinline__ void mark_slot(uint32_t* mk, int slot, uint32_t code) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(mk + slot), "r"(code));
}
__device__ __forceinline__ void cp_async16(uint32_t smem, const void* g) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem), "l"(g));
}

// ---- R-B2 判别臂：只改 B 的**拷贝粒度**（地址逐字节相同），不改环/不改消费者 ----
//   0 = 归档口径（cp.async 16 B/线程，256 条/4KB chunk）
//   1 = 生产腿的做法（cp.async 4 B/线程 ⇒ 1024 条/4KB chunk；腿的注释：36 B 记录不是 16 B 对齐）
#ifndef T4_V9F_GRAN4
#define T4_V9F_GRAN4 0
#endif
__device__ __forceinline__ void cp_async4_g(uint32_t smem, const void* g) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(smem), "l"(g));
}
__device__ __forceinline__ void cp_b16(uint32_t smem, const void* g) {
#if T4_V9F_GRAN4
#pragma unroll
  for (int u = 0; u < 4; ++u)
    cp_async4_g(smem + (uint32_t)(4 * u), (const unsigned char*)g + 4 * u);
#else
  cp_async16(smem, g);
#endif
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }

// ---- R-B2 H-mem 判别臂：在**同一份源码**里加第三个变体，只改 B 的**全局取数寻址**（生产口径）----
//   0 = 归档/线性口径（gran 臂用的那个：src + k*B_CHUNK + t*16）
//   1 = 生产寻址：行主序真实张量，行内每 64-K 记录 36 B = [4 B sf][32 B nib]，行步长 row_stride；
//       板内 32 行 × 每行 8 个 4 B 单元；t → (u = t&7, row = (t>>3)&31)，
//       源地址 = row*row_stride + cc*144 + r*36 + 4 + 4u（与 t4_mmq_ring.cuh:352-366 逐字同构）。
//       smem 落点仍走 canonical（= gran1 落点），故数值判据不变（d_mismatch 必须仍为 0）。
//   事实来源：model-nvfp4.gguf 头部直读（2026-09-17，/tmp/gguf_dims.py）
//     type40 NVFP4: K=5120 → row_stride=2880 = 20*144（无补零），128 张量 / 6120 MiB（占 NVFP4 字节 62%）
//                   K=17408 → row_stride=9792 = 68*144，64 张量 / 3060 MiB
//     CARD §7 原写「行距 8704」= 取自 q8 KV smem 预算，**与本模型无关，已作废**。
#ifndef T4_V9F_PRODADDR
#define T4_V9F_PRODADDR 0
#endif
// 生产口径搬一个 chunk（4096 B payload）：256 线程各 4 B × 4 子块 = 1024 请求（粒度与 gran1 相同）
__device__ __forceinline__ void cp_chunk_prod(unsigned char* d, const unsigned char* src, int t) {
  const int u = t & 7, row = (t >> 3) & 31;
  const unsigned char* s2 = src + (size_t)row * (size_t)t4prod::ROW_STRIDE + t4prod::src_off_in_rec(u);
  unsigned char* d2 = d + t4prod::b_smem_off(row, u);
#pragma unroll
  for (int r = 0; r < NSUB; ++r) cp_async4_g(cvta_smem(d2 + r * B_SUB), s2 + (size_t)r * t4prod::REC);
}
template <int NPG>
__device__ __forceinline__ void cp_wait_group() {
  asm volatile("cp.async.wait_group %0;" ::"n"(NPG));
}
__device__ __host__ __forceinline__ uint32_t pack4_sfa(int m, int kb0) {
  uint32_t w = 0;
  for (int i = 0; i < 4; ++i) w |= (uint32_t)genSFA(m, kb0 + i) << (8 * i);
  return w;
}
__device__ __host__ __forceinline__ uint32_t pack4_sfb(int n, int kb0) {
  uint32_t w = 0;
  for (int i = 0; i < 4; ++i) w |= (uint32_t)genSFB(n, kb0 + i) << (8 * i);
  return w;
}
__device__ __host__ __forceinline__ float ref_D_kt(int m, int n) {
  float acc = 0.f;
  for (int k = 0; k < KT; ++k)
    acc += fp4_val(genA_nib(m, k)) * fp4_val(genB_nib(n, k)) *
           sf_val(genSFA(m, k / 16)) * sf_val(genSFB(n, k / 16));
  return acc;
}

extern "C" __global__ void __launch_bounds__(NTHR) k_b9f_fp4_ws32k(
    const unsigned char* __restrict__ gB, unsigned chunks, float* D, uint32_t* mk, unsigned* diag,
    unsigned long long* ts, unsigned flags) {
  if (T4_V9F_SMEM_REQUIRED > (int)(T4_V9F_SMEM_HOST_REQ)) return;   // 契约守护（事故 #5 后强制）
  const bool do_mma = (flags & 1u) != 0u;
  const bool trace = (flags & 2u) != 0u;               // bit1：只写痕迹，不改协议
  const bool warp_arrive = (flags & 4u) != 0u;         // bit2：每 warp 1 次 arrive
  extern __shared__ __align__(1024) unsigned char smem[];          // 动态 smem（>48 KB 必须）
  const int t = threadIdx.x, warp = t / 32, blk = blockIdx.x;
  unsigned char* smA = smem + OFF_A;
  unsigned char* smB = smem + OFF_B;
  const uint32_t mfull = cvta_smem(smem + OFF_MFULL);
  const uint32_t membty = cvta_smem(smem + OFF_MEMPTY);
  const uint32_t mbar = cvta_smem(smem + OFF_MBAR);
  unsigned char* slot = smem + OFF_TADDR;
  const uint32_t n_arrive = warp_arrive ? (uint32_t)NBW_WARP : (uint32_t)NPROD;

  if (t == 0) {
    mark(mk, (uint32_t)MK_START);
    *(volatile uint32_t*)slot = TADDR_POISON;
    mbar_init(mbar, 1u);
    for (int i = 0; i < STAGES; ++i) {
      mbar_init(mfull + 8 * i, n_arrive);
      mbar_init(membty + 8 * i, 1u);
    }
  }
  __syncthreads();

  // 1) A（激活，常驻）：4 个 K=64 子块，各按 canonical fp4 布局（与 B7/B9c 同）
  for (int i = t; i < A_TILE; i += blockDim.x) {
    const int s = i / A_SUB, j = i % A_SUB, m = j / (K / 2), p = j % (K / 2);
    smA[s * A_SUB + off_fp4(m, 2 * p)] =
        (uint8_t)(genA_nib(m, 64 * s + 2 * p) | (genA_nib(m, 64 * s + 2 * p + 1) << 4));
  }
  proxy_fence_shared();
  __syncthreads();

  // 2) TMEM 分配（warp0 全活跃收敛；F1）
  if (warp == 0) { tmem_alloc(cvta_smem(slot), NCOLS); tmem_relinquish(); }
  __syncthreads();
  const uint32_t taddr = *(volatile uint32_t*)slot;
  const bool taddr_bad = (taddr == TADDR_POISON) || ((taddr >> 16) != 0u) ||
                         ((taddr & 0xFFFFu) >= TADDR_NCOLS_MAX) || ((taddr & 31u) != 0u);
  if (t == 0) {
    int st = 0;
    if (!taddr_bad) st |= ST_TADDR_OK;
    diag[DG_STATUS] = (unsigned)st;
    diag[DG_SMEM_A] = (unsigned)cvta_smem(smA);
    diag[DG_SMEM_B] = (unsigned)cvta_smem(smB);
    mark(mk, (uint32_t)MK_TADDR);
  }

  // 3) SF（每 K=64 步基址 +4 列 = F6；SFB 复制到全 128 lane = F5）
  if (!taddr_bad && warp < 4) {
    const int m = warp * 32 + (t % 32);
    for (int s = 0; s < NSUB; ++s)
      tmem_st_32x32b_x1(taddr + ((uint32_t)(warp * 32) << 16) + COL_SFA + 4u * s + (uint32_t)warp,
                        pack4_sfa(m, 4 * s));
    const int n = t % 32;
    for (int s = 0; s < NSUB; ++s)
      tmem_st_32x32b_x1(taddr + ((uint32_t)(warp * 32) << 16) + COL_SFB + 4u * s, pack4_sfb(n, 4 * s));
    tmem_wait_st();
  }
  fence_before_thread_sync();
  __syncthreads();
  if (t == 0) mark(mk, (uint32_t)MK_SF);

  // 4) 描述符（每块每 K=64 子步一个；B 随 stage/块变）——所有线程算也无害
  uint64_t ad[NSUB], bd[STAGES][CPS][NSUB];
  if (!taddr_bad) {
    for (int s = 0; s < NSUB; ++s) ad[s] = sdesc_via_cutlass<M>(smA + s * A_SUB);
    for (int st = 0; st < STAGES; ++st)
      for (int cp = 0; cp < CPS; ++cp)
        for (int s = 0; s < NSUB; ++s)
          bd[st][cp][s] = sdesc_via_cutlass<N>(smB + st * B_STAGE + cp * B_CHUNK + s * B_SUB);
  }

  // PRODADDR=1 时源缓冲按「n-tile（32 行 × row_stride）」排布，每块占 ceil(chunks/CPR) 个 tile
  const unsigned char* gB_blk =
#if T4_V9F_PRODADDR
      gB + (size_t)blk * (size_t)((chunks + (unsigned)t4prod::CPR - 1u) / (unsigned)t4prod::CPR) *
               (size_t)t4prod::TILE_BYTES;
#else
      gB + (size_t)blk * (size_t)chunks * B_CHUNK;
#endif
  const unsigned nstages = (chunks + (unsigned)CPS - 1u) / (unsigned)CPS;
  unsigned long long t0 = 0, t1 = 0;
  uint32_t ok_all = 1;
  if (!taddr_bad) {
    if (t >= NPROD) {
      // ---------------- 消费者（warp8）：等 full[st] → 发 used×4 个 K64 MMA → 通知 empty[st] ----------------
      //   依赖距离 = 一整轮（STAGES 个 stage）⇒ 生产者等 empty[st] 时对应 MMA 早已完成，
      //   不会把「MMA 完成」压进关键路径（B8 首版正是栽在这一点：6.72 GB/s）。
      t0 = gtimer_ns();
      if (t == NPROD) mark(mk, (uint32_t)MK9F_STREAM);
      unsigned fails = 0;
      for (unsigned j = 0; j < nstages; ++j) {
        const int st = (int)(j % (unsigned)STAGES);
        const unsigned round = j / (unsigned)STAGES;
        if (!mbar_wait_bounded_p(mfull + 8 * st, BOUND, round & 1u)) { fails++; break; }
        if (trace && t == NPROD) mark_slot(mk, 2, j + 1u);
        const unsigned c0 = j * (unsigned)CPS;
        const int used = (int)((c0 + (unsigned)CPS <= chunks) ? (unsigned)CPS : (chunks - c0));
        if (do_mma) {
          if (cute::elect_one_sync()) {
            for (int cp = 0; cp < used; ++cp)
              for (int s = 0; s < NSUB; ++s)
                mma_mxf4nvf4(taddr + COL_D, ad[s], bd[st][cp][s], IDESC_B5, taddr + COL_SFA + 4u * s,
                             taddr + COL_SFB + 4u * s, (s == 0) ? 0u : 1u);
            commit_mbar(membty + 8 * st);   // 该 stage 的 MMA 全完成 → 生产者可覆盖
          }
        } else {
          if (cute::elect_one_sync()) mbar_arrive(membty + 8 * st);   // 纯拷贝臂也必须放行生产者
        }
      }
      if (do_mma && cute::elect_one_sync()) {
        commit_mbar(mbar);
        ok_all = (mbar_wait_bounded_p(mbar, BOUND, 0u) && fails == 0u) ? 1u : 0u;
      } else {
        ok_all = (fails == 0u) ? 1u : 0u;
      }
      t1 = gtimer_ns();
    } else {
      // ---------------- 生产者（warp0..7）：等 empty[st] → CPS 块 cp.async → commit → wait → 通知 full ----------------
      //   cp.async.wait_group<S-2> 保证「最老的若干组已落地」⇒ n 轮时能确认的是 stage (n-(S-2))%S，
      //   而 (n-(S-2))%S == (n+2)%S ⇒ arrive 落在 full[(n+2)%S]。**不能落在刚发出的 n 上**。
      //   ★ F12：消费者第 j 个 stage 的通知发生在 n = j + (S-2) ⇒ 必须跑 nstages + (S-2) 轮，
      //     尾部 (S-2) 轮不再搬数据（避免越界读 gB），只做「等全落地 → arrive」。
      unsigned fails = 0;
      const unsigned n_iter = nstages + (unsigned)(STAGES - 2);
      for (unsigned n = 0; n < n_iter; ++n) {
        const int st = (int)(n % (unsigned)STAGES);
        const unsigned round = n / (unsigned)STAGES;
        if (n < nstages) {
          if (round > 0 && !mbar_wait_bounded_p(membty + 8 * st, BOUND, (round - 1u) & 1u)) {
            fails++;
            break;
          }
          const unsigned c0 = n * (unsigned)CPS;
          unsigned char* dst = smB + (size_t)st * B_STAGE;
#if T4_V9F_PRODADDR
          // 生产寻址：chunk ci → tile = ci/CPR（32 行板），cc = ci%CPR；块内不做线性前缀
          //   与归档口径**同结构**（满 stage / 尾 stage 两支）⇒ SASS 的 LDGSTS 计数可与 gran1 逐条对照
          if (c0 + (unsigned)CPS <= chunks) {                 // 满 stage（计时臂走这条）
#pragma unroll
            for (int k = 0; k < CPS; ++k) {
              const unsigned ci = c0 + (unsigned)k;
              const unsigned char* src = gB_blk +
                  (size_t)(ci / (unsigned)t4prod::CPR) * (size_t)t4prod::TILE_BYTES +
                  (size_t)(ci % (unsigned)t4prod::CPR) * (size_t)t4prod::CHUNK_ROWS;
              cp_chunk_prod(dst + (size_t)k * B_CHUNK, src, t);
            }
          } else {                                            // 尾 stage 可能不满：只搬存在的块
            for (int k = 0; k < CPS && c0 + (unsigned)k < chunks; ++k) {
              const unsigned ci = c0 + (unsigned)k;
              const unsigned char* src = gB_blk +
                  (size_t)(ci / (unsigned)t4prod::CPR) * (size_t)t4prod::TILE_BYTES +
                  (size_t)(ci % (unsigned)t4prod::CPR) * (size_t)t4prod::CHUNK_ROWS;
              cp_chunk_prod(dst + (size_t)k * B_CHUNK, src, t);
            }
          }
#else
          const unsigned char* src = gB_blk + (size_t)c0 * B_CHUNK;
          if (c0 + (unsigned)CPS <= chunks) {                 // 满 stage（计时臂走这条）
#pragma unroll
            for (int k = 0; k < CPS; ++k)
              cp_b16(cvta_smem(dst + k * B_CHUNK + t * 16), src + k * B_CHUNK + t * 16);
          } else {                                            // 尾 stage 可能不满：只搬存在的块
            for (int k = 0; k < CPS && c0 + (unsigned)k < chunks; ++k)
              cp_b16(cvta_smem(dst + k * B_CHUNK + t * 16), src + k * B_CHUNK + t * 16);
          }
#endif
          cp_commit();
        }
        if (n >= (unsigned)(STAGES - 2)) {
          if (n < nstages) cp_wait_group<STAGES - 2>();
          else            cp_wait_group<0>();                  // 尾流：无新提交，等全部 cp.async 落地
          const int st_done = (int)((n + 2u) % (unsigned)STAGES);
          if (!warp_arrive || (t & 31u) == 0u) mbar_arrive(mfull + 8 * st_done);
        }
        if (trace && t == 0) mark_slot(mk, 1, n + 1u);
      }
      if (t == 0 && fails != 0u) diag[12] = 0xFFFFFFFFu;   // 生产者侧超时（宿主读 prod_fail）
    }
  } else {
    __syncthreads();
  }
  // 主循环里没有任何 __syncthreads ⇒ 全线程在同一 PC 收敛
  __syncthreads();
  if (t == NPROD) {
    diag[11] = ok_all;
    ts[blk * 2 + 0] = t0;
    ts[blk * 2 + 1] = t1;
    mark(mk, (uint32_t)MK_DONE);
  }
  fence_after_thread_sync();
  __syncthreads();

  // 5) 读回 D（lane=m，列=n）
  uint32_t r[32];
  for (int i = 0; i < 32; ++i) r[i] = 0u;
  if (!taddr_bad && warp < 4) {
    tmem_ld_32x32b_x32(taddr + ((uint32_t)(warp * 32) << 16) + COL_D, r);
    tmem_wait_ld();
  }
  __syncthreads();
  if (!taddr_bad && D && warp < 4)
    for (int n = 0; n < N; ++n)
      D[(size_t)blk * M * N + (warp * 32 + (t % 32)) * N + n] = __uint_as_float(r[n]);
  __threadfence();
  __syncthreads();

  // 6) 收官 + dealloc（warp0 收敛，F1）
  if (warp == 0) {
    const unsigned ok_u = __shfl_sync(0xFFFFFFFFu, taddr_bad ? 0u : 1u, 0);
    if (ok_u) tmem_dealloc(taddr, NCOLS);
    if (t == 0) mark(mk, (uint32_t)MK_DEALLOC);
  }
}

// 参考件：**纯读带宽天花板**（与 B7/B9 同件，用于同会话标尺）
extern "C" __global__ void __launch_bounds__(256) k_b9f_readbw(
    const uint4* __restrict__ p, unsigned long long n16, unsigned long long* out) {
  unsigned long long acc = 0;
  const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
  for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n16;
       i += stride) {
    const uint4 v = p[i];
    acc += (unsigned long long)(v.x ^ v.y ^ v.z ^ v.w);
  }
  if (acc == 0xDEADBEEFDEADBEEFull) out[0] = acc;
}
}  // namespace v9f
