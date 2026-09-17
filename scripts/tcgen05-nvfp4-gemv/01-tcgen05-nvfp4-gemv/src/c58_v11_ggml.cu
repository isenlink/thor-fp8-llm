// c58_v11_ggml.cu — T4-4 原型件 **v11：生产者直接吃真实 GGML 张量（零预处理）**
//
// 血统：v10r（evidence/20260916-77）——结构、warp 分工、环深协议、F12 尾流、F14 握手单元全部照抄。
//   v10r 吃的是**预先重排好的 canonical 分块流**（mk_stream 产物）；v11 把这一层预处理去掉：
//     ① **B 侧纯拷贝**：源 = 真实 GGML NVFP4 张量字节（行主序，每行每 64-K 块 36 B = 4 B 缩放 + 32 B nibble），
//        16 B granule 原样搬进 canonical smem 槽、4 B 缩放字原样搬进 sf 区（判据 results/20260916-109：
//        canonical 的**片段边界**与 ggml 一致，逐 nibble 全等）。
//     ② **A 侧改用 ggml 落位**：canonical 与 ggml 在 16 B granule 内部的 nibble 归属不同
//        （canonical：字节 i ↔ 元素 2i/2i+1；ggml：字节 j ↔ 元素 j/j+8，见 §2b）。
//        内核算 Σ_slot A[slot]·B[slot] ⇒ **只要两侧同约定，D 就是真值**（点积对索引置换不变）。
//        ⇒ 走纯拷贝 + A 同约定，两侧都不需要 nibble 重排。
//   ⚠ 这两处必须**成对改动**：只改一侧 ⇒ f_A ≠ f_B ⇒ D 全错。
//   ⚠ 缩放路径保持 v10r 原样（warp0 独占搬、bar_sync_sf 交接）：4 个 warp 读同一份 sf，1× 读取。
//   ⇒ 判据：与 canonical 流同源的宿主参考 `host/verify_t4_4.py d` 必须 mism=0/4096。
//
// 不变量（照抄 B9f/TECH §10）：arrive ∈ full[(n+2)%S]；生产侧跑 nstages+(S-2) 轮，尾部不搬数据。
#include "c58_v5_fp4_mma.cu"
#include "t4_smem_layout.h"

namespace v11 {
using namespace v5;

constexpr int KT = 256;
constexpr int NSUB = KT / K;                 // 4 个 K=64 子步
constexpr int A_SUB = 4096;                  // M*K/2
constexpr int B_SUB = N * K / 2;             // 1024（一个 K=64 的 canonical 板）
constexpr int A_TILE = A_SUB * NSUB;         // 16384
constexpr int QS_CHUNK = B_SUB * NSUB;       // 4096
constexpr int SF_SUB = 128;                  // 32 n × 4 B
constexpr int SF_CHUNK = SF_SUB * NSUB;      // 512
constexpr int B_CHUNK = QS_CHUNK + SF_CHUNK; // 4608
constexpr int CPS = T4_V10R_CPS;             // 8
constexpr int B_STAGE = B_CHUNK * CPS;       // 36864
constexpr int STAGES = T4_V10R_STAGES;       // 3
constexpr int NPROD = T4_V10R_NPROD;         // 256
constexpr int NTHR = T4_V10R_NTHR;           // 288
constexpr int OFF_A = 0;
constexpr int OFF_B = A_TILE;
constexpr int OFF_MFULL = OFF_B + STAGES * B_STAGE;
constexpr int OFF_MEMPTY = OFF_MFULL + 8 * STAGES;
constexpr int OFF_MBAR = OFF_MEMPTY + 8 * STAGES;
constexpr int OFF_TADDR = OFF_MBAR + 8;
// ★ 83：epilogue 握手（每个 n-tile 一组）
// ★ 84：**每个 D 缓冲一个 mbarrier**（epi_rdy[0..1] / epi_done[0..1]）——
//   单 mbarrier 时「生产者已连续落盘 2 个边界」会让相位奇偶回到同值 ⇒ 消费者的 parity 判据永久假（实测死等）。
//   每个缓冲一个后，等待方与到达方在同一 barrier 上的相位差 ≤1 ⇒ 奇偶判据无歧义（见 results/20260916-84）。
constexpr int OFF_EPI_RDY  = OFF_TADDR + 8;          // 消费者 commit → 生产者（MMA 完成 ⇒ D 可读）
constexpr int OFF_EPI_DONE = OFF_EPI_RDY + 16;       // 生产者读回 D → 消费者（可以复位覆盖 D）
constexpr uint32_t BOUND = 200000000u;
// ★ TMEM 列布局（F15：SF 基址按 **4 列**步进 —— B5/B9f 的 +4 约定；用 +1 会让 sf_b_tmem 基址非 4 对齐，
//   实测后果 = MMA 停摆 + 生产者的 tcgen05.st 阻塞 ⇒ 整核死锁（见 results/20260916-77 §3）。
//   每个 (chunk, K64 子步) 占 4 列、只用其中第 1 列（与 B9f 相同），故每 stage = CPS*NSUB*4 列。
// ★ 79/80：SF 区整体下移（SFA 256→32、SFB 272→48）。旧约束「SFB 必须 ≥ 272」源自 B5/B9f 时期，
//   当时的「低基址打坏 lane group 1..3」极可能是 F16 拷贝竞态的误归因（见 results/20260916-78 §3）。
//   下移的目的：释放 TMEM 列预算 ⇒ 允许更深/更宽的环（CPS=8/STAGES=3 或 CPS=5/STAGES=5）。
// ★ 83c：D 双缓冲（每个 n-tile 交替写 D0/D1，epilogue 有整整一个 n-tile 的余量）⇒ 边界可以落在 stage 中间
constexpr uint32_t COL_D0      = 0u;        // D 缓冲 A（N=32 列）
constexpr uint32_t COL_D1      = 32u;       // D 缓冲 B
constexpr uint32_t COL_SFA     = 64u;
constexpr uint32_t COL_SFB_ST  = 80u;
constexpr uint32_t SF_COLS_STEP = 4u;       // 每个 (cp,s) 的列步进
constexpr uint32_t SF_COLS_STAGE = (uint32_t)(CPS * NSUB) * SF_COLS_STEP;   // 128
constexpr uint32_t MK10_STREAM = 9;
constexpr int NBW_WARP = NPROD / 32;

static_assert(OFF_EPI_DONE + 16 == (int)T4_V10R_SMEM_REQUIRED, "Offsets must match the single-source smem header");
static_assert((int)(COL_SFB_ST + STAGES * SF_COLS_STAGE) <= (int)TADDR_NCOLS_MAX, "SFB columns must fit TMEM");
static_assert(COL_SFA % 4u == 0u && COL_SFB_ST % 4u == 0u, "SF bases must be 4-column aligned");
static_assert(COL_SFA >= COL_D1 + 32u && COL_SFB_ST >= COL_SFA + 16u, "regions must not overlap D0/D1/SFA");

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
// 痕迹槽（宿主映射内存，sys-scope release）；slot>=1，mk[0] 留给 mark()
__device__ __forceinline__ void mark_slot(uint32_t* mk, int slot, uint32_t code) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(mk + slot), "r"(code));
}
__device__ __forceinline__ void cp_async16(uint32_t smem, const void* g) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem), "l"(g));
}
__device__ __forceinline__ void cp_async4(uint32_t smem, const void* g) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(smem), "l"(g));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
// 生产者内部的 128 线程命名屏障（只用 warp0..3；id=1 避开 __syncthreads 的 id=0）
// 用途：跨 warp 的 smem 交接 —— cp.async 的完成只对「发起它的线程」可见（results/20260916-78 §7.2）
__device__ __forceinline__ void bar_sync_sf() {
  asm volatile("bar.sync 1, 128;" ::: "memory");
}
// 生产者内部的第二个命名屏障（id=2）：epilogue 的 D 回读（仅 warp0..3）
__device__ __forceinline__ void bar_sync_epi() {
  asm volatile("bar.sync 2, 128;" ::: "memory");
}
#ifndef T4_V10R_NO_DREAD
// ★ 83：D 回读 + 落盘（lane=m，列=n）。loop 内 epilogue 与收尾共用同一份实现。
__device__ __forceinline__ void epilogue_store_d(uint32_t dtmem, float* D, unsigned blk, int warp,
                                                 int t) {
  uint32_t r[32];
  for (int i = 0; i < 32; ++i) r[i] = 0u;
  if (warp < 4) {
    tmem_ld_32x32b_x32(dtmem + ((uint32_t)(warp * 32) << 16), r);
    tmem_wait_ld();
  }
  // ★ 83b：D 落盘用 **N 主序**（D_T[n*M + m]）——同一指令内 32 条 lane 的 m 连续 ⇒ 合并写；
  //   行主序时相邻 lane 相隔 N*4=128 B（每 lane 一个 sector）⇒ 8× 写放大，实测每次 epilogue 5.5 µs。
  if (D && warp < 4)
    for (int n = 0; n < N; ++n)
      D[(size_t)blk * M * N + (size_t)n * M + (warp * 32 + (t % 32))] = __uint_as_float(r[n]);
}
#endif
template <int NPG>
__device__ __forceinline__ void cp_wait_group() {
  asm volatile("cp.async.wait_group %0;" ::"n"(NPG));
}
__device__ __host__ __forceinline__ uint32_t pack4_sfa(int m, int kb0) {
  uint32_t w = 0;
  for (int i = 0; i < 4; ++i) w |= (uint32_t)genSFA(m, kb0 + i) << (8 * i);
  return w;
}

extern "C" __global__ void __launch_bounds__(NTHR) k_b11_ggml(
    const unsigned char* __restrict__ gB, unsigned chunks, float* D, uint32_t* mk, unsigned* diag,
    unsigned long long* ts, unsigned flags, float* sfdump, unsigned cpr,
    unsigned row_stride) {
  if (T4_V10R_SMEM_REQUIRED > (int)(T4_V10R_SMEM_HOST_REQ)) return;   // 契约守护（事故 #5 后强制）
  const bool do_mma = (flags & 1u) != 0u;
  const bool trace = (flags & 2u) != 0u;
  const bool warp_arrive = (flags & 4u) != 0u;
  const bool dbg_sf = (flags & 8u) != 0u;      // bit3：回读 SFA/SFB 的 TMEM 内容（诊断用）
  extern __shared__ __align__(1024) unsigned char smem[];
  const int t = threadIdx.x, warp = t / 32, blk = blockIdx.x;
  unsigned char* smA = smem + OFF_A;
  unsigned char* smB = smem + OFF_B;
  const uint32_t mfull = cvta_smem(smem + OFF_MFULL);
  const uint32_t membty = cvta_smem(smem + OFF_MEMPTY);
  const uint32_t mbar = cvta_smem(smem + OFF_MBAR);
  const uint32_t epi_rdy = cvta_smem(smem + OFF_EPI_RDY);
  const uint32_t epi_done = cvta_smem(smem + OFF_EPI_DONE);
  // ★ 83：内核内 epilogue（每 cpr 个 chunk 一个 n-tile）——要求 cpr 是 CPS 的整数倍（边界落在 stage 边界上）
  // ★ 83c：D 双缓冲后不再要求 cpr 是 CPS 的整数倍（边界可落在 stage 中间）；
  //   只要求块内 chunk 数是 cpr 的整数倍（否则最后一个残块没有 epilogue）。
  const bool epi_on = (cpr != 0u) && (chunks % cpr == 0u) && (chunks >= 2u * cpr);
  unsigned epi_count = 0u;
  unsigned char* slot = smem + OFF_TADDR;
  const uint32_t n_arrive = warp_arrive ? (uint32_t)NBW_WARP : (uint32_t)NPROD;

  if (t == 0) {
    mark(mk, (uint32_t)MK_START);
    diag[13] = 0u;   // ★ 83：epilogue 计数先清零（否则 CPR=0 时读到未初始化值）
    *(volatile uint32_t*)slot = TADDR_POISON;
    mbar_init(mbar, 1u);
    for (int i = 0; i < 2; ++i) { mbar_init(epi_rdy + 8u * i, 1u); mbar_init(epi_done + 8u * i, 1u); }
    for (int i = 0; i < STAGES; ++i) {
      mbar_init(mfull + 8 * i, n_arrive);
      mbar_init(membty + 8 * i, 1u);
    }
  }
  __syncthreads();

  // 1) A（激活，常驻，合成；周期 256 与参考同源）
#ifndef T4_V10R_NO_ASY      // 仅用于「每块固定开销分解」的对照臂（编译期开关；定版不带此宏）
  for (int i = t; i < A_TILE; i += blockDim.x) {
    const int s = i / A_SUB, j = i % A_SUB, m = j / (K / 2), p = j % (K / 2);
    // ★ v11：**ggml 落位**（与 B 侧纯拷贝同约定）——旧 canonical 是「字节 i ↔ 元素 2i/2i+1」；
    //   ggml 是「字节 j ↔ 元素 j（低半字节）/ j+8（高半字节）」。地址算式不变，只换元素来源。
    const int bb = p & 15, gg = p >> 4;
    const int e0 = 64 * s + 32 * gg + bb + ((bb >= 8) ? 8 : 0);
    smA[s * A_SUB + off_fp4(m, 2 * p)] =
        (uint8_t)(genA_nib(m, e0) | (genA_nib(m, e0 + 8) << 4));
  }
#endif
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

  // 3) SFA（A 常驻 ⇒ 与 B9f 同：4 个子步 × 4 个 m 组 = 16 列，全程不变；SFB 由生产者在环里按 stage 写）
  if (!taddr_bad && warp < 4) {
    const int m = warp * 32 + (t % 32);
    for (int s = 0; s < NSUB; ++s)
      tmem_st_32x32b_x1(taddr + ((uint32_t)(warp * 32) << 16) + COL_SFA + 4u * s + (uint32_t)warp,
                        pack4_sfa(m, 4 * s));
    tmem_wait_st();
  }
  __syncthreads();
  if (t == 0) mark(mk, (uint32_t)MK_SF);

  // 4) 描述符
  uint64_t ad[NSUB], bd[STAGES][CPS][NSUB];
  if (!taddr_bad) {
    for (int s = 0; s < NSUB; ++s) ad[s] = sdesc_via_cutlass<M>(smA + s * A_SUB);
    for (int st = 0; st < STAGES; ++st)
      for (int cp = 0; cp < CPS; ++cp)
        for (int s = 0; s < NSUB; ++s)
          bd[st][cp][s] = sdesc_via_cutlass<N>(smB + st * B_STAGE + cp * B_CHUNK + s * B_SUB);
  }

  // ★ v11：每个 block 吃真实张量的**一个 32 行 tile**（行主序，行步长 row_stride 字节）
  const unsigned char* gB_tile = gB + (size_t)blk * 32u * (size_t)row_stride;
  const unsigned nstages = (chunks + (unsigned)CPS - 1u) / (unsigned)CPS;
  unsigned long long t0 = 0, t1 = 0;
  uint32_t ok_all = 1;
  if (!taddr_bad) {
    if (t >= NPROD) {
      // ---------------- 消费者（warp8）----------------
      t0 = gtimer_ns();
#ifdef T4_V10R_CYC
      const unsigned long long _ct0 = (t == NPROD) ? clock64() : 0ull;
#endif
      if (t == NPROD) mark(mk, (uint32_t)MK10_STREAM);
      unsigned fails = 0;
      for (unsigned j = 0; j < nstages; ++j) {
        const int st = (int)(j % (unsigned)STAGES);
        const unsigned round = j / (unsigned)STAGES;
#ifdef T4_V10R_CYC
        const unsigned long long _cw0 = clock64();
#endif
        if (!mbar_wait_bounded_p(mfull + 8 * st, BOUND, round & 1u)) { fails++; break; }
#ifdef T4_V10R_CYC
        if (t == NPROD) diag[7] += (unsigned)(clock64() - _cw0);
#endif
        fence_after_thread_sync();          // 与生产者侧的 fence::before_thread_sync 配对（缩放经 TMEM 传递）
        if (trace && t == NPROD) mark_slot(mk, 2, j + 1u);
        const unsigned c0 = j * (unsigned)CPS;
        const int used = (int)((c0 + (unsigned)CPS <= chunks) ? (unsigned)CPS : (chunks - c0));
        bool cp_fail = false;
#ifdef T4_V10R_CYC
        const unsigned long long _ci0 = (t == NPROD) ? clock64() : 0ull;
#endif
        if (do_mma) {
          if (cute::elect_one_sync()) {
            // ★ 84c：把 n-tile 序号/残差做成**每 stage 一次的递增计数器**，内层循环里不再做 32 位除法
            //   （`g/cpr`、`g%cpr` 在 780 次内层迭代里各算一次，实测拖慢内层 MMA 发射 ~2 倍）。
            unsigned k = epi_on ? (c0 / cpr) : 0u;      // 本 stage 首个 chunk 的 n-tile 序号
            unsigned rem = epi_on ? (c0 % cpr) : 0u;    // 本 stage 首个 chunk 在 n-tile 内的偏移
            for (int cp = 0; cp < used; ++cp) {
              const unsigned g = c0 + (unsigned)cp;                 // 块内 chunk 序号（仅跟踪/诊断用）
              if (epi_on && rem == 0u && k >= 2u) {
                // ★ 84：D 双缓冲 ⇒ 复用同一块 D 之前，上一次（k-2）的 epilogue 必须已完成；
                //   等待对象 = k-2 号 D 缓冲自己的 mbarrier，相位 = (k-2) 在该 barrier 上的第几次到达。
                const unsigned wb = k - 2u;
#ifdef T4_V10R_CYC
                const unsigned long long _ce0 = clock64();
#endif
                if (!mbar_wait_bounded_p(epi_done + 8u * (wb & 1u), BOUND, (wb >> 1) & 1u)) { fails++; cp_fail = true; break; }
#ifdef T4_V10R_CYC
                if (t == NPROD) diag[5] += (unsigned)(clock64() - _ce0);
#endif
                fence_after_thread_sync();
              }
#ifdef T4_V10R_FIXD      // 仅计时探针：accumulator 固定在同一块 D（数值必错）
              const uint32_t dtm = taddr + COL_D0;
#else
              const uint32_t dtm = taddr + (epi_on ? ((k & 1u) ? COL_D1 : COL_D0) : COL_D0);
#endif
              for (int s = 0; s < NSUB; ++s)
                mma_mxf4nvf4(dtm, ad[s], bd[st][cp][s], IDESC_B5,
                             taddr + COL_SFA + 4u * s,
                             taddr + COL_SFB_ST + (uint32_t)st * SF_COLS_STAGE +
                                 (uint32_t)(cp * NSUB + s) * SF_COLS_STEP,
                             epi_on ? ((rem == 0u && s == 0) ? 0u : 1u)
                                    : ((j == 0u && cp == 0 && s == 0) ? 0u : 1u));   // ★ 跨 chunk 累加 / 每 n-tile 复位
#ifdef T4_V10R_PLAIN_RDY  // 仅计时探针：用普通 arrive 代替 tcgen05.commit（数值必错）
              if (epi_on && rem == (cpr - 1u)) mbar_arrive(epi_rdy + 8u * (k & 1u));
#else
              if (epi_on && rem == (cpr - 1u)) commit_mbar(epi_rdy + 8u * (k & 1u));   // 该 n-tile 的 D 就绪（按 D 缓冲分组）
#endif
              if (epi_on) { if (++rem == cpr) { rem = 0u; ++k; } }   // 递增计数器（无除法）
            }
            commit_mbar(membty + 8 * st);
          }
        } else {
          if (cute::elect_one_sync()) mbar_arrive(membty + 8 * st);
        }
#ifdef T4_V10R_CYC
        if (t == NPROD) diag[14] += (unsigned)(clock64() - _ci0);
#endif
        if (cp_fail) break;
      }
      if (do_mma && cute::elect_one_sync()) {
        commit_mbar(mbar);
        ok_all = (mbar_wait_bounded_p(mbar, BOUND, 0u) && fails == 0u) ? 1u : 0u;
      } else {
        ok_all = (fails == 0u) ? 1u : 0u;
      }
      t1 = gtimer_ns();
#ifdef T4_V10R_CYC
      if (t == NPROD) { diag[8] = (unsigned)(t1 - t0); diag[9] = (unsigned)(clock64() - _ct0); }
#endif
    } else {
      // ---------------- 生产者（warp0..7）----------------
      unsigned fails = 0;
#ifdef T4_V10R_CYC
      const unsigned long long _pt0 = (t == 0) ? clock64() : 0ull;
#endif
      unsigned next_b = 0u;                       // ★ 83c：下一个待办边界（D 双缓冲后按序处理）
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
          const unsigned nblk = (c0 + (unsigned)CPS <= chunks) ? (unsigned)CPS : (chunks - c0);
          // ★ v11：**纯拷贝**——源是真实 GGML 张量，不是预重排的流。
          //   ⚠ 粒度纪律（事故志「A 类：非对齐向量访存」）：GGML 的 36 B K64 记录 = [4 B 缩放][32 B nibble]，
          //     granule 起点在 +4/+20 ⇒ **不是 16 B 对齐**，`cp.async ... 16` 会 LL/对齐 fault（v11 首臂实测
          //     整核挂 + GR reset 失败，见 INCIDENTS）。⇒ 全部改成 **4 B 粒度**（4 | 36、4 | row_stride ⇒ 天然 4 对齐）。
          //   qs 区（每 chunk 4096 B）：256 个线程各搬 4 个 4B 单元（r 循环 = K64 子步 0..3）：
          //     u  = t & 7      （K64 记录 [4B sf][32B nibble] 内的第几个 4B → 源偏移 4 + 4u）
          //     row= (t>>3)&31  （行 0..31）
          //     每个 4B 单元：src = tile + row*row_stride + (4c+r)*36 + 4 + 4u
          //                   dst = r*1024 + (row%8)*16 + (u>=4 ? 128 : 0) + (row/8)*256 + (u&3)*4
          //     （核验：u<4 ⇒ granule A 的 16 B 内；u>=4 ⇒ granule B；r 步进 ⇒ 源 +36 B、目的 +1024 B = 一个板）
          {
            const int u_ = t & 7, row_ = (t >> 3) & 31;      // u_: 记录内第几个 4B；row_: 行（0..31）
            const unsigned char* s2 = gB_tile + (size_t)row_ * row_stride +
                                      (size_t)(4u * c0) * 36u + 4u + 4u * (unsigned)u_;
            unsigned char* d2 = dst + (row_ & 7) * 16 + ((u_ >= 4) ? 128 : 0) +
                                (row_ >> 3) * 256 + (u_ & 3) * 4;
            for (unsigned k = 0; k < nblk; ++k) {
              const size_t ksrc = (size_t)k * 144u, kdst = (size_t)k * B_CHUNK;
#pragma unroll
              for (int r = 0; r < NSUB; ++r)                 // r = K64 子步（源步长 36 B，目的步长一个板 1024 B）
                cp_async4(cvta_smem(d2 + kdst + (unsigned)r * B_SUB), s2 + ksrc + (size_t)r * 36u);
            }
          }
          // ★ 缩放区（每 chunk 512 B = 4 子步 × 32 行 × 4 B UE4M3）**由 warp0 独占搬运 + 128 线程命名屏障交接**：
          //   源不连续（每行每 K64 块的头 4 B，步长 row_stride），4 B 粒度天然对齐；1× 读取与 v10r 同理由
          //   （4 个 warp 读同一份）。跨 warp 读别人的 cp.async 必须先有屏障（见 results/20260916-79）。
          if (warp == 0) {
            const int row_ = t & 31;
            // ★ 修正（20260916-112）：源是**行主序真实张量**，本 stage 的 chunk 在行内基址 = c0*144 B，
            //   必须显式加上。v11 首版漏了这一项 ⇒ stage≥1 的缩放字取自本 stage 的第 k 块而非第 c0+k 块；
            //   stage0 因 c0=0 恰好正确 ⇒ 表象为「chunks≤8 全对、≥9（nstages≥2）全错」。
            const unsigned char* rs = gB_tile + (size_t)row_ * row_stride + (size_t)c0 * 144u;
            for (unsigned k = 0; k < nblk; ++k)
              for (int r = 0; r < NSUB; ++r)
                cp_async4(cvta_smem(dst + (size_t)k * B_CHUNK + QS_CHUNK + r * SF_SUB + row_ * 4),
                          rs + (size_t)k * 144u + (size_t)r * 36u);
          }
          cp_commit();
        }
        if (n >= (unsigned)(STAGES - 2)) {
          if (n < nstages) cp_wait_group<STAGES - 2>();
          else            cp_wait_group<0>();
          const int st_done = (int)((n + 2u) % (unsigned)STAGES);
          // ★ 真实缩放：把 stage st_done 的 8×4 个 SF 字搬进 TMEM（warp0..3 覆盖 128 lane）
          //   上一次用到该 stage 的 MMA 早已提交（empty[] 已放行本轮），不会覆盖在执行中的缩放。
          if (warp < 4) {
            bar_sync_sf();     // ★ 缩放区由 warp0 单独搬（1×）⇒ 读前与 warp0 对齐（cp.async 完成只对发起线程可见）
            const int nn = t & 31;
            const unsigned char* sfs = smB + (size_t)st_done * B_STAGE + QS_CHUNK;
            for (int cp = 0; cp < CPS; ++cp)
              for (int s = 0; s < NSUB; ++s) {
                uint32_t w32;
                const unsigned char* pw = sfs + (size_t)cp * B_CHUNK + (size_t)s * SF_SUB + (size_t)nn * 4;
                w32 = (uint32_t)pw[0] | ((uint32_t)pw[1] << 8) | ((uint32_t)pw[2] << 16) | ((uint32_t)pw[3] << 24);
                tmem_st_32x32b_x1(taddr + ((uint32_t)((warp & 3) * 32) << 16) + COL_SFB_ST +
                                      (uint32_t)st_done * SF_COLS_STAGE +
                                      (uint32_t)(cp * NSUB + s) * SF_COLS_STEP, w32);
              }
            tmem_wait_st();
            fence_before_thread_sync();
          }
          if (!warp_arrive || (t & 31u) == 0u) mbar_arrive(mfull + 8 * st_done);
          // ★ 83：本轮 n 对应消费者 stage jc = n-(STAGES-2)；若 jc 收尾了一个 n-tile ⇒ warp0..3 读回 D
          if (epi_on && warp < 4) {
            const int jc = (int)n - (STAGES - 2);      // 本轮对应的消费者 stage 序号
            if (jc >= 0) {
              // ★ 84b：**滞后 2 个 stage** 再取边界。本迭代开头刚等到 membty[stage n-3] ⇒ 消费者的 MMA
              //   **执行**已越过 chunk 8(n-3)+8 = 8*(jc-1)。此时这些边界的 tcgen05.commit 必然已到达
              //   ⇒ 下面的 epi_rdy 等待恒为 0（实测：不滞后要等 MMA 排空，5.2 µs/边界；滞后后见 results/20260916-84）。
              //   尾轮（n >= nstages，无 membty 等待）用全量 chunks：此时本来就要等消费者跑完，等待由 epi_rdy 兜底。
              const unsigned chunks_done = ((unsigned)(jc + 1) < nstages)
                                               ? (jc >= 1 ? (unsigned)(jc - 1) * (unsigned)CPS : 0u)
                                               : (unsigned)chunks;
              const unsigned b_max = chunks_done / cpr;   // 消费者**已执行完**的 n-tile 数
              bool ok_b = true;
              while (next_b < b_max) {
#ifdef T4_V10R_CYC
                const unsigned long long _cp0 = (t == 0) ? clock64() : 0ull;
#endif
                if (!mbar_wait_bounded_p(epi_rdy + 8u * (next_b & 1u), BOUND, (next_b >> 1) & 1u)) { fails++; ok_b = false; break; }
#ifdef T4_V10R_CYC
                const unsigned long long _cp1 = (t == 0) ? clock64() : 0ull;
                if (t == 0) diag[3] += (unsigned)(_cp1 - _cp0);
#endif
#ifndef T4_V10R_NO_DREAD
                epilogue_store_d(taddr + ((next_b & 1u) ? COL_D1 : COL_D0), D, (unsigned)blk, warp, t);
#endif
                fence_before_thread_sync();
                bar_sync_epi();                        // 128 线程必须全部到达
                if (warp == 0 && (t & 31u) == 0u) {
                  mbar_arrive(epi_done + 8u * (next_b & 1u));
                  epi_count++;
                  if (t == 0) diag[13] = epi_count;
                }
#ifdef T4_V10R_CYC
                if (t == 0) { diag[4] += (unsigned)(clock64() - _cp1); diag[6]++; }
#endif
                next_b++;
              }
              if (!ok_b) break;
            }
          }
        }
        if (trace && t == 0) mark_slot(mk, 1, n + 1u);
      }
#ifdef T4_V10R_CYC
      if (t == 0) diag[10] += (unsigned)(clock64() - _pt0);
#endif
      if (t == 0 && fails != 0u) diag[12] = 0xFFFFFFFFu;
    }
  } else {
    __syncthreads();
  }
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
#ifndef T4_V10R_NO_DREAD    // 仅用于「每块固定开销分解」的对照臂（编译期开关；定版不带此宏）
  if (!taddr_bad && !epi_on) epilogue_store_d(taddr + COL_D0, D, (unsigned)blk, warp, t);  // 无 epilogue 时的收尾
  __threadfence();
  __syncthreads();
#endif

  // 5b) 诊断（flags bit3）：回读 SFA(列 256..271) 与 SFB stage0(列 272..351) 的 TMEM 真实内容。
  //     sfdump 布局 = [lane(128)][col(96)]，col 0..15 = SFA，col 16..95 = SFB stage0。
  if (dbg_sf && !taddr_bad && sfdump && warp < 4) {
    uint32_t sfr[32];
    for (int b = 0; b < 96; b += 32) {
      tmem_ld_32x32b_x32(taddr + ((uint32_t)(warp * 32) << 16) + COL_SFA + (uint32_t)b, sfr);
      tmem_wait_ld();
      for (int i = 0; i < 32; ++i)
        sfdump[(size_t)(warp * 32 + (t % 32)) * 96u + (size_t)(b + i)] = __uint_as_float(sfr[i]);
    }
  }
  __syncthreads();

  // 6) 收官 + dealloc（warp0 收敛，F1）
  if (warp == 0) {
    const unsigned ok_u = __shfl_sync(0xFFFFFFFFu, taddr_bad ? 0u : 1u, 0);
    if (ok_u) tmem_dealloc(taddr, NCOLS);
    if (t == 0) mark(mk, (uint32_t)MK_DEALLOC);
  }
}
}  // namespace v11
