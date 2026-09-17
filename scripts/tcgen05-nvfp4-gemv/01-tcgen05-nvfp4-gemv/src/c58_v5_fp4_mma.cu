// c58_v5_fp4_mma.cu — T4-3 · B5：`tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X`
//   **最小数值件**（单 block × 128 thr = 4 warp；M=128, N=32, K=64, 块缩放 16 元素 = NVFP4 口径）
//
// 与 v2/v3/v4 的关系：全新件，不改存量。alloc/relinquish/dealloc 照 **v4 模板**（全 warp 收敛，F1）；
//   A/B 的 smem canonical 布局照 **v2 i8 同构**（K-major SWIZZLE_NONE，LBO=128B / SBO=256B，
//   行 16B、K 块 128B、行组 256B —— 只是每行 16B 里塞 32 个 fp4 而不是 16 个 i8）。
//
// 三个「算出来而非猜」的定值（host 侧证据 = host/*.txt，用 CUTLASS 头在本机跑出来的）：
//   ① idesc = 0x08080480 = CUTLASS `make_instr_desc_block_scaled<e2m1,e2m1,f32,ue4m3,128,32,K,K>` 逐位相同
//      （字段：a/b_format=1(E2M1)<<7/<<10 | n_dim=4<<17 | scale_format=0(UE4M3) | m_dim=8<<24 | sf_id=0）
//   ② SF 的 TMEM 布局（CUTLASS `UMMA::tmem_sf_frg<ue4m3,16,1,*/4x1>`）：
//        SFA：lane m 的 32b 字在**列 base + m/32**，字内 byte kb = 第 kb 个 16 元素块的缩放
//        SFB：lane n 的 32b 字在**列 base + 0**（N≤32 时列恒为 0），字内 byte kb 同上
//        每多一个 K=64 的 MMA 步，SF 基址 = base + 4 列（与社区方 PTX 的 +4 步进逐位吻合）
//   ③ smem 描述符：SWIZZLE_NONE / version=1 / LBO=8 / SBO=16（16B 单位）——与 B2/B4 已绿灯的 i8 同值
//
// 判据（单点）：D 全 128×32 与宿主 CPU 参考（同布局假设反解 NVFP4）逐元素一致，且 taddr 无泄漏。
#include <cstdint>

// 次序要求同 v2：util.hpp 提供 cast_smem_ptr_to_uint，必须先于 desc 头
#include <cute/arch/util.hpp>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/mma_sm100_umma.hpp>

namespace v5 {
using cute::UMMA::Major;
using cute::UMMA::LayoutType;

constexpr int M = 128, N = 32, K = 64;
constexpr int NC = 4;                 // 缩放块数 = K/16
constexpr uint32_t NCOLS = 512u;
constexpr uint32_t COL_D = 0u, COL_SFA = 256u, COL_SFB = 272u;
constexpr uint32_t LBO_BYTES = 128u, SBO_BYTES = 256u;
constexpr int A_BYTES = 4096, B_BYTES = 1024;
constexpr int OFF_A = 0, OFF_B = 4096, OFF_MBAR = 5120, OFF_TADDR = 5128;
constexpr int SMEM_BYTES = 16384;
constexpr uint32_t TADDR_POISON = 0xA5A5A5A5u;
constexpr uint32_t TADDR_NCOLS_MAX = 512u;
// idesc（host 对拍定值）
constexpr uint32_t IDESC_B5 = 0x08080480u;

// status 位（host 侧同序打印）
enum : int {
  ST_TADDR_OK        = 1 << 0,
  ST_TADDR_POISON    = 1 << 1,
  ST_TADDR_LANE_NZ   = 1 << 2,
  ST_TADDR_COL_BAD   = 1 << 3,
  ST_MBAR_TIMEOUT    = 1 << 5,
  ST_DEALLOC_SKIPPED = 1 << 11,
  ST_SMEM_BASE_NZ    = 1 << 12,
  ST_IDESC_EQ_HAND   = 1 << 13,
};
// diag 索引（host 同序）
enum : int {
  DG_STATUS = 0, DG_TADDR_W0, DG_TADDR_W1, DG_TADDR_W2, DG_TADDR_W3,
  DG_ADESC_LO, DG_ADESC_HI, DG_BDESC_LO, DG_BDESC_HI, DG_IDESC_CUT, DG_IDESC_HAND,
  DG_MBAR_OK, DG_MISM, DG_SMEM_A, DG_SMEM_B, DG_SFA_W0, DG_SFB_N0, DG_N
};
// 进度标记（宿主映射内存；F4）
enum : int { MK_START = 0, MK_FILLED = 1, MK_ALLOC = 2, MK_TADDR = 3, MK_SF = 4,
             MK_MMA = 5, MK_DONE = 6, MK_DEALLOC = 7, MK_N = 8 };

// ---------------- PTX 包装（与 CUTLASS 头同源） ----------------
__device__ __forceinline__ uint32_t cvta_smem(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
// cute/arch/tmem_allocator_sm100.hpp Allocator1Sm
__device__ __forceinline__ void tmem_alloc(uint32_t dst, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(dst), "r"(ncols));
}
__device__ __forceinline__ void tmem_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}
__device__ __forceinline__ void tmem_dealloc(uint32_t taddr, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols));
}
__device__ __forceinline__ void proxy_fence_shared() {
  asm volatile("fence.proxy.async.shared::cta;");
}
__device__ __forceinline__ void fence_before_thread_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;");
}
__device__ __forceinline__ void fence_after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;");
}
__device__ __forceinline__ void mbar_init(uint32_t mbar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(mbar), "r"(count));
}
__device__ __forceinline__ void commit_mbar(uint32_t mbar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" ::"r"(mbar));
}
__device__ __forceinline__ uint32_t mbar_wait_bounded(uint32_t mbar, uint32_t bound) {
  uint32_t ok = 0;
  asm volatile(
      "{\n\t.reg .pred p;\n\t.reg .u32 cnt;\n\tmov.u32 cnt, %2;\n"
      "L1_%=:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0;\n\t"
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
      : "r"(mbar), "r"(bound));
  return ok;
}
// cute/arch/copy_sm100.hpp SM100_TMEM_LOAD_32dp32b32x
__device__ __forceinline__ void tmem_ld_32x32b_x32(uint32_t taddr, uint32_t* r) {
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
__device__ __forceinline__ void tmem_wait_ld() {
  asm volatile("tcgen05.wait::ld.sync.aligned;");
}
// cute/arch/copy_sm100.hpp SM100_TMEM_STORE_32dp32b1x
__device__ __forceinline__ void tmem_st_32x32b_x1(uint32_t taddr, uint32_t v) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};" ::"r"(taddr), "r"(v));
}
__device__ __forceinline__ void tmem_wait_st() {
  asm volatile("tcgen05.wait::st.sync.aligned;");
}
// cute/arch/mma_sm100_umma.hpp SM100_MMA_MXF4_SS::fma, VS=16 分支（CUDA<12.9 用 scale_vec::4X）
__device__ __forceinline__ void mma_mxf4nvf4(uint32_t d_tmem, uint64_t adesc, uint64_t bdesc,
                                             uint32_t idesc, uint32_t tsfa, uint32_t tsfb,
                                             uint32_t scaleC) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X "
      "[%0], %1, %2, %3, [%5], [%6], p;\n\t}"
      ::"r"(d_tmem), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(scaleC), "r"(tsfa), "r"(tsfb));
}

// 描述符（字段名同 cute::UMMA::SmemDescriptor）
__device__ __host__ __forceinline__ uint64_t sdesc_fields(uint32_t smem_byte_addr, uint32_t lbo_bytes,
                                                          uint32_t sbo_bytes) {
  cute::UMMA::SmemDescriptor d;
  d.desc_ = 0;
  d.version_ = 1u;
  d.lbo_mode_ = 0u;
  d.layout_type_ = uint8_t(LayoutType::SWIZZLE_NONE);
  d.start_address_ = uint16_t(smem_byte_addr >> 4);
  d.base_offset_ = 0u;
  d.leading_byte_offset_ = uint16_t(lbo_bytes >> 4);
  d.stride_byte_offset_ = uint16_t(sbo_bytes >> 4);
  return d.desc_;
}
// CUTLASS 真构图路径（同 v2）：canonical K-major SWIZZLE_NONE = ((8,rows/8),(2,16)):((16,SBO),(LBO,1)) 元素
template <int ROWS>
__device__ __forceinline__ uint64_t sdesc_via_cutlass(uint8_t* smem) {
  using namespace cute;
  // 字节级 canonical K-major SWIZZLE_NONE：((8,ROWS/8),(2,16)) : ((16,SBO),(LBO,1))
  // 每行 16B（=32 个 fp4），K 块 128B，行组 256B
  auto layout = make_layout(
      make_shape(make_shape(Int<8>{}, Int<ROWS / 8>{}), make_shape(Int<2>{}, Int<16>{})),
      make_stride(make_stride(Int<16>{}, Int<int(SBO_BYTES)>{}), make_stride(Int<int(LBO_BYTES)>{}, Int<1>{})));
  auto t = make_tensor(make_smem_ptr((uint8_t*)smem), layout);
  return (uint64_t)cute::UMMA::make_umma_desc<Major::K>(t);
}

// ---------------- canonical 布局（fp4：每行 16B = 32 个 fp4；K 块 128B；行组 256B） ----------------
__device__ __host__ __forceinline__ int off_fp4(int mn, int k) {
  return (mn % 8) * 16 + (k / 32) * 128 + (mn / 8) * 256 + (k % 32) / 2;
}
// 数据发生器（host 同源，用于 CPU 参考）
__device__ __host__ __forceinline__ int genA_nib(int m, int k) {   // e2m1 nibble
  if (m == 64) return 4;                  // 2.0（整行）—— 测行组 (m/8)=8 的 SBO 路径
  return (k % 2 == 0) ? 2 : 4;            // 偶数 k=1.0, 奇数 k=2.0 —— 测字节内高低半字节
}
__device__ __host__ __forceinline__ int genB_nib(int n, int k) {
  return (n == 24) ? 4 : 2;               // B 行 24 整行 2.0 —— 测 B 的行组 (n/8)=3
}
__device__ __host__ __forceinline__ uint8_t genSFA(int m, int kb) {  // e4m3 byte
  return (m == 0 && kb == 0) ? 0x40 : 0x38;   // 2.0 / 1.0
}
__device__ __host__ __forceinline__ uint8_t genSFB(int n, int kb) {
  return (n == 5 && kb == 3) ? 0x48 : 0x38;   // 4.0 / 1.0
}
__device__ __host__ __forceinline__ float fp4_val(int nib) {
  const float t[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                       -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
  return t[nib & 15];
}
__device__ __host__ __forceinline__ float sf_val(uint8_t b) {  // e4m3 解码（本件只用 0x38/0x40/0x48）
  const int s = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 7;
  if (e == 0) return s ? -0.f : 0.f;
  const float v = (float)(1.0 + m / 8.0) * (float)(1 << (e - 7));
  return s ? -v : v;
}
__device__ __host__ __forceinline__ float ref_D(int m, int n) {
  float acc = 0.f;
  for (int k = 0; k < K; ++k)
    acc += fp4_val(genA_nib(m, k)) * fp4_val(genB_nib(n, k)) *
           sf_val(genSFA(m, k / 16)) * sf_val(genSFB(n, k / 16));
  return acc;
}

__device__ __forceinline__ void mark(uint32_t* mk, uint32_t code) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(mk), "r"(code));
}

// ---------------- 主核 ----------------
extern "C" __global__ void __launch_bounds__(128) k_b5_fp4_mma(float* D, uint32_t* mk, unsigned* diag, uint32_t* SFD) {
  __shared__ __align__(1024) unsigned char smem[SMEM_BYTES];
  const int t = threadIdx.x;
  const int warp = t / 32;
  uint8_t* smA = smem + OFF_A;
  uint8_t* smB = smem + OFF_B;
  const uint32_t mbar = cvta_smem(smem + OFF_MBAR);
  unsigned char* slot = smem + OFF_TADDR;

  if (t == 0) { mark(mk, MK_START); *(volatile uint32_t*)slot = TADDR_POISON; mbar_init(mbar, 1); }
  __syncthreads();

  // 1) 填 A/B（每线程一个字节 = 两个相邻 k）
  for (int i = t; i < M * (K / 2); i += blockDim.x) {
    const int m = i / (K / 2), p = i % (K / 2);     // p = k/2
    smA[off_fp4(m, 2 * p)] = (uint8_t)(genA_nib(m, 2 * p) | (genA_nib(m, 2 * p + 1) << 4));
  }
  for (int i = t; i < N * (K / 2); i += blockDim.x) {
    const int n = i / (K / 2), p = i % (K / 2);
    smB[off_fp4(n, 2 * p)] = (uint8_t)(genB_nib(n, 2 * p) | (genB_nib(n, 2 * p + 1) << 4));
  }
  proxy_fence_shared();
  if (t == 0) mark(mk, MK_FILLED);
  __syncthreads();

  // 2) TMEM 分配（warp 0，全活跃收敛；F1）
  if (warp == 0) { tmem_alloc(cvta_smem(slot), NCOLS); tmem_relinquish(); }
  __syncthreads();
  if (t == 0) mark(mk, MK_ALLOC);

  const uint32_t taddr = *(volatile uint32_t*)slot;
  if (t % 32 == 0) diag[DG_TADDR_W0 + warp] = taddr;
  const bool taddr_bad = (taddr == TADDR_POISON) || ((taddr >> 16) != 0u) ||
                         ((taddr & 0xFFFFu) >= TADDR_NCOLS_MAX) || ((taddr & 31u) != 0u);
  if (t == 0) {
    int st = 0;
    if (taddr == TADDR_POISON) st |= ST_TADDR_POISON;
    if ((taddr >> 16) != 0u) st |= ST_TADDR_LANE_NZ;
    if ((taddr & 0xFFFFu) >= TADDR_NCOLS_MAX || (taddr & 31u) != 0u) st |= ST_TADDR_COL_BAD;
    if (!taddr_bad) st |= ST_TADDR_OK;
    diag[DG_STATUS] = (unsigned)st;
    mark(mk, MK_TADDR);
  }

  // 3) 描述符 + idesc（对拍）
  const uint32_t a_smem = cvta_smem(smA), b_smem = cvta_smem(smB);
  const uint64_t adesc_cut = sdesc_via_cutlass<M>(smA);
  const uint64_t bdesc_cut = sdesc_via_cutlass<N>(smB);
  const uint64_t adesc_hand = sdesc_fields(a_smem, LBO_BYTES, SBO_BYTES);
  const uint64_t bdesc_hand = sdesc_fields(b_smem, LBO_BYTES, SBO_BYTES);
  if (t == 0) {
    diag[DG_ADESC_LO] = (unsigned)(adesc_cut & 0xFFFFFFFFu);
    diag[DG_ADESC_HI] = (unsigned)(adesc_cut >> 32);
    diag[DG_BDESC_LO] = (unsigned)(bdesc_cut & 0xFFFFFFFFu);
    diag[DG_BDESC_HI] = (unsigned)(bdesc_cut >> 32);
    diag[DG_IDESC_CUT] = IDESC_B5;        // CUTLASS 定值（host 对拍见 host/*.txt）
    diag[DG_IDESC_HAND] = IDESC_B5;
    diag[DG_SMEM_A] = a_smem;
    diag[DG_SMEM_B] = b_smem;
    unsigned st = (unsigned)diag[DG_STATUS];
    if (adesc_cut == adesc_hand) st |= ST_IDESC_EQ_HAND;   // 复用位：smem desc 构图==手编码
    if (a_smem != 0u || b_smem != OFF_B) st |= ST_SMEM_BASE_NZ;
    diag[DG_STATUS] = st;
  }

  // 4) 缩放进 TMEM（SFA: 每个 warp 一列；SFB: warp0 一列）
  if (!taddr_bad) {
    {
      const int m = warp * 32 + (t % 32);
      uint32_t w32 = 0;
      for (int kb = 0; kb < NC; ++kb) w32 |= (uint32_t)genSFA(m, kb) << (8 * kb);
      tmem_st_32x32b_x1(taddr + ((uint32_t)(warp * 32) << 16) + COL_SFA + (uint32_t)warp, w32);
      if (m == 0 && t == 0) diag[DG_SFA_W0] = w32;
    }
    // SFB：N=32 但 **必须复制到全部 128 lane**（CUTLASS UMMA::tmem_sf_frg 的 REP=4：
    //   "Replication factor. Data is always replicated across subpartitions"）。
    //   2026-09-16 板端首跑实测：只写 lane0..31 ⇒ D 的 lane32..127 全 0（SFB 按输出行所在的 lane 取数）。
    {
      const int n = t % 32;                       // 每个 warp 各写自己的 32 lane
      uint32_t w32 = 0;
      for (int kb = 0; kb < NC; ++kb) w32 |= (uint32_t)genSFB(n, kb) << (8 * kb);
      tmem_st_32x32b_x1(taddr + ((uint32_t)(warp * 32) << 16) + COL_SFB, w32);
      if (n == 0 && warp == 0) diag[DG_SFB_N0] = w32;
    }
    tmem_wait_st();
  }
  __syncthreads();
  // 3.5) SF 区转储（列 256..287，逐 lane 32 字）→ 看 STTM 落点（被动观测，不改行为）
  if (SFD && !taddr_bad) {
    uint32_t rs[32];
    for (int i = 0; i < 32; ++i) rs[i] = 0u;
    tmem_ld_32x32b_x32(taddr + ((uint32_t)(warp * 32) << 16) + COL_SFA, rs);
    tmem_wait_ld();
    __syncthreads();
    for (int i = 0; i < 32; ++i) SFD[(warp * 32 + (t % 32)) * 32 + i] = rs[i];
  }
  fence_before_thread_sync();
  if (t == 0) mark(mk, MK_SF);
  __syncthreads();

  // 5) 发 mma（warp 0，elect_one）+ commit
  if (!taddr_bad && warp == 0) {
    fence_after_thread_sync();
    if (cute::elect_one_sync()) {
      mma_mxf4nvf4(taddr + COL_D, adesc_cut, bdesc_cut, IDESC_B5, taddr + COL_SFA, taddr + COL_SFB, 0u);
      commit_mbar(mbar);
      mark(mk, MK_MMA);
    }
  }

  // 6) 有界等待（绝不死等）
  uint32_t ok = 1;
  if (!taddr_bad) ok = mbar_wait_bounded(mbar, 200000000u);
  if (t == 0) { diag[DG_MBAR_OK] = ok; if (!ok) diag[DG_STATUS] |= (unsigned)ST_MBAR_TIMEOUT; }
  fence_after_thread_sync();
  __syncthreads();

  // 7) 读回 D（lane=m，列=n，32b 不打包）
  uint32_t r[32];
  for (int i = 0; i < 32; ++i) r[i] = 0u;
  if (!taddr_bad && D) {
    tmem_ld_32x32b_x32(taddr + ((uint32_t)(warp * 32) << 16), r);
    tmem_wait_ld();
    __syncthreads();
    for (int n = 0; n < N; ++n) D[(warp * 32 + (t % 32)) * N + n] = __uint_as_float(r[n]);
  } else {
    __syncthreads();
  }

  // 8) 本核内 CPU 参考比对（残差计数）
  int bad = 0;
  if (D) {
    const int m = warp * 32 + (t % 32);
    for (int n = 0; n < N; ++n) {
      const float ref = ref_D(m, n);
      const float got = D[m * N + n];
      if (!(got == ref)) bad++;
    }
  }
  if (bad) atomicAdd((int*)&diag[DG_MISM], bad);
  if (t == 0) mark(mk, (uint32_t)MK_DONE);

  // 9) dealloc（warp 0 全 32 lane 收敛；F1）
  __syncthreads();
  if (warp == 0) {
    const unsigned ok_u = __shfl_sync(0xFFFFFFFFu, taddr_bad ? 0u : 1u, 0);
    if (ok_u) tmem_dealloc(taddr, NCOLS);
    if (t == 0) {
      if (!ok_u) diag[DG_STATUS] |= (unsigned)ST_DEALLOC_SKIPPED;
      mark(mk, MK_DEALLOC);
    }
  }
}
}  // namespace v5
