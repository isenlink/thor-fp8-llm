// t4_mmq.cu — T4（自研 tcgen05 NVFP4 快路）与生产 MMQ/MMVQ 的接缝。
//
// 阶段 1（2026-09-16 19:0x）：只落接缝，不落内核 —— 行为与打补丁前完全一致。
// 阶段 2a（20:3x）：形状条件抽成无依赖纯判定 `t4_mmq_pred.h`（可被宿主单测直接编译）。
// 阶段 2b（20:4x–21:2x，本形态）：**内核接线**。
//   · 权重（B）：生产 GGML NVFP4 字节**纯拷贝**进 smem（v11 已证逐位同构，results/20260916-112）
//     ⇒ 零 relayout、零额外常驻内存；canonical 只存在于 smem。
//   · 激活（A）：`t4_mmq_quant_a` 产出 canonical 板布局（ggml nibble 约定，与 B 对齐）。
//   · D：内核出 N 主序 `[n][m]`（F21）→ `t4_mmq_transpose_nm` 落到生产 dst（行主序 float）。
//
// ⚠ 2026-09-16 21:0x 板上实测（T4-MMQ 一次性日志）定下的两件事：
//   ① **接缝必须挂两处**：NVFP4 的 decode（M ≤ MMVQ_MAX_BATCH_SIZE=8）走
//      `ggml_cuda_mul_mat_vec_q`（mmvq.cu），**根本不过 mmq.cu**；只有 M > 8（prefill/大 verify）
//      才走 `ggml_cuda_mul_mat_q`。只挂 mmq ⇒ 段级读数纹丝不动（实测：decode 90 步零命中）。
//   ② `mmq_args.stride_row_x` 是**块数**（nb[1]/36），不是字节。本文件现在直接吃张量，不再经 mmq_args。
//
// 开关：环境变量 `T4_MMQ=1` 显式启用；未设置 ⇒ 一律走原路（与原二进制逐位同行为）。
// occ=2 需 `T4_ALLOW_OCC2=1`；`T4_DIAG=1` 时把内核 diag[] 拉回宿主并打日志（诊断用，会同步流）。
// 回退：删掉本文件 + 还原 mmq.cu/mmvq.cu/common.cuh 的备份即可（见 T4-M2c-rollout.md §4）。

#include "t4_mmq.cuh"

#include "t4_mmq_pred.h"

#include "mmq.cuh"

#include "t4/t4_mmq_aq.cuh"
#include "t4/t4_mmq_ring.cuh"

#include <cstdlib>

// 纯判定头里的常量必须与 ggml/CUDA 真值逐字对齐（防漂移）。
static_assert(T4_MMQ_CC_THOR    == GGML_CUDA_CC_THOR_SM101A,   "T4_MMQ_CC_THOR 与 common.cuh 不一致");
static_assert(T4_MMQ_TYPE_NVFP4 == (int) GGML_TYPE_NVFP4,      "T4_MMQ_TYPE_NVFP4 与 ggml.h 不一致");
static_assert(T4_MMQ_TYPE_F32   == (int) GGML_TYPE_F32,        "T4_MMQ_TYPE_F32 与 ggml.h 不一致");
static_assert(T4_MMQ_KCHUNK     == t4mmq::KT,                  "T4_MMQ_KCHUNK 与 canonical 环粒度不一致");
static_assert(T4_MMQ_NTILE      == t4mmq::MMA_N,               "T4_MMQ_NTILE 与 MMA N 不一致");
static_assert(T4_MMQ_M_MAX      == t4mmq::MMA_M,               "T4_MMQ_M_MAX 与 MMA M 不一致");
static_assert(sizeof(block_nvfp4) == (size_t) t4mmq::B_REC,    "block_nvfp4 字节数与 B_REC 不一致");

static bool t4_mmq_env1(const char * name) {
    const char * e = getenv(name);
    return e != nullptr && e[0] == '1';
}

namespace {

// 环配置（2026-09-16 22:0x，M2c 阶段 4a：在服务端路径上扫「等待组深度」）。
// 背景：生产环 CPS=3/STAGES=2 ⇒ `cp_wait_group<STAGES-2>` = wait_group<0>，每个 stage 的拷贝必须
// 整段排空才交接 ⇒ DRAM 延迟无法摊薄。实测（results/20260916-113 §5）：生产 627 ns/chunk，
// 而隔离原型的长流（CPS=8/STAGES=3 = wait_group<1>）是 262.6 ns/chunk（同板、同 chunk 尺寸）。
// 未设 T4_CPS/T4_STAGES 时 = 默认 3/2，与打补丁前行为逐字节一致。
constexpr int T4_CPS_DEF    = 3;   // 每个 stage 装几个 chunk（宿主 T2 实测选型）
constexpr int T4_STAGES_DEF = 2;   // 环深

// occ=1 硬保证（只对非默认配置启用）：动态 smem 申请抬到每 SM 上限的一半以上 ⇒ 硬件不可能把两个 CTA
// 放在同一 SM 上。必要性：CPS/STAGES 变大后 NCOLS 可能 > 256，两个 CTA 的 TMEM 合计超 512 ⇒
// 后者永久卡在 `tcgen05.alloc`（事故 §九/#8 的形态）。默认配置 NCOLS=256（256+256=512 恰好装得下）
// 不需要，故默认路径保持原样（申请值不变）。
constexpr int T4_SMEM_OCC1 = 118784;

// A 常驻（T4-M2c-4b）的 chunk 上限：A 区 = ACAP × A_CHUNK，入口搬一次、之后所有 n-tile 复用。
// 68 = 生产最大 cpr（ffn_down K=17408 / 256）；更大的 K 不接管（回落原路），绝不做运行期截断。
constexpr int T4_ACAP = 68;

// 正整数环境变量（未设/非法 ⇒ 默认值）
inline int t4_mmq_env_int(const char * name, const int def) {
    const char * e = getenv(name);
    if (e == nullptr || e[0] == '\0') {
        return def;
    }
    const int v = atoi(e);
    return v > 0 ? v : def;
}

// A 环的行档位：取 ≥M 的最小档（多出的行由内核的 m_real 守卫丢弃）
inline int t4_mmq_abd_for(const int m) {
    if (m <= 8)  return 8;
    if (m <= 16) return 16;
    if (m <= 32) return 32;
    if (m <= 64) return 64;
    return 128;
}

// 首次接线时"没命中却看不出为什么"白花掉一次板端臂 ⇒ 前 16 次调用逐次打印，之后每个原因码一次。
void t4_mmq_log_reject(const int why, const t4_mmq_pred_in & s, const int type_y) {
    static int  n_call = 0;
    static bool seen[T4_MMQ_REJ_LAST] = { false };
    const bool by_count = n_call < 16;
    if (!by_count && (why <= 0 || why >= T4_MMQ_REJ_LAST || seen[why])) {
        return;
    }
    ++n_call;
    if (why > 0 && why < T4_MMQ_REJ_LAST) {
        seen[why] = true;
    }
    GGML_LOG_WARN("T4-MMQ: 未命中(%s) K=%lld N=%lld M=%lld type_x=%d type_y=%d stride=%lld nb=%lld "
                  "ne12=%lld ne13=%lld ne02=%lld ne03=%lld act32=%d ptr4=%d #%d\n",
                  t4_mmq_pred_reason_name(why), s.ncols_x, s.nrows_x, s.ncols_dst, s.type_x, type_y,
                  s.stride_row_x, (long long) (s.stride_row_x * t4mmq::B_REC), s.nchannels_y, s.nsamples_y,
                  s.nchannels_x, s.nsamples_x, (int) s.act32_ok, (int) s.ptr4_ok, n_call);
}

template <int ABD, int CPS, int STAGES, bool PAD, int ACAP = 0>
void t4_mmq_launch_cfg(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * x,
                       ggml_tensor * dst, const float * x_f32, const int64_t s_x_f32,
                       cudaStream_t stream, const int nblk) {
    using G = t4mmq::RingGeom<ABD, CPS, STAGES, ACAP>;
    // A 常驻或非默认环 ⇒ 一律 pad 到 occ=1（NCOLS 可能 >256，两个 CTA 同 SM 会卡死在 tcgen05.alloc）
    const bool pad = PAD || (ACAP > 0);
    const int smem_req = (pad && G::SMEM_REQ < T4_SMEM_OCC1) ? T4_SMEM_OCC1 : G::SMEM_REQ;

    const int K = (int) w->ne[0];
    const int N = (int) w->ne[1];
    const int M = (int) x->ne[1];
    const int cpr     = K / t4mmq::KT;          // 每个 n-tile 的 chunk 数
    const int ntiles  = N / t4mmq::MMA_N;
    const int nboards = K / t4mmq::MMA_K;
    const size_t row_stride = (size_t) w->nb[1];                                   // 权重行字节步长
    const size_t sd = (size_t) (dst->nb[1] / (int64_t) ggml_type_size(dst->type));  // dst 行步长（float 数）

    ggml_cuda_pool_alloc<unsigned char> a_nib(ctx.pool(), (size_t) nboards * t4mmq::A_BD_STRIDE);
    ggml_cuda_pool_alloc<unsigned char> a_sf (ctx.pool(), (size_t) nboards * t4mmq::A_SF_STRIDE);
    ggml_cuda_pool_alloc<float>         rsc  (ctx.pool(), t4mmq::MMA_M);
    ggml_cuda_pool_alloc<float>         dtmp (ctx.pool(), (size_t) ntiles * t4mmq::MMA_N * M);
    ggml_cuda_pool_alloc<unsigned>      diag (ctx.pool(), 8);

    t4mmq::t4_mmq_quant_a<<<M, 256, 0, stream>>>(x_f32, (size_t) s_x_f32,
                                                 a_nib.get(), a_sf.get(), rsc.get(), M, K);
    CUDA_CHECK(cudaGetLastError());

    // A 常驻（T4-M2c-4b）运行期正证据：每个实例只报一次（证明常驻核真的被发射，而不是只在 env 上设了值）
    if constexpr (ACAP > 0) {
        static bool ares_logged = false;
        if (!ares_logged) {
            ares_logged = true;
            GGML_LOG_WARN("T4-MMQ: A 常驻核上线 ABD=%d CPS=%d STAGES=%d ACAP=%d cpr=%d K=%d N=%d M=%d nblk=%d smem_req=%d\n",
                          ABD, CPS, STAGES, ACAP, cpr, K, N, M, nblk, smem_req);
        }
    }
    CUDA_SET_SHARED_MEMORY_LIMIT((t4mmq::t4_mmq_ring<ABD, CPS, STAGES, ACAP>), smem_req);
    t4mmq::t4_mmq_ring<ABD, CPS, STAGES, ACAP><<<nblk, t4mmq::NTHR, smem_req, stream>>>(
        reinterpret_cast<const unsigned char *>(w->data), a_nib.get(), a_sf.get(), rsc.get(), dtmp.get(),
        (unsigned) cpr, ntiles, 0, ntiles, M, row_stride, diag.get());
    CUDA_CHECK(cudaGetLastError());

    const int64_t total = (int64_t) M * (int64_t) N;
    const unsigned nthr = 256;
    t4mmq::t4_mmq_transpose_nm<<<(unsigned) ((total + nthr - 1) / nthr), nthr, 0, stream>>>(
        dtmp.get(), reinterpret_cast<float *>(dst->data), M, N, sd);
    CUDA_CHECK(cudaGetLastError());

    if (t4_mmq_env1("T4_DIAG")) {
        unsigned d[8] = {0};
        CUDA_CHECK(cudaMemcpy(d, diag.get(), sizeof(d), cudaMemcpyDeviceToHost));
        GGML_LOG_WARN("T4-MMQ[diag]: taddr_ok=%u fails=0x%08x ok_all=%u (注：多块同址累加，仅定性)\n",
                      d[0], d[1], d[2]);
    }
}

// 配置选择器：把 (CPS,STAGES) 映射到编译期实例。未列入集合的取值 ⇒ 返回 false，调用方回落默认配置
//（**不静默跑一个没审过的几何**：NCOLS/smem 两个预算都要先过宿主审计，见 t4_m2c_ring_audit）。
// ABD 档位限制：A_CHUNK(ABD) = 4×(ABD×32 + ABD×4) ⇒ ABD=32/64/128 时一个 chunk 就要 4.6/9.2/18.4 KB，
// 更宽/更深的环放不下（`RingGeom` 的 smem static_assert 会拦）。故**新配置只对小 M 档开放**
//（部署形状 M ≤ 8 ⇒ ABD=8、M=13 ⇒ ABD=16，正是 decode 唯二会走的路）；大 M 档一律走默认配置。
template <int ABD>
bool t4_mmq_launch_sel(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * x,
                       ggml_tensor * dst, const float * x_f32, const int64_t s_x_f32,
                       cudaStream_t stream, const int nblk, const int cps, const int stages, const bool ares) {
    if constexpr (ABD <= 16) {
        if (ares) {                             // A 常驻变体（本版只实例化 3/2 与 4/3 两个环配置）
            switch (cps * 16 + stages) {
                case T4_CPS_DEF * 16 + T4_STAGES_DEF:
                    t4_mmq_launch_cfg<ABD, T4_CPS_DEF, T4_STAGES_DEF, false, T4_ACAP>(
                        ctx, w, x, dst, x_f32, s_x_f32, stream, nblk);
                    return true;
                case 4 * 16 + 3:
                    t4_mmq_launch_cfg<ABD, 4, 3, false, T4_ACAP>(ctx, w, x, dst, x_f32, s_x_f32, stream, nblk);
                    return true;
                default:
                    break;                      // 未实例化的组合 ⇒ 落回下面的环形态（单变量：不静默换几何）
            }
        }
        switch (cps * 16 + stages) {
            case T4_CPS_DEF * 16 + T4_STAGES_DEF:
                t4_mmq_launch_cfg<ABD, T4_CPS_DEF, T4_STAGES_DEF, false>(ctx, w, x, dst, x_f32, s_x_f32, stream, nblk);
                return true;
            case 4 * 16 + 3:                    // wait_group<1>：一个组在飞
                t4_mmq_launch_cfg<ABD, 4, 3, true>(ctx, w, x, dst, x_f32, s_x_f32, stream, nblk);
                return true;
            case 3 * 16 + 4:                    // wait_group<2>：两个组在飞
                t4_mmq_launch_cfg<ABD, 3, 4, true>(ctx, w, x, dst, x_f32, s_x_f32, stream, nblk);
                return true;
            case 7 * 16 + 2:                    // 单元最大（32 KB/stage），仍 wait_group<0>
                t4_mmq_launch_cfg<ABD, 7, 2, true>(ctx, w, x, dst, x_f32, s_x_f32, stream, nblk);
                return true;
            default:
                return false;
        }
    } else {
        if (cps == T4_CPS_DEF && stages == T4_STAGES_DEF) {
            t4_mmq_launch_cfg<ABD, T4_CPS_DEF, T4_STAGES_DEF, false>(ctx, w, x, dst, x_f32, s_x_f32, stream, nblk);
            return true;
        }
        static bool warned = false;             // ABD ≥ 32 档不支持覆盖值 ⇒ 交回调用方（回落 generic 路）
        if (!warned) {
            warned = true;
            GGML_LOG_WARN("T4-MMQ: ABD=%d 档不支持 T4_CPS=%d T4_STAGES=%d（只支持默认 %d/%d）⇒ 本次不接管\n",
                          ABD, cps, stages, T4_CPS_DEF, T4_STAGES_DEF);
        }
        return false;
    }
}

}  // namespace

bool t4_mmq_try_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * x,
                       ggml_tensor * dst, const int cc, cudaStream_t stream) {
    if (!t4_mmq_env1("T4_MMQ")) {
        return false;
    }
    const t4_mmq_pred_in s = {
        cc,
        ggml_cuda_highest_compiled_arch(cc),
        (int) w->type,
        false,                              // 本入口只从非 ids 分支调用
        w->ne[0],                           // K
        w->ne[1],                           // N（权重行数）
        x->ne[1],                           // M（本步 token 数）
        w->nb[1] / ggml_type_size(w->type), // 权重行步长（块数）
        x->ne[2],                           // ne12
        x->ne[3],                           // ne13
        w->ne[2],                           // ne02
        w->ne[3],                           // ne03
        (int) x->type,
        (x->nb[1] % 32u == 0) && (reinterpret_cast<uintptr_t>(x->data) % 32u == 0),
        (reinterpret_cast<uintptr_t>(w->data) % 4u == 0) &&
            (reinterpret_cast<uintptr_t>(x->data) % 4u == 0) &&
            (reinterpret_cast<uintptr_t>(dst->data) % 4u == 0),
    };
    const int why = t4_mmq_pred_reason(s);
    if (why != T4_MMQ_OK) {
        t4_mmq_log_reject(why, s, (int) x->type);
        return false;
    }

    const int id  = ggml_cuda_get_device();
    const int nsm = ggml_cuda_info().devices[id].nsm;
    int nblk = t4_mmq_env1("T4_ALLOW_OCC2") ? 2 * nsm : nsm;
    const int ntiles = (int) (w->ne[1] / T4_MMQ_NTILE);
    if (nblk > ntiles) {
        nblk = ntiles;
    }
    if (nblk <= 0) {
        return false;
    }

    const int M = (int) x->ne[1];
    {
        static bool logged = false;
        if (!logged) {
            logged = true;
            GGML_LOG_WARN("T4-MMQ: 命中 K=%lld N=%lld M=%lld abd=%d nblk=%d nsm=%d cpr=%lld ntiles=%lld\n",
                          (long long) w->ne[0], (long long) w->ne[1], (long long) x->ne[1],
                          t4_mmq_abd_for(M), nblk, nsm, (long long) (w->ne[0] / T4_MMQ_KCHUNK),
                          (long long) (w->ne[1] / T4_MMQ_NTILE));
        }
    }

    // 环配置（默认 3/2 = 与打补丁前逐字节一致；覆盖值必须落在已验证集合里，否则回落默认）
    int cps    = t4_mmq_env_int("T4_CPS",    T4_CPS_DEF);
    int stages = t4_mmq_env_int("T4_STAGES", T4_STAGES_DEF);
    {
        const bool known = (cps == T4_CPS_DEF && stages == T4_STAGES_DEF) ||
                           (cps == 4 && stages == 3) || (cps == 3 && stages == 4) || (cps == 7 && stages == 2);
        static bool warned = false;
        if (!known && !warned) {
            warned = true;
            GGML_LOG_WARN("T4-MMQ: 环配置 %d/%d 未列入已验证集合 ⇒ 回落默认 %d/%d\n",
                          cps, stages, T4_CPS_DEF, T4_STAGES_DEF);
        }
        if (!known) {
            cps = T4_CPS_DEF;
            stages = T4_STAGES_DEF;
        }
        if ((cps != T4_CPS_DEF || stages != T4_STAGES_DEF) && !warned) {
            warned = true;
            GGML_LOG_WARN("T4-MMQ: 环配置覆盖 T4_CPS=%d T4_STAGES=%d（默认 %d/%d，smem pad 强制 occ=1）\n",
                          cps, stages, T4_CPS_DEF, T4_STAGES_DEF);
        }
    }

    const float * x_f32 = reinterpret_cast<const float *>(x->data);
    const int64_t s_x   = x->nb[1] / (int64_t) ggml_type_size(x->type);
    // A 常驻：要求 cpr ≤ T4_ACAP（否则不接管，绝不截断）
    bool ares = t4_mmq_env1("T4_ARES");
    if (ares && (int) (w->ne[0] / T4_MMQ_KCHUNK) > T4_ACAP) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            GGML_LOG_WARN("T4-MMQ: T4_ARES=1 但 cpr=%lld > ACAP=%d ⇒ 本张量用环形态\n",
                          (long long) (w->ne[0] / T4_MMQ_KCHUNK), T4_ACAP);
        }
        ares = false;
    }
    bool launched = false;
    switch (t4_mmq_abd_for(M)) {
        case 8:   launched = t4_mmq_launch_sel<8>  (ctx, w, x, dst, x_f32, s_x, stream, nblk, cps, stages, ares); break;
        case 16:  launched = t4_mmq_launch_sel<16> (ctx, w, x, dst, x_f32, s_x, stream, nblk, cps, stages, ares); break;
        case 32:  launched = t4_mmq_launch_sel<32> (ctx, w, x, dst, x_f32, s_x, stream, nblk, cps, stages, ares); break;
        case 64:  launched = t4_mmq_launch_sel<64> (ctx, w, x, dst, x_f32, s_x, stream, nblk, cps, stages, ares); break;
        default:  launched = t4_mmq_launch_sel<128>(ctx, w, x, dst, x_f32, s_x, stream, nblk, cps, stages, ares); break;
    }
    return launched;
}
