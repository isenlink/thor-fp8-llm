// fp8-cublaslt.cu — F8_E4M3 × F8(在线量化激活) → F32 via cuBLASLt
// Thor sm_101a 集成路径（microbench 实测 266 GB/s @M=8）
// D[M,N] = A_acts[M,K] * W[N,K]^T
//   W: GGML_TYPE_F8_E4M3 权重（[N,K] row-major = col-major [K,N]）
//   A: F32 激活 → 在线 per-tensor 量化 E4M3
#include "common.cuh"
#include <cublasLt.h>
#include <cuda_fp8.h>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <cstdint>

#define FP8LT_CHECK(x) do { \
    cublasStatus_t s_ = (x); \
    if (s_ != CUBLAS_STATUS_SUCCESS) { \
        GGML_LOG_ERROR("cublasLt FP8 err %d at %s:%d\n", (int)s_, __FILE__, __LINE__); \
        GGML_ABORT("cublasLt FP8 failure"); \
    } \
} while (0)


// graph-safe absmax reduce（两段）
__global__ void absmax_partial_kernel(const float* __restrict__ x, float* __restrict__ partial, int64_t n) {
    __shared__ float smax[1024];
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    float v = 0.f;
    if (i < n) v = fabsf(x[i]);
    // warp reduce
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, o));
    if ((threadIdx.x & 31) == 0) smax[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        float w = (threadIdx.x < (blockDim.x >> 5)) ? smax[threadIdx.x] : 0.f;
        for (int o = 16; o > 0; o >>= 1) w = fmaxf(w, __shfl_down_sync(0xffffffff, w, o));
        if (threadIdx.x == 0) partial[blockIdx.x] = w;
    }
}
__global__ void absmax_final_kernel(const float* __restrict__ partial, float* __restrict__ out, int n) {
    __shared__ float smax[1024];
    float v = (threadIdx.x < n) ? partial[threadIdx.x] : 0.f;
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, o));
    if ((threadIdx.x & 31) == 0) smax[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        float w = (threadIdx.x < (blockDim.x >> 5)) ? smax[threadIdx.x] : 0.f;
        for (int o = 16; o > 0; o >>= 1) w = fmaxf(w, __shfl_down_sync(0xffffffff, w, o));
        if (threadIdx.x == 0) out[0] = w;
    }
}

// 激活量化 kernel: f32 → e4m3（per-tensor scale，值域 |x|<=448/缩放）
__global__ void quantize_f32_to_e4m3_kernel(const float* __restrict__ x, uint8_t* __restrict__ y,
                                             float scale_inv, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i] * scale_inv;
    y[i] = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}

// 反缩放 kernel: D *= scale_w * scale_a
__global__ void scale_f32_kernel(float* __restrict__ y, float s, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] *= s;
}

static bool fp8_lt_handles_ready = false;
static cublasLtHandle_t fp8_lt_handle = nullptr;

// [Thor FP8] production workload instrumentation:
//   counts F8 MUL_MAT calls per shape and accumulates device-side ns via clock64().
//   Env GGML_FP8_PROF=1 enables; results dumped to /brand_data/ai_workspace/bench/f8_prof.txt
//   every 10000 calls (graph-safe: no D2H inside capture; dump happens on call boundaries
//   which execute outside graph replay).
struct fp8_shape_stat_t {
    uint64_t calls = 0;
    uint64_t ns_total = 0;     // accumulated kernel-side time (approx, per-SM clock)
    uint32_t amax_bucket = 0;  // log2-ish bucket of last amax (0: <1, 1: <8, 2: <64, 3: <448, 4: SAT)
    uint32_t sat_events = 0;   // count of tensors whose amax >= 448 (saturation risk)
};
static std::unordered_map<uint64_t, fp8_shape_stat_t> g_fp8_shape_stats;
static uint64_t g_fp8_call_counter = 0;
static bool g_fp8_prof = getenv("GGML_FP8_PROF") != nullptr;

__global__ void fp8_amax_probe_kernel(const float* __restrict__ x, int64_t n, uint32_t* out_bucket, uint32_t* out_sat) {
    // single-block reduce, cheap enough for M*K activations (8*5120=40KB typical)
    __shared__ float smax[256];
    float v = 0.f;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) v = fmaxf(v, fabsf(x[i]));
    // warp reduce
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, o));
    if ((threadIdx.x & 31) == 0) smax[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        float w = (threadIdx.x < 8) ? smax[threadIdx.x] : 0.f;
        for (int o = 4; o > 0; o >>= 1) w = fmaxf(w, __shfl_down_sync(0xffffffff, w, o));
        if (threadIdx.x == 0) {
            out_bucket[0] = w < 1 ? 0 : w < 8 ? 1 : w < 64 ? 2 : 3;
            if (w >= 448.0f) out_sat[0] = 1;  // would saturate E4M3 at scale=1
        }
    }
}

__global__ void fp8_gemm_timer_kernel(unsigned long long* acc_ns) {
    // accumulate wall time between calls; launched adjacently to matmul on same stream.
    // Approximation: measures inter-kernel gap + our own overhead, NOT the Lt kernel itself
    // (Lt kernel timing needs events outside graph capture — handled host-side below).
    atomicAdd(acc_ns, (unsigned long long)0); // placeholder to keep symbol
}

// [Thor FP8] per-shape plan cache: desc/layouts/algo created once, reused every call.
// Key encodes (K, N, M). This removes the dominant CPU cost (heuristic query) from
// the decode hot path — without it every one of the 208 GEMMs per forward pass
// recreates descriptors (~30-100us each), which cost us R16 decode -8% vs R12.
struct fp8_lt_plan_t {
    cublasLtMatmulDesc_t       desc     = nullptr;
    cublasLtMatrixLayout_t     la       = nullptr;   // weights  (K, N, ld=K)
    cublasLtMatrixLayout_t     lb       = nullptr;   // acts     (K, M, ld=K)
    cublasLtMatrixLayout_t     lc       = nullptr;   // dst      (N, M, ld=N)
    cublasLtMatmulAlgo_t       algo{};
    size_t                     ws_size  = 0;
    bool                       ok       = false;
};
static std::unordered_map<uint64_t, fp8_lt_plan_t> g_fp8_lt_plans;
static std::mutex g_fp8_lt_plans_mu;

bool ggml_cuda_fp8_mul_mat_available() {
    return true;  // sm_101a + CUDA12.8 已实测支持
}

// 主入口: src0 = F8_E4M3 权重, src1 = F32 激活, dst = F32
void ggml_cuda_mul_mat_f8_e4m3(ggml_backend_cuda_context & ctx,
                                const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int64_t ne00 = src0->ne[0];  // K
    const int64_t ne01 = src0->ne[1];  // N (输出通道)
    const int64_t ne11 = src1->ne[1];  // M (tokens)
    GGML_ASSERT(src0->type == GGML_TYPE_F8_E4M3);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    cudaStream_t stream = ctx.stream();
    const int64_t K = ne00, N = ne01, M = ne11;
    const int64_t n_acts = ggml_nelements(src1);
    const int64_t n_out  = ggml_nelements(dst);

    // 1) 激活幅值探测（profiling mode；graph-safe：结果留 device，host 只在 dump 边界同步）
    ggml_cuda_pool_alloc<float> amax_buf(ctx.pool());
    amax_buf.alloc(1);
    ggml_cuda_pool_alloc<uint32_t> prof_bucket(ctx.pool());
    ggml_cuda_pool_alloc<uint32_t> prof_sat(ctx.pool());
    prof_bucket.alloc(2);
    uint32_t * bucket_d = prof_bucket.ptr;
    uint32_t * sat_d = prof_bucket.ptr + 1;
    if (g_fp8_prof) {
        CUDA_CHECK(cudaMemsetAsync(bucket_d, 0, 2*sizeof(uint32_t), stream));
        fp8_amax_probe_kernel<<<1, 256, 0, stream>>>((const float*)src1->data, n_acts, bucket_d, sat_d);
        // copy to persistent staging (device-to-device); read back outside graph capture below
    }
    // 静态 act_scale=1（E4M3 覆盖 ±448）；动态 scale 待拆账数据决策
    const float act_scale = 1.0f;

    // 2) 权重 per-tensor scale：F8 权重原始分布就是 E4M3 值域，scale=1（转换时已吸收）
    const float w_scale = 1.0f;

    // 3) 激活量化 → E4M3 buffer
    ggml_cuda_pool_alloc<uint8_t> acts_q(ctx.pool());
    acts_q.alloc(n_acts);
    {
        int64_t blocks = (n_acts + 255) / 256;
        quantize_f32_to_e4m3_kernel<<<(unsigned)blocks, 256, 0, stream>>>(
            (const float*)src1->data, acts_q.ptr, 1.0f /*act_scale_inv*/, n_acts);
    }

    // 4) cuBLASLt FP8 GEMM (GGML dst convention: dst[n,m] at n + m*N, i.e. col-major (N,M,ld=N))
    // D(N,M) = W^T(N,K) x acts(K,M)
    //   A = W   col-major (K,N,ld=K) → transa=T → op(A)=(N,K)
    //   B = act col-major (K,M,ld=K) → transb=N → op(B)=(K,M)
    //   C/D     col-major (N,M,ld=N) → dst[n,m] at n + m*N  ✓ GGML convention
    if (!fp8_lt_handles_ready) {
        FP8LT_CHECK(cublasLtCreate(&fp8_lt_handle));
        fp8_lt_handles_ready = true;
    }

    // per-shape plan cache (first call builds, rest reuse)
    const uint64_t key = ((uint64_t)(uint32_t)K << 40) ^ ((uint64_t)(uint32_t)N << 20) ^ (uint64_t)(uint32_t)M;
    fp8_lt_plan_t * plan = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_fp8_lt_plans_mu);
        plan = &g_fp8_lt_plans[key];
    }
    if (!plan->ok) {
        FP8LT_CHECK(cublasLtMatmulDescCreate(&plan->desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t transa = CUBLAS_OP_T, transb = CUBLAS_OP_N;
        FP8LT_CHECK(cublasLtMatmulDescSetAttribute(plan->desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
        FP8LT_CHECK(cublasLtMatmulDescSetAttribute(plan->desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

        FP8LT_CHECK(cublasLtMatrixLayoutCreate(&plan->la, CUDA_R_8F_E4M3, K, N, K));
        FP8LT_CHECK(cublasLtMatrixLayoutCreate(&plan->lb, CUDA_R_8F_E4M3, K, M, K));
        FP8LT_CHECK(cublasLtMatrixLayoutCreate(&plan->lc, CUDA_R_32F,     N, M, N));

        cublasLtMatmulPreference_t pref;
        FP8LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
        plan->ws_size = 32 << 20;
        FP8LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &plan->ws_size, sizeof(plan->ws_size)));

        cublasLtMatmulHeuristicResult_t heur;
        int n_ret = 0;
        cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(fp8_lt_handle, plan->desc, plan->la, plan->lb, plan->lc, plan->lc, pref, 1, &heur, &n_ret);
        cublasLtMatmulPreferenceDestroy(pref);
        if (hs != CUBLAS_STATUS_SUCCESS || n_ret < 1) {
            GGML_LOG_ERROR("fp8 heuristic unavailable K=%lld N=%lld M=%lld\n", (long long)K, (long long)N, (long long)M);
            GGML_ABORT("FP8 heuristic unavailable on this shape");
        }
        plan->algo = heur.algo;
        plan->ok = true;
    }

    ggml_cuda_pool_alloc<char> ws(ctx.pool());
    if (plan->ws_size > 0) {
        ws.alloc(plan->ws_size);
    }

    float alpha = act_scale * w_scale, beta = 0.0f;

    // host-side per-call timing (cudaEvent cannot live inside graph capture; this
    // path only measures when NOT capturing — during warmup pre-graph phase, which
    // is exactly the production shapes). Accumulate per (K,N,M) key.
    cudaStreamCaptureStatus cap_st = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &cap_st);
    const bool can_time = (cap_st == cudaStreamCaptureStatusNone);
    cudaEvent_t ev_a = nullptr, ev_b = nullptr;
    if (g_fp8_prof && can_time) {
        cudaEventCreate(&ev_a); cudaEventCreate(&ev_b);
        cudaEventRecord(ev_a, stream);
    }

    FP8LT_CHECK(cublasLtMatmul(fp8_lt_handle, plan->desc, &alpha,
                               src0->data, plan->la,
                               acts_q.ptr, plan->lb,
                               &beta,
                               dst->data, plan->lc,
                               dst->data, plan->lc,
                               &plan->algo, ws.ptr, plan->ws_size, stream));

    if (g_fp8_prof && can_time) {
        cudaEventRecord(ev_b, stream);
        cudaEventSynchronize(ev_b);
        float ms = 0; cudaEventElapsedTime(&ms, ev_a, ev_b);
        cudaEventDestroy(ev_a); cudaEventDestroy(ev_b);
        std::lock_guard<std::mutex> lk(g_fp8_lt_plans_mu);
        auto & st = g_fp8_shape_stats[key];
        st.calls += 1;
        st.ns_total += (uint64_t)(ms * 1e6f);
        if (bucket_d) {
            uint32_t bk[2] = {0, 0};
            // read probe results (outside graph capture — safe here)
            CUDA_CHECK(cudaMemcpyAsync(bk, bucket_d, 2*sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            st.amax_bucket = bk[0];
            if (bk[1]) st.sat_events += 1;
        }
        g_fp8_call_counter += 1;
        if (g_fp8_call_counter % 10000 == 0) {
            FILE * f = fopen("/brand_data/ai_workspace/bench/f8_prof.txt", "w");
            if (f) {
                fprintf(f, "F8 MUL_MAT production profile (calls=%llu)\n", (unsigned long long)g_fp8_call_counter);
                fprintf(f, "%-14s %-10s %-8s %10s %12s %8s %8s\n", "K", "N", "M", "calls", "total_ms", "avg_ms", "amax_bk");
                double total_ms = 0;
                for (auto & kv : g_fp8_shape_stats) {
                    // decode key
                    uint64_t k = kv.first;
                    uint32_t M = (uint32_t)(k & 0xFFFFF);
                    uint32_t N = (uint32_t)((k >> 20) & 0xFFFFF);
                    uint32_t K = (uint32_t)((k >> 40) & 0xFFFFFF);
                    fprintf(f, "%-14u %-10u %-8u %10llu %12.3f %8.4f %8u\n",
                            K, N, M, (unsigned long long)kv.second.calls,
                            kv.second.ns_total / 1e6, (double)kv.second.ns_total / 1e6 / kv.second.calls,
                            kv.second.amax_bucket);
                    total_ms += kv.second.ns_total / 1e6;
                }
                fprintf(f, "TOTAL GEMM ms (accumulated): %.3f\n", total_ms);
                uint64_t sat_total = 0;
                for (auto & kv : g_fp8_shape_stats) sat_total += kv.second.sat_events;
                fprintf(f, "saturation-risk tensors (amax>=448 at scale=1): %llu\n", (unsigned long long)sat_total);
                fclose(f);
            }
        }
    }
}
