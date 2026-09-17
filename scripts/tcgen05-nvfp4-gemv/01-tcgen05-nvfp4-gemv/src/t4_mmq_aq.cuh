// t4_mmq_aq.cuh — T4 生产接线的**激活（A）量化器**：canonical 板布局 + GGML nibble 约定
//
// 语义来源：`quantize.cu:128-330` 的 `quantize_mmq_nvfp4`（scatter=false + use_aligned_float8 路径）逐字照抄
//   （F30：算子体只 cvt/shfl/fma，与 mma 无关；evidence/20260916-90 已证明可原样复用，且与从零手写的
//    CPU 语义逐字节一致）。**唯一改动 = 落点**：从「给 warp-MMA 的 mmq 布局」改成 canonical 板布局，
//   并且 nibble 配对保持 ggml 文件格式的 **(元素 l, 元素 l+8)**（与权重侧同约定；T4 的 (元素 → 槽位) 映射
//   由 t4_mmq_canon.h 唯一确定，宿主单测 T1/T3a/T4 已钉死）。
//
// 判据（上板）：与生产件同一次调用的 **int8/dp4a 路**相比，D 的相对误差在量化噪声量级；A 的 (m,k) nibble
//   与 sf 码与上游量化器**逐位相同**（同一条 x）。
#pragma once

#include "t4_mmq_canon.h"

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

namespace t4mmq {

// ---- 上游小工具（逐字照抄；原处为 static，故此处复述）----
struct __builtin_align__(32) aq_float8 { float x, y, z, w, p, q, r, s; };

__device__ __forceinline__ float aq_ue4m3_to_fp32(uint8_t x) {
    const uint32_t bits = x * (x != 0x7F && x != 0xFF);
    const __nv_fp8_e4m3 xf = *reinterpret_cast<const __nv_fp8_e4m3 *>(&bits);
    return static_cast<float>(xf) / 2;
}
__device__ __forceinline__ uint8_t aq_fp32_to_ue4m3(float x) {
    if (!(x > 0.0f)) return 0;
    const __nv_fp8_e4m3 xf(x);
    return xf.__x;
}
__device__ __forceinline__ float aq_warp_reduce_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
    return v;
}
#if CUDART_VERSION >= 12080
__device__ __forceinline__ float aq_nvfp4_native_scale_error(const float vals[K_SUB], const float inv_col_scale,
                                                             const float inv_scale, const float scale) {
    const float scale_dequant = 2.0f * scale;
    float err = 0.0f;
#pragma unroll
    for (int k = 0; k < K_SUB; k += 4) {
        const __nv_fp4x4_e2m1 q(make_float4(vals[k + 0] * inv_col_scale * inv_scale,
                                           vals[k + 1] * inv_col_scale * inv_scale,
                                           vals[k + 2] * inv_col_scale * inv_scale,
                                           vals[k + 3] * inv_col_scale * inv_scale));
        const __nv_fp4x4_storage_t q_storage = q.__x;
        const __nv_fp4x2_storage_t q_lo = static_cast<__nv_fp4x2_storage_t>(q_storage);
        const __nv_fp4x2_storage_t q_hi = static_cast<__nv_fp4x2_storage_t>(q_storage >> 8U);
        const __half2_raw hraw2_lo = __nv_cvt_fp4x2_to_halfraw2(q_lo, __NV_E2M1);
        const __half2_raw hraw2_hi = __nv_cvt_fp4x2_to_halfraw2(q_hi, __NV_E2M1);
        const __half2 h2_lo = *reinterpret_cast<const __half2 *>(&hraw2_lo);
        const __half2 h2_hi = *reinterpret_cast<const __half2 *>(&hraw2_hi);
        const float2 f2_lo = __half22float2(h2_lo);
        const float2 f2_hi = __half22float2(h2_hi);
        const float d0 = fmaf(f2_lo.x, scale_dequant, -vals[k + 0] * inv_col_scale);
        const float d1 = fmaf(f2_lo.y, scale_dequant, -vals[k + 1] * inv_col_scale);
        const float d2 = fmaf(f2_hi.x, scale_dequant, -vals[k + 2] * inv_col_scale);
        const float d3 = fmaf(f2_hi.y, scale_dequant, -vals[k + 3] * inv_col_scale);
        err = fmaf(d0, d0, err);
        err = fmaf(d1, d1, err);
        err = fmaf(d2, d2, err);
        err = fmaf(d3, d3, err);
    }
    return err;
}
#else
__device__ __forceinline__ float aq_nvfp4_native_scale_error(const float vals[K_SUB], const float inv_col_scale,
                                                             const float inv_scale, const float scale) {
    float err = 0.0f;
#pragma unroll
    for (int k = 0; k < K_SUB; ++k) {
        const float v = vals[k] * inv_col_scale;
        const __nv_fp4x4_e2m1 q(make_float4(v * inv_scale, 0.f, 0.f, 0.f));
        const uint8_t qb = static_cast<uint8_t>(q.__x & 0xF);
        static const float kv[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                                     -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
        const float d = fabsf(v) - fabsf(kv[qb & 0x7]) * scale;
        err = fmaf(d, d, err);
    }
    return err;
}
#endif

// ---- A 量化核体：**一行一个 block**（256 线程）----
//   抽成 device 函数是为了让板端对拍驱动（t4/m2c，Driver API 单 launch）能复用同一个核体，
//   避免"生产内核"与"对拍内核"两份实现漂移。
__device__ __forceinline__ void t4_mmq_quant_a_row(
        const float * __restrict__ x, size_t s01, unsigned char * __restrict__ a_nib,
        unsigned char * __restrict__ a_sf, float * __restrict__ row_scale, int M, int K, const int m) {
    const float * x_row = x + (size_t) m * s01;

    float amax = 0.0f;
    for (int i0 = 8 * threadIdx.x; i0 < K; i0 += 8 * blockDim.x) {
        const aq_float8 v = *reinterpret_cast<const aq_float8 *>(x_row + i0);
        amax = fmaxf(amax, fabsf(v.x)); amax = fmaxf(amax, fabsf(v.y));
        amax = fmaxf(amax, fabsf(v.z)); amax = fmaxf(amax, fabsf(v.w));
        amax = fmaxf(amax, fabsf(v.p)); amax = fmaxf(amax, fabsf(v.q));
        amax = fmaxf(amax, fabsf(v.r)); amax = fmaxf(amax, fabsf(v.s));
    }
    amax = aq_warp_reduce_max(amax);
    __shared__ float warp_amax[256 / 32];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (lane == 0) warp_amax[warp] = amax;
    __syncthreads();
    if (warp == 0) {
        amax = threadIdx.x < 256 / 32 ? warp_amax[lane] : 0.0f;
        amax = aq_warp_reduce_max(amax);
        if (lane == 0) warp_amax[0] = amax / (6.0f * 448.0f);
    }
    __syncthreads();

    const float row_scale_v = warp_amax[0];
    if (threadIdx.x == 0 && row_scale) row_scale[m] = row_scale_v;
    const float inv_col_scale = row_scale_v > 0.0f ? 1.0f / row_scale_v : 0.0f;

    const int n_subblocks = K / K_SUB;             // 每 16 元素一个子块
    for (int isb = threadIdx.x; isb < n_subblocks; isb += blockDim.x) {
        const int k0 = isb * K_SUB;
        float vals[K_SUB];
        const aq_float8 v0 = *reinterpret_cast<const aq_float8 *>(x_row + k0);
        const aq_float8 v1 = *reinterpret_cast<const aq_float8 *>(x_row + k0 + 8);
        vals[0] = v0.x; vals[1] = v0.y; vals[2] = v0.z;  vals[3] = v0.w;
        vals[4] = v0.p; vals[5] = v0.q; vals[6] = v0.r;  vals[7] = v0.s;
        vals[8] = v1.x; vals[9] = v1.y; vals[10] = v1.z; vals[11] = v1.w;
        vals[12] = v1.p; vals[13] = v1.q; vals[14] = v1.r; vals[15] = v1.s;

        float amax_sub = 0.0f;
#pragma unroll
        for (int k = 0; k < K_SUB; ++k) amax_sub = fmaxf(amax_sub, fabsf(vals[k] * inv_col_scale));

        static constexpr int test_offsets[5] = {0, -1, 1, -2, 2};
        const int first_fp8_code = (int) aq_fp32_to_ue4m3(amax_sub / 6.0f);
        uint8_t fp8_code = (uint8_t) first_fp8_code;
        float subblock_scale = aq_ue4m3_to_fp32(fp8_code);
        float inv_scale_err = subblock_scale > 0.0f ? 0.5f / subblock_scale : 0.0f;
        float best_err = aq_nvfp4_native_scale_error(vals, inv_col_scale, inv_scale_err, subblock_scale);
#pragma unroll
        for (int i = 1; i < 5; ++i) {
            const int test_code = first_fp8_code + test_offsets[i];
            if (test_code < 0 || test_code > 0x7e) continue;
            const float test_scale = aq_ue4m3_to_fp32((uint8_t) test_code);
            const float test_inv = test_scale > 0.0f ? 0.5f / test_scale : 0.0f;
            const float cur_err = aq_nvfp4_native_scale_error(vals, inv_col_scale, test_inv, test_scale);
            if (cur_err < best_err) { best_err = cur_err; fp8_code = (uint8_t) test_code; subblock_scale = test_scale; }
        }

        const float inv_scale = subblock_scale > 0.0f ? 0.5f / subblock_scale : 0.0f;
        const float s = inv_col_scale * inv_scale;
        // 上游同款打包：q0 的字节 l = (元素 l, 元素 l+8)；q1 的字节 l = (元素 4+l, 元素 12+l)
        const __nv_fp4x4_e2m1 p0(make_float4(vals[0] * s, vals[8]  * s, vals[1] * s, vals[9]  * s));
        const __nv_fp4x4_e2m1 p1(make_float4(vals[2] * s, vals[10] * s, vals[3] * s, vals[11] * s));
        const __nv_fp4x4_e2m1 p2(make_float4(vals[4] * s, vals[12] * s, vals[5] * s, vals[13] * s));
        const __nv_fp4x4_e2m1 p3(make_float4(vals[6] * s, vals[14] * s, vals[7] * s, vals[15] * s));
        const char2 c0 = *reinterpret_cast<const char2 *>(&p0);
        const char2 c1 = *reinterpret_cast<const char2 *>(&p1);
        const char2 c2 = *reinterpret_cast<const char2 *>(&p2);
        const char2 c3 = *reinterpret_cast<const char2 *>(&p3);

        const int k64 = k0 / MMA_K, sub = (k0 % MMA_K) / K_SUB;   // 板号 / 板内子块号
        unsigned char * dst = a_nib + a_nib_off_in_board(m, sub, 0) + (size_t) k64 * A_BD_STRIDE;
        dst[0] = (unsigned char) c0.x; dst[1] = (unsigned char) c0.y;
        dst[2] = (unsigned char) c1.x; dst[3] = (unsigned char) c1.y;
        dst[4] = (unsigned char) c2.x; dst[5] = (unsigned char) c2.y;
        dst[6] = (unsigned char) c3.x; dst[7] = (unsigned char) c3.y;
        if (a_sf) a_sf[(size_t) k64 * A_SF_STRIDE + (size_t) m * 4 + sub] = fp8_code;
    }
}

// 生产入口：一个 block 一行（grid = M，256 线程）
__global__ void __launch_bounds__(256) t4_mmq_quant_a(
        const float * __restrict__ x, size_t s01, unsigned char * __restrict__ a_nib,
        unsigned char * __restrict__ a_sf, float * __restrict__ row_scale, int M, int K) {
    const int m = blockIdx.x;
    if (m >= M) return;
    t4_mmq_quant_a_row(x, s01, a_nib, a_sf, row_scale, M, K, m);
}

// ---- D 转置：N 主序 [n][m] → 生产 dst 的行主序 [m*sd + n]（F21：内核绝不行主序散写）----
__global__ void t4_mmq_transpose_nm(const float * __restrict__ D, float * __restrict__ dst,
                                    int M, int N, size_t sd) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) M * (size_t) N) return;
    const int m = (int) (i / (size_t) N), n = (int) (i % (size_t) N);
    dst[(size_t) m * sd + (size_t) n] = D[(size_t) n * (size_t) M + (size_t) m];
}

}  // namespace t4mmq
