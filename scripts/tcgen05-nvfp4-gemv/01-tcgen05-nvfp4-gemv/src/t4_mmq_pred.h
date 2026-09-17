#pragma once

// t4_mmq_pred.h — T4（自研 tcgen05 NVFP4 快路）接缝的**纯判定**。
//
// 为什么单独一个头：让"什么形状/类型/对齐才算有资格"这件事**可以被宿主单测直接编译验证**
// （本头无 CUDA、无 ggml 依赖）。生产侧只在 t4_mmq.cu 里把 mmq_args 填进 t4_mmq_pred_in，
// 并用 static_assert 把下面这几个"魔法常量"钉死在 ggml/CUDA 的真值上（防漂移）。
//
// 语义：返回 true = 落在**已实测验证过**的集合内（"有资格"）。**命中不等于会走 T4** ——
// 是否真的换路，由 t4_mmq_try_launch 的开关（T4_MMQ + 内核已接线）决定。
//
// 已实测验证的来源（改动这些条件前先回去读）：
//   - 类型/形状/对齐：results/20260916-108（M2b 跨板）、109（canonical 片段映射）、112（真实张量纯拷贝逐位全对）
//   - 真实生产形状：gate/up K=5120 · down K=17408 · 权重行步长（字节）= (K/64)*36（K=5120 → 2880 B）
//     · decode 每步 token 数 M=13（spec n_max=7），MMA 侧按 MREAL 压 A；M=128 为原型全宽
// ⚠ 口径（2026-09-16 20:4x 更正，之前写错导致 can_handle 恒 false）：
//   `stride_row_x` 传进来的是 **mmq_args.stride_row_x = nb[1]/ggml_type_size(NVFP4)**，即**块数**
//   （K=5120 → 2880/36 = **80**），**不是字节数**。字节步长 = `stride_row_x * 36`（36 = sizeof(block_nvfp4)）。
//   历史版本按字节校验（要求 ≥ (K/64)*36 = 2880）⇒ 真实形状 80 被拒 ⇒ 整条路静默不触发。
//   块的字节步长恒为 4 的倍数（36·n），故**行步长不再有对齐约束**；只要装得下 K/64 条记录即可。

#define T4_MMQ_CC_THOR        1010    // = GGML_CUDA_CC_THOR_SM101A
#define T4_MMQ_TYPE_NVFP4     40      // = GGML_TYPE_NVFP4
#define T4_MMQ_TYPE_F32        0      // = GGML_TYPE_F32（激活只读对过 F32）
#define T4_MMQ_KCHUNK         256     // 环的 K chunk 粒度（chunks ≤ K/256）
#define T4_MMQ_NTILE          32      // TMEM/SF 的 n-tile 宽度
#define T4_MMQ_M_MAX          128     // TMEM 累加器 lane 数（A 侧 ≤ M_MAX，超出部分补零）

struct t4_mmq_pred_in {
    int       cc;              // 运行时 compute capability
    int       arch;            // 本 TU 编译期支持的最高 arch（ggml_cuda_highest_compiled_arch(cc)）
    int       type_x;          // src0 的 ggml_type
    bool      ids_branch;      // ids_dst / expert_bounds 非空（batched/ids 分支）
    long long ncols_x;         // K（ne00）
    long long nrows_x;         // N（ne01，权重行数 / 输出维）
    long long ncols_dst;       // M（ne1，本步 token 数）
    long long stride_row_x;    // 权重行步长，**单位 = NVFP4 块**（= nb[1]/36）；字节 = 36×此值
    long long nchannels_y;     // ne12（>1 = 通道批，未验证）
    long long nsamples_y;      // ne13（>1 = 多样本，未验证）
    long long nchannels_x;     // 权重的 ne02（>1 = 通道批权重，未验证）
    long long nsamples_x;      // 权重的 ne03（>1 = 多样本权重，未验证）
    int       type_y;          // src1（激活）的 ggml_type：只支持 F32（0）——其余类型我们没读对过
    bool      act32_ok;        // src1（fp32 激活）基址 32 B 对齐且行步长（float 数）% 8 == 0
    bool      ptr4_ok;         // 权重/激活/dst 基址 4 B 对齐
};

// 判定结果（2026-09-16 21:0x：从 bool 改成**原因码**——首次接线时"没命中却看不出为什么"，
// 白白花掉一次板端臂；t4_mmq.cu 现在按原因码打一次性日志）。
enum {
    T4_MMQ_OK = 0,
    T4_MMQ_REJ_CC,
    T4_MMQ_REJ_ARCH,
    T4_MMQ_REJ_TYPE_X,
    T4_MMQ_REJ_TYPE_Y,
    T4_MMQ_REJ_IDS,
    T4_MMQ_REJ_NDIM,
    T4_MMQ_REJ_NDIM_X,
    T4_MMQ_REJ_K,
    T4_MMQ_REJ_N,
    T4_MMQ_REJ_M,
    T4_MMQ_REJ_STRIDE,
    T4_MMQ_REJ_ACT32,
    T4_MMQ_REJ_PTR4,
    T4_MMQ_REJ_LAST
};

static inline const char * t4_mmq_pred_reason_name(const int r) {
    switch (r) {
        case T4_MMQ_OK:          return "OK";
        case T4_MMQ_REJ_CC:      return "cc!=1010";
        case T4_MMQ_REJ_ARCH:    return "arch!=1010";
        case T4_MMQ_REJ_TYPE_X:  return "type_x!=NVFP4";
        case T4_MMQ_REJ_TYPE_Y:  return "type_y!=F32";
        case T4_MMQ_REJ_IDS:     return "ids 分支";
        case T4_MMQ_REJ_NDIM:    return "ne12/ne13!=1";
        case T4_MMQ_REJ_NDIM_X:  return "权重 ne02/ne03!=1";
        case T4_MMQ_REJ_K:       return "K%256!=0";
        case T4_MMQ_REJ_N:       return "N%32!=0";
        case T4_MMQ_REJ_M:       return "M 越界(0,128]";
        case T4_MMQ_REJ_STRIDE:  return "行步长装不下 K/64";
        case T4_MMQ_REJ_ACT32:   return "激活非 32B 对齐";
        case T4_MMQ_REJ_PTR4:    return "基址非 4B 对齐";
        default:                 return "?";
    }
}

static inline int t4_mmq_pred_reason(const t4_mmq_pred_in & s) {
    if (s.cc != T4_MMQ_CC_THOR || s.arch != T4_MMQ_CC_THOR) {
        return s.cc != T4_MMQ_CC_THOR ? T4_MMQ_REJ_CC : T4_MMQ_REJ_ARCH;   // 只在 Thor/sm_101a 上验证过
    }
    if (s.type_x != T4_MMQ_TYPE_NVFP4) {
        return T4_MMQ_REJ_TYPE_X;                       // MXFP4/Q4_0 等未验，不接（避免顺手吸进来）
    }
    if (s.type_y != T4_MMQ_TYPE_F32) {
        return T4_MMQ_REJ_TYPE_Y;                       // 激活只读对过 F32（F16 会被当 F32 读 = 静默数值灾难）
    }
    if (s.ids_branch) {
        return T4_MMQ_REJ_IDS;                          // ids/batched 分支未覆盖
    }
    if (s.nchannels_y != 1 || s.nsamples_y != 1) {
        return T4_MMQ_REJ_NDIM;                         // 只验证过 2D（单通道单样本）
    }
    if (s.nchannels_x != 1 || s.nsamples_x != 1) {
        return T4_MMQ_REJ_NDIM_X;                       // 权重也必须是 2D（批权重未验证）
    }
    if (s.ncols_x < T4_MMQ_KCHUNK || s.ncols_x % T4_MMQ_KCHUNK != 0) {
        return T4_MMQ_REJ_K;                            // K 必须是 KCHUNK 的整数倍
    }
    if (s.nrows_x <= 0 || s.nrows_x % T4_MMQ_NTILE != 0) {
        return T4_MMQ_REJ_N;                            // N 必须按 n-tile 整除
    }
    if (s.ncols_dst <= 0 || s.ncols_dst > T4_MMQ_M_MAX) {
        return T4_MMQ_REJ_M;                            // A 侧 0 < M ≤ 128（其余补零）
    }
    if (s.stride_row_x < s.ncols_x / 64) {
        return T4_MMQ_REJ_STRIDE;                       // 行步长（块数）至少要装得下 K/64 条 36 B 记录
    }
    if (!s.act32_ok) {
        return T4_MMQ_REJ_ACT32;                        // 激活行不足 32 B ⇒ aq_float8 向量载入会非对齐（A 类事故）
    }
    if (!s.ptr4_ok) {
        return T4_MMQ_REJ_PTR4;                         // 基址 4 B 对齐
    }
    return T4_MMQ_OK;
}

static inline bool t4_mmq_pred_ok(const t4_mmq_pred_in & s) {
    return t4_mmq_pred_reason(s) == T4_MMQ_OK;
}
