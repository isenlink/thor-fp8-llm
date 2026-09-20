// fp8-mma.cuh — FP8 E4M3 MMA kernel for Thor sm_101a
// 基于 mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
// 每个 MMA tile: D[16x8] = A[16x32] * B[32x8] + C[16x8]

#ifndef FP8_MMA_CUH
#define FP8_MMA_CUH

#include "common.cuh"
#include "ggml-common.h"

// FP8 MMA tile dimensions
#define FP8_MMA_M 16
#define FP8_MMA_N 8
#define FP8_MMA_K 32

// 每个 warp thread 持有的寄存器数:
// A: 16x32 FP8 = 512 bytes / 32 threads = 16 bytes = 4 × int32
// B: 32x8 FP8 = 256 bytes / 32 threads = 8 bytes = 2 × int32
// D: 16x8 FP32 = 512 bytes / 32 threads = 16 bytes = 4 × float

struct fp8_mma_frag_a {
    int x[4];  // 4 × int32, each holding 4 × FP8 bytes
};

struct fp8_mma_frag_b {
    int x[2];  // 2 × int32, each holding 4 × FP8 bytes
};

struct fp8_mma_frag_d {
    float x[4];  // 4 × float accumulator
};

// 执行单个 FP8 MMA 指令
static __device__ __forceinline__ void fp8_mma(
        fp8_mma_frag_d & D,
        const fp8_mma_frag_a & A,
        const fp8_mma_frag_b & B) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(D.x[0]), "+f"(D.x[1]), "+f"(D.x[2]), "+f"(D.x[3])
        : "r"(A.x[0]), "r"(A.x[1]), "r"(A.x[2]), "r"(A.x[3]),
          "r"(B.x[0]), "r"(B.x[1])
    );
}

// 从共享内存加载 A fragment (16x32 FP8)
// 每个 warp thread 加载自己的部分
// 寄存器布局: 参考 PTX ISA Figure 90/91
// 简化版: 连续内存加载 (实际需要按 MMA fragment layout 重排)
static __device__ __forceinline__ void fp8_load_a_frag(
        fp8_mma_frag_a & frag,
        const uint8_t * shmem,
        int row,      // tile row index
        int col,      // tile column index (in K dimension)
        int stride) { // shared memory stride (in bytes)
    // 每个 thread 加载 4 个 int32 (16 bytes = 16 FP8 values)
    // 简化: 假设连续布局 (需要根据实际 MMA fragment layout 调整)
    int tid = threadIdx.x % 32;  // warp lane id
    int src_row = row * FP8_MMA_M + (tid / 4);
    int src_col = col * FP8_MMA_K + (tid % 4) * 4;
    
    const uint8_t * src = shmem + src_row * stride + src_col;
    frag.x[0] = *((const int *)(src + 0));
    frag.x[1] = *((const int *)(src + 16));
    frag.x[2] = *((const int *)(src + 32));
    frag.x[3] = *((const int *)(src + 48));
}

// 从共享内存加载 B fragment (32x8 FP8)
static __device__ __forceinline__ void fp8_load_b_frag(
        fp8_mma_frag_b & frag,
        const uint8_t * shmem,
        int row,      // tile row index (in K dimension)
        int col,      // tile column index
        int stride) {
    int tid = threadIdx.x % 32;
    int src_row = row * FP8_MMA_K + (tid % 8) * 4;
    int src_col = col * FP8_MMA_N;
    
    const uint8_t * src = shmem + src_row * stride + src_col;
    frag.x[0] = *((const int *)(src + 0));
    frag.x[1] = *((const int *)(src + stride * 4));
}

// 初始化 D fragment 为零
static __device__ __forceinline__ void fp8_mma_frag_d_zero(fp8_mma_frag_d & frag) {
    frag.x[0] = 0.0f;
    frag.x[1] = 0.0f;
    frag.x[2] = 0.0f;
    frag.x[3] = 0.0f;
}

// 累加两个 D fragments
static __device__ __forceinline__ void fp8_mma_frag_d_add(
        fp8_mma_frag_d & dst,
        const fp8_mma_frag_d & src) {
    dst.x[0] += src.x[0];
    dst.x[1] += src.x[1];
    dst.x[2] += src.x[2];
    dst.x[3] += src.x[3];
}

#endif // FP8_MMA_CUH
