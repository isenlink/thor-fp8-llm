// fp8-mmq-production.cu — FP8 MMQ production kernel for Thor sm_101a
// 基于验证过的 fragment 布局，实现完整 tiled matmul
// 编译: nvcc-xc128.sh -arch=sm_101a -c fp8-mmq-production.cu
//
// 计算: D[M×N] = A[M×K] × B[K×N]^T
// A: FP8 E4M3 (input activations, quantized from FP32 on-the-fly)
// B: FP8 E4M3 (weight matrix, pre-quantized)
// D: FP32 (output)
//
// Tile sizes:
// BM=128, BN=128, BK=64 (每步处理64个K元素)
// 每个 warp 处理16×8 MMA tile (m16n8k32)
// 每个 thread block 包含8×16=128个 MMA tiles

#include <cstdio>
#include <cstdint>
#include <cuda_fp8.h>

// MMA tile dimensions
#define MMA_M 16
#define MMA_N 8
#define MMA_K 32

// Block tile dimensions (in MMA tiles)
#define BM 128  // 8 MMA tiles in M dimension
#define BN 128  // 16 MMA tiles in N dimension
#define BK 64   // 2 MMA tiles in K dimension

// Thread block dimensions
#define BLOCK_M 8   // warps in M dimension
#define BLOCK_N 16  // warps in N dimension
#define BLOCK_K 2   // MMA tiles in K per iteration
#define NUM_WARPS (BLOCK_M * BLOCK_N)

// Shared memory dimensions
#define SMEM_A_M BM
#define SMEM_A_K BK
#define SMEM_B_K BK
#define SMEM_B_N BN

// FP8 E4M3 转换 (device)
__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t val) {
    int sign = (val >> 7) & 1;
    int exp = (val >> 3) & 0xF;
    int man = val & 0x7;
    float result;
    if (exp == 0) {
        result = man * 0.001953125f;  // 2^-9
    } else {
        result = (1.0f + man * 0.125f) * ldexpf(1.0f, exp - 7);
    }
    return sign ? -result : result;
}

__device__ __forceinline__ uint8_t float_to_fp8_e4m3(float val) {
    // 快速近似转换 (用于输入量化)
    int bits = __float_as_int(val);
    int sign = (bits >> 31) & 1;
    int exp = ((bits >> 23) & 0xFF) - 127 + 7;
    int man = (bits >> 16) & 0x7F;
    
    if (exp <= 0) return (sign << 7);  // underflow to zero
    if (exp > 15) { exp = 15; man = 7; }  // clamp to max
    
    return (sign << 7) | (exp << 3) | (man >> 4);
}

// FP8 MMA 指令 (单个 tile)
__device__ __forceinline__ void fp8_mma_tile(
    float &d0, float &d1, float &d2, float &d3,
    int a0, int a1, int a2, int a3,
    int b0, int b1) 
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
}

// FP8 MMQ kernel
// grid: (BN/BN_tile, BM/BM_tile, 1) = (1, 1, 1) for 128×128 output
// block: (NUM_WARPS * 32, 1, 1) = (128 * 32, 1, 1)
__global__ void fp8_mmq_kernel(
    float * __restrict__ D,           // [M × N] output
    const uint8_t * __restrict__ A,   // [M × K] input (FP8)
    const uint8_t * __restrict__ B,   // [N × K] weight (FP8, transposed)
    int M, int N, int K,
    int stride_a,  // A stride in bytes
    int stride_b,  // B stride in bytes
    int stride_d)  // D stride in floats
{
    // 共享内存
    __shared__ uint8_t smem_a[SMEM_A_M * SMEM_A_K];  // 128×64 = 8192 bytes
    __shared__ uint8_t smem_b[SMEM_B_K * SMEM_B_N];  // 64×128 = 8192 bytes
    
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    // 每个 warp 的 MMA tile 索引
    int warp_m = warp_id / BLOCK_N;  // 0..7
    int warp_n = warp_id % BLOCK_N;  // 0..15
    
    // 累加器 (每个 warp 处理一个16×8 tile)
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    
    // 主循环: 沿 K 维度分块
    for (int k_base = 0; k_base < K; k_base += BK) {
        // 协作加载 A 到共享内存
        // A[block_m*BM : block_m*BM+BM, k_base : k_base+BK]
        for (int i = threadIdx.x; i < SMEM_A_M * SMEM_A_K; i += blockDim.x) {
            int row = i / SMEM_A_K;
            int col = i % SMEM_A_K;
            int global_row = blockIdx.y * BM + row;
            int global_col = k_base + col;
            
            if (global_row < M && global_col < K)
                smem_a[i] = A[global_row * stride_a + global_col];
            else
                smem_a[i] = 0;
        }
        
        // 协作加载 B 到共享内存
        // B[block_n*BN : block_n*BN+BN, k_base : k_base+BK]
        for (int i = threadIdx.x; i < SMEM_B_K * SMEM_B_N; i += blockDim.x) {
            int row = i / SMEM_B_N;
            int col = i % SMEM_B_N;
            int global_row = blockIdx.x * BN + row;
            int global_col = k_base + col;
            
            if (global_row < N && global_col < K)
                smem_b[i] = B[global_row * stride_b + global_col];
            else
                smem_b[i] = 0;
        }
        
        __syncthreads();
        
        // MMA 计算
        for (int k_inner = 0; k_inner < BK; k_inner += MMA_K) {
            // 加载 A fragment (16×32)
            int g = lane_id / 4;
            int t = lane_id % 4;
            
            int a_row0 = warp_m * MMA_M + g;
            int a_row1 = a_row0 + 8;
            int a_col_base = k_inner + 4 * t;
            
            uint8_t a_bytes[16];
            for (int i = 0; i < 4; i++) {
                a_bytes[i]    = smem_a[a_row0 * SMEM_A_K + a_col_base + i];
                a_bytes[4+i]  = smem_a[a_row1 * SMEM_A_K + a_col_base + i];
                a_bytes[8+i]  = smem_a[a_row0 * SMEM_A_K + a_col_base + 16 + i];
                a_bytes[12+i] = smem_a[a_row1 * SMEM_A_K + a_col_base + 16 + i];
            }
            
            int a0 = *(int*)(a_bytes);
            int a1 = *(int*)(a_bytes+4);
            int a2 = *(int*)(a_bytes+8);
            int a3 = *(int*)(a_bytes+12);
            
            // 加载 B fragment (32×8)
            int b_row_base = k_inner + 4 * t;
            int b_col = warp_n * MMA_N + g;
            
            uint8_t b_bytes[8];
            for (int i = 0; i < 4; i++) {
                b_bytes[i]   = smem_b[(b_row_base + i) * SMEM_B_N + b_col];
                b_bytes[4+i] = smem_b[(b_row_base + 16 + i) * SMEM_B_N + b_col];
            }
            
            int b0 = *(int*)(b_bytes);
            int b1 = *(int*)(b_bytes+4);
            
            // 执行 MMA
            fp8_mma_tile(acc0, acc1, acc2, acc3, a0, a1, a2, a3, b0, b1);
        }
        
        __syncthreads();
    }
    
    // 写回结果
    // D fragment: d0=(g,2t), d1=(g,2t+1), d2=(g+8,2t), d3=(g+8,2t+1)
    int g = lane_id / 4;
    int t = lane_id % 4;
    
    int out_row0 = blockIdx.y * BM + warp_m * MMA_M + g;
    int out_row1 = out_row0 + 8;
    int out_col0 = blockIdx.x * BN + warp_n * MMA_N + 2 * t;
    int out_col1 = out_col0 + 1;
    
    if (out_row0 < M && out_col0 < N) D[out_row0 * stride_d + out_col0] = acc0;
    if (out_row0 < M && out_col1 < N) D[out_row0 * stride_d + out_col1] = acc1;
    if (out_row1 < M && out_col0 < N) D[out_row1 * stride_d + out_col0] = acc2;
    if (out_row1 < M && out_col1 < N) D[out_row1 * stride_d + out_col1] = acc3;
}
