// fp8-mmq.cu — FP8 MMQ kernel for Thor sm_101a
// FP8 E4M3 weights × FP8 input → FP32 output via MMA
// 编译: nvcc -arch=sm_101a -c fp8-mmq.cu

#include "fp8-mma.cuh"
#include "dequantize.cuh"
#include <cstdint>

// FP8 MMQ kernel parameters
// M = rows of output (batch size)
// N = cols of output (ne01 = number of rows in weight matrix)  
// K = inner dimension (ne00 = number of cols in weight matrix)
// 
// 计算: C[M×N] = A[M×K] × B[N×K]^T
// 其中 A 是输入 (FP8), B 是权重 (FP8), C 是输出 (FP32)

// Tile sizes (per warp-block)
#define TILE_M 64   // 4 MMA tiles in M dimension (4 × 16)
#define TILE_N 32   // 4 MMA tiles in N dimension (4 × 8)
#define TILE_K 128  // 4 MMA tiles in K dimension (4 × 32)

// 共享内存大小
#define SHMEM_A_SIZE (TILE_M * TILE_K)  // 8192 bytes
#define SHMEM_B_SIZE (TILE_N * TILE_K)  // 4096 bytes

// FP8 MMQ kernel
// grid: (N/TILE_N, M/TILE_M, 1)
// block: (256, 1, 1) = 8 warps
__global__ void fp8_mmq_kernel(
    float * __restrict__ C,           // [M × N] output
    const uint8_t * __restrict__ A,   // [M × K] input (FP8 E4M3)
    const uint8_t * __restrict__ B,   // [N × K] weight (FP8 E4M3)
    int M, int N, int K,
    int stride_a,  // A stride in bytes (K dimension)
    int stride_b,  // B stride in bytes (K dimension)
    int stride_c)  // C stride in floats (N dimension)
{
    // 共享内存
    __shared__ uint8_t shmem_a[SHMEM_A_SIZE];
    __shared__ uint8_t shmem_b[SHMEM_B_SIZE];
    
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    // 块索引
    int block_m = blockIdx.y * TILE_M;
    int block_n = blockIdx.x * TILE_N;
    
    // 每个 warp 处理 16×32 的 MMA tile
    // 8 个 warp 排列: 4 行 (M) × 2 列 (N)
    int warp_row = warp_id / 2;  // 0-3
    int warp_col = warp_id % 2;  // 0-1
    
    // 累加器初始化
    fp8_mma_frag_d acc[2][4];  // [N_tiles][M_tiles]
    for (int ni = 0; ni < 2; ni++)
        for (int mi = 0; mi < 4; mi++)
            fp8_mma_frag_d_zero(acc[ni][mi]);
    
    // 主循环: 沿 K 维度分块
    for (int k_offset = 0; k_offset < K; k_offset += TILE_K) {
        // 协作加载 A 到共享内存
        // A[block_m : block_m+TILE_M, k_offset : k_offset+TILE_K]
        for (int i = threadIdx.x; i < SHMEM_A_SIZE; i += blockDim.x) {
            int row = i / TILE_K;
            int col = i % TILE_K;
            int global_row = block_m + row;
            int global_col = k_offset + col;
            
            if (global_row < M && global_col < K)
                shmem_a[i] = A[global_row * stride_a + global_col];
            else
                shmem_a[i] = 0;
        }
        
        // 协作加载 B 到共享内存
        // B[block_n : block_n+TILE_N, k_offset : k_offset+TILE_K]
        for (int i = threadIdx.x; i < SHMEM_B_SIZE; i += blockDim.x) {
            int row = i / TILE_K;
            int col = i % TILE_K;
            int global_row = block_n + row;
            int global_col = k_offset + col;
            
            if (global_row < N && global_col < K)
                shmem_b[i] = B[global_row * stride_b + global_col];
            else
                shmem_b[i] = 0;
        }
        
        __syncthreads();
        
        // MMA 计算: 每个 warp 处理自己的 tile
        for (int k_inner = 0; k_inner < TILE_K; k_inner += FP8_MMA_K) {
            fp8_mma_frag_a frag_a;
            fp8_mma_frag_b frag_b;
            
            // 加载 A fragment (16×32)
            fp8_load_a_frag(frag_a, shmem_a, warp_row, k_inner / FP8_MMA_K, TILE_K);
            
            // 加载 B fragment (32×8)
            fp8_load_b_frag(frag_b, shmem_b, warp_col, k_inner / FP8_MMA_K, TILE_K);
            
            // 执行 MMA
            fp8_mma(acc[warp_col][warp_row], frag_a, frag_b);
        }
        
        __syncthreads();
    }
    
    // 写回结果
    int out_row = block_m + warp_row * FP8_MMA_M + (lane_id / 4);
    int out_col = block_n + warp_col * FP8_MMA_N + (lane_id % 4) * 2;
    
    if (out_row < M) {
        for (int ni = 0; ni < 2; ni++) {
            int col = out_col + ni;
            if (col < N) {
                // 简化写回: 直接写 acc 的前两个元素
                // 实际需要按 MMA fragment layout 反变换
                C[out_row * stride_c + col] = acc[ni][warp_row].x[0];
            }
        }
    }
}
