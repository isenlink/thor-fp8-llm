// b3-microbench4.cu — 验证"y 向量重复 L2 读取"假设
// 真实 kernel（mmvq.cu:696）：每个行 block 除读权重外，还要读整段 y（Q8_1, 5120B）
//   y 只有 5KB 全驻 L2，但 17408 个 block 各读一遍 → y 的 L2 流量 ≈ 28MB（权重 DRAM 流量的 55%）
// 3 变体（权重读取模式与 microbench3 完全一致，保证可比）：
//   V1_full_y   : 权重(LUT+dp4a) + y 读取（真实 kernel 的访存）
//   V2_full_noy : 权重(LUT+dp4a)，无 y（= microbench3 V1，对照）
//   V3_noy      : 权重纯 load，无 y（= microbench3 V3，对照）
// V1-V2 = y 重复 L2 读取的成本
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

#define CK(x) do { auto e_=(x); if(e_!=0){printf("CUDA err %d @%d\n",(int)e_,__LINE__);exit(1);} } while(0)

static const int N = 17408;
static const int K = 5120;
static const int NB = K / 64;   // 80 个 NVFP4 block
static const int NY = K / 32;   // 160 个 Q8_1 block（32B）
struct block_nvfp4 { uint8_t d[4]; uint8_t qs[32]; };  // 36B
struct block_q8_1  { int16_t d; uint8_t qs[32]; };     // 34B

static const int8_t h_lut[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
__device__ int8_t d_lut[16];

// block_q8_1.qs 在 offset 2（非 4B 对齐），禁止 (const int*) 直接读（向量化后非法访问）
// 真实 kernel 用 get_int_b4 逐字节构造，这里复刻
static __device__ __forceinline__ int ld4(const uint8_t * p) {
    return (int)(p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24));
}

static __device__ __forceinline__ int2 lut(const int & q4) {
    const uint32_t * t = (const uint32_t *) d_lut;
    uint32_t tmp[2];
    const uint32_t sel = (0x32103210 | ((q4 & 0x88888888) >> 1));
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t sh = 16 * i;
        const uint32_t low  = __byte_perm(t[0], t[1], q4 >> sh);
        const uint32_t high = __byte_perm(t[2], t[3], q4 >> sh);
        tmp[i] = __byte_perm(low, high, sel >> sh);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

enum MODE { FULL_Y=0, FULL_NOY=1, NOY=2 };

template<int MODE>
__global__ void k(const block_nvfp4 * __restrict__ w, const block_q8_1 * __restrict__ y, float * __restrict__ out) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;  // 0..31
    const block_nvfp4 * bq4 = w + (size_t)row * NB;
    const block_q8_1 * by  = y;  // M=1：所有行共享同一 y
    const int kqs = 4 * (tid % 2);
    float sum = 0.0f;
    for (int iter = 0; iter < NB/16; iter++) {  // 5
        int kbx = tid/2 + iter*16;
        const int * qs = (const int *) bq4[kbx].qs;
        // y 读取（精确复刻 vecdotq.cuh:349-355）：
        //   y block 索引 = 2*kbx + (tid%2) = tid + iter*32
        //   每线程读该 block 完整 32B qs（8 个 int）
        const uint8_t * yq = by[tid + iter*32].qs;
        if (MODE == NOY) {
#pragma unroll
            for (int i = 0; i < 8; i++) sum += (float)qs[i];
        } else {
            float ys = 0.0f;
            if (MODE == FULL_Y) {
#pragma unroll
                for (int i = 0; i < 8; i++) ys += (float)ld4(yq + 4*i);  // 每线程 32B y
            }
#pragma unroll
            for (int i = 0; i < 2; i++) {
                int iqs0 = kqs + 2*i;
                int iqs1 = iqs0 + 1;
                int v0 = lut(qs[iqs0]).x;
                int v1 = lut(qs[iqs1]).x;
                int sumi = __dp4a(v0, 0x11111111, 0);
                sumi = __dp4a(v1, 0x22222222, sumi);
                sum += (float)sumi * (float)bq4[kbx].d[iqs0>>1];
            }
            sum += ys * 0.0001f;  // 防 DCE
        }
    }
    out[row] = sum;
}

static double run(int mode, int iters) {
    size_t wbytes = (size_t)N * NB * sizeof(block_nvfp4);
    size_t ybytes = (size_t)NY * sizeof(block_q8_1);  // 5440B
    block_nvfp4 * d_w; block_q8_1 * d_y; float * d_out;
    CK(cudaMalloc(&d_w, wbytes));
    CK(cudaMalloc(&d_y, ybytes));
    CK(cudaMalloc(&d_out, N * sizeof(float)));
    std::vector<uint8_t> hw(wbytes);
    for (auto & b : hw) b = (uint8_t)rand();
    CK(cudaMemcpy(d_w, hw.data(), wbytes, cudaMemcpyHostToDevice));
    std::vector<uint8_t> hy(ybytes);
    for (auto & b : hy) b = (uint8_t)rand();
    CK(cudaMemcpy(d_y, hy.data(), ybytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpyToSymbol(d_lut, h_lut, 16));

    int blocks = N, threads = 32;
    auto launch = [&](){
        switch (mode) {
            case FULL_Y:   k<FULL_Y><<<blocks,threads>>>(d_w, d_y, d_out); break;
            case FULL_NOY: k<FULL_NOY><<<blocks,threads>>>(d_w, d_y, d_out); break;
            case NOY:      k<NOY><<<blocks,threads>>>(d_w, d_y, d_out); break;
        }
    };
    for (int i = 0; i < 5; i++) launch();
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0));
    for (int i = 0; i < iters; i++) launch();
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms=0; CK(cudaEventElapsedTime(&ms,e0,e1)); ms/=iters;
    double gbs = wbytes / (ms*1e-3) / 1e9;
    printf("  V%d  %8.3f ms/iter  =  %7.1f GB/s (权重口径)\n", mode, ms, gbs);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaFree(d_w); cudaFree(d_y); cudaFree(d_out);
    return gbs;
}

int main(int argc, char ** argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 200;
    printf("=== B3 microbench4: y 重复 L2 读取假设 (N=%d K=%d iters=%d) ===\n", N, K, iters);
    printf("  权重=%.1fMB  y=%.1fKB(驻L2)  y总L2流量=%.1fMB\n",
           (double)N*NB*sizeof(block_nvfp4)/1e6, (double)NY*sizeof(block_q8_1)/1e3,
           (double)N*NY*sizeof(block_q8_1)/1e6);
    double v1 = run(FULL_Y, iters);
    double v2 = run(FULL_NOY, iters);
    double v3 = run(NOY, iters);
    printf("--- 拆解 ---\n");
    printf("  纯 load 无 y (V3)        = %7.1f GB/s\n", v3);
    printf("  full 无 y (V2)           = %7.1f GB/s\n", v2);
    printf("  full 有 y (V1, 真实访存) = %7.1f GB/s\n", v1);
    printf("  y 重复读取成本 (V2-V1)   = %7.1f GB/s  (占 V1 的 %.0f%%)\n", v2 - v1, (v2-v1)/v1*100);
    printf("  对照 standalone 真实 kernel = 144 GB/s\n");
    printf("=== done ===\n");
    return 0;
}
