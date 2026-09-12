// b3-microbench5.cu — 验证"增大 rows_per_block 降低 y 重复 L2 读取"优化
// 真实 kernel（mmvq.cu）M=1 NVFP4：
//   - y 读 2 路拆分：偶线程读 y block A，奇线程读 y block B（不是 32 路）
//   - rows_per_block 行/block，y 每 block 读一次跨行复用
// 本 bench：固定权重读取（同 mb3），变 rows_per_block R ∈ {1,2,4,8}
//   每 block 处理 R 行，y 只读一次（R 行共享）→ y L2 流量 = (N/R)*5KB
// 对照：R=1 应 ≈ 真实 kernel 144；R=8 若显著回升 → 优化成立
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

#define CK(x) do { auto e_=(x); if(e_!=0){printf("CUDA err %d @%d\n",(int)e_,__LINE__);exit(1);} } while(0)

static const int N = 17408;
static const int K = 5120;
static const int NB = K / 64;   // 80 NVFP4 block
static const int NY = K / 32;   // 160 Q8_1 block
struct block_nvfp4 { uint8_t d[4]; uint8_t qs[32]; };  // 36B
struct block_q8_1  { int16_t d; uint8_t qs[32]; };     // 34B

static const int8_t h_lut[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
__device__ int8_t d_lut[16];

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

// R = rows per block。每 block 处理 R 行，y 读一次跨行复用
// y 读 32 路拆分（复刻真实 kernel）：y block 索引 = tid + iter*32，每线程读整块 32B
// （vecdotq.cuh:349 推导：bq8 = bq8_1 + (is>>1)，is=2*(tid%2)+i → is>>1=tid%2；kby=2*kbx → 2*kbx+tid%2 = tid+iter*32）
template<int R>
__global__ void k(const block_nvfp4 * __restrict__ w, const block_q8_1 * __restrict__ y, float * __restrict__ out) {
    int row0 = R * blockIdx.x;
    if (row0 >= N) return;
    int tid = threadIdx.x;  // 0..31
    const int kqs = 4 * (tid % 2);
    float sum[R];
#pragma unroll
    for (int r = 0; r < R; r++) sum[r] = 0.0f;

    for (int iter = 0; iter < NB/16; iter++) {  // 5
        int kbx = tid/2 + iter*16;
        const uint8_t * yq = y[tid + iter*32].qs;  // 32 路拆分，每线程 32B
        float ys0 = 0.0f, ys1 = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            int v = ld4(yq + 4*i);
            if (i < 4) ys0 += (float)v; else ys1 += (float)v;
        }
#pragma unroll
        for (int r = 0; r < R; r++) {
            int row = row0 + r;
            if (row >= N) break;
            const block_nvfp4 * bq4 = w + (size_t)row * NB;
            const int * qs = (const int *) bq4[kbx].qs;
#pragma unroll
            for (int i = 0; i < 2; i++) {
                int iqs0 = kqs + 2*i;
                int iqs1 = iqs0 + 1;
                int v0 = lut(qs[iqs0]).x;
                int v1 = lut(qs[iqs1]).x;
                int sumi = __dp4a(v0, 0x11111111, 0);
                sumi = __dp4a(v1, 0x22222222, sumi);
                float yscale = (i==0) ? ys0 : ys1;
                sum[r] += (float)sumi * (float)bq4[kbx].d[iqs0>>1] + yscale*0.0001f;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; r++) {
        int row = row0 + r;
        if (row < N) out[row] = sum[r];
    }
}

static double run(int R, int iters) {
    size_t wbytes = (size_t)N * NB * sizeof(block_nvfp4);
    size_t ybytes = (size_t)NY * sizeof(block_q8_1);
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

    int blocks = (N + R - 1) / R, threads = 32;
    auto launch = [&](){
        switch (R) {
            case 1: k<1><<<blocks,threads>>>(d_w, d_y, d_out); break;
            case 2: k<2><<<blocks,threads>>>(d_w, d_y, d_out); break;
            case 4: k<4><<<blocks,threads>>>(d_w, d_y, d_out); break;
            case 8: k<8><<<blocks,threads>>>(d_w, d_y, d_out); break;
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
    double ytraffic = (double)(N/R) * NY * sizeof(block_q8_1) / 1e6;
    printf("  R=%d  blocks=%-6d  %8.3f ms/iter  =  %7.1f GB/s  (y L2流量=%.1fMB)\n",
           R, blocks, ms, gbs, ytraffic);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaFree(d_w); cudaFree(d_y); cudaFree(d_out);
    return gbs;
}

int main(int argc, char ** argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 200;
    printf("=== B3 microbench5: rows_per_block 优化验证 (N=%d K=%d iters=%d) ===\n", N, K, iters);
    printf("  权重=%.1fMB  y=%.1fKB  对照真实 kernel=144 GB/s\n",
           (double)N*NB*sizeof(block_nvfp4)/1e6, (double)NY*sizeof(block_q8_1)/1e3);
    run(1, iters);
    run(2, iters);
    run(4, iters);
    run(8, iters);
    printf("=== done ===\n");
    return 0;
}
