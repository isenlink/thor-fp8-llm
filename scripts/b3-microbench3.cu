// b3-microbench3.cu — 精确复制真实 mul_mat_vec_q 映射，分离 load / LUT / dp4a
// 真实映射（mmvq.cu, NVFP4 M=1）：grid(17408,1,1) block(32,1,1)
//   tid=threadIdx.x; kbx=tid/2+iter*16; kqs=4*(tid%2); 5 iter 覆盖 80 block
//   每线程每 iter 读 bq4[kbx].qs 的 8 个 int（iqs=0..7）+ d[4]
// 3 变体（同访存，只差 compute）：
//   V1_full   : LUT(6 byte_perm) + dp4a  = 真实 kernel
//   V2_nolut  : nibble 直接 dp4a（无 LUT）
//   V3_nocompute : 只 load qs 累加（无 LUT 无 dp4a）
// V1-V2 = LUT 成本；V2-V3 = dp4a 成本；V3 = 纯 load 上限
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

#define CK(x) do { auto e_=(x); if(e_!=0){printf("CUDA err %d @%d\n",(int)e_,__LINE__);exit(1);} } while(0)

static const int N = 17408;
static const int K = 5120;
static const int NB = K / 64;  // 80
struct block_nvfp4 { uint8_t d[4]; uint8_t qs[32]; };  // 36B

static const int8_t h_lut[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
__device__ int8_t d_lut[16];

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

enum MODE { FULL=0, NO_LUT=1, NO_COMPUTE=2 };

template<int MODE>
__global__ void k(const block_nvfp4 * __restrict__ w, float * __restrict__ out) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;  // 0..31
    const block_nvfp4 * bq4 = w + (size_t)row * NB;
    const int kqs = 4 * (tid % 2);
    float sum = 0.0f;
    for (int iter = 0; iter < NB/16; iter++) {  // 5
        int kbx = tid/2 + iter*16;
        const int * qs = (const int *) bq4[kbx].qs;
        if (MODE == NO_COMPUTE) {
            // 只 load 8 个 int 累加（无 LUT 无 dp4a）
#pragma unroll
            for (int i = 0; i < 8; i++) sum += (float)qs[i];
        } else {
#pragma unroll
            for (int i = 0; i < 2; i++) {
                int iqs0 = kqs + 2*i;
                int iqs1 = iqs0 + 1;
                int v0, v1;
                if (MODE == FULL) {
                    v0 = lut(qs[iqs0]).x;
                    v1 = lut(qs[iqs1]).x;
                } else { // NO_LUT: 低 4 bit 直接当 int8
                    v0 = qs[iqs0] & 0x0F0F0F0F;
                    v1 = qs[iqs1] & 0x0F0F0F0F;
                }
                int sumi = __dp4a(v0, 0x11111111, 0);
                sumi = __dp4a(v1, 0x22222222, sumi);
                sum += (float)sumi * (float)bq4[kbx].d[iqs0>>1];
            }
        }
    }
    out[row] = sum;
}

static double run(int mode, int iters) {
    size_t wbytes = (size_t)N * NB * sizeof(block_nvfp4);
    block_nvfp4 * d_w; float * d_out;
    CK(cudaMalloc(&d_w, wbytes));
    CK(cudaMalloc(&d_out, N * sizeof(float)));
    std::vector<uint8_t> hw(wbytes);
    for (auto & b : hw) b = (uint8_t)rand();
    CK(cudaMemcpy(d_w, hw.data(), wbytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpyToSymbol(d_lut, h_lut, 16));

    int blocks = N, threads = 32;
    auto launch = [&](){
        switch (mode) {
            case FULL:      k<FULL><<<blocks,threads>>>(d_w, d_out); break;
            case NO_LUT:    k<NO_LUT><<<blocks,threads>>>(d_w, d_out); break;
            case NO_COMPUTE:k<NO_COMPUTE><<<blocks,threads>>>(d_w, d_out); break;
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
    printf("  V%d  %8.3f ms/iter  =  %7.1f GB/s\n", mode, ms, gbs);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaFree(d_w); cudaFree(d_out);
    return gbs;
}

int main(int argc, char ** argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 200;
    printf("=== B3 microbench3: 精确真实映射, load/LUT/dp4a 分离 (N=%d K=%d iters=%d) ===\n", N, K, iters);
    printf("  权重=%.1fMB  grid(%d,1,1) block(32,1,1)\n", (double)N*NB*sizeof(block_nvfp4)/1e6, N);
    double v1 = run(FULL, iters);
    double v2 = run(NO_LUT, iters);
    double v3 = run(NO_COMPUTE, iters);
    printf("--- 拆解（同访存模式）---\n");
    printf("  纯 load 上限 (V3)      = %7.1f GB/s\n", v3);
    printf("  +dp4a 成本 (V3-V2)     = %7.1f GB/s\n", v3 - v2);
    printf("  +LUT  成本 (V2-V1)     = %7.1f GB/s  (占 V1 的 %.0f%%)\n", v2 - v1, (v2-v1)/v1*100);
    printf("  完整 kernel (V1)       = %7.1f GB/s\n", v1);
    printf("=== done ===\n");
    return 0;
}
