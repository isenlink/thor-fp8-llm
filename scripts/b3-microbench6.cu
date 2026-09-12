// b3-microbench6.cu - NVFP4 E2M1 LUT vs arithmetic decode microbench
// Goal: test whether the fixed FP4 table {0,1,2,3,4,6,8,12,+sign}
// can avoid the generic 6x __byte_perm table lookup in the real MMVQ mapping.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

#define CK(x) do { auto e_=(x); if(e_!=0){printf("CUDA err %d @%d\n",(int)e_,__LINE__);exit(1);} } while(0)

static const int N = 17408;
static const int K = 5120;
static const int NB = K / 64;
struct block_nvfp4 { uint8_t d[4]; uint8_t qs[32]; };

static const int8_t h_lut[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
__device__ int8_t d_lut[16];

static __device__ __forceinline__ int2 lut_decode(const int q4) {
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

static __device__ __forceinline__ uint32_t fp4_e2m1_mag4(uint32_t idx) {
    idx &= 0x07070707u;
    const uint32_t eq5 = __vcmpeq4(idx, 0x05050505u);
    const uint32_t eq6 = __vcmpeq4(idx, 0x06060606u);
    const uint32_t eq7 = __vcmpeq4(idx, 0x07070707u);
    uint32_t corr = 0;
    corr |= eq5 & 0x01010101u;
    corr |= eq6 & 0x02020202u;
    corr |= eq7 & 0x05050505u;
    return __vadd4(idx, corr);
}

static __device__ __forceinline__ uint32_t fp4_e2m1_apply_sign4(uint32_t idx, uint32_t mag) {
    const uint32_t sign = __vcmpne4(idx & 0x08080808u, 0u);
    const uint32_t neg  = __vsub4(0u, mag);
    return (mag & ~sign) | (neg & sign);
}

static __device__ __forceinline__ int2 arith_decode(const int q4) {
    const uint32_t qe = ((uint32_t) q4) & 0x0F0F0F0Fu;
    const uint32_t qo = (((uint32_t) q4) >> 4) & 0x0F0F0F0Fu;
    const uint32_t ve = fp4_e2m1_apply_sign4(qe, fp4_e2m1_mag4(qe));
    const uint32_t vo = fp4_e2m1_apply_sign4(qo, fp4_e2m1_mag4(qo));
    return make_int2((int) ve, (int) vo);
}

__global__ void check_decode(unsigned int * mismatches, unsigned int * first_q, int * first_lut_x, int * first_lut_y, int * first_arith_x, int * first_arith_y) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int x = tid * 747796405u + 2891336453u;
    for (int i = 0; i < 64; ++i) {
        x = x * 1664525u + 1013904223u;
        const int q = (int) x;
        const int2 a = lut_decode(q);
        const int2 b = arith_decode(q);
        if (a.x != b.x || a.y != b.y) {
            if (atomicAdd(mismatches, 1u) == 0u) {
                *first_q = x;
                *first_lut_x = a.x;
                *first_lut_y = a.y;
                *first_arith_x = b.x;
                *first_arith_y = b.y;
            }
        }
    }
}

enum MODE { LUT=0, ARITH=1, LOAD_ONLY=2 };

template<int MODE>
__global__ void k(const block_nvfp4 * __restrict__ w, float * __restrict__ out) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const block_nvfp4 * bq4 = w + (size_t) row * NB;
    const int kqs = 4 * (tid % 2);
    float sum = 0.0f;
    for (int iter = 0; iter < NB/16; iter++) {
        const int kbx = tid/2 + iter*16;
        const int * qs = (const int *) bq4[kbx].qs;
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int iqs0 = kqs + 2*i;
            const int iqs1 = iqs0 + 1;
            int v0, v1;
            if (MODE == LOAD_ONLY) {
                v0 = qs[iqs0];
                v1 = qs[iqs1];
            } else if (MODE == LUT) {
                v0 = lut_decode(qs[iqs0]).x;
                v1 = lut_decode(qs[iqs1]).x;
            } else {
                v0 = arith_decode(qs[iqs0]).x;
                v1 = arith_decode(qs[iqs1]).x;
            }
            int sumi = __dp4a(v0, 0x11111111, 0);
            sumi = __dp4a(v1, 0x22222222, sumi);
            sum += (float) sumi * (float) bq4[kbx].d[iqs0 >> 1];
        }
    }
    out[row] = sum;
}

static double run(int mode, int iters) {
    const size_t wbytes = (size_t) N * NB * sizeof(block_nvfp4);
    block_nvfp4 * d_w; float * d_out;
    CK(cudaMalloc(&d_w, wbytes));
    CK(cudaMalloc(&d_out, N * sizeof(float)));
    std::vector<uint8_t> hw(wbytes);
    for (auto & b : hw) b = (uint8_t) rand();
    CK(cudaMemcpy(d_w, hw.data(), wbytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpyToSymbol(d_lut, h_lut, 16));

    auto launch = [&](){
        switch (mode) {
            case LUT:       k<LUT><<<N,32>>>(d_w, d_out); break;
            case ARITH:     k<ARITH><<<N,32>>>(d_w, d_out); break;
            case LOAD_ONLY: k<LOAD_ONLY><<<N,32>>>(d_w, d_out); break;
        }
    };
    for (int i = 0; i < 5; ++i) launch();
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0));
    for (int i = 0; i < iters; ++i) launch();
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1)); ms /= iters;
    const double gbs = wbytes / (ms*1e-3) / 1e9;
    printf("  V%d  %8.3f ms/iter  =  %7.1f GB/s\n", mode, ms, gbs);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaFree(d_w); cudaFree(d_out);
    return gbs;
}

int main(int argc, char ** argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 200;
    CK(cudaMemcpyToSymbol(d_lut, h_lut, 16));
    unsigned int * mismatches, * first_q;
    int * lx, * ly, * ax, * ay;
    CK(cudaMallocManaged(&mismatches, sizeof(unsigned int)));
    CK(cudaMallocManaged(&first_q, sizeof(unsigned int)));
    CK(cudaMallocManaged(&lx, sizeof(int))); CK(cudaMallocManaged(&ly, sizeof(int)));
    CK(cudaMallocManaged(&ax, sizeof(int))); CK(cudaMallocManaged(&ay, sizeof(int)));
    *mismatches = 0; *first_q = 0; *lx = *ly = *ax = *ay = 0;
    check_decode<<<256,256>>>(mismatches, first_q, lx, ly, ax, ay);
    CK(cudaDeviceSynchronize());
    printf("=== B3 microbench6: NVFP4 LUT vs arithmetic decode (N=%d K=%d iters=%d) ===\n", N, K, iters);
    printf("  correctness mismatches=%u", *mismatches);
    if (*mismatches) {
        printf(" first_q=0x%08x lut=(0x%08x,0x%08x) arith=(0x%08x,0x%08x)",
               *first_q, *lx, *ly, *ax, *ay);
    }
    printf("\n");
    run(LUT, iters);
    run(ARITH, iters);
    run(LOAD_ONLY, iters);
    printf("=== done ===\n");
    return *mismatches ? 1 : 0;
}
