// b3-microbench2.cu — 验证"36B NVFP4 block stride 导致访存不 coalesce"假设
// 真实 MMVQ kernel 映射：17408 block × 32 线程，线程 tid 读 base + (tid/2)*36 + (tid%2)*4
// 对照：
//   A_real_stride36 : 复刻真实映射（2 线程/36B block，stride 36B）
//   B_coalesced     : 理想映射（每线程连续 4B，无 36B stride）
//   C_full36        : 每线程读完整 36B（测 36B 对齐本身）
// 同总字节数 50MB（> 24MB L2 → DRAM-bound），看 A vs B 差距 = coalescing 损失
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

#define CK(x) do { auto e_=(x); if(e_!=0){printf("CUDA err %d @%d\n",(int)e_,__LINE__);exit(1);} } while(0)

static const int N = 17408;    // 行（输出通道）
static const int K = 5120;
static const int NB = K / 64;  // 80 block/行
struct block_nvfp4 { uint8_t d[4]; uint8_t qs[32]; };  // 36B

// A: 真实映射 — 每 warp 处理 1 行，tid 读 (tid/2)*36 + (tid%2)*4，5 次迭代
__global__ void k_real_stride36(const block_nvfp4 * __restrict__ w, float * __restrict__ out) {
    int row = blockIdx.x;  // 1 block/行
    if (row >= N) return;
    int tid = threadIdx.x;  // 0..31
    const block_nvfp4 * bq4 = w + (size_t)row * NB;
    float sum = 0.0f;
    for (int iter = 0; iter < NB/16; iter++) {  // 80/16 = 5
        int kbx = tid/2 + iter*16;
        const int iqs = 4 * (tid % 2);  // 0 或 4
        const int * qs = (const int *) bq4[kbx].qs;
        // 复刻 vec_dot_nvfp4_q8_1 的 4 次 4B 读 + dp4a（LUT 简化为恒等，纯测访存）
#pragma unroll
        for (int i = 0; i < 2; i++) {
            int iqs0 = iqs + 2*i;
            int iqs1 = iqs0 + 1;
            int v0 = qs[iqs0];
            int v1 = qs[iqs1];
            int sumi = __dp4a(v0, 0x11111111, 0);
            sumi = __dp4a(v1, 0x22222222, sumi);
            sum += (float)sumi * (float)bq4[kbx].d[iqs0>>1];
        }
    }
    out[row] = sum;
}

// B: 理想 coalesced — 每线程连续读 4B，无 36B stride（同样 50MB 总读）
__global__ void k_coalesced(const int * __restrict__ w, float * __restrict__ out, int total_ints, int rpt) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int i = 0; i < rpt; i++) {
        int idx = tid + i * (gridDim.x * blockDim.x);
        if (idx < total_ints) {
            int v = w[idx];
            sum += (float)__dp4a(v, 0x11111111, 0);
        }
    }
    if (tid < N) out[tid] = sum;
}

static double run(const char * name, int iters, bool is_real) {
    size_t wbytes = (size_t)N * NB * sizeof(block_nvfp4);
    block_nvfp4 * d_w; float * d_out;
    CK(cudaMalloc(&d_w, wbytes));
    CK(cudaMalloc(&d_out, N * sizeof(float)));
    std::vector<uint8_t> hw(wbytes);
    for (auto & b : hw) b = (uint8_t)rand();
    CK(cudaMemcpy(d_w, hw.data(), wbytes, cudaMemcpyHostToDevice));

    int blocks, threads;
    if (is_real) { blocks = N; threads = 32; }
    else { threads = 256; blocks = (N*5 + threads - 1)/threads; }

    // B 要读满 50MB：total_ints = 50MB/4，rpt = 每线程读几次
    int total_ints = (int)(wbytes / 4);
    int b_threads = 256, b_blocks = (N*5 + b_threads - 1)/b_threads;
    int rpt = (total_ints + b_blocks*b_threads - 1) / (b_blocks*b_threads);

    auto launch = [&](){
        if (is_real) k_real_stride36<<<blocks, threads>>>((const block_nvfp4*)d_w, d_out);
        else k_coalesced<<<b_blocks, b_threads>>>((const int*)d_w, d_out, total_ints, rpt);
    };
    for (int i = 0; i < 5; i++) launch();
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0));
    for (int i = 0; i < iters; i++) launch();
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms=0; CK(cudaEventElapsedTime(&ms,e0,e1)); ms/=iters;
    double gbs = wbytes / (ms*1e-3) / 1e9;
    printf("  %-16s %8.3f ms/iter  =  %7.1f GB/s\n", name, ms, gbs);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaFree(d_w); cudaFree(d_out);
    return gbs;
}

int main(int argc, char ** argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 200;
    printf("=== B3 microbench2: 36B stride coalescing 验证 (N=%d K=%d iters=%d) ===\n", N, K, iters);
    printf("  权重=%.1fMB (>24MB L2, DRAM-bound)\n", (double)N*NB*sizeof(block_nvfp4)/1e6);
    printf("A_real_stride36 : 真实映射（2线程/36B block, stride 36B）\n");
    printf("B_coalesced     : 理想映射（连续 4B, 无 stride）\n");
    double a = run("A_real_stride36", iters, true);
    double b = run("B_coalesced",     iters, false);
    printf("--- 结论 ---\n");
    printf("  coalescing 损失 (B-A) = %+.1f GB/s（A 是 B 的 %.0f%%）\n", b - a, a/b*100);
    printf("=== done ===\n");
    return 0;
}
