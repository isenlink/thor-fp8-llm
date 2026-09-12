// b3-standalone.cu — NVFP4 MMVQ 单算子基准（B3 第一步：复现 95GB/s + 上限实验）
// 走 ggml backend 公开 API，与 llama-server 完全同 dispatch 路径。
// 形状对齐 R32 per-call 数据：
//   A: N=17408 K=5120  (ffn_gate/up, R32: 46MB/488µs=95GB/s)
//   B: N=5120  K=17408 (ffn_down,    R32: 46MB/433µs=103GB/s)
//   C: N=248320 K=5120 (lm_head,     R32: 635MB/6.7ms=94GB/s)
// 上限实验：同形状 Q8_0（预展开，无 NVFP4 解包）——若 Q8_0 到 200+ 而 NVFP4 95，
//          实锤"喂 dp4a 的流水线"是根因（GPT R4.5 建议）。
#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-alloc.h>
#include <ggml-cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <chrono>

#define CK(x) do { auto e_ = (x); if (e_ != 0) { printf("CUDA err %d @%s:%d\n", (int)e_, __FILE__, __LINE__); exit(1); } } while (0)

static void fill_rand(uint8_t * p, size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> d(0, 255);
    for (size_t i = 0; i < n; i++) p[i] = (uint8_t) d(rng);
}

struct Case { const char * name; int N; int K; };

// L2 flush buffer（> L2 24MB，用 128MB）
static uint8_t * g_flush_buf = nullptr;
static const size_t FLUSH_SIZE = 128 * 1024 * 1024;
static void flush_l2() {
    if (!g_flush_buf) CK(cudaMalloc(&g_flush_buf, FLUSH_SIZE));
    CK(cudaMemset(g_flush_buf, 0, FLUSH_SIZE));  // 写满 128MB 冲刷 L2
}

static void run_case(ggml_backend_t backend, const Case & c, ggml_type wtype, int iters, bool cold) {
    // 权重 A [K, N]（行主：每行 K 个元素 = 一个输出通道的权重）
    int64_t neA[2] = { c.K, c.N };
    int64_t neB[2] = { c.K, 1 };

    ggml_init_params gp = { .mem_size = 256 * 1024 * 1024, .no_alloc = true };
    ggml_context * ctx = ggml_init(gp);
    ggml_tensor * A = ggml_new_tensor(ctx, wtype, 2, neA);
    ggml_tensor * B = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, neB);
    ggml_tensor * C = ggml_mul_mat(ctx, A, B);

    // 构建 graph：build_forward_expand 自动遍历依赖 + 设 COMPUTE flag + 加节点
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, C);

    // 标准分配路径：为 ctx 所有张量分配 GPU 内存（设置 tensor->buffer 和 data）
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        printf("[%s %s] alloc_ctx_tensors failed\n", c.name, wtype == GGML_TYPE_NVFP4 ? "NVFP4" : "Q8_0");
        ggml_free(ctx);
        return;
    }

    // 填数据：host 缓冲 → tensor_set 拷到 GPU（alloc 后 A->data 指向 GPU）
    size_t nA = ggml_nbytes(A), nB = ggml_nbytes(B);
    std::vector<uint8_t> hA(nA), hB(nB);
    fill_rand(hA.data(), nA, 42);
    fill_rand(hB.data(), nB, 7);
    ggml_backend_tensor_set(A, hA.data(), 0, nA);
    ggml_backend_tensor_set(B, hB.data(), 0, nB);

    printf("  [dbg] graph nodes=%d  A.data=%p  A.buffer=%p\n",
           ggml_graph_n_nodes(gf), (void*)A->data, (void*)A->buffer);
    printf("  [dbg] C.op=%s C.flags=0x%x (COMPUTE=0x%x)  C.ne=[%lld,%lld]\n",
           ggml_op_name(C->op), (unsigned)C->flags, (unsigned)GGML_TENSOR_FLAG_COMPUTE,
           (long long)C->ne[0], (long long)C->ne[1]);

    // warmup
    for (int i = 0; i < 20; i++) ggml_backend_graph_compute(backend, gf);
    CK(cudaDeviceSynchronize());

    // 检查单次 compute 返回值
    enum ggml_status st = ggml_backend_graph_compute(backend, gf);
    CK(cudaDeviceSynchronize());
    printf("  [dbg] compute status=%d (%s)\n", (int)st, ggml_status_to_string(st));

    // host 时钟计时（CUDA backend 用 per-thread stream，cudaEvent 在 default stream 上不同步）
    // cold 模式：每次迭代前 flush L2（计时外），复现全模型"每权重每 token 只读一次"的冷缓存场景
    double total_ms = 0.0;
    for (int i = 0; i < iters; i++) {
        if (cold) { flush_l2(); CK(cudaDeviceSynchronize()); }
        auto s = std::chrono::high_resolution_clock::now();
        ggml_backend_graph_compute(backend, gf);
        CK(cudaDeviceSynchronize());
        auto e = std::chrono::high_resolution_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(e - s).count();
    }
    double ms = total_ms / iters;

    double bytes = (double) nA + (double) nB;
    double gbs = bytes / (ms * 1e-3) / 1e9;
    printf("[%s %s] N=%d K=%d  weights=%.1fMB  %.3f ms/iter  =  %.1f GB/s\n",
           c.name, wtype == GGML_TYPE_NVFP4 ? "NVFP4" : "Q8_0", c.N, c.K, nA / 1e6, ms, gbs);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 300;
    printf("=== B3 standalone: NVFP4 MMVQ vs Q8_0 上限实验 (iters=%d) ===\n", iters);
    printf("env GGML_MMVQ_MAX=%s\n", getenv("GGML_MMVQ_MAX") ? getenv("GGML_MMVQ_MAX") : "(unset, default 8)");

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { printf("cuda backend init failed\n"); return 1; }

    // 设备属性（L2 大小决定 standalone 缓存热度的解释力）
    int l2 = 0, sms = 0, smclock = 0, memclock = 0, membus = 0;
    CK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0));
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    CK(cudaDeviceGetAttribute(&smclock, cudaDevAttrClockRate, 0));
    CK(cudaDeviceGetAttribute(&memclock, cudaDevAttrMemoryClockRate, 0));
    CK(cudaDeviceGetAttribute(&membus, cudaDevAttrGlobalMemoryBusWidth, 0));
    printf("device: SMs=%d  L2=%dMB  SMclk=%.0fMHz  memclk=%.0fMHz  bus=%d-bit  (theoretical=%.0f GB/s)\n",
           sms, l2 / (1024*1024), smclock / 1e3, memclock / 1e3, membus,
           (double)memclock * 1e3 * (membus / 8) / 1e9);

    Case cases[] = {
        { "A-ffn_gate", 17408, 5120 },
        { "B-ffn_down", 5120, 17408 },
        { "C-lm_head",  248320, 5120 },
    };

    printf("--- [HOT] NVFP4（权重热在 L2，standalone 上限）---\n");
    for (auto & c : cases) run_case(backend, c, GGML_TYPE_NVFP4, iters, false);

    printf("--- [HOT] Q8_0 上限实验（预展开，无 NVFP4 解包）---\n");
    for (auto & c : cases) run_case(backend, c, GGML_TYPE_Q8_0, iters, false);

    int cold_iters = iters / 3;  // cold 模式每次迭代含 flush，少跑点
    printf("--- [COLD] NVFP4（每次迭代前 flush L2，复现全模型 95 GB/s 基线）---\n");
    for (auto & c : cases) run_case(backend, c, GGML_TYPE_NVFP4, cold_iters, true);

    printf("--- [COLD] Q8_0（flush L2 对照）---\n");
    for (auto & c : cases) run_case(backend, c, GGML_TYPE_Q8_0, cold_iters, true);

    ggml_backend_free(backend);
    printf("=== done ===\n");
    return 0;
}
