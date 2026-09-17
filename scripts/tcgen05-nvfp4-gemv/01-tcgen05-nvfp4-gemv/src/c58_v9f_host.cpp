// c58_v9f_host.cpp — B9f 宿主驱动（板端 aarch64；Driver API + dlopen）
// 用法: c58_v9f_host stream <cubin> <chunks/block> <grid> <timeout_ms> [flags] [bw 1] [breps]
//   flags bit0=1 发 MMA / =0 纯拷贝臂；bit1=2 写痕迹；bit2=4 每 warp 1 次 arrive。
// 全程：进度轮询（宿主映射内存）+ 硬超时 + D 全量 CPU 参考比对（K=256）+ 设备侧 %globaltimer 计时
// 动态 smem 尺寸来自 t4/common/t4_smem_layout.h（单一来源；事故 #5 的根因就是宿主申请值 < 内核需求值）
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <unistd.h>
#include <dlfcn.h>
#include <sys/time.h>
#include <cuda.h>
#include <vector>
#include "t4_smem_layout.h"

#define M 128
#define N 32
#define KT 256
#define A_SUB 4096
#define B_SUB 1024
#define B_CHUNK 4096
#define NTHR T4_V9F_NTHR
#define MK_N 8
#define DG_N 17

static void* g_lib = nullptr;
template <typename T> static T sym(const char* n) {
  void* p = dlsym(g_lib, n);
  if (!p) { printf("{\"tag\":\"fatal\",\"sym\":\"%s\",\"err\":\"%s\"}\n", n, dlerror()); exit(2); }
  return reinterpret_cast<T>(p);
}
static CUresult (*p_cuInit)(unsigned);
static CUresult (*p_cuDeviceGet)(CUdevice*, int);
static CUresult (*p_cuCtxCreate)(CUcontext*, unsigned, CUdevice);
static CUresult (*p_cuCtxSynchronize)(void);
static CUresult (*p_cuModuleLoad)(CUmodule*, const char*);
static CUresult (*p_cuModuleGetFunction)(CUfunction*, CUmodule, const char*);
static CUresult (*p_cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned,
                                    unsigned, unsigned, CUstream, void**, void**);
static CUresult (*p_cuMemAlloc)(CUdeviceptr*, size_t);
static CUresult (*p_cuMemFree)(CUdeviceptr);
static CUresult (*p_cuMemcpyHtoD)(CUdeviceptr, const void*, size_t);
static CUresult (*p_cuMemcpyDtoH)(void*, CUdeviceptr, size_t);
static CUresult (*p_cuMemsetD32)(CUdeviceptr, unsigned, size_t);
static CUresult (*p_cuDeviceGetAttribute)(int*, int, CUdevice);
static CUresult (*p_cuGetErrorString)(CUresult, const char**);
static CUresult (*p_cuMemHostAlloc)(void**, size_t, unsigned);
static CUresult (*p_cuMemHostGetDevicePointer)(CUdeviceptr*, void*, unsigned);
static CUresult (*p_cuFuncSetAttribute)(CUfunction, int, int);
static CUresult (*p_cuOccupancy)(int*, CUfunction, int, size_t);
static void bind() {
  g_lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
  if (!g_lib) { printf("{\"tag\":\"fatal\",\"dlopen\":\"%s\"}\n", dlerror()); exit(2); }
  p_cuInit = sym<decltype(p_cuInit)>("cuInit");
  p_cuDeviceGet = sym<decltype(p_cuDeviceGet)>("cuDeviceGet");
  p_cuCtxCreate = sym<decltype(p_cuCtxCreate)>("cuCtxCreate_v2");
  p_cuCtxSynchronize = sym<decltype(p_cuCtxSynchronize)>("cuCtxSynchronize");
  p_cuModuleLoad = sym<decltype(p_cuModuleLoad)>("cuModuleLoad");
  p_cuModuleGetFunction = sym<decltype(p_cuModuleGetFunction)>("cuModuleGetFunction");
  p_cuLaunchKernel = sym<decltype(p_cuLaunchKernel)>("cuLaunchKernel");
  p_cuMemAlloc = sym<decltype(p_cuMemAlloc)>("cuMemAlloc_v2");
  p_cuMemFree = sym<decltype(p_cuMemFree)>("cuMemFree_v2");
  p_cuMemcpyHtoD = sym<decltype(p_cuMemcpyHtoD)>("cuMemcpyHtoD_v2");
  p_cuMemcpyDtoH = sym<decltype(p_cuMemcpyDtoH)>("cuMemcpyDtoH_v2");
  p_cuMemsetD32 = sym<decltype(p_cuMemsetD32)>("cuMemsetD32_v2");
  p_cuDeviceGetAttribute = sym<decltype(p_cuDeviceGetAttribute)>("cuDeviceGetAttribute");
  p_cuGetErrorString = sym<decltype(p_cuGetErrorString)>("cuGetErrorString");
  p_cuMemHostAlloc = sym<decltype(p_cuMemHostAlloc)>("cuMemHostAlloc");
  p_cuMemHostGetDevicePointer = sym<decltype(p_cuMemHostGetDevicePointer)>("cuMemHostGetDevicePointer_v2");
  p_cuFuncSetAttribute = sym<decltype(p_cuFuncSetAttribute)>("cuFuncSetAttribute");
  p_cuOccupancy = sym<decltype(p_cuOccupancy)>("cuOccupancyMaxActiveBlocksPerMultiprocessor");
}
static long now_ms() { struct timeval tv; gettimeofday(&tv, nullptr); return tv.tv_sec * 1000L + tv.tv_usec / 1000L; }

// ---- 与设备件同源的生成/参考 ----
static int genA_nib(int m, int k) { if (m == 64) return 4; return (k % 2 == 0) ? 2 : 4; }
static int genB_nib(int n, int k) { return (n == 24) ? 4 : 2; }
// 顺序验证用：块号相关数据（mode=stream1）——同一 (n,k) 的 nibble 随块号平移
static int genB_nibb(int n, int kk, int blk) { return (genB_nib(n, kk) + blk) & 15; }
static unsigned char genSFA(int m, int kb) { return (m == 0 && kb == 0) ? 0x40 : 0x38; }
static unsigned char genSFB(int n, int kb) { return (n == 5 && kb == 3) ? 0x48 : 0x38; }
static float fp4_val(int nib) {
  static const float t[16] = {0.f,0.5f,1.f,1.5f,2.f,3.f,4.f,6.f,-0.f,-0.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
  return t[nib & 15];
}
static float sf_val(unsigned char b) {
  const int s = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 7;
  if (e == 0) return 0.f;
  const float v = (float)(1.0 + m / 8.0) * (float)(1 << (e - 7));
  return s ? -v : v;
}
static float ref_D(int m, int n) {
  float acc = 0.f;
  for (int k = 0; k < KT; ++k)
    acc += fp4_val(genA_nib(m, k)) * fp4_val(genB_nib(n, k)) *
           sf_val(genSFA(m, k / 16)) * sf_val(genSFB(n, k / 16));
  return acc;
}
static float ref_D_b(int m, int n, int blk) {
  float acc = 0.f;
  for (int k = 0; k < KT; ++k)
    acc += fp4_val(genA_nib(m, k)) * fp4_val(genB_nibb(n, k, blk)) *
           sf_val(genSFA(m, k / 16)) * sf_val(genSFB(n, k / 16));
  return acc;
}
// B 的 canonical 四字节块（= 设备件 smB 一 stage 的内容）
static void fill_stage(uint8_t* dst) {
  memset(dst, 0, B_CHUNK);
  for (int s = 0; s < 4; ++s)
    for (int n = 0; n < N; ++n)
      for (int k = 0; k < 64; ++k) {
        const int byte = (n % 8) * 16 + (k / 32) * 128 + (n / 8) * 256 + (k % 32) / 2;
        const int nib = genB_nib(n, 64 * s + k);
        uint8_t& b = dst[s * B_SUB + byte];
        b = (uint8_t)((k % 2 == 0) ? ((b & 0xF0) | (nib & 15)) : ((b & 0x0F) | (nib << 4)));
      }
}
// 同布局，但 nibble 随块号平移（mode=stream1 的顺序验证）
static void fill_stage_b(uint8_t* dst, int blk) {
  memset(dst, 0, B_CHUNK);
  for (int s = 0; s < 4; ++s)
    for (int n = 0; n < N; ++n)
      for (int k = 0; k < 64; ++k) {
        const int byte = (n % 8) * 16 + (k / 32) * 128 + (n / 8) * 256 + (k % 32) / 2;
        const int nib = genB_nibb(n, 64 * s + k, blk);
        uint8_t& b = dst[s * B_SUB + byte];
        b = (uint8_t)((k % 2 == 0) ? ((b & 0xF0) | (nib & 15)) : ((b & 0x0F) | (nib << 4)));
      }
}

int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "stream";
  const char* cubin = argc > 2 ? argv[2] : "c58_v7.cubin";
  const long chunks = argc > 3 ? atol(argv[3]) : 4096;
  const int grid = argc > 4 ? atoi(argv[4]) : 1;
  const long timeout_ms = argc > 5 ? atol(argv[5]) : 30000;
  // mode: stream=各块同数据(计时口径)；stream1=块号相关数据(顺序验证)；stream1b=反证（故意用错块的参考）
  const int order_mode = strcmp(mode, "stream1") == 0 ? 1 : (strcmp(mode, "stream1b") == 0 ? 2 : 0);
  if (strcmp(mode, "stream") != 0 && order_mode == 0) { printf("{\"tag\":\"bad_mode\",\"mode\":\"%s\"}\n", mode); return 1; }
  bind();
  if (p_cuInit(0) != CUDA_SUCCESS) { printf("{\"tag\":\"init\",\"ok\":false}\n"); return 2; }
  CUdevice dev = 0; p_cuDeviceGet(&dev, 0);
  CUcontext ctx = nullptr; p_cuCtxCreate(&ctx, 0, dev);
  int ccM = 0, ccm = 0, sms = 0, smem_sm = 0;
  p_cuDeviceGetAttribute(&ccM, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
  p_cuDeviceGetAttribute(&ccm, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
  p_cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, dev);
  p_cuDeviceGetAttribute(&smem_sm, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, dev);
  int nsm = 0;
  p_cuDeviceGetAttribute(&nsm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
  const int DYN_SMEM = (int)T4_V9F_SMEM_HOST_REQ;
  printf("{\"tag\":\"env\",\"mode\":\"stream\",\"cc\":\"%d.%d\",\"cuda_ver\":%d,\"shape\":\"M=%d,N=%d,K=%d\","
         "\"sm_count\":%d,\"max_thr_per_sm\":%d,\"max_smem_per_sm\":%d,\"chunks\":%ld,\"grid\":%d,\"chunk_bytes\":%d,"
         "\"threads\":%d,\"smem_required\":%d,\"smem_host_req\":%d}\n",
         ccM, ccm, (int)CUDA_VERSION, M, N, KT, nsm, sms, smem_sm, chunks, grid, B_CHUNK, (int)NTHR,
         (int)T4_V9F_SMEM_REQUIRED, DYN_SMEM);
  fflush(stdout);

  CUmodule mod = nullptr; CUfunction fn = nullptr;
  if (p_cuModuleLoad(&mod, cubin) != CUDA_SUCCESS) { printf("{\"tag\":\"stream\",\"ok\":false,\"stage\":\"module_load\"}\n"); return 2; }
  if (p_cuModuleGetFunction(&fn, mod, "k_b9f_fp4_ws32k") != CUDA_SUCCESS) { printf("{\"tag\":\"stream\",\"ok\":false,\"stage\":\"getfn\"}\n"); return 2; }

  void* h_mk = nullptr; CUdeviceptr d_mk = 0;
  if (p_cuMemHostAlloc(&h_mk, MK_N * 4, 0x02) != CUDA_SUCCESS) { printf("{\"tag\":\"stream\",\"ok\":false,\"stage\":\"hostalloc\"}\n"); return 2; }
  memset(h_mk, 0, MK_N * 4);
  p_cuMemHostGetDevicePointer(&d_mk, h_mk, 0);

  const size_t total_bytes = (size_t)chunks * (size_t)grid * B_CHUNK;
  CUdeviceptr d_gB = 0, d_D = 0, d_diag = 0, d_ts = 0;
  if (p_cuMemAlloc(&d_gB, total_bytes) != CUDA_SUCCESS) { printf("{\"tag\":\"stream\",\"ok\":false,\"stage\":\"alloc_gB\",\"bytes\":%zu}\n", total_bytes); return 2; }
  p_cuMemAlloc(&d_D, (size_t)grid * M * N * 4);
  p_cuMemAlloc(&d_diag, (size_t)DG_N * 4);
  p_cuMemAlloc(&d_ts, (size_t)grid * 2 * 8);
  p_cuMemsetD32(d_D, 0, (size_t)grid * M * N);
  p_cuMemsetD32(d_diag, 0, DG_N);
  p_cuMemsetD32(d_ts, 0, (size_t)grid * 2 * 2);

  std::vector<uint8_t> stage(B_CHUNK); fill_stage(stage.data());
  std::vector<uint8_t> host_buf(total_bytes);
  if (order_mode == 0) {
    for (long i = 0; i < chunks * grid; ++i)
      memcpy(host_buf.data() + (size_t)i * B_CHUNK, stage.data(), B_CHUNK);
  } else {
    for (long i = 0; i < chunks * grid; ++i)
      fill_stage_b(host_buf.data() + (size_t)i * B_CHUNK, (int)(i % chunks));
  }
  p_cuMemcpyHtoD(d_gB, host_buf.data(), total_bytes);
  p_cuCtxSynchronize();
  printf("{\"tag\":\"filled\",\"bytes\":%zu}\n", total_bytes); fflush(stdout);

  if (p_cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, DYN_SMEM) != CUDA_SUCCESS) {
    printf("{\"tag\":\"stream\",\"ok\":false,\"stage\":\"funcattr\",\"dyn_smem\":%d}\n", DYN_SMEM); return 2;
  }
  unsigned u_chunks = (unsigned)chunks;
  unsigned u_flags = (unsigned)(argc > 6 ? atol(argv[6]) : 1);
  void* args[7] = {&d_gB, &u_chunks, &d_D, &d_mk, &d_diag, &d_ts, &u_flags};
  // 参考件：纯读带宽（可选第 7 参数 bw=1 时先跑）
  if (argc > 7 && atoi(argv[7]) == 1) {
    CUfunction fnbw = nullptr;
    if (p_cuModuleGetFunction(&fnbw, mod, "k_b9f_readbw") == CUDA_SUCCESS) {
      unsigned long long n16 = (unsigned long long)(total_bytes / 16);
      unsigned long long* d_out = nullptr;
      p_cuMemAlloc((CUdeviceptr*)&d_out, 8);
      void* ab[3] = {&d_gB, &n16, &d_out};
      const int breps = (argc > 8 ? atoi(argv[8]) : 20);
      {   // 批量计时（单发只有 1 ms 分辨率，不可用）
        long t_a = now_ms();
        for (int rep = 0; rep < breps; ++rep) {
          CUresult rb = p_cuLaunchKernel(fnbw, 14 * 8, 1, 1, 256, 1, 1, 0, nullptr, ab, nullptr);
          if (rb != CUDA_SUCCESS) { printf("{\"tag\":\"bw\",\"ok\":false,\"rep\":%d}\n", rep); break; }
        }
        p_cuCtxSynchronize();
        const long dt = now_ms() - t_a;
        const double by = (double)total_bytes * (double)breps;
        printf("{\"tag\":\"bw\",\"ok\":true,\"reps\":%d,\"bytes_total\":%.0f,\"host_ms\":%ld,\"gbps\":%.2f,"
               "\"per_sm_gbps\":%.2f}\n", breps, by, dt,
               dt > 0 ? (by / 1e9) / ((double)dt * 1e-3) : 0.0,
               dt > 0 ? (by / 1e9) / ((double)dt * 1e-3) / nsm : 0.0);
        fflush(stdout);
      }
    }
  }
  CUresult sr = p_cuLaunchKernel(fn, (unsigned)grid, 1, 1, (unsigned)NTHR, 1, 1, (unsigned)DYN_SMEM, nullptr, args, nullptr);
  if (sr != CUDA_SUCCESS) {
    const char* s = "?"; p_cuGetErrorString(sr, &s);
    printf("{\"tag\":\"stream\",\"ok\":false,\"stage\":\"launch\",\"cu_rc\":%d,\"cu_err\":\"%s\"}\n", (int)sr, s);
    return 2;
  }
  volatile unsigned* mk = (volatile unsigned*)h_mk;
  const long t0 = now_ms();
  unsigned last = 0xFFFFFFFFu; bool done = false;
  while (now_ms() - t0 < timeout_ms) {
    const unsigned st = mk[0];
    if (st != last) {
      printf("{\"tag\":\"progress\",\"stage\":%u,\"t_ms\":%ld}\n", st, now_ms() - t0);
      fflush(stdout); last = st;
      if (st == 7u) { done = true; break; }   // MK_DEALLOC
    }
    usleep(2000);
  }
  if (!done) {
    printf("{\"tag\":\"HANG\",\"elapsed_ms\":%ld,\"stage\":%u,\"prod_n\":%u,\"cons_c\":%u}\n",
           now_ms() - t0, mk[0], mk[1], mk[2]);
    printf("--- DONE (HANG) ---\n"); fflush(stdout);
    _exit(2);
  }
  sr = p_cuCtxSynchronize();
  const long el = now_ms() - t0;
  if (sr != CUDA_SUCCESS) {
    const char* s = "?"; p_cuGetErrorString(sr, &s);
    printf("{\"tag\":\"stream\",\"ok\":false,\"stage\":\"sync\",\"cu_rc\":%d,\"cu_err\":\"%s\"}\n", (int)sr, s);
    printf("--- DONE (SYNC_ERR) ---\n"); fflush(stdout);
    _exit(2);
  }

  std::vector<float> D((size_t)grid * M * N);
  unsigned dg[DG_N] = {0};
  std::vector<unsigned long long> ts((size_t)grid * 2, 0);
  p_cuMemcpyDtoH(D.data(), d_D, D.size() * 4);
  p_cuMemcpyDtoH(dg, d_diag, sizeof(dg));
  p_cuMemcpyDtoH(ts.data(), d_ts, ts.size() * 8);

  unsigned long long max_dwell = 0, min_dwell = ~0ull;
  int ts_valid = 1;
  for (int b = 0; b < grid; ++b) {
    if (ts[b * 2 + 1] <= ts[b * 2]) { ts_valid = 0; continue; }
    const unsigned long long d = ts[b * 2 + 1] - ts[b * 2];
    if (d > max_dwell) max_dwell = d;
    if (d < min_dwell) min_dwell = d;
  }
  int occ = -1;
  if (p_cuOccupancy(&occ, fn, (int)NTHR, (size_t)DYN_SMEM) != CUDA_SUCCESS) occ = -1;
  unsigned long long gmin = ~0ull, gmax = 0;
  for (int b = 0; b < grid; ++b) {
    if (ts[b * 2 + 0] < gmin) gmin = ts[b * 2 + 0];
    if (ts[b * 2 + 1] > gmax) gmax = ts[b * 2 + 1];
  }
  const unsigned long long span = (gmax > gmin) ? (gmax - gmin) : 0;
  // 顺序验证：order_mode!=0 时 D 必须等于「最后一块」的参考（每块数据随块号平移）
  const int ref_blk = (order_mode == 2) ? (int)(chunks - 2) : (int)(chunks - 1);   // stream1b 故意用错块 ⇒ 反证
  int ref_distinct = 0;
  if (order_mode) {
    for (int m = 0; m < M; ++m)
      for (int n = 0; n < N; ++n)
        if (ref_D_b(m, n, (int)(chunks - 1)) != ref_D_b(m, n, (int)(chunks - 2))) ref_distinct++;
  }
  int mism = 0, first_bad = -1;
  for (int b = 0; b < grid; ++b)
    for (int m = 0; m < M; ++m)
      for (int n = 0; n < N; ++n) {
        const float ref = order_mode ? ref_D_b(m, n, ref_blk) : ref_D(m, n);
        const float got = D[(size_t)b * M * N + m * N + n];
        if (!(got == ref)) { mism++; if (first_bad < 0) first_bad = b * M * N + m * N + n; }
      }
  const double dwell_s = (double)max_dwell * 1e-9;
  const double gbps = dwell_s > 0 ? ((double)total_bytes / 1e9) / dwell_s : 0.0;
  printf("{\"tag\":\"stream\",\"ok\":%s,\"grid\":%d,\"chunks\":%ld,\"bytes\":%zu,"
         "\"max_dwell_ns\":%llu,\"min_dwell_ns\":%llu,\"span_ns\":%llu,\"gbps_span\":%.2f,"
         "\"occ_blocks_per_sm\":%d,\"ts_valid\":%d,\"gbps\":%.2f,"
         "\"per_sm_gbps\":%.2f,\"d_mismatch\":%d,\"first_bad_idx\":%d,\"mbar_ok\":%u,\"prod_fail\":%u,\"status\":\"0x%04X\","
         "\"smem_a\":%u,\"smem_b\":%u,\"prod_n\":%u,\"cons_c\":%u,\"order_mode\":%d,\"ref_blk\":%d,"
         "\"ref_distinct\":%d,\"host_elapsed_ms\":%ld}\n",
         (mism == 0 && ts_valid && dg[11] == 1u) ? "true" : "false", grid, chunks, total_bytes,
         max_dwell, (min_dwell == ~0ull ? 0ull : min_dwell), span,
         span > 0 ? ((double)total_bytes / 1e9) / ((double)span * 1e-9) : 0.0, occ, ts_valid, gbps,
         grid > 0 ? gbps / grid : 0.0, mism, first_bad, dg[11], dg[12], dg[0], dg[13], dg[14],
         mk[1], mk[2], order_mode, order_mode ? ref_blk : -1, ref_distinct, el);
  printf("{\"tag\":\"done\",\"rc\":%d}\n", (mism == 0 && ts_valid) ? 0 : 3);
  fflush(stdout);
  return (mism == 0 && ts_valid) ? 0 : 3;
}
