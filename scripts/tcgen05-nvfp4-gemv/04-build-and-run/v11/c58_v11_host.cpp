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
#include <cmath>
#include <vector>
#include "t4_smem_layout.h"

#define M 128
#define N 32
#define KT 256
#define A_SUB 4096
#define B_SUB 1024
#define B_CHUNK 4608   // T4-4：qs 4096 + 真实缩放 512
#define NTHR T4_V10R_NTHR
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
      }// ---------------------------------------------------------------------------
}

// T4-4 原型宿主：吃**真实张量流**（stream_big.bin / stream_num.bin），用宿主 double 参考做容差比对。
// 用法: c58_v11_host <t11> <cubin> <chunks/tile> <ntiles> <timeout_ms> <flags> <raw.bin> <ref> [dump] [row_stride]
//   t11 = **真实张量**臂：设备侧输入是 GGML NVFP4 张量的前 ntiles×32 行（行主序，行步长 row_stride）；
//         块 b 吃第 b 个 32 行 tile 的 chunks 个 K-chunk（chunks ≤ K/256）。
//         参考 <ref> 由 canonical 流同源产出（`verify_t4_4.py ref stream_num.bin 20 2 <prefix>`）。
// flags: bit0=1 发 MMA（=0 纯拷贝臂）；bit1=2 写痕迹；bit2=4 每 warp 1 次 arrive。
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "t11";
  const char* cubin = argc > 2 ? argv[2] : "c58_v11.cubin";
  const long chunks = argc > 3 ? atol(argv[3]) : 20;          // 每 tile 的 K-chunk 数（K/256）
  const int grid = argc > 4 ? atoi(argv[4]) : 2;              // tile 数 = 块数
  const long timeout_ms = argc > 5 ? atol(argv[5]) : 30000;
  const unsigned flags = argc > 6 ? (unsigned)atol(argv[6]) : 1u;
  const char* raw_file = argc > 7 ? argv[7] : "/tmp/t4work/raw_t2.bin";
  const char* ref_file = argc > 8 ? argv[8] : "/tmp/t4work/ref_num20.bin";
  const unsigned row_stride = argc > 10 ? (unsigned)atol(argv[10]) : 2880u;   // GGML 行步长 = (K/64)*36
  const bool real_mode = (strcmp(mode, "t11") == 0);
  if (!real_mode) { printf("{\"tag\":\"bad_mode\",\"mode\":\"%s\"}\n", mode); return 1; }

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
  const int DYN_SMEM = (int)T4_V10R_SMEM_HOST_REQ;
  printf("{\"tag\":\"env\",\"mode\":\"%s\",\"cc\":\"%d.%d\",\"cuda_ver\":%d,\"shape\":\"M=%d,N=%d,KCHUNK=%d\","
         "\"sm_count\":%d,\"max_thr_per_sm\":%d,\"max_smem_per_sm\":%d,\"chunks\":%ld,\"grid\":%d,\"chunk_bytes\":%d,"
         "\"threads\":%d,\"smem_required\":%d,\"smem_host_req\":%d,\"flags\":%u,\"stream\":\"%s\",\"row_stride\":%u}\n",
         mode, ccM, ccm, (int)CUDA_VERSION, M, N, KT, nsm, sms, smem_sm, chunks, grid, B_CHUNK, (int)NTHR,
         (int)T4_V10R_SMEM_REQUIRED, DYN_SMEM, flags, raw_file, row_stride);
  fflush(stdout);

  CUmodule mod = nullptr; CUfunction fn = nullptr;
  if (p_cuModuleLoad(&mod, cubin) != CUDA_SUCCESS) { printf("{\"tag\":\"stream\",\"ok\":false,\"stage\":\"module_load\"}\n"); return 2; }
  if (p_cuModuleGetFunction(&fn, mod, "k_b11_ggml") != CUDA_SUCCESS) { printf("{\"tag\":\"stream\",\"ok\":false,\"stage\":\"getfn\"}\n"); return 2; }

  void* h_mk = nullptr; CUdeviceptr d_mk = 0;
  if (p_cuMemHostAlloc(&h_mk, MK_N * 4, 0x02) != CUDA_SUCCESS) { printf("{\"tag\":\"stream\",\"ok\":false,\"stage\":\"hostalloc\"}\n"); return 2; }
  memset(h_mk, 0, MK_N * 4);
  p_cuMemHostGetDevicePointer(&d_mk, h_mk, 0);

  // ★ v11：设备输入 = GGML 张量的前 grid×32 行（不是分块流）
  const size_t total_bytes = (size_t)grid * 32u * (size_t)row_stride;
  CUdeviceptr d_gB = 0, d_D = 0, d_diag = 0, d_ts = 0, d_sf = 0;
  if (p_cuMemAlloc(&d_gB, total_bytes) != CUDA_SUCCESS) { printf("{\"tag\":\"stream\",\"ok\":false,\"stage\":\"alloc_gB\",\"bytes\":%zu}\n", total_bytes); return 2; }
  p_cuMemAlloc(&d_D, (size_t)grid * M * N * 4);
  p_cuMemAlloc(&d_diag, (size_t)DG_N * 4);
  p_cuMemAlloc(&d_ts, (size_t)grid * 2 * 8);
  p_cuMemsetD32(d_D, 0, (size_t)grid * M * N);
  p_cuMemsetD32(d_diag, 0, DG_N);
  p_cuMemsetD32(d_ts, 0, (size_t)grid * 2 * 2);
  const bool dbg_sf = (flags & 8u) != 0u;
  std::vector<float> sfv((size_t)128 * 96, 0.f);
  if (dbg_sf && p_cuMemAlloc(&d_sf, sfv.size() * 4) != CUDA_SUCCESS) {
    printf("{\"tag\":\"stream\",\"ok\":false,\"stage\":\"alloc_sf\"}\n"); return 2;
  }
  if (dbg_sf) p_cuMemsetD32(d_sf, 0, sfv.size());

  std::vector<uint8_t> host_buf(total_bytes);
  { FILE* f = fopen(raw_file, "rb");
    if (!f) { printf("{\"tag\":\"stream\",\"ok\":false,\"stage\":\"open_stream\",\"file\":\"%s\"}\n", raw_file); return 2; }
    const size_t got = fread(host_buf.data(), 1, total_bytes, f);
    fclose(f);
    if (got != total_bytes) { printf("{\"tag\":\"stream\",\"ok\":false,\"stage\":\"short_stream\",\"got\":%zu,\"want\":%zu}\n", got, total_bytes); return 2; }
  }
  p_cuMemcpyHtoD(d_gB, host_buf.data(), total_bytes);
  p_cuCtxSynchronize();
  printf("{\"tag\":\"filled\",\"bytes\":%zu,\"src\":\"file\"}\n", total_bytes); fflush(stdout);

  if (p_cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, DYN_SMEM) != CUDA_SUCCESS) {
    printf("{\"tag\":\"stream\",\"ok\":false,\"stage\":\"funcattr\",\"dyn_smem\":%d}\n", DYN_SMEM); return 2;
  }
  unsigned u_chunks = (unsigned)chunks;
  unsigned u_flags = flags;
  const char* cpr_env = getenv("T4_V10R_CPR");
  unsigned u_cpr = (unsigned)(cpr_env ? strtoul(cpr_env, nullptr, 10) : 0ul);   // ★ 83：内核内 epilogue
  unsigned u_row_stride = row_stride;
  void* args[10] = {&d_gB, &u_chunks, &d_D, &d_mk, &d_diag, &d_ts, &u_flags, &d_sf, &u_cpr, &u_row_stride};

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
      if (st == 7u) { done = true; break; }
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

  std::vector<float> ref((size_t)grid * M * N, 0.f);
  { FILE* f = fopen(ref_file, "rb");
    if (!f) { printf("{\"tag\":\"stream\",\"ok\":false,\"stage\":\"open_ref\",\"file\":\"%s\"}\n", ref_file); return 2; }
    const size_t want = (grid > 1) ? ref.size() * 4 : (size_t)M * N * 4;
    const size_t got = fread(ref.data(), 1, want, f);
    fclose(f);
    if (got != want) { printf("{\"tag\":\"stream\",\"ok\":false,\"stage\":\"short_ref\",\"got\":%zu,\"want\":%zu}\n", got, want); return 2; }
    if (want < ref.size() * 4 && grid > 1)
      for (int b = 1; b < grid; ++b) memcpy(&ref[(size_t)b * M * N], &ref[0], sizeof(float) * M * N);
  }

  int nmis = 0, first_bad = -1;
  double maxabs = 0.0, sumabs = 0.0;
  for (size_t i = 0; i < D.size(); ++i) {
    const double g = D[i], r = ref[i];
    const double d = fabs(g - r);
    if (d > maxabs) maxabs = d;
    sumabs += fabs(r);
    if (!(d <= 1e-3 * (1.0 + fabs(r)))) { nmis++; if (first_bad < 0) first_bad = (int)i; }
  }
  const double mean_abs_ref = sumabs / (double)D.size();
  if (argc > 9) {                                   // 诊断：把 D 与参考落盘
    FILE* fo = fopen(argv[9], "wb");
    if (fo) { fwrite(D.data(), 4, D.size(), fo); fwrite(ref.data(), 4, ref.size(), fo); fclose(fo); }
  }
  if (dbg_sf) {                                     // 诊断：SFA/SFB 的 TMEM 内容落盘
    p_cuMemcpyDtoH(sfv.data(), d_sf, sfv.size() * 4);
    char sp2[1200];
    if (argc > 9) snprintf(sp2, sizeof(sp2), "%s.sf", argv[9]); else snprintf(sp2, sizeof(sp2), "/tmp/sfdump.bin");
    FILE* fs2 = fopen(sp2, "wb");
    if (fs2) { fwrite(sfv.data(), 4, sfv.size(), fs2); fclose(fs2); printf("{\"tag\":\"sfdump\",\"file\":\"%s\",\"bytes\":%zu}\n", sp2, sfv.size()*4); }
    else printf("{\"tag\":\"sfdump\",\"ok\":false}\n");
  }
  { int shown = 0;                                  // 诊断：头几个 got/ref 对
    printf("{\"tag\":\"probe\","); 
    for (int i = 0; i < 6 && i < (int)D.size(); ++i) {
      const int m = i / N, n = i % N;
      printf("%s\"m%d_n%d\":[%.6g,%.6g]", shown++ ? "," : "", m, n, D[i], ref[i]);
    }
    printf("}\n"); }
  const double dwell_s = (double)max_dwell * 1e-9;
  const double gbps = dwell_s > 0 ? ((double)total_bytes / 1e9) / dwell_s : 0.0;
  printf("{\"tag\":\"stream\",\"ok\":%s,\"grid\":%d,\"chunks\":%ld,\"bytes\":%zu,"
         "\"max_dwell_ns\":%llu,\"min_dwell_ns\":%llu,\"span_ns\":%llu,\"gbps_span\":%.2f,"
         "\"occ_blocks_per_sm\":%d,\"ts_valid\":%d,\"gbps\":%.2f,\"per_sm_gbps\":%.2f,"
         "\"d_mismatch\":%d,\"first_bad_idx\":%d,\"maxabs\":%.6g,\"mean_abs_ref\":%.6g,"
         "\"mbar_ok\":%u,\"prod_fail\":%u,\"epi\":%u,\"cyc\":[%u,%u,%u,%u,%u,%u,%u,%u,%u],\"status\":\"0x%04X\",\"host_elapsed_ms\":%ld,\"flags\":%u}\n",
         (nmis == 0 && ts_valid && dg[11] == 1u) ? "true" : "false", grid, chunks, total_bytes,
         max_dwell, (min_dwell == ~0ull ? 0ull : min_dwell), span,
         span > 0 ? ((double)total_bytes / 1e9) / ((double)span * 1e-9) : 0.0, occ, ts_valid, gbps,
         grid > 0 ? gbps / grid : 0.0, nmis, first_bad, maxabs, mean_abs_ref,
         dg[11], dg[12], dg[13], dg[3], dg[4], dg[5], dg[6], dg[7], dg[8], dg[9], dg[10], dg[14], dg[0], el, flags);
  printf("{\"tag\":\"done\",\"rc\":%d}\n", (nmis == 0 && ts_valid) ? 0 : 3);
  fflush(stdout);
  return 0;
}
