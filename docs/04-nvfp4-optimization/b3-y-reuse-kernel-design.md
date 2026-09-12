# B3 y 复用 kernel 改造设计文档

日期：2026-09-12
目标：把 NVFP4 MMVQ kernel 的 y 向量重复 L2 读取消除，decode 16.4→~22 t/s

## 1. 问题（已实锤）

真实 kernel（mmvq.cu:728-730）的 y 读取在**行循环内部**：
```cpp
for (int i = 0; i < rows_per_cuda_block; ++i) {
    tmp[j][i] += vec_dot_q_cuda(vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
}
```
M=1 时 y 对所有行相同，但每行重读整段 y（5120B）。17408 行 × 5120B = **89MB L2 流量**（权重 DRAM 50MB 的 1.8 倍）。

microbench5 实测：y 提到行循环外 + rows_per_block=8 → 108→202 GB/s（+88%）。

## 2. y 访问模式（NVFP4 MMVQ，已推导核实）

- 每 kbx，线程 tid 读 y block `2*kbx + (tid%2)` 的 **32B**（qs[0..15] + qs[32..47]，2×16B 非连续）
  - i=0: i8=0 → qs[0..3]（16B）
  - i=1: i8=8 → qs[8..11]（16B）
- 32 线程 × 32B = 1024B/iter，5 iter = 5120B = 完整 y ✓
- kbx = tid/2 + iter*16，kby = 2*kbx

## 3. 改动方案（最小侵入，仅 NVFP4 + ncols_dst==1 + !small_k）

### 3.1 calc_rows_per_block（mmvq.cu:563）
```cpp
case 1:
    return small_k ? nwarps : 1;
```
改为：
```cpp
case 1:
    if (small_k) return nwarps;
    // [B3] M=1 decode：增大 rows_per_block 让 y 跨行复用（microbench5: R=8 → +88%）
    return 8;
```
**host/device 一致性**：`calc_launch_params`（:987）和 kernel（:602）都调同一函数 → 自动一致。✓

### 3.2 kernel 主循环（mmvq.cu:699-739）
在 kbx 循环内、行循环外，预读 y 到寄存器；行循环调新 vec_dot 变体：
```cpp
for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
    const int kby = kbx * (qk/QK8_1);
    const int kqs = vdr * (tid % (qi/vdr));

    // [B3] NVFP4 + rows>1：预读 y 到寄存器（跨行复用，消除重复 L2 读取）
    int y_pre[8];
    if constexpr (type == GGML_TYPE_NVFP4 && rows_per_cuda_block > 1) {
        const block_q8_1 * bq8 = y + kby + (tid % 2);
        const int * qsy = (const int *) bq8->qs;
        y_pre[0]=qsy[0]; y_pre[1]=qsy[1]; y_pre[2]=qsy[2]; y_pre[3]=qsy[3];  // i=0: i8=0
        y_pre[4]=qsy[8]; y_pre[5]=qsy[9]; y_pre[6]=qsy[10]; y_pre[7]=qsy[11]; // i=1: i8=8
    }

    for (int j = 0; j < ncols_dst; ++j) {
        for (int i = 0; i < rows_per_cuda_block; ++i) {
            if constexpr (type == GGML_TYPE_NVFP4 && rows_per_cuda_block > 1) {
                tmp[j][i] += vec_dot_nvfp4_q8_1_preload(vx, y_pre, kbx_offset + i*stride_row_x + kbx, kqs);
            } else {
                tmp[j][i] += vec_dot_q_cuda(vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
            }
        }
    }
}
```

**注意**：`y_pre` 的读取 `(const int*) bq8->qs` —— block_q8_1.qs 在 offset 2（非 4B 对齐）！
**必须用 get_int_b4 逐字节构造**（vecdotq.cuh:27 `get_int_b4` 假设 4B 对齐，但 qs 实际 offset 2）。
→ 用 `ld4` 逐字节读（同 microbench4 教训）：
```cpp
static __device__ __forceinline__ int ld4u(const uint8_t * p) {
    return (int)(p[0] | (p[1]<<8) | (p[2]<<16) | (p[3]<<24));
}
y_pre[0]=ld4u(bq8->qs+0); y_pre[1]=ld4u(bq8->qs+4); y_pre[2]=ld4u(bq8->qs+8); y_pre[3]=ld4u(bq8->qs+12);
y_pre[4]=ld4u(bq8->qs+32); y_pre[5]=ld4u(bq8->qs+36); y_pre[6]=ld4u(bq8->qs+40); y_pre[7]=ld4u(bq8->qs+44);
```
等等——qs 是 32B（32 字节），qs[32..47] 越界！qs 只有 32 字节（index 0..31）。
重新核对：i8=8 → get_int_b4(qs, 8) = ((const int*)qs)[8] = qs[32..35]？

**错误**：get_int_b4(x, i32) = ((const int*)x)[i32]，i32 是 **int 索引**不是字节。
i8=8 → ((const int*)qs)[8] = 第 8 个 int = 字节 32..35。但 qs 只有 32 字节（8 个 int，index 0..7）！
→ i8=8 越界？

重新看 vecdotq.cuh:352-355：
```cpp
int sumi = ggml_cuda_dp4a(v0.x, get_int_b4(bq8->qs, i8 + 0), 0);
sumi = ggml_cuda_dp4a(v0.y, get_int_b4(bq8->qs, i8 + 2), sumi);
sumi = ggml_cuda_dp4a(v1.x, get_int_b4(bq8->qs, i8 + 1), sumi);
sumi = ggml_cuda_dp4a(v1.y, get_int_b4(bq8->qs, i8 + 3), sumi);
```
i=0: i8=0 → qs[0],qs[1],qs[2],qs[3]（int 索引 0..3 = 字节 0..15）
i=1: i8=8 → qs[8],qs[9],qs[10],qs[11]（int 索引 8..11 = 字节 32..47）

qs 是 32 字节 = 8 个 int（index 0..7）。i8=8 → index 8..11 **越界**！

这说明我的推导有误。重新看 is 和 i8：
```cpp
const int32_t is = iqs0 >> 1;
const block_q8_1 * bq8 = bq8_1 + (is >> 1);
const int32_t i8 = ((is & 1) << 2);
```
i=0: iqs0 = iqs + 0 = kqs。kqs = 4*(tid%2)。
- tid%2=0: iqs0=0, is=0, bq8=bq8_1+0, i8=0 → qs[0..3]
- tid%2=1: iqs0=4, is=2, bq8=bq8_1+1, i8=0 → qs[0..3]
i=1: iqs0 = kqs+2。
- tid%2=0: iqs0=2, is=1, bq8=bq8_1+0, i8=8 → qs[8..11] **越界**
- tid%2=1: iqs0=6, is=3, bq8=bq8_1+1, i8=8 → qs[8..11] **越界**

这不可能——真实 kernel 能跑，不会越界。我哪里错了？

**重新看 QK8_1**：block_q8_1 的 qs 是 `int8_t qs[QK8_1]`，QK8_1=32。所以 qs 是 32 字节。
get_int_b4(qs, i8) = ((const int*)qs)[i8]。i8 最大 = 8+3 = 11。((const int*)qs)[11] = 字节 44..47。**越界**（qs 只有 32 字节）。

除非... QK8_1 不是 32？查 ggml-common.h：`int8_t qs[QK8_1]`，QK8_1 定义在哪？

实际上 Q8_1 的 QK=32，但 block_q8_1 的 qs 可能是 32 字节。让我重新确认。

**关键**：我可能搞错了 i8 的范围。重新看：
- i=0: iqs0 = kqs + 0。is = iqs0>>1。
  - tid%2=0: kqs=0, iqs0=0, is=0, i8=(0&1)<<2=0
  - tid%2=1: kqs=4, iqs0=4, is=2, i8=(2&1)<<2=0
- i=1: iqs0 = kqs + 2。is = iqs0>>1。
  - tid%2=0: kqs=0, iqs0=2, is=1, i8=(1&1)<<2=8
  - tid%2=1: kqs=4, iqs0=6, is=3, i8=(3&1)<<2=8

所以 i=1 时 i8=8，读 qs[8..11]（int 索引）。如果 qs 只有 8 个 int（32 字节），则越界。

**结论**：要么 QK8_1 > 32（qs 更大），要么我的 kqs 推导有误。

让我重新确认 kqs。mmvq.cu:703：`const int kqs = vdr * (tid % (qi/vdr));`
NVFP4: vdr = VDR_NVFP4_Q8_1_MMVQ = 4。qi = ggml_cuda_type_traits<NVFP4>::qi。

qi 是 NVFP4 的 quant index 数。NVFP4 block = 64 值 = 32 字节 qs = 8 个 int。所以 qi=8。
kqs = 4 * (tid % (8/4)) = 4 * (tid % 2)。✓ 我的推导没错。

那 i8=8 越界怎么解释？除非 get_int_b4 的 i32 不是 int 索引而是别的。

重新看 vecdotq.cuh:27：
```cpp
static __device__ __forceinline__ int get_int_b4(const void * x, const int & i32) {
    return ((const int *) x)[i32]; // assume at least 4 byte alignment
}
```
确实是 int 索引。i32=8 → 第 8 个 int。

**唯一解释**：QK8_1 不是 32，或者 block_q8_1 的 qs 比 32 字节大。

让我查 QK8_1 的定义。实际上在 ggml 里 Q8_1 的 QK=32，但 CUDA 的 block_q8_1 可能有 padding 或不同布局。

**我需要实际查代码确认**，不能猜。这是用户铁律：技术结论必须有依据。

让我在实现前查 QK8_1 和 block_q8_1 的确切定义。

## 4. 推导纠错记录（实现前查清，09-12 14:30）

1. **i8 算术纠错**：`i8 = ((is&1)<<2)`，is=1 时 = `1<<2 = 4`（不是 8）。y 读 int 偏移 0-3 和 4-7，全在 32B qs 内，**无越界**
2. **block_q8_1 布局纠错**：`ds` 是 `ggml_half2` = **4 字节**（ggml-common.h:265-275 核实），qs 在 **offset 4，4B 对齐**，真实 kernel 的 `(const int*)qs` 合法。microbench4 崩溃根因 = 我自定义结构体用了 2 字节 d（qs 落 offset 2 非对齐），非真实 kernel 问题
3. **y 映射定案**：每线程每 kbx 恰好 1 个 y block = `y + kby + (tid%2)`，读全部 8 个 int（i8=0/4）+ ds 低半。i8 只依赖 i（=i*4），不依赖 tid

## 5. 新函数 vec_dot_nvfp4_q8_1_preload（vecdotq.cuh，已实现）

```cpp
// [B3] 预读 y 变体：y 已在寄存器 y_pre[8]（跨行复用），只读权重
static __device__ __forceinline__ float vec_dot_nvfp4_q8_1_preload(
        const void * __restrict__ vbq, const int * y_pre, const int32_t & kbx, const int32_t & iqs) {
    const block_nvfp4 * bq4 = (const block_nvfp4 *) vbq + kbx;
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < VDR_NVFP4_Q8_1_MMVQ/2; i++) {
        const int32_t iqs0 = iqs + 2*i;
        const int32_t iqs1 = iqs0 + 1;
        const int32_t is = iqs0 >> 1;
        const int2 v0 = get_int_from_table_16(get_int_b4(bq4->qs, iqs0), kvalues_mxfp4);
        const int2 v1 = get_int_from_table_16(get_int_b4(bq4->qs, iqs1), kvalues_mxfp4);
        const int32_t i8 = ((is & 1) << 2);
        int sumi = ggml_cuda_dp4a(v0.x, y_pre[i8 + 0], 0);
        sumi = ggml_cuda_dp4a(v0.y, y_pre[i8 + 2], sumi);
        sumi = ggml_cuda_dp4a(v1.x, y_pre[i8 + 1], sumi);
        sumi = ggml_cuda_dp4a(v1.y, y_pre[i8 + 3], sumi);
        const float d = ggml_cuda_ue4m3_to_fp32(bq4->d[is]) * __low2float(bq8_ds);  // 需传 ds
        sum += d * float(sumi);
    }
    return sum;
}
```
**问题**：原函数用 `__low2float(bq8->ds)`（y block 的 scale）。预读变体也需要 ds。
→ y_pre 要包含 ds，或单独传 ds。

**修正**：y_pre[8] 存 qs，另传 `half2 y_ds`（y block 的 scale）。
但 ds 取决于 y block（is>>1），而 is 随 i 变化（i=0: is=0/2, i=1: is=1/3）。
→ 每个 i 的 y block 不同（i=0: bq8_1+(is>>1), i=1: bq8_1+(is>>1)）。
- tid%2=0: i=0 → bq8_1+0, i=1 → bq8_1+0（is=1, is>>1=0）
- tid%2=1: i=0 → bq8_1+1, i=1 → bq8_1+1（is=3, is>>1=1）

所以每个线程只用 1 个 y block 的 ds（tid%2 决定）。→ 传 1 个 half2 ds 即可。

**最终签名**：
```cpp
static __device__ __forceinline__ float vec_dot_nvfp4_q8_1_preload(
        const void * __restrict__ vbq, const int * y_pre, half2 y_ds, const int32_t & kbx, const int32_t & iqs);
```

## 6. 风险与回滚

- 改动仅影响 NVFP4 + ncols_dst==1 + !small_k 路径（M=1 decode）
- 其他类型/路径走原 vec_dot_q_cuda（不变）
- 回滚：git diff 3 处（calc_rows_per_block、kernel 主循环、vecdotq.cuh 新函数），revert 即可
- 正确性验证：standalone 对比改前/改后输出（bit-exact 或 rel_err < 1e-5）

## 7. 验证计划

1. 交叉编译（~37 分钟）
2. standalone 跑 ffn_gate/ffn_down/lm_head（预期 144→200+）
3. 正确性：standalone 输出对比（改前 vs 改后）
4. 全模型 A/B：128K decode 3 连测（预期 16.4→~22 t/s）
5. 温度监控（每 5 分钟）
