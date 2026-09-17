#pragma once

// T4（自研 tcgen05 NVFP4 快路）与生产 MMQ 的接缝。
// 背景/约束/回退步骤见 project 仓的 tasks/T4-M2c-rollout.md；数值判据见 results/20260916-108。

#include "common.cuh"

#include <cuda_runtime.h>

struct mmq_args;

// 由 mmq.cu / mmvq.cu 在各自原路之前调用（**两个接缝**：2026-09-16 21:1x 实测
// NVFP4 decode 的 M ≤ MMVQ_MAX_BATCH_SIZE 走 mmvq，根本不过 mmq ⇒ 只挂 mmq 会静默失效）。
// 返回 true = 本次已由 T4 处理完，调用方必须直接 return。
// 返回 false = 不适用/未启用，调用方按原路继续（默认路径，行为与打此补丁前一致）。
//
// 条件判定（不看环境变量）在 t4_mmq.cu 里把张量字段填进 `t4_mmq_pred_in`（纯判定头，
// 由宿主单测 `t4/m2c/host/test_t4_mmq_pred.cpp` 覆盖真值表）；任何一条不满足 ⇒ 原路。
//   `x`（src1）的行必须 32 B 对齐且行步长 %8==0（`aq_float8` 向量载入；非对齐会触发 A 类事故，
//   见 INCIDENTS §十四）——A 由我们自己量化，故不吃上游的 q8_1/MMQ 布局。
bool t4_mmq_try_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * x,
                       ggml_tensor * dst, int cc, cudaStream_t stream);
