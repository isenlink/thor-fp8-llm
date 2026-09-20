# tcgen05 分支 — 自研 T4 tcgen05 实验件（只记录，不公开）

> ⚠️ **定位声明**：本分支内容是**自研 T4 tcgen05 实验件**——**不是最终件，不对外发布**。
> 仅作内部实验记录：过程数据、结论、失败尝试。仓库 `main` 分支的
> FP8 权重侧快路（`docs/08-fp8-fastpath-server/`）才是公开件。

## 背景

- tcgen05（第 5 代 Tensor Core）NVFP4 快路内核线，与 main 分支的 FP8/MMQ 路线并行
- 同板实测参照（vs FP8 路线）：8K 41.8 / 128K 27.4 / 200K 24.1 t/s（FP8 路线：22.18 / 19.70）
- 不公开的原因：部分实验件**精度结论未定论**（长上下文下有退化案例），按"宁可慢也不能错"原则只记录、不分发

## 现状

- `main` 分支 `scripts/tcgen05-nvfp4-gemv/` 已收录 GEMV 快路内核的代码级整理包（−49 ms/步）与测量纪律工具
- 本分支待整理：tcgen05-fp4 内核源码（`ggml/src/ggml-cuda/tcgen05-fp4.cu`）、
  `GGML_CUDA_TCGEN05*` 环境变量门控、sm_101a 构建配置——**仅作记录归档，不构建分发**

## 起草器（DFlash2）获取说明

起草器权重体积超出 GitHub 限制，不入仓。需要时按以下详细名称自行获取：

- **推荐档**：`Qwen3.8-27B-DFlash2-Q2_K_S-MIX.gguf`（约 536 MB / 0.52 GiB，81 张量 = DFlash2 形态，
  来源 HuggingFace `HermiHg/Qwen3.8-27B-DFlash2-Q2_K_S-MIX-GGUF`）
- 备选档（同一来源系列）：`Qwen3.8-27B-DFlash2-Q4_K_M`（约 1.06 GiB）/ `-Q8_0`（约 2.06 GiB）/ `-BF16`（约 3.60 GiB）
- 判别方法：读 GGUF 头第 3 个数（张量数），**81 = DFlash2 可用**；58 = DFlash1；19 = MTP 侧车（不适用本用法）
- 上游 PR 参考：llama.cpp #27342（`--spec-type draft-dflash` 支持在 main 分支的 pin `72797e89` 中已自带）

## 口径提醒

- 报速度必须带上下文长度与请求长度；跨臂比较看 ms/步，不看 t/s
- 精度未定论的实验件不发布（"宁可慢也不能错"）
