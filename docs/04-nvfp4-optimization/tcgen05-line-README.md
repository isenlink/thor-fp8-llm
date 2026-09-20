# tcgen05 内核线（独立分支）

> 本分支承载 tcgen05（第 5 代 Tensor Core）NVFP4 快路内核线的完整源码与构建材料。
> 与 `main` 分支的 FP8 权重侧快路线（docs/08-prebuilt-server）是**两条并行路线**。

## 现状

- `main` 分支 `scripts/tcgen05-nvfp4-gemv/` 已收录 GEMV 快路内核的代码级整理包（−49 ms/步）与测量纪律工具
- 同板实测参照（vs FP8 路线）：8K 41.8 / 128K 27.4 / 200K 24.1 t/s（FP8 路线：22.18 / 19.70）
- 本分支待整理：tcgen05-fp4 内核源码（`ggml/src/ggml-cuda/tcgen05-fp4.cu`）、
  `GGML_CUDA_TCGEN05*` 环境变量门控、sm_101a 构建配置

## 材料

完整 tcgen05 内核线材料整理中（2026-09-20 立项，材料确认后入库）。

## 口径提醒

- 报速度必须带上下文长度与请求长度；跨臂比较看 ms/步，不看 t/s
- 精度未定论的实验件不发布（"宁可慢也不能错"）
