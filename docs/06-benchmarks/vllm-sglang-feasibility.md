# vLLM / SGLang 可行性调研

> One-liner: Feasibility analysis of running vLLM/SGLang on DRIVE Thor
> (sm_101a, DriveOS 7.0.3) — theoretically possible, no official support,
> and why we stayed with llama.cpp.
>
> 日期：2026-09-08（本仓库版：脱敏整理）

## 结论

vLLM/SGLang 在 DRIVE Thor（sm_101a）上**理论上可行，但无官方支持、无社区先例**，属高难度自研路线。

## 关键事实

1. Jetson Thor（T5000, sm_110）有官方路线：JetPack 7.0 + CUDA 13 + NGC 容器支持 vLLM/SGLang。但那是 JetPack 栈，与 DRIVE Thor 的 DriveOS 7.0.3（CUDA 12.8）**不同**。
2. vLLM 官方 wheel 不含 sm_110 更不含 sm_101a，必须源码编译（`TORCH_CUDA_ARCH_LIST="11.0"` 是 Jetson 的配法，DRIVE Thor 需要 10.1）。
3. SGLang 依赖 Triton，其 bundled ptxas 不认 sm_101a/110a，需 `TRITON_PTXAS_PATH` 指向系统 ptxas（板无 ptxas，但交叉编译环境里有）。
4. 参考教程：HackMD johnnynunez "Run SGLang Thor & Spark"（Jetson Thor 版蓝本）；vLLM 多节点 Thor 笔记（同作者）。
5. 依赖链深：PyTorch（aarch64+cu128+sm101）→ vLLM/SGLang → Python 3.12 环境，全部要装到数据区避免撑爆系统分区。
6. 社区实测反馈"vLLM/SGLang 未安装成功"——最终选择魔改 llama.cpp 路线，侧面说明此路成本高。

## 决策

短期坚持 llama.cpp（已验证可行 + 编译链已通）；SGLang 作为中期实验，前置条件：板载数据分区工作区 + PyTorch 交叉编译成功。

## 附：GitHub 下载难题记录（国内网络）

GitHub 直连超时；gh-proxy.com / ghfast.top 截断；ghproxy.net 可用但 ~10MB 截断，需断点续传循环（`curl -C -` + `gzip -t` 校验）。完整 master.tar.gz 约 25MB。
