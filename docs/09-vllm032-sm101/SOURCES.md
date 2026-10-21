# 源码与许可

- vLLM: https://github.com/vllm-project/vllm ，提交5df63a7203cdb9328d824a27b682e77c7b9928be，Apache-2.0。原生NVFP4参考测试由其tests/kernels/quantization适配，保留SPDX及许可证。
- vllm-project/flash-attention: vLLM上述提交的CMake固定9cd61de38763d712bb6ce56e2a02cc2bf718c89f，BSD-3-Clause。SM101 FA2补丁关闭本模型不使用的ALiBi/softcap/local/非对齐head分支。编译出的本轮扩展SHA256为5a9c46c84e005c28c8db8cf1c8645002b8bf284c9594040e34db97e9a97b94ba。
- 5090参考：https://github.com/adrienbrault/qwen3.8-27b-rtx5090 ，研究时固定c67ab758cf42208bfc9a2ad2d7d75733fb6ccc17；仅作为优化方向参考，本轮未照搬SM120/NVFP4-KV内核。

build-reference保留本次已执行的宿主构建命令及精确路径；它们是构建记录，不是通用一键安装器。使用者需准备合法DriveOS/CUDA12.8 SDK、aarch64 sysroot、GCC14、aarch64 PyTorch2.13目标库及对应源码依赖，并将记录中的本地路径映射到自己的目录。仓库不分发SDK、凭证、模型或3.5GB私有运行环境。
