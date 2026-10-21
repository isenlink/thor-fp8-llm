# 交叉编译与二进制来源

本轮确有x86宿主→Linux AArch64/SM101交叉编译。vLLM基线源码提交5df63a7203cdb9328d824a27b682e77c7b9928be；Python3.12、aarch64 PyTorch2.13、CUDA12.8私有工具链、GCC14/sysroot。vLLM wheel中6个ELF均已检查e_machine=183（AArch64）。wheel名+cu120是遗留元数据，实际工具链12.8，不宣称12.0已测。

GitHub docs/09-vllm032-sm101/build-reference保存本次构建脚本和toolchain记录；patches/vllm032-sm101/vllm-build-worktree.patch保存本机构建树相对该提交的完整改动（含C++/CUDA跨编译相关修改），fa2-native-sm101.patch保存FA2独立CMake改动。构建脚本中的本机路径是记录，不是通用一键脚本，需使用者映射到合法SDK/sysroot依赖。构建工作树快照不能单独证明wheel每个对象均由该时点源码重建；未进行独立可复现重构验证，二进制以SHA和实际测试工件为准。

FA2源码由vLLM CMake固定vllm-project/flash-attention 9cd61de38763d712bb6ce56e2a02cc2bf718c89f。独立FA2扩展SHA为5a9c46c84e005c28c8db8cf1c8645002b8bf284c9594040e34db97e9a97b94ba，来自实测环境。闭掉本模型不用的ALiBi/softcap/local/不对齐head分支，不能宣称这些功能受支持。

与基础wheel逐文件比对实测vLLM包的.py/.so共6处差异：FA2另供.so，其他5个Python文件收入vllm-tested-site-overlay.tar.gz，精确路径与SHA见TESTED-OVERLAY-MANIFEST.json；对应源码patch收入GitHub runtime/tested-site-overlay.patch。这使发布者不会漏掉warmup、配置、平台检测和hugetlb内存统计差异。

公开二进制只覆盖vLLM/FA2；运行仍需匹配Python/PyTorch/CUDA/DriveOS环境及代码插件，基础wheel本身不是完整实测环境。完整私有venv/runtime只留模型库，SDK/第三方运行库不随公开包再分发。独立公共安装路径尚未重新端到端验证，不能标为通用一键部署。
