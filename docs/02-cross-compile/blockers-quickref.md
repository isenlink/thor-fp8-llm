# 交叉编译连环坑速查表

> One-liner: Quick-reference of the four CMake blocking issues + the glibc/cospi
> header conflict when cross-compiling CUDA 12.8 (ARM64 nvcc via qemu-user) for
> sm_101a on a Debian host. Full narrative in cross-compile-log.md.
>
> 日期：2026-09-08（本仓库版：脱敏整理）

【环境】主机 Debian 13 (x86_64)，目标：Thor 板 aarch64 sm_101a，CUDA 12.8.93（ARM64 版 nvcc 经 qemu-user 运行）

## 四个连环坑

【坑1】CMake 找不到 CUDA Toolkit → 设 `CUDAToolkit_ROOT` + `CUDAToolkit_LIBRARY_DIR`（指向 targets/sbsa-linux/lib）

【坑2】CUDA::cublas 缺失 → 下载 libcublas-dev-12-8 deb 解包进 nvcc-extract

【坑3】CUDA::cuda_driver 缺失 → 从板上 scp 真 libcuda.so.1 放进 sbsa lib 目录

【坑4】C/C++ 编译器无 sysroot → `CMAKE_C_FLAGS`/`CXX_FLAGS` 显式传 `--sysroot`（注意 CMake 里 `$SR` 变量不展开，**必须写绝对路径**！`CMAKE_SYSROOT` 单独设不生效）

## cospi 冲突终极解法

Debian glibc 2.41 的 mathcalls.h（`__MATHCALL_VEC`）与 CUDA 12.8 crt/math_functions.h 的 noexcept 声明在 C++ host 预处理时冲突。

**解法**：patch CUDA 头（.h 和 .hpp 两处），把 sinpi/sinpif/cospi/cospif/sincospi/sincospif 共 6 个 device-only 函数声明和 .hpp 里 3 个 inline helper（773-786 行）用 `#if 0` 包掉——host 代码不调用这些 device 函数，安全无副作用。备份 .bak 保留。

## nvcc-wrapper 最终形态

```bash
# nvcc-wrapper.sh 要点：
# - QEMU_LD_PREFIX + noble sysroot -L
# - -ccbin aarch64-linux-gnu-g++ --sysroot=noble（经 -Xcompiler）
# - -I/-L 指向 CUDA targets/sbsa-linux
# - host glibc 用 Debian 2.41（恢复原状），CUDA 冲突头已 patch
# 与失败方案的差异：不再试图让 glibc 头全部走 noble，只 patch CUDA 头
```

完整工具链搭建过程见 [cross-compile-log.md](./cross-compile-log.md)。
