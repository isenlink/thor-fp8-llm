# GPU 验证与交叉编译环境首夜记录

> One-liner: First-night deviceQuery verification of the Thor GPU (sm_101a,
> 14 SM, 20GiB visible) and the initial working cross-compile command template.
>
> 日期：2026-09-07 深夜（本仓库版：脱敏整理）

## GPU 验证 PASS

板端运行自制 deviceQuery（纯 CUDA runtime，无依赖）：

```
GPU 名: "Thor"
compute capability: 10.1 (SM101a)
SM: 14 @ 1530MHz
L2: 24MiB
GPU 可见显存: 20GiB（统一内存分区，非全部物理内存）
总线: 256-bit
CUDA 计算内核测试: PASS
```

## 关键发现

1. **板上没有任何编译器**（无 gcc/nvcc，只有 cuda-gdb），运行库齐全。符合 DriveOS 设计哲学：主机编译、板端运行。
2. 板载数据分区实际是 rw 挂载（官方资料说"默认不可写"是指**权限**）：目录属主为系统预置账号，普通用户写不进。需要 sudo 一次创建工作区并 chown。
3. 板上 /tmp 是 30G tmpfs（RAM 盘），可用于临时文件，重启丢失。

## 主机交叉编译链（首夜打通版）

- qemu-user-static + aarch64-linux-gnu-gcc (Debian 14.2) + libc6-arm64-cross
- ARM64 版 nvcc 12.8.93（从 NVIDIA sbsa repo 下载 deb 解包）：nvcc+crt+cccl+cudart-dev+nvvm(cicc)，共 5 个 deb
- Ubuntu noble (24.04) glibc 2.39 sysroot（Debian 13 的 glibc 2.41 与 CUDA 12.8 头文件冲突，所以必须用 noble 的 2.39）
- 编译命令模板：

```bash
qemu-aarch64 -L $SR $CUDA/bin/nvcc \
  -ccbin aarch64-linux-gnu-g++ \
  -arch=sm_101a \
  -isystem $SR/usr/include \
  -isystem $SR/usr/include/aarch64-linux-gnu \
  -I$CUDA/include \
  -o out src.cu
```

- 产出验证程序已 scp 到板端运行 PASS

> 后续完整版工具链（含 cospi patch、libcublas/libcuda 处理）见
> [cross-compile-log.md](../02-cross-compile/cross-compile-log.md) 与
> [blockers-quickref.md](../02-cross-compile/blockers-quickref.md)。
