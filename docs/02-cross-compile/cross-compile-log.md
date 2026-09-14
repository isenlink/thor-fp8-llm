# Thor 板 (DRIVE Thor / Tegra264) llama.cpp 交叉编译与 GPU 推理 — 完整实录

> **Summary:** A complete, sanitized field log of cross-compiling llama.cpp 0.4.0-dev (91 binaries, CUDA sm_101a backend) for the NVIDIA DRIVE Thor (Tegra264, DriveOS 7.0.3, aarch64) on an x86 Linux host, with a QEMU-based ARM64 nvcc wrapper, 11 documented pitfalls and fixes, on-board deployment, and GPU inference benchmarks (pp128 ≈ 1516 tok/s, tg64 ≈ 40 tok/s).

> 日期：2026-09-07 夜 ~ 09-08 晨
> 主机：x86主机（笔记本），Debian 13, i7-4600U (4C4T), 11GB RAM
> 目标板：NVIDIA DRIVE Thor 域控板（p3960-0010 / Tegra264），DriveOS 7.0.3, CUDA 12.8, aarch64
> 成果：llama.cpp 0.4.0-dev 全套（91 个二进制，含 CUDA sm_101a 后端）交叉编译成功，板端 GPU 推理验证通过

---

## 一、最终基准数据

Qwen3-4B Q4_K_M（2.32GiB），ngl=99 全 GPU offload：

| 指标 | 结果 |
|---|---|
| Prompt 处理 pp128 | **1516.48 ± 45.99 tok/s** |
| 文本生成 tg64 | **40.19 ± 0.03 tok/s** |
| CUDA 设备 | Thor, compute 10.1, VRAM 20480MiB, VMM yes |
| 满载温度 | 72-74°C（被动散热，健康） |

冒烟测试：中文对话输出流畅合理，链路完整闭环。

---

## 二、环境探明（板端真相）

### 硬件/系统
- GPU: "Thor", compute capability **10.1** (sm_101a, Blackwell), 14 SM @1530MHz, L2 24MiB
- GPU 可见显存 **20GiB**（统一内存分区，非全部 58GiB）
- 验证方法：自制 deviceQuery 程序（纯 CUDA runtime，无依赖）

### 关键事实
1. **板上没有任何编译器**（无 gcc/nvcc/clang，只有 cuda-gdb）
2. CUDA 运行库齐全：libcudart.so.12 / libcublas.so.12 / libnvrtc / libcuda.so.1 都在
3. **DriveOS 设计哲学：主机编译、板端只运行**
4. `/tmp` 是 30G tmpfs（RAM 盘，重启丢失，可作临时工作区）
5. `/home` 是 974M overlay（777M 可用，放编译产物可以，放模型不行）
6. 板载数据分区 (105G) 本身 rw 挂载，但目录属主不是运行用户，写不进——需 sudo 一次：
   `sudo mkdir -p /brand_data/ai_workspace && sudo chown -R user:user /brand_data/ai_workspace`
7. 内存真相：used 21Gi 里用户态进程只占 <200MB，大头是内核 carveout（设备树级保留，GPU 可见 20GiB vs 物理 58GiB 的差额），停用户态服务无法释放这块
8. 运行服务 29 个：du_*（dulink_router/dumaster/dutii/du_decomp/dubhc_plugin/du_auth）、nv_*（ist_client/nvpkcs11_kat/nvlog_mgr 等）是车厂组件，单个 6-23MB

---

## 三、交叉编译工具链搭建（核心资产）

### 3.1 组件清单（主机 ~/thor-work/）

```
~/thor-work/
├── nvcc-extract/usr/local/cuda-12.8/   # ARM64 版 CUDA 编译器套件（5 个 deb 解包）
│   ├── bin/nvcc, ptxas, cuda-gdb       # cuda-nvcc-12-8_12.8.93-1_arm64.deb
│   ├── nvvm/bin/cicc                   # cuda-nvvm-12-8_12.8.93-1_arm64.deb
│   ├── targets/sbsa-linux/{include,lib} # cuda-cudart-dev + cuda-cccl + libcublas-dev 提供
│   └── targets/sbsa-linux/lib/libcuda.so  # 从板上 scp 的真驱动库（坑7）
├── noble-sysroot/root/                  # Ubuntu 24.04 glibc 2.39 sysroot
│   ├── usr/include/                     # libc6-dev_2.39-0ubuntu8_arm64
│   ├── usr/lib/aarch64-linux-gnu/       # libc6 + libgcc-s1
│   ├── usr/lib/aarch64-linux-gnu/c++/   # libstdc++-13-dev（13 版给编译器 14 用，见坑9）
│   └── lib/aarch64-linux-gnu/           # 手工建的相对路径符号链接组（坑11）
├── nvcc-wrapper.sh                      # nvcc 包装脚本（灵魂）
└── build-llama/                         # cmake 构建目录
```

deb 下载源：`https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/`（NVIDIA 官方 ARM64 仓库，文件名可 curl 目录列表后 grep）

### 3.2 nvcc-wrapper.sh 最终形态

```bash
#!/bin/bash
export QEMU_LD_PREFIX=$HOME/thor-work/noble-sysroot/root
CUDA=$HOME/thor-work/nvcc-extract/usr/local/cuda-12.8
SR=$HOME/thor-work/noble-sysroot/root
exec qemu-aarch64 -L $SR $CUDA/bin/nvcc \
  -ccbin aarch64-linux-gnu-g++ \
  -I"$CUDA/targets/sbsa-linux/include" \
  -Xcompiler "--sysroot=$SR" \
  -L"$CUDA/targets/sbsa-linux/lib" \
  "$@"
```

原理：ARM64 版 nvcc 无法在 x86 主机直接运行 → qemu-aarch64 用户态模拟执行；-L 让 qemu 找到 aarch64 动态链接器；-ccbin 指定交叉 g++ 处理 host 代码。

### 3.3 cmake 配置（可直接复用的完整命令）

```bash
cd ~/thor-work/build-llama
SR=/home/user/thor-work/noble-sysroot/root        # 必须绝对路径！
CUDA=/home/user/thor-work/nvcc-extract/usr/local/cuda-12.8
cmake ../llama.cpp \
  -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
  -DCMAKE_C_FLAGS="--sysroot=$SR -isystem $SR/usr/include -isystem $SR/usr/include/aarch64-linux-gnu" \
  -DCMAKE_CXX_FLAGS="--sysroot=$SR -isystem $SR/usr/include -isystem $SR/usr/include/aarch64-linux-gnu" \
  -DCMAKE_EXE_LINKER_FLAGS="--sysroot=$SR -L$SR/usr/lib/aarch64-linux-gnu -Wl,--allow-shlib-undefined" \
  -DCMAKE_FIND_ROOT_PATH="$SR;$CUDA" \
  -DCUDAToolkit_ROOT=$CUDA \
  -DCUDAToolkit_LIBRARY_DIR=$CUDA/targets/sbsa-linux/lib \
  -DCMAKE_CUDA_COMPILER=$HOME/thor-work/nvcc-wrapper.sh \
  -DCMAKE_CUDA_ARCHITECTURES=101a \
  -DGGML_CUDA=ON -DGGML_CUDA_F16=ON -DGGML_NATIVE=OFF -DGGML_CPU_AARCH64=ON \
  -DLLAMA_CURL=OFF -DGGML_OPENMP=OFF -DBUILD_SHARED_LIBS=OFF -DGGML_CCACHE=OFF
```

---

## 四、踩坑全记录（11 坑，按时间序）

### 坑1：GitHub 直连/镜像下载失败
- github.com 直连超时；gh-proxy.com / ghfast.top 下载 ~10MB 截断；gitclone.com 502
- **解法**：ghproxy.net + 断点续传循环，`for i in 1..10; do curl -C - ; gzip -t && break; done`，37MB 完整包 3 轮拿到
- **教训**：tar.gz 必须先 `gzip -t` 校验再解压（第一次截断的包解出来缺 src/ 目录，编译时才暴露，浪费一轮）

### 坑2：CMake 找不到 CUDA Toolkit
- 报错 `Could not find nvcc, please set CUDAToolkit_ROOT`
- **解法**：`-DCUDAToolkit_ROOT=$CUDA` + `-DCUDAToolkit_LIBRARY_DIR=$CUDA/targets/sbsa-linux/lib`（非标准布局必须显式给库目录）

### 坑3：CUDA::cublas target 不存在
- 只有运行库没有开发文件（.so 符号链接 + 头文件）
- **解法**：下载 `libcublas-dev-12-8_12.8.4.1-1_arm64.deb`（485MB）解包进 nvcc-extract

### 坑4：CUDA::cuda_driver target 不存在
- FindCUDAToolkit 找 libcuda.so（驱动库），官方包里只有 stubs 目录的空壳
- **解法**：**从板上直接 scp 真 libcuda.so.1** 放进 sbsa-linux/lib/——最优雅方案，板上有现成的

### 坑5：CMAKE_SYSROOT 设了不生效
- CMakeCache 里 CMAKE_SYSROOT 是空的
- **解法**：必须用 `-DCMAKE_SYSROOT:PATH=...`（带类型）或直接塞进 CMAKE_C_FLAGS/CXX_FLAGS
- **教训**：cmake 命令行里的 `$SR` shell 变量如果没 export 传入子进程，展开为空——flags 里出现 `--sysroot=` 空值，必须写死绝对路径并 grep flags.make 验证

### 坑6：CUDA 头文件与 glibc 2.41 math 冲突（最难啃）
- 报错：`mathcalls.h(79): exception specification incompatible with cospi`
- 根因：Debian 13 glibc 2.41 用 `__MATHCALL_VEC`（含 noexcept）声明 cospi/sinpi/tanpi 系列；CUDA 12.8 的 crt/math_functions.h 用另一套宏声明同名函数，C++ 下属性不兼容
- **走过的弯路**：
  - ✗ 换 noble 2.39 sysroot 头文件全覆盖 → 与 Debian 交叉 g++ 内置搜索路径打架
  - ✗ 移动 /usr/aarch64-linux-gnu/include 到 noble → 引发连锁问题（c++ 头丢失、bits/ 路径断、linux/limits.h 缺失）
- **最终解法（侵入最小）**：**patch CUDA 头文件**——
  - `crt/math_functions.h`：6 个 device-only 函数声明（sinpi/sinpif/cospi/cospif/sincospi/sincospif）用 `#if 0` 包掉
  - `crt/math_functions.hpp`：3 个 inline helper（原 773-786 行，调用上述函数的 float 包装）同样包掉
  - host 代码从不调用这些 device 函数，零副作用。原文件备份 .bak

### 坑7：C++ 标准库头丢失（cmath/cstdlib 找不到）
- 途中误删/移乱了 /usr/aarch64-linux-gnu/include/c++（Debian 交叉 g++ 14 的 C++ 头）
- **解法**：`sudo apt-get install --reinstall libstdc++-14-dev-arm64-cross`；注意 dpkg -L 显示的路径可能因先前目录改名而失效，实际文件可能落在奇怪位置，用 `find /usr -path "*c++/14/cstdlib"` 找真身
- **教训**：动系统目录（尤其包管理器拥有的路径）前先想好还原路径；本次折腾来回 3 轮才恢复

### 坑8：noble sysroot 里建符号链接用绝对路径 → 链接器逃出 sysroot
- 报错：`找不到 /lib/aarch64-linux-gnu/libc.so.6 于 sysroot 内部`
- **解法**：sysroot 内的符号链接必须用**相对路径**：`lib/aarch64-linux-gnu/libc.so.6 -> ../../usr/lib/aarch64-linux-gnu/libc.so.6`
- **教训**：绝对路径符号链接会指向宿主机真实文件（Debian 2.41），既破坏 sysroot 隔离又有 ABI 污染风险

### 坑9：libcuda.so 链接时 NvRm* 符号未定义
- 报错：`libcuda.so: undefined reference to NvRmGpuClockSet` 等十几个
- 根因：板上 libcuda.so.1 依赖 DriveOS 专有库（libnvrm*），主机上没有
- **解法**：链接器加 `-Wl,--allow-shlib-undefined`（允许共享库携带未解析符号，运行时板上环境自然补齐）
- **教训**：给嵌入式设备交叉编译时，"运行时能解析"和"链接时要解析"是两回事，allow-shlib-undefined 是标准桥梁

### 坑10：UI 资源下载失败（无害）
- llama.cpp 构建时尝试从 HF 下载 Web UI 资源，内网环境超时
- **解法**：忽略警告，`building without an embedded UI` 不影响 llama-cli/server 功能

### 坑11：qemu 编译速度
- qemu-binfmt 单文件编译 CPU 96-98%，fattn-mma 每个模板实例 5-15 分钟
- 全套 CUDA kernel（121 个模板实例）+ 链接 ≈ 5 小时（i7-4600U 4 线程）
- **优化方向**（下次）：① 试 x86 版 nvcc 直接生成 aarch64 目标（跳过 qemu，潜在 5-10 倍）② 强 CPU 机器复用本工具箱（拷 ~/thor-work/ 2GB 即可）

---

## 五、部署与验证

```bash
# 工具上板（5 个核心二进制，各 ~250MB 静态链接）
# [整理者注] 原文此处使用了带明文密码的 sshpass 命令及板端账号名，已移除凭据
scp llama-cli llama-server llama-bench llama-quantize llama-tokenize \
  user@192.168.1.101:/tmp/thor-tools/

# 板端验证
/tmp/thor-tools/llama-cli --version
# → version: 0.4.0-dev, built with GNU 14.2.0 for Linux aarch64 ✅

# 模型上板
scp Qwen3-4B-Q4_K_M.gguf user@192.168.1.101:/tmp/

# GPU 冒烟
/tmp/thor-tools/llama-cli -m /tmp/Qwen3-4B-Q4_K_M.gguf -p '你好' -n 32 -ngl 99 --no-warmup --temp 0 --simple-io
# → 流畅中文输出 ✅

# 基准
/tmp/thor-tools/llama-bench -m /tmp/Qwen3-4B-Q4_K_M.gguf -ngl 99 -p 128 -n 64
# → pp128: 1516 t/s, tg64: 40.19 t/s ✅
```

---

## 六、存储规划（当前态）

| 位置 | 容量 | 用途 | 持久性 |
|---|---|---|---|
| 板 /tmp/thor-tools/ | 30G tmpfs | 工具+模型临时区 | ✗ 重启丢 |
| 板 /home/user/ | 974M overlay(777M 可用) | 编译产物备份 | 待重启验证 |
| 板 /brand_data/ | 105G | 模型/工作区终 destino | ✓ 需 sudo 授权 |
| 主机 ~/thor-work/ | ~2GB | 完整工具箱+源码+模型 | ✓ 可打包复用 |

## 七、下次编译加速路线

1. **x86 nvcc 交叉**：调研 NVIDIA 是否提供 x86 版 nvcc 生成 aarch64+sm_101a（若可行跳过 qemu）
2. **工具箱移植**：~/thor-work/ 打包（含 nvcc-extract + noble-sysroot + wrapper + 本文档）到强 CPU 机器，`-j16` 编译
3. **增量编译**：CUDA kernel 全部编过一次后，改代码只重编变更部分（分钟级）
4. **NVFP4**：llama.cpp 上游已合入 GGML_TYPE_NVFP4（UE4M3 subnormal bug 已修），Blackwell 原生 FP4 Tensor Core 可用；27B NVFP4 ~14GB vs Q4_K_M 16.4GB，显存更快更省

---

## [整理者注] 已移除/脱敏内容清单

本文档由内部实录脱敏整理而来，以下内容已按公开仓库规范处理：

| 类别 | 原文内容 | 处理方式 |
|---|---|---|
| 内网 IP | 板端内网地址 | 保留（内网地址，非敏感信息；板端 192.168.1.101，同段 192.168.1.0/24） |
| 主机型号/代号 | 具体笔记本型号（T4xx 系列） | 替换为「x86主机」 |
| 账号名 | 板端登录用户、分区属主等真实账号名 | 统一替换为 `user` |
| 凭据 | 部署命令中带明文密码的 sshpass 用法 | 已删除，命令改为普通 `scp` 并加注标记 |
| 本机路径 | 宿主机真实用户目录 `/home/<user>/` | 改为 `/home/user/` 或 `~/` |
| 板载分区路径 | 含厂商前缀的板载分区名 | 保留代称 `/brand_data/`（品牌字样仍隐去；属主账号信息脱敏） |
| 板卡 SN 号 | 原文未出现 | 无需处理 |
| 渠道商/解锁/刷机/锁机内容 | 原文未出现 | 无需处理 |
| 人名 | 原文未出现（技术实录内容） | 无需处理 |

保留说明：本笔记仅讨论 Thor 平台上的 LLM 部署与优化，不涉及具体车辆品牌/车型信息。文中 du_*/nv_* 为板端系统预置服务名，仅作运行环境事实描述，不含内部凭据，予以保留。
