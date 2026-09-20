# 构建配方（从 pin 到随包二进制）

## 1. 工具链

| 组件 | 版本/路径要求 |
|---|---|
| 主机 | x86_64 Linux（交叉编译） |
| 交叉编译器 | `aarch64-linux-gnu-gcc-14` / `aarch64-linux-gnu-g++-14` |
| CUDA 交叉编译 | `nvcc` **12.8.93**（走 `toolchain/nvcc-xc128.sh` 包装），CUDA sbsa 包见 `UPSTREAM-LINKS.md` |
| aarch64 sysroot | Debian/Ubuntu arm64 rootfs（`--sysroot=<root>`），用于交叉链接板端 glibc |
| 目标架构 | `CMAKE_CUDA_ARCHITECTURES=101a`（件里只有 `sm_101a` 的 cubin） |

`toolchain/toolchain-thor-cuda128.cmake` 是本包提供的工具链文件，里面的路径变量改成你自己的：

| 变量 | 含义 |
|---|---|
| `THOR_XBUILD` | 你放 sbsa/sysroot 的目录 |
| `SR` | aarch64 sysroot 根 |
| `SBSA` | CUDA sbsa 目录（`include/` + `lib/libcudart.so`） |

## 2. 构建

```bash
bash toolchain/build-thor-cuda128.sh /home/dev/llama-cpp-build build-out -j 64
# 等价于：
# cmake -DCMAKE_TOOLCHAIN_FILE=toolchain-thor-cuda128.cmake -B build-out -S /home/dev/llama-cpp-build \
#       -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=101a -DBUILD_SHARED_LIBS=OFF \
#       -DGGML_CUDA=ON -DGGML_CPU=ON -DGGML_CPU_AARCH64=ON
# cmake --build build-out -j 64
```
产物：`build-out/bin/llama-server`。

## 3. 运行期依赖（板端已自带；缺一个都起不来）

`libgomp.so.1` · `libcudart.so.12` · `libcublas.so.12` · `libcublasLt.so.12` · `libcuda.so.1` · `libstdc++.so.6` · `libm.so.6` · `libgcc_s.so.1` · `libc.so.6`

## 4. 验收（比"编译成功"更强的三条）

```bash
# ① 结构自检
bash toolchain/verify.sh /path/to/llama-server
#    期望：arch=aarch64、sm_101a cubin 存在、ldd 里没有 not found、
#    strings 里 draft-dflash 命中 > 0、NEEDED 列表与上表一致
# ② 数值自检（在板上跑，跟我们同一个 NVFP4 target 时）
./bin/b3-correctness-perj | tail -3      # 期望 cpu_failures=0
# ③ 性能口径自检（在板上跑）
#    128K 上下文、f16 KV、DFlash2 起草器：步耗应落在 235 ms 档（≈22.2 t/s）
#    只看 ms/步 = 1000 × mean_len / t/s，不要跨 mean_len 比 t/s
```

**逐字节 sha256 相同：本包做不到，原因有两条（都是事实，不是保守说法）**

1. 件是 `not stripped`，DWARF 里带**绝对路径**（源码目录、构建目录、sysroot、CUDA 目录）⇒ 这些路径必须逐字一致；
   件里嵌的是 `…/xbuild/llama-cpp-latest/...`，即源码要放在 `<你的中性前缀>/xbuild/llama-cpp-latest`。
2. 有一部分文件在构建**之后**又被改过（`patches/PATCH-FIDELITY.tsv` 里 `IN_EVIDENCE` 与 `UNKNOWN` 那些），
   构建当时的那一版我们**没有留存快照**，只能给当前版本（是超集）。其中已确认有证据的：
   - 在件里：`GGML_CUDA_GRAPH_OPT`、`GGML_MMVQ_MAX`、`GGML_FP8_FORCE_F16`、`GGML_OP_PROF`、`GGML_FP8_PROF`
   - 不在件里：`GGML_NVFP4_WIDE_MAX`、`GGML_NVFP4_WIDE_CFG`、`T4_MMQ/T4_CPS/T4_STAGES`、`LLAMA_SPEC_TIMING`、
     `GGML_FATTN_MMA_Q8KV*`、`GGML_FUSE_CONV_STATE_CPY`、`LLAMA_MTP_VOCAB_CROP`
   （判据 = 字符串级反查；`GGML_NVFP4_WIDE_MAX` 只存在于构建后的 `mmvq.cu`，所以随包件**不读它**，启动脚本里那条 export 是空操作）

所以验收请用上面三条（结构 + 数值 + 步耗），**别用 sha256 不同就断定代码不对**。
真要做到逐字节相同，需要我们来导出一份"构建时快照"（当前工作树已经回不到那个状态）。
