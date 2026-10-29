# 11.0 · 纯 CUDA 13 底座（不带框架）

> 本文是 [本目录 README](README.md)（llama.cpp 视角）的**框架无关版**：只讲"让 CUDA 13 在
> DRIVE Thor（CUDA 12.8 内核驱动）上跑起来"这件事本身。搭好之后上面想跑什么框架
> （llama.cpp / vLLM / ComfyUI / TensorRT-LLM / 自研 CUDA 程序）由你决定。
>
> 方法来源致谢：[souyou233 (EnkiMa)](https://github.com/souyou233) 的公开工作
> （[Minimax-H3-on-thor-u](https://github.com/souyou233/Minimax-H3-on-thor-u)）。

## 原理（30 秒版）

Thor 的内核驱动是 CUDA 12.8 时代的（`libnvrm_*` + 旧内核 ABI）。**不改内核、不动系统
CUDA 12.8**，靠 NVIDIA 官方 forward-compat 包提供 CUDA 13 的**用户态驱动**
（`libcuda.so.1.1` + 12 个 JIT 伴侣库），进程内 `LD_LIBRARY_PATH` 隔离加载：

```
LD_LIBRARY_PATH = /你的目录/compat/compat_orin : 系统库…
                       ↑ CUDA 13 用户态驱动（排最前）   ↑ libnvrm_* 殿后
```

进程启动时 CUDA 13 runtime/框架拿到 compat 里的新版 `libcuda` → 13.x 全链可用。
**compat 顺序铁律**：compat 排最前；宿主 12.8 的 `libcuda` 抢先 = 悄悄退回 CUDA 12.8；
宿主 `libcuda` 绝不复制进 compat 目录。

## 适用性自查（动手前 1 分钟）

| 你的情况 | 能不能用 |
|---|---|
| DRIVE Thor / Orin（CUDA 12.x 驱动，想跑 13.x 用户态） | ✅ 本方法目标场景 |
| Jetson（JetPack 自带官方 CUDA 13 路线） | ❌ 不需要本方法，走 NGC/JetPack |
| 板上系统 CUDA 已是 13.x | ❌ 不需要 |
| 想升级系统驱动本身 | ❌ 本方法不改内核/系统，失效即回滚 |

硬件/架构注记：CUDA 12.9 及更早叫 `sm_101/sm_101a`；**CUDA 13.0 起同一硬件改名
`sm_110/sm_110a`**——编 CUDA 13 程序时用后者。板卡在 hypervisor 下（如 Thor01）GPU
可见数会小于物理值，以实测为准（Thor01 实测 14 SM / 46 GiB）。

## 下载底座（ModelScope，免编译直接用）

**<https://www.modelscope.cn/models/navyyang/thor-cuda132-shared>**

| 件 | 是什么 | 大小 | sha256（首 8） |
|---|---|---|---|
| `compat/compat-orin-13.2.tar.gz` | NVIDIA 官方 `cuda-compat-orin-13-2` deb 解包归档（`libcuda.so.1.1` + 12 个 JIT 伴侣库），原样未改 | 134 MB | `c1177ba6` |
| `runtime/cu132-runtime.tar` | CUDA 13.2.86 runtime + toolkit（cudart/cublas/cublasLt/nvrtc + include + nvcc） | 1.17 GB | `e64540a0` |
| `scripts/minimal_cuda.cu` | 30 秒自检探针（下文） | — | — |
| `scripts/build-thor01-cuda132.sh` 等 | 参考构建/交叉编译脚本 | — | — |

官方原件出处（想自己从源头取，不经过我们的归档）：

```
cuda-compat-orin-13-2_13.2.44290101-1_arm64.deb   82,021,804 B
sha256 3331f3ac94eecc0e3c0d1976c8fa306c2bf33dd6f771e0d61bb583cab3590037
（NVIDIA 官方 arm64 CUDA 13.2 forward-compat 包；官方仓库直链会随版本更迭，
  以 developer.nvidia.com/cuda-downloads 选 Linux→aarch64→DriveOS/Debian 系检索）
```

下载后 `sha256sum -c SHA256SUMS.txt` 再使用。许可：NVIDIA 专有软件，按
[CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/index.html) 原样二进制再分发。

## 板端部署（四个动作，≤10 分钟）

```bash
# 0) 持久层（ DriveOS 板 /zeekr_data 类持久分区；/home /etc 是 overlay 断电会丢）
BASE=/zeekr_data/cuda13-base        # 目录名随意，持久即可
mkdir -p $BASE/compat $BASE/runtime

# 1) 底座解包
tar xzf compat-orin-13.2.tar.gz -C $BASE/compat --strip-components=1   # 得 compat_orin/
tar xf cu132-runtime.tar        -C $BASE/runtime

# 2) dev 符号链接（13.x 系编译期 -lcudart 找库的关键，pip wheel 也一样需要）
ln -sf $BASE/compat/compat_orin/libcuda.so.1.1 $BASE/runtime/lib/libcuda.so
ln -sfn lib $BASE/runtime/lib64                 # 部分构建系统硬编码 lib64

# 3) 环境模板（写进你的启动脚本，进程内隔离，不污染全局）
cat > $BASE/env.sh <<'EOF'
export CUDA13_BASE=/zeekr_data/cuda13-base
export LD_LIBRARY_PATH=$CUDA13_BASE/compat/compat_orin:$LD_LIBRARY_PATH
export PATH=$CUDA13_BASE/runtime/bin:$PATH
export CUDA_CACHE_PATH=/zeekr_data/cuda13-base/cache   # JIT 缓存落持久层
EOF

# 4) 探针自检（见下）
```

> 断电自愈提醒：若板子有 overlay 吞 /etc 的行为，本方案**全部东西都在持久分区 +
> 环境变量随进程走**，没有 systemd unit、没有 /etc 依赖——脏断电后只要持久分区在，
> 重新 `source env.sh` 即完整恢复，无需恢复脚本。

## 自检探针：30 秒证明 CUDA 13 真的通了

`scripts/minimal_cuda.cu`（81 行，无任何框架依赖）：`cudaGetDeviceCount` →
`cudaGetDeviceProperties` → `cudaMalloc` → 一个 kernel 写 42 → `cudaMemcpy` 回来。
**通过判据 = 打印 `runtime=13020`（或更高 13.x）且 `kernel_value=42`，退出码 0。**
`driver` 字段显示的仍是宿主 12.8——正常，正是本方案的形态。

两条编译路线（选一）：

```bash
# 路线 A：板端 nvcc 自编译（runtime tar 自带 nvcc，板上 CPU 慢，此探针 1~2 分钟可接受）
source $BASE/env.sh
nvcc -arch=sm_110a -O2 -o /tmp/minimal_cuda $BASE/scripts/minimal_cuda.cu
/tmp/minimal_cuda

# 路线 B：x86 主机交叉编译，板上只跑（板上零编译负载）
#   x86 侧：装 CUDA 13 x86_64 toolkit 后
nvcc -arch=sm_110a -target-dir=aarch64 --compiler-bindings /path/aarch64-gcc -O2 \
     -o minimal_cuda $BASE/scripts/minimal_cuda.cu   # 详见底座 scripts/nvcc-xc132-x86.sh
#   scp 到板执行同上
```

失败速查：

| 症状 | 含义 | 处置 |
|---|---|---|
| 探针报 `runtime=12080` | compat 没抢到最前 | 查 `echo $LD_LIBRARY_PATH` 首段；查宿主 libcuda 是否被复制进 compat |
| `error 35 / CUDA driver version is insufficient` | 同上（compat 未生效） | 同上 |
| `libcudart.so: cannot open` | dev 符号链接没建 | 重做部署第 2 步 |
| `sm_110a not supported` | 用了 CUDA 12.x 的 nvcc | PATH 须指向 13.x runtime/bin |
| kernel 跑了但 `kernel_value` 错 | 架构不匹配 | 确认 `-arch=sm_110a`（13.x 命名） |

## 之后：在这上面搭框架

| 你要跑的 | 去哪篇 |
|---|---|
| llama.cpp（预编译 sm_110a 二进制） | 本目录 README（ModelScope 仓直接带 81MB 二进制） |
| vLLM 0.32（NVFP4 + MTP 生产线） | [vllm032-line/vllm032-cu13-deployment.md](vllm032-line/vllm032-cu13-deployment.md)（A0→A7 全流程） |
| ComfyUI / 图像生成（官方 wheel 路线） | [thor-minimax-h3 docs/08](https://github.com/isenlink/thor-minimax-h3/tree/main/docs/08-cuda13-forward-compat) |
| 自研 CUDA 程序 | 本文探针即最小模板（compile→run→verify 三步） |

## 已知边界

1. **T4-MMQ 快路不生效**：compat 把 Thor 报成 `cc=11.0`，llama.cpp 的 T4 守卫只认
   `10.1a` → 走通用路径（速度非最优，不是错误）。细见 README 边界 §1。
2. **上游 issue 27763**：13.2 编译的 SM110 cubin 在 >8 层 offload 时有输出垃圾的社区
   报告（13.0.88 正常、13.3 修复）。**换模型/换构建必须做真实输出验收**，不能只看
   `/health`。
3. compat 是**进程内隔离**：每次启动都要带 `LD_LIBRARY_PATH` 前缀——脚本化是唯一
   正确姿势，手工敲命令容易漏。
