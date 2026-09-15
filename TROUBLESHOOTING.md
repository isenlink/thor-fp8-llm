# Troubleshooting / 踩坑速查

> **How to use this file**: 遇到问题时，直接用**报错原文**或症状关键词搜索本文件
> （`Ctrl+F` / GitHub 仓库搜索 / `grep -i "报错原文" TROUBLESHOOTING.md`）。
> 每条给出：症状 → 根因 → 解法 → 详细文档链接。
>
> 本文汇总了在 NVIDIA DRIVE Thor（p3960-0010 / Tegra264 / sm_101a / DriveOS 7.0.3）
> 上部署 LLM 过程中**实际踩过并解决**的坑，含失败实验与反直觉结论。
> 英文报错字符串**逐字保留原文**，以便搜索引擎命中。

---

## 🚀 快速症状索引

| 你遇到的现象 | 跳到 |
|---|---|
| `Read-only file system`（连 sudo mkdir 都失败） | [A1](#a1-根分区只读sudo-mkdir-也失败) |
| 断电重启后 /etc 里的改动、账号、配置**回滚了** | [A2](#a2-硬断电后-etc-改动消失) |
| 物理内存 58G，GPU 却只有 20G 可用，杀掉所有进程也放不出来 | [A3](#a3-gpu-可用显存远小于物理内存) |
| 板子上找不到 curl / dhclient / parted / fdisk / mkfs / mount.nfs | [A4](#a4-板上缺常用工具) |
| 网络断了重启 networkd 才恢复；或有线口抢了默认路由导致断网 | [A5](#a5-网络自己断了路由被抢) |
| 温度飙高 / 需要监控 GPU 但没有 nvidia-smi | [A6](#a6-温度与-gpu-监控) |
| 运行中突然断电，MCU 串口先打 `0x40B6` 再 `0x40B4` | [A8](#a8-高温断电事件链0x40b6--0x40b4真实温度阈值表) |
| 摘下 USB 设备后，同一链路上的其他设备消失 | [A7](#a7-usb-热插拔导致链路设备消失) |
| `Could not find nvcc, please set CUDAToolkit_ROOT` | [B1](#b1-cmake-找不到-cuda-toolkit) |
| `CUDA::cublas` / `CUDA::cuda_driver` target 不存在 | [B2](#b2-cuda-target-缺失cublas--cuda_driver) |
| `exception specification incompatible with cospi` | [B3](#b3-glibc-与-cuda-头文件冲突-cospi) |
| `undefined reference to NvRmGpuClockSet`（libcuda.so 一堆未定义符号） | [B4](#b4-libcudaso-链接时-nvrm-符号未定义) |
| 链接器报找不到 `/lib/aarch64-linux-gnu/libc.so.6`（明明 sysroot 里有） | [B5](#b5-sysroot-内符号链接用绝对路径导致逃逸) |
| `CMAKE_SYSROOT` 设了但 CMakeCache 里是空的 / `--sysroot=` 后面是空值 | [B6](#b6-cmake_sysroot-设了不生效) |
| 交叉编译找不到 `cmath` / `cstdlib` 等 C++ 标准头 | [B7](#b7-c-标准库头丢失) |
| GitHub 直连超时、镜像下载到 10MB 就截断 | [B8](#b8-github-下载中断国内网络) |
| qemu 交叉编译慢到离谱（全套 kernel 要 5 小时） | [B9](#b9-qemu-交叉编译太慢) |
| `building without an embedded UI` 警告 | [B10](#b10-ui-资源下载失败无害) |
| `TypeError: expected np.ndarray (got numpy.ndarray)` | [C1](#c1-numpytorch-abi-错配最坑的一个) |
| 模型转换导出 0 张量，只得到 10MB 元数据 | [C2](#c2-导出-0-张量只有元数据) |
| config 改了还是 0 张量 | [C3](#c3-非标准分片命名) |
| NVFP4 转换时 memmap 相关类型错误 | [C4](#c4-nvfp4-触发-mmap-类型错误) |
| pip 装 torch 卡在 79MB/196MB 零速；requirements 钉版本装不上 | [C5](#c5-依赖下载假死国内网络) |
| **MTP 投机解码调参后反而变慢** | [D1](#d1-mtp-深起草反而变慢负收益) |
| 实测数据前后矛盾 / A/B 对比结论不可信 | [D2](#d2-测出来的数不可信parallel-污染) |
| 长任务看到"完成"回显，但其实没完成 | [D3](#d3-监控盯错目标假完成) |
| 模型一样但速度差一倍（社区能跑 2× 我们不行） | [D4](#d4-同模型速度差一倍) |
| 输出速度在创作类任务上远低于问答类 | [D5](#d5-创作任务加速失效) |
| `sudo mkdir /mnt/xxx` 失败，挂载点建不了 | [E1](#e1-挂载点必须建在-media) |
| fstab 里写了挂载但开机没挂上（静默跳过） | [E2](#e2-fstab-条目静默失效nfs-helper-缺失) |
| USB 盘写速只有 32MB/s（理论值 ~400MB/s） | [E3](#e3-usb-盘写入慢) |

---

## A. 系统 / 环境类

### A1. 根分区只读，`sudo mkdir` 也失败

```
mkdir: cannot create directory '/mnt/xxx': Read-only file system
```

**根因**：DriveOS 根文件系统只读挂载（安全设计），`/mnt`、`/sbin`、`/usr/local` 全在只读根下。
`sudo` 也改不了——这是挂载属性，不是权限问题。

**更深一层（实测）**：根设备挂在 hypervisor 虚拟存储后端（`tegra_virt_storageNN`）上，
写保护在 **hypervisor 层**。所以连 `sudo mount -o remount,rw /`、
`blockdev --setrw`、`dd` 写裸设备、`debugfs -w` **全部失败**
（`write-protected` 或 `Operation not permitted`）。**这条路彻底堵死，别浪费时间试。**

**解法**：需要写的文件放**可写区 + 全路径引用**。

```bash
sudo mkdir -p /media/xxx     # ✅ 可写 overlay 区
```

⚠️ **本坑我们踩了两次**（`/mnt/ssd_repo`、`/mnt/ai-station_models`，后来又犯一次 `/sbin/mount.nfs`）。
**正确思路永远是"放可写区 + 全路径引用"**，不要试图 remount,rw——恒只读，不该碰。

⚠️ **但要注意**：`/etc`、`/home`、`/media`、`/var` 是 **overlay 可写层**，**脏断电后会被固件格式化**
（见 [A2](#a2-硬断电后-etc-改动消失)）。放这里只保证"本次开机期间可写"，
**不保证跨断电存活**。要扛断电请用独立非易失分区（如 `/brand_data/`）。

📄 详见 [docs/05-system-tuning/storage-mounting.md](docs/05-system-tuning/storage-mounting.md)

### A2. 硬断电后 /etc 改动消失

**症状**：修改了 `/etc` 下的配置（账号、静态 IP、sysctl），断电重启后回滚到出厂状态。

**根因（实测，不是 sync 问题）**：`/etc` 是 overlay，可写上层是一个独立分区。
**脏断电后该分区 fsck 判错，固件直接把它格式化重建**——整个可写层归零。
dmesg 铁证：

```
[overlay] partition errors,ready to format     ← 本次 boot 被格式化（配置已丢）
[overlay] partition no errors                  ← 干净复位（配置保留）
```

**关键推论**：`sync` 救不了（不是缓存问题）；**板上任何钩子也救不了**——
systemd unit / cron / rc.local 全都活在 overlay 上，格式化时一起被抹掉。

**解法**（要扛脏断电，必须两步都做）：
1. 把配置**打包放独立非易失分区**（`/brand_data/`，该分区在多次脏断电中实测存活），
   写一个 restore 脚本从包里还原；
2. 触发器放**板外常电主机**（串口探测 boot → 判定 overlay 被吞 → 远程跑 restore）。
   板内插钩子这条路已实测堵死：根分区只读保护在 **hypervisor 层**
   （`dd` / `debugfs -w` / `remount rw` 全部 `Operation not permitted`）。

按此架构实测：断电后 **约 3 分钟无人值守自动恢复**，账号/IP/大页池全部复原。

📄 全套实证（10 条死路 + 启动链解剖 + 架构设计 + 踩坑）见
[docs/05-system-tuning/overlay-power-loss-recovery.md](docs/05-system-tuning/overlay-power-loss-recovery.md)

### A3. GPU 可用显存远小于物理内存

**症状**：物理内存 58GiB，但 GPU 只能看到 20GiB；杀掉所有用户态服务也放不出来。

**根因**：那 38GiB 差额是**内核级 carveout**（设备树级保留：GPU/SATA/CAM/安全岛等），
用户态服务总共只占 <200MB。停服务毫无用处；要动 carveout 需改设备树，风险极高。

**解法（安全路线）**：通过 **hugepages 扩容 GPU 池**，不动设备树。我们做到了
20GiB → 42GiB（冷启动全量分配）。

注意运行时扩容有**碎片化上限**：在线加只能到 ~32GiB，再高必须冷启动。

📄 详见 [docs/01-hardware-recon/storage-memory-recon.md](docs/01-hardware-recon/storage-memory-recon.md)、
[docs/05-system-tuning/](docs/05-system-tuning/)

### A4. 板上缺常用工具

**症状**：`curl`、`dhclient`、`parted`、`fdisk`、`mkfs`、`sshfs`、`mount.nfs` **全部没有**。
板上也没有**任何编译器**（无 gcc/nvcc/clang，只有 cuda-gdb）。

**根因**：DriveOS 精简系统 + 设计哲学"主机编译、板端只运行"。

**解法**：交叉编译所需 aarch64 二进制放进可写区（如 `~/bin/`），用**全路径调用**
（因为 `/sbin`、`/usr/local` 只读放不进去）。

📄 详见 [docs/02-cross-compile/cross-compile-log.md](docs/02-cross-compile/cross-compile-log.md) §二

### A5. 网络自己断了／路由被抢

**症状 1**：DHCP 失败后长时间不重试，网络一直不通。
**解法**：`sudo systemctl restart systemd-networkd`（networkd 有长退避机制）。

**症状 2**：断网，`ping` 不通。
**根因**：板载 mgbe 车规网口会**抢占默认路由**。
**解法**：先查 `ip route` 确认默认路由指向哪里。

**症状 3**：路由修好了但还是解析不了域名。
**根因**：`resolv.conf` 指向板内网 DNS。
**解法**：手改为局域网网关 + 公共 DNS。

📄 详见 [docs/01-hardware-recon/hardware-archive.md](docs/01-hardware-recon/hardware-archive.md) §5

### A6. 温度与 GPU 监控

**要点**：
- 板上**没有** `nvidia-smi` / `nvtop`，用 **`tegrastats`**（`GR3D_FREQ` = GPU 占用率）
- 空载 tj 61-66°C 正常；**>75°C 暂停重负载**（我们的人工熔断线）
- 满载推理 72-74°C（被动散热，健康）

📄 详见 [docs/01-hardware-recon/hardware-archive.md](docs/01-hardware-recon/hardware-archive.md) §3

### A8. 高温断电事件链：0x40B6 → 0x40B4（真实温度阈值表）

**症状**：运行中板子突然整机断电，MCU 串口（NvShell）先打高温预警
`0x40B6`，随后打断电码 `0x40B4`，无任何 Linux 侧日志（断电发生在内核
有机会落盘之前）。

**我们的一次真实事件**（水冷失效 + OOM 满载）：

```
20:48  MCU 高温预警 0x40B6
22:26  MCU 断电 0x40B4（推算已过 114°C）
```

**真实温度阈值表**（来自 NVIDIA IGX/Thor 官方文档核定，2026-09-10）：

| 状态 | 温度 | 说明 |
|---|---|---|
| 水冷正常 idle | 60-66°C | 实测典型区间 |
| **负载 80°C** | **正常工作区间** | 不是异常，无需紧张 |
| 满载推理 80-95°C | 正常（水冷压得住的前提下） | 离降频线余量充足 |
| 109°C | 软件降频 | |
| 113°C | 硬件降频 | |
| **114.5-115°C** | **关机** | MCU Safetyservice 抢先断电 |
| Linux zone critical | 125°C | 永远轮不到它，MCU 先动手 |

**关键认知**：
- **DRIVE OS 没有渐进降频缓冲**——温度到线就是直接断电，不像桌面 GPU
  先降频挣扎很久。监控必须盯 MCU 串口侧（热点传感器 = EXT0/TMP451，
  正常 52-53°C），只看 Linux 侧 tegrastats 会漏掉预警窗口。
- 断电后 `/etc` overlay 可能被吞（见 [A2](#a2-硬断电后-etc-改动消失)），
  高温断电 = 变相脏断电。
- ⚠️ 0x40B6/0x40B4 事件链全网无公开资料（GitHub/论坛均搜不到），
  本条是独份记录。阈值数据以官方文档为准，事件链为我们实测。

**解法**：
1. 确保水冷在位且真正工作（我们的失效案例：水冷泵停转）；
2. 长任务盯 MCU 串口温度而非只盯 Linux 侧；
3. 设人工熔断线（我们用 75°C，保守但安全——注意这是**自定纪律**，
   板子真实降频线在 109°C，不要混淆）。

📄 详见 [docs/01-hardware-recon/hardware-archive.md](docs/01-hardware-recon/hardware-archive.md) §3

### A7. USB 热插拔导致链路设备消失

**症状**：摘掉一个 USB 设备后，**同一链路上的其他设备**也枚举失败，报 `error -22` 死循环。

**根因**：Tegra xUSB 控制器的热拔处理 bug。

**解法**：物理重插恢复；**重要操作后建议 `reboot` 而非热插拔**。

📄 详见 [docs/01-hardware-recon/hardware-archive.md](docs/01-hardware-recon/hardware-archive.md) §5

---

## B. 交叉编译类

> 完整工具链搭建过程、nvcc-wrapper.sh、cmake 完整命令见
> [docs/02-cross-compile/cross-compile-log.md](docs/02-cross-compile/cross-compile-log.md)

### B1. CMake 找不到 CUDA Toolkit

```
Could not find nvcc, please set CUDAToolkit_ROOT
```

**根因**：ARM64 CUDA 安装在非标准路径。

**解法**：同时显式指定**根目录**和**库目录**（缺一不可）：
```bash
-DCUDAToolkit_ROOT=$CUDA \
-DCUDAToolkit_LIBRARY_DIR=$CUDA/targets/sbsa-linux/lib
```

### B2. CUDA target 缺失（cublas / cuda_driver）

**症状**：`CUDA::cublas` 或 `CUDA::cuda_driver` target 不存在。

**根因**：
- cublas：只有运行库，没有开发文件（`.so` 符号链接 + 头文件）
- cuda_driver：官方包里 `stubs/` 只有空壳

**解法**：
- cublas：下载 `libcublas-dev-12-8_*_arm64.deb`（~485MB）解包
- cuda_driver：**从板子上直接 scp 真的 `libcuda.so.1`** —— 板上有现成的，最优雅

### B3. glibc 与 CUDA 头文件冲突（cospi）

```
mathcalls.h(79): exception specification incompatible with cospi
```

**根因**：Debian 13 glibc 2.41 用 `__MATHCALL_VEC`（含 `noexcept`）声明 `cospi/sinpi/tanpi` 系列；
CUDA 12.8 的 `crt/math_functions.h` 用另一套宏声明同名函数 → C++ 下属性不兼容。

**走过的弯路（不要重复）**：
- ✗ 换整个 sysroot 的头文件 → 与交叉 g++ 内置搜索路径打架
- ✗ 移动 `/usr/aarch64-linux-gnu/include` → 连锁问题（C++ 头丢失、`bits/` 路径断、`linux/limits.h` 缺失）

**最终解法（侵入最小）**：**patch CUDA 头**，把冲突声明包掉：
- `crt/math_functions.h`：6 个 device-only 函数声明（`sinpi/sinpif/cospi/cospif/sincospi/sincospif`）用 `#if 0` 包掉
- `crt/math_functions.hpp`：3 个 inline helper（调用上述函数的 float 包装）同样包掉
- host 代码从不调用这些 device 函数 → **零副作用**。原文件留 `.bak`

### B4. libcuda.so 链接时 NvRm 符号未定义

```
libcuda.so: undefined reference to `NvRmGpuClockSet'
...（十几个同类符号）
```

**根因**：板上 `libcuda.so.1` 依赖 DriveOS 专有库（`libnvrm*`），主机上没有。

**解法**：链接器加 `-Wl,--allow-shlib-undefined`（允许共享库携带未解析符号，运行时板上环境自然补齐）。

**教训**：给嵌入式设备交叉编译时，**"运行时能解析"和"链接时要解析"是两回事**。

### B5. sysroot 内符号链接用绝对路径导致逃逸

**症状**：链接器报找不到 `/lib/aarch64-linux-gnu/libc.so.6`，而 sysroot 里明明有。

**根因**：sysroot 内用了**绝对路径**符号链接 → 链接器跟随链接跑出 sysroot，指向宿主机真实文件。

**解法**：sysroot 内符号链接必须用**相对路径**：
```bash
lib/aarch64-linux-gnu/libc.so.6 -> ../../usr/lib/aarch64-linux-gnu/libc.so.6
```

**教训**：绝对路径符号链接既破坏 sysroot 隔离，又有 ABI 污染风险。

### B6. CMAKE_SYSROOT 设了不生效

**症状**：`CMakeCache.txt` 里 `CMAKE_SYSROOT` 是空的；或 flags 里出现 `--sysroot=`（空值）。

**根因**：两个独立问题：
1. `CMAKE_SYSROOT` 需带类型：`-DCMAKE_SYSROOT:PATH=...`
2. **cmake 命令行里的 `$SR` shell 变量如果没 export，展开为空**

**解法**：写死绝对路径，或把 sysroot 塞进 `CMAKE_C_FLAGS`/`CMAKE_CXX_FLAGS`；
**改完 `grep flags.make` 验证实际展开值**。

### B7. C++ 标准库头丢失

**症状**：找不到 `cmath` / `cstdlib` 等 C++ 标准头。

**根因**：途中移动/删乱了交叉 g++ 的 C++ 头目录。

**解法**：
```bash
sudo apt-get install --reinstall libstdc++-14-dev-arm64-cross
```
⚠️ `dpkg -L` 显示的路径可能因先前目录改名而失效，用 `find` 找真身：
```bash
find /usr -path "*c++/14/cstdlib"
```

**教训**：动系统目录（尤其包管理器拥有的路径）前先想好还原路径——我们来回折腾了 3 轮。

### B8. GitHub 下载中断（国内网络）

**症状**：`github.com` 直连超时；`gh-proxy.com` / `ghfast.top` 下载约 10MB 就截断；`gitclone.com` 502。

**解法**：`ghproxy.net` + **断点续传循环 + 完整性校验**：
```bash
for i in $(seq 1 10); do
  curl -C - -O <url>
  gzip -t <file> && break
done
```

**教训**：**tar.gz 必须先 `gzip -t` 校验再解压**。截断的包解压出来缺目录，
要等到编译时才暴露，白白浪费一轮。

### B9. qemu 交叉编译太慢

**症状**：qemu-binfmt 单文件编译 CPU 96-98%；`fattn-mma` 每个模板实例 5-15 分钟；
全套 CUDA kernel（121 个模板实例）+ 链接 ≈ **5 小时**（i7-4600U 4 线程）。

**优化方向**：
1. 试 x86 版 nvcc 直接生成 aarch64 目标（跳过 qemu，潜在 5-10 倍）
2. 工具箱移植到强 CPU 机器复用（拷 `~/thor-work/` 约 2GB 即可），`-j16` 编译
3. **增量编译**：CUDA kernel 全编过一次后，改代码只重编变更部分（分钟级）

### B10. UI 资源下载失败（无害）

```
building without an embedded UI
```

**说明**：构建时尝试从 HF 下载 Web UI 资源，内网超时。**忽略即可**，不影响
`llama-cli` / `llama-server` 功能。

---

## C. 模型转换类

> 完整转换流程见 [docs/03-model-conversion/README.md](docs/03-model-conversion/README.md) 与
> [docs/04-nvfp4-optimization/nvfp4-optimization-log.md](docs/04-nvfp4-optimization/nvfp4-optimization-log.md) 阶段 1-3

### C1. numpy/torch ABI 错配（最坑的一个）

```
TypeError: expected np.ndarray (got numpy.ndarray)
```

**症状**：`torch.from_numpy` 失败。报错**字面自我矛盾**——expected 和 got 是同一个类型名。

**根因**：torch 的 cp312 轮子是用 **numpy 2.x ABI** 编译的，运行时 numpy 必须是 2.x。
换 torch 版本没用；**降级 numpy 反而制造错误**。

**正确组合**：`numpy 2.5.3` + `torch 2.11.0+cpu`

**教训（通用）**：
- 报错文案字面矛盾（"expected X (got X)"）= **类型对象来自不同 ABI 编译**，去查编译侧与运行时的库版本配套
- pip 装完**必须用最小用例验证**，不能只看 `import` 成功：
```bash
python3 -c "import torch, numpy as np; print(torch.from_numpy(np.arange(10)))"
```

### C2. 导出 0 张量（只有元数据）

**症状**：`convert_hf_to_gguf.py` 成功退出，但产出只有约 10MB 元数据，张量为 0。

**根因**：`config.json` 的 `architectures = Qwen3_5ForConditionalGeneration`，
该名字在 llama.cpp 注册表里同时注册给了 `Qwen3VLVisionModel`（多模态视觉类）和
`Qwen3_5TextModel`。多模态分发器把它分给了 VisionModel → **纯文本权重全被跳过**。

**解法**：patch config，指向纯文本路径：
```bash
python3 -c "import json; c=json.load(open('config.json')); \
  c['architectures']=['Qwen3_5ForCausalLM']; json.dump(c,open('config.json','w'),indent=2)"
```

### C3. 非标准分片命名

**症状**：patch 了 config，**仍然是 0 张量**。

**根因**：llama.cpp `conversion/base.py` 硬编码 `prefix = "model"` 找分片，
而权重分片命名为 `layers-*.safetensors`。

**解法**：symlink 成标准命名 + 重写 index（Python 脚本 1 分钟）：
```bash
# 66 个分片 → model-XXXXX-of-00066.safetensors
# 并重写 model.safetensors.index.json 的 weight_map
```

> 原生标准命名的模型（如 `model-000XX-of-00005`）不需要这步，只需 config patch。

### C4. NVFP4 触发 mmap 类型错误

**症状**：转换 NVFP4 真量化权重时，`base.py` 里 memmap 类型错误：
```
base.py:2653  byteswap_tensor(tensor.mmap_bytes(), ...)
```

**解法**：patch 为 `np.asarray(tensor.mmap_bytes())`。
（实际上根因也是 ABI 问题，见 [C1](#c1-numpytorch-abi-错配最坑的一个)——ABI 修好后此 patch 保留无害。）

### C5. 依赖下载假死（国内网络）

| 现象 | 解法 |
|---|---|
| pytorch.org 官方源 79MB/196MB **零速停滞** | 切清华源 `pypi.tuna.tsinghua.edu.cn/simple` |
| requirements 钉死 `torch==2.11.0+cpu`，装了 2.14 后又要回 pytorch.org 下 190MB，56KB/s 假死 | **不要跟 requirements 死磕版本**，前台手动装 |
| 清华主索引**没有 `+cpu` 变体** | **SJTU 镜像直连**：`https://mirror.sjtu.edu.cn/pytorch-wheels/cpu/`（实测 10MB/s+，20 秒 190MB） |

---

## D. 推理调优类

> 完整调优过程与实验矩阵见
> [docs/04-nvfp4-optimization/nvfp4-optimization-log.md](docs/04-nvfp4-optimization/nvfp4-optimization-log.md)

### D1. MTP 深起草反而变慢（负收益）

**症状**：把 MTP 投机解码的 draft 深度从 3 提到 7，速度**从 11.42 掉到 7.24 tok/s**，
acceptance 只有 18%。

**根因**：**没有置信门控时，深起草 = 大量低置信 draft 浪费验证算力**。
起草得越深，无效草稿越多，验证开销吃掉收益。

**解法**：加 **`--spec-draft-p-min` 门控**（低置信度直接跳过深起草）。
这是深起草能生效的**前提条件**，不是可选优化。

**实测矩阵**：

| 配置 | 高可预测任务 | acceptance | 创作任务 |
|---|---|---|---|
| n-max 3（默认，无门控） | 11.42 t/s | 40.6% | 11.42 t/s |
| n-max 7（无门控） | 7.24 t/s ❌ | 18% | — |
| **n-max 12 + p-min 0.6 + fa on + parallel 1** | **26.48 t/s ✅** | **85.9%** | 11.02 t/s |
| n-max 8 + p-min 0.5 | ~21.2 t/s | 81.1% | 11.27 t/s |

**关键洞察**：`mean accepted length` 从 2.19 → **9.34**（一次验证出 9 个 token，带宽效率 ×4）。

**黄金配方（带宽受限机型）**：深起草（n-max 8-12）+ 高置信门控（p-min 0.6）+ `-fa on` + `--parallel 1`

⚠️ **跨栈抄参数会负收益**：社区"n-max 7"的成绩来自 vLLM 树形投机栈；
llama.cpp 链式投机直接抄参数是负收益。**但机型瓶颈画像一致时配方可迁移**。

### D2. 测出来的数不可信（parallel 污染）

**症状**：基线测出来偏低，A/B 对比结论全错。

**根因**：`--parallel` 不为 1 时，**基线虚低约 20%**。

**解法**：测量时**必须显式 `--parallel 1`**（并行 4 以上投机优势直接消失）。

### D3. 监控盯错目标（假"完成"）

**症状**：长任务监控里看到"完成"回显，实际并未完成。

**根因**：`journal` 日志会回显命令里的 heredoc 内容，**污染 grep 完成标记**。

**解法**：盯**硬指标**——systemd unit 状态、文件数/大小、端口状态，不要 grep 日志文本。

### D4. 同模型速度差一倍

**症状**：同一个模型，社区能跑出 2× 速度，我们不行。

**排查顺序**（先运行时，后模型）：
1. **`--spec-draft-p-min` 门控**是否设置（带宽受限机型影响巨大）
2. **`--parallel 1`** 是否显式设置
3. **`-fa on`**（flash attention）是否显式开启
4. **`-ctk/-ctv` KV 量化**是否配合显存容量
5. 最后才考虑换模型/换量化

**实例**：带宽受限 APU 类机型基线 11.5 → 调优后 23.7 tok/s（**纯调参翻倍**，没换模型）。

### D5. 创作任务加速失效

**症状**：问答类任务加速明显（26.48 t/s），创作类任务（高温采样）掉回基线。

**根因**：**物理上限**。temperature 0.8 时 acceptance 自然掉到 ~55%（MTP 头预测力的天花板），
深起草无处发力。

**结论**：这不是 bug。EXP1 配置下创作任务不掉速（11.02 ≈ 基线），等价于"免费期权"。

---

## E. 挂载 / 存储类

### E1. 挂载点必须建在 /media

见 [A1](#a1-根分区只读sudo-mkdir-也失败)。`/mnt` 在只读根下，`/media` 是 overlay 可写区。

### E2. fstab 条目静默失效（NFS helper 缺失）

**症状**：`/etc/fstab` 里挂载条目一直在，但开机**没有挂载**，也没有明显报错。

**根因链**：
```
mount.nfs helper 只在 ~/bin/（因为 /sbin 只读，放不进去）
→ 开机 fstab 挂载找不到 helper
→ nofail 选项让它"静默跳过"（不报错、不阻塞启动）
```

**❌ 错误方案（我们踩坑重犯）**：`ln -sf ... /sbin/mount.nfs` + `remount,rw`
——根分区恒只读，且**不该碰**。

**✅ 正确方案**：用 **systemd service 单元**（放可写的 `/etc/systemd/system/`），
`ExecStart` 全路径调用：

```ini
# /etc/systemd/system/models-mount.service
[Unit]
Description=Mount NFS models repository
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/home/user/bin/mount.nfs <NFS-HOST>:/path/models /media/models -o vers=4
Restart=on-failure
RestartSec=15

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now models-mount.service
```

> 说明：`oneshot` 成功后显示 `inactive` 是**正常状态**，不是失败。

### E3. USB 盘写入慢

**症状**：USB 3.0 SSD（理论 ~400MB/s）实测写入只有 **32.6 MB/s**。

**根因**：mount 带 sync 标志 + 盘本身缓外性能。

**解法 / 取舍**：
- 读取为主要场景时影响不大；但拷入大模型会慢（16GB ≈ 8-9 分钟）
- 可选：`sudo mount -o remount,async /media/ssd_repo`（需验证稳定性）
- **更好的方案：改用 NFS**。千兆内网实测写 65MB/s、读 ~110MB/s，完胜 USB，且不掉盘

**挂载规范**：一律 **UUID 挂载 + `nofail`**（杜绝设备号漂移和启动卡死）
```
UUID=<uuid> /media/ssd_repo ext4 defaults,nofail 0 2
```

---

## 📌 方法论总结（比单个坑更值钱的部分）

1. **报错字面矛盾 = ABI 问题**。"expected X (got X)" 这种自我矛盾报错，
   去查编译侧与运行时库版本配套，不要在业务代码里找。
2. **只读根分区上的任何"写"操作，正确思路永远是"放可写区 + 全路径引用"**，
   不要试图 remount rw。我们在这上面栽了三次。
3. **测速前先确认没被测量污染**：`--parallel`、KV 量化、flash-attn 开关都会显著影响基线。
4. **深起草必须配门控**（`p-min`），否则负收益。这是反直觉但可复现的结论。
5. **社区实测库 > 官方文档**：一把 53 配置的实测表，10 分钟解决我们半天扫不出的调参问题。
   同时注意**问清对方的运行栈**——树形投机和链式投机的参数不可直接迁移。
6. **长任务监控盯硬指标**（unit 状态 / 文件数 / 端口），不要 grep 日志文本。
7. **嵌入式的链接期与运行期是两回事**：`--allow-shlib-undefined` 是标准桥梁。
8. **动系统目录前先想好还原路径**（我们为此来回折腾 3 轮）。

---

## 相关文档

- [README.md](README.md) — 仓库总览与实测数据
- [docs/01-hardware-recon/](docs/01-hardware-recon/) — 板端环境摸底
- [docs/02-cross-compile/](docs/02-cross-compile/) — 交叉编译全链路（11 坑详录）
- [docs/03-model-conversion/](docs/03-model-conversion/) — FP8→GGUF 转换
- [docs/04-nvfp4-optimization/](docs/04-nvfp4-optimization/) — NVFP4 调优实录
- [docs/05-system-tuning/](docs/05-system-tuning/) — 系统调优
- [docs/06-benchmarks/](docs/06-benchmarks/) — 基准与社区对照

---

> 本文档持续补充。**新增踩坑记录时，请把英文报错原文逐字保留**——
> 别人是靠报错原文搜到这里的。
