# Thor 车载域控 LLM 推理实验——硬件档案与测试数据汇总

> One-liner: Full hardware recon + measured benchmark archive for running LLM
> inference on an NVIDIA DRIVE Thor (p3960-0010, Tegra264, sm_101a) automotive
> domain controller, including the hugepage pool expansion from 20 GiB to 42 GiB.
>
> 整理日期：2026-09-08（本仓库版：脱敏整理）

---

## 1. 硬件平台

### 1.1 Thor 车载域控板（被测设备）

| 项 | 规格 |
|---|---|
| 型号 | NVIDIA DRIVE Thor（p3960-0010 / Tegra264） |
| 架构 | aarch64，12 核（Blackwell SM101a，CUDA Compute 10.1a） |
| GPU | 集成 Blackwell 架构 GPU（Tegra 统一内存，无独立显存） |
| 物理内存 | 58 GiB 可见（MemTotal 61524140 kB，另有 ~19G 固件 carveout 在 MemTotal 之外） |
| GPU 统一内存池 | **42 GiB**（hugepages 21504×2MB，出厂 20G → 扩容 110%） |
| 内核 | 6.1.119-rt45-prod-rt-tegra（PREEMPT_RT 实时内核） |
| 系统 | DriveOS 7.0.3（build 40740537），根分区只读 + overlay 可写层 |
| 网络 | USB 千兆网卡（DHCP）+ 车载 mgbe 多路 VLAN |
| 温控 | 空载 tj 61-66°C，满载推理 <75°C（75°C 为人工熔断线） |
| USB 串口 | 板载 CH34x 双串口（115200 8N1）：口1=主 SoC Linux 控制台，口2=MCU/AURIX nvshell。注意：默认量产固件下实测两口均静默（控制台被禁用）——**串口静默 ≠ 接线/硬件问题** |

**存储布局：**

| 设备 | 容量 | 用途 |
|---|---|---|
| 板载数据分区 | 104G | 模型仓库 + 工具（40G 已用） |
| NFS ← 局域网工作站 | 5.3T | 共享模型仓库（实测 65MB/s） |
| USB SSD（已摘除） | 477G | 曾用仓库（写入仅 32MB/s，已淘汰） |
| 根分区 | 4G | 只读 + overlay（/etc /media /home /var 可写） |

### 1.2 辅助工作站（转换/编译机）

| 项 | 规格 |
|---|---|
| CPU | Intel i5-13600KF（20 线程） |
| 内存 | 46 GiB |
| GPU | RTX 3090 24G（转换/编译时未用，推理对照用） |
| 角色 | 交叉编译机、FP8→GGUF 转换机、模型下载中转、NFS 服务器 |

---

## 2. 软件栈

| 层 | 版本/说明 |
|---|---|
| llama.cpp | 自编译，CUDA 12.8 / sm_101a（tcgen05 已点亮），aarch64 交叉编译 |
| 推理服务 | llama-server，MTP 投机解码（`--spec-type draft-mtp`） |
| 编译链 | Ubuntu noble sysroot + aarch64 gcc-14 + nvcc-wrapper |

---

## 3. 实测性能数据（2026-09-08）

### 3.1 推理速度（llama-bench / 实测）

| 模型 | 配置 | prefill (pp128) | 生成 (tg64) | 备注 |
|---|---|---|---|---|
| Qwen3.8-27B NVFP4 | COMPACT-LOW，KV FP16 | **210.19 t/s** | **10.72 t/s** | 基准 |
| 同上 + MTP | `--spec-type draft-mtp` | — | **12.77 t/s** | **+19.6%**，acceptance 38.7%（技术文本）/ 75%（短问答） |
| Qwen3-4B Q4_K_M | 对照 | — | 40.2 t/s | 小模型对照 |

**关键结论**：
- MTP 加速 +19.6%（低于预期 +30%），瓶颈在显存带宽（底层单速 ~10.7 t/s 被带宽锁死）
- 剩余优化路线：tcgen05 kernel（prefill ×2 潜力）+ dflash2 树形投机（~~llama.cpp 暂不支持，已搁置~~ → ⚠️**2026-09-16 修订：llama.cpp 已原生支持** `--spec-type draft-dflash` / `draft-dspark`，本平台实测跑通且为当前最快草稿方案，见 `04-nvfp4-optimization/dflash2-revalidation-2026-09-16.md`）
- 社区对照：FP8+MTP 34.91 t/s（不同板卡/配置，参考值）

### 3.2 GPU 计算能力验证

| 测试 | 结果 |
|---|---|
| tcgen05 BF16 matmul（自写 kernel v5） | **60.4 / 60.2 TFLOPS**（可复现） |
| 对照 B200 | 1302 TFLOPS（同 kernel，Thor 约为 B200 的 1/21.5） |

### 3.3 显存池扩容历程（hugepages）

| 阶段 | 池大小 | 操作 |
|---|---|---|
| 出厂 | 20 GiB | sysctl vm.nr_hugepages=10240 |
| 运行时 +2G | 22 GiB | sysctl -w（在线生效） |
| 运行时 +4G | 24 GiB | 同上 |
| 运行时 +12G | **32 GiB** | 碎片化前上限实测 |
| 冷启动 +22G | **42 GiB（当前）** | 21504 页，冷启动全量分配成功 |

**42G 池容量核算**：HIGHEST 档权重 22.1G + 16K 上下文 KV ~4G + 余量 6G ✓
（200-256K 上下文需 KV 量化 q8_0：128K≈8.3G、192K≈12.5G）

### 3.4 存储性能

| 路径 | 写入 | 读取 | 备注 |
|---|---|---|---|
| 板载数据分区（vblkdev） | — | — | 稳定，服务热模型 |
| USB 3.0 SSD（已淘汰） | 32.6 MB/s | — | sync 标志 + 盘本身瓶颈，摘除 |
| NFS（千兆局域网） | **65.4 MB/s** | ~110 MB/s（理论） | 首选仓库方案 |

### 3.5 网络拓扑（拓扑形态，地址已脱敏）

```
路由器（LAN，DHCP）
├── Thor 板（USB 千兆网卡，DHCP 持久化）
├── 转换工作站（NFS 服务器，5.3T 导出）
└── x86 管理机（自动化代理宿主）
```

---

## 4. 当前模型资产

| 文件 | 大小 | 位置 | 状态 |
|---|---|---|---|
| Qwen3.8-27B-NVFP4-MTP-COMPACT-LOW.gguf | 15.2G | 板载数据分区 | **服务运行中**（MTP，4096 ctx） |
| Qwen3.8-27B-NVFP4-MTP-HIGHEST.gguf | 23.2G | 板载数据分区 | 已上板待测 |
| Qwen3.8-27B-FP8（safetensors 分片） | 29G | 转换工作站 | 待转 GGUF |
| DFlash2-FP8 draft（5 层） | 2.1G | 转换工作站 | ~~搁置（llama.cpp 不支持该架构）~~ → ⚠️**2026-09-16 修订：llama.cpp 已原生支持 `draft-dflash`，本平台实测为最快草稿**（见 `04-nvfp4-optimization/dflash2-revalidation-2026-09-16.md`）；GGUF 格式 draft 头可直接用 |
| Qwen3-4B-Q4_K_M.gguf | 2.5G | 板载 | 对照用 |

---

## 5. 踩坑记录（重要！后来人必读）

1. **根分区只读**：/mnt 与根下所有路径只读，`sudo mkdir` 也失败。持久挂载点/文件一律放 /media、/home、/var、/etc（overlay 可写区）。⚠️ 已两次踩坑
2. **硬断电丢 /etc 改动**：overlay 未 sync 时断电 = 用户账号/配置回滚。改完必 `sync`，重启一律 `sudo reboot`
3. **/sbin、/usr/local 只读**：二进制放 `~/bin` 或直接全路径调用（mount.nfs 先例）
4. **Tegra xUSB 热拔故障**：摘 USB 设备可致同链路其他设备枚举死循环（error -22），物理重插恢复；重要操作后建议 reboot 而非热插拔
5. **networkd DHCP 退避**：DHCPv4 失败后长退避不重试，恢复用 `sudo systemctl restart systemd-networkd`
6. **默认路由被车载网抢占**：车载 mgbe 接口（10.x 内网段）会抢默认路由，断网时先查 `ip route`
7. **DNS 跟路由失效**：resolv.conf 指向车载内网 DNS（10.x 段），路由修正后需手动改为局域网网关 + 公共 DNS
8. **板上没有**：curl、dhclient、parted、fdisk、mkfs、sshfs、mount.nfs（需自行交叉编译 aarch64 版放入 ~/bin）
9. **llama-bench 勿加 -c 参数**；>75°C 暂停重负载
10. **GPU 监控用 tegrastats**（GR3D_FREQ=GPU 占用率），无 nvidia-smi/nvtop

---

## 6. 待办 / 后续路线

- [x] FP8 主模型 → GGUF 转换（转换环境搭建 → 已打通，见 03-model-conversion）
- [ ] HIGHEST vs COMPACT-LOW 质量盲测（HIGHEST 已上板）
- [ ] 128K-192K 上下文实测（KV q8_0 量化）
- [ ] tcgen05 kernel 集成 llama.cpp 主线（prefill ×2）
- [x] dflash2 树形投机：⚠️**2026-09-16 已可做**——llama.cpp 原生支持 `--spec-type draft-dflash`，本平台实测为当前最快草稿方案（见 `04-nvfp4-optimization/dflash2-revalidation-2026-09-16.md`）；无需自写转换器，GGUF draft 头直接可用

---

## [整理者注] 已移除内容

按脱敏规则移除，原文含以下类别信息：

- 具体内网 IP 地址（Thor 板 / 工作站 / 路由器 / 车载网段）→ 改为拓扑描述
- USB 网卡 MAC 地址
- 内网 SSH 访问方式与设备账号名 → 统一为 user/删除
- "凭证索取"内部协作说明
- 维护人员真实姓名/单位信息

技术数据（性能、hugepages、存储、温度阈值、踩坑）100% 保留，未做任何删改。
