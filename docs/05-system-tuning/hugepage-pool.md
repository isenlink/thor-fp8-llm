# GPU 大页池（hugepages）扩容与固化

> One-liner: How we grew the GPU unified-memory pool on DRIVE Thor boards from the
> factory 20 GiB to 46 GiB, then to 52 / 54 / 56 GiB across three boards — including the
> fragmentation ceiling (and why it depends on pool occupancy, not on the target tier),
> the swap-first expansion method, the five-source persistence method, the per-tier health
> criteria, and the measured 54 GiB ceiling for resident LLM services.
>
> 适用：DRIVE Thor（p3960-0010 / Tegra264 / sm_101a），DriveOS 7.0.3
> 本仓库版：脱敏整理
>
> **v2（2026-09-27）改了什么**：
> ① 新增 2026-09-26 三块板 52 / 54 / 56 GiB 阶梯扩池与 swap 部署的全套实测；
> ② 修正 v1 的「运行时扩容上限 ~32 GiB」——碎片天花板是**池占用 + 开机新鲜度**的属性，
>    池全空闲时 live 扩到 54/56 GiB 一轮直达（v1 结论作为它当时的条件读数原样保留）；
> ③ 修正 v1 的「三层固化」→ **五源头固化**（v1 的三层做法在断电恢复后被两处旧源打回，
>    实证见 §三）。

---

## 一、为什么需要大页池

Thor 是**统一内存**架构（无独立显存），GPU 能用的内存由内核 carveout + hugepages 决定：

- 物理内存 58 GiB（58.68 GiB 实测）
- 出厂 GPU 可见池 **20 GiB**（hugepages 10240 × 2MB）
- 差额 ~38 GiB 是设备树级内核 carveout（GPU/SATA/CAM/安全岛），**用户态动不了**

要跑 27B 级模型（权重 15-23G + KV cache），20G 池不够。
**安全路线是扩 hugepages 池，不动设备树**（改设备树风险极高，不推荐）。

---

## 二、扩容历程（实测）

| 阶段 | 池大小 | 页数（2MB/页） | 操作 | 结果 |
|---|---|---|---|---|
| 出厂 | 20 GiB | 10240 | sysctl vm.nr_hugepages | 基准 |
| 运行时 +2G | 22 GiB | — | `sysctl -w`（在线生效） | ✅ |
| 运行时 +4G | 24 GiB | — | 同上 | ✅ |
| 运行时 +12G | **32 GiB** | — | 同上 | ✅ **当时的碎片化上限**（见关键结论 1 修正） |
| 冷启动 +22G | **42 GiB** | 21504 | 冷启动全量分配 | ✅ |
| 冷启动 +4G | **46 GiB** | 23552 | 同上 | ✅（长期基线） |
| live +6G（swap 后） | **52 GiB** | 26624 | 开机 30 分钟的 fresh 板，`sysctl -w` 首轮拿满 | ✅ 2026-09-26 |
| live（swap 先行） | **54 GiB** | 27648 | 一块板池被占用 → 撞天花板 plateau 后转**干净重启**；另一块板池全空闲 → **echo 一轮直达** | ✅ 2026-09-26 |
| live（swap 先行） | **56 GiB** | 28672 | 池全空闲的板一轮直达；池被占用的板 live 卡 26836 → 转重启 | ✅ 2026-09-26 |

**档位速查**：46G=23552、52G=26624、54G=27648、56G=28672（GiB × 512）。
板子 58.68 GiB 总内存，系统真实开销 ≈6-8G；**57G（29184）不建议**——系统侧 <2G，
kswapd/journald 饿死、OOM 实质化【2026-09-26 测算】。

### 关键结论 1（v2 修正）：碎片天花板是「池占用 + 开机新鲜度」的属性，不是目标档位的属性

v1 的结论是「运行时扩容只能到 ~32 GiB」——那是**池被占用 + 长 uptime** 条件下的读数。
2026-09-26 三块板的对照实测把它拆开了：

| 条件 | 实测结果 |
|---|---|
| **池全空闲**（HugePages_Free = Total）+ swap 先开 | 54G(27648) 和 56G(28672) 都是 `echo` **round 1 一轮直达**，三板斧用不上、不用重启 |
| **池被占用**（推理服务占着页） | 撞碎片天花板：一块板 54G 目标 plateau 在 27529（差 119 页 = 238MB 推不动）；另一块板 56G 目标卡 26836（6 轮爬升递减） |
| **开机新鲜度**（同一块板） | fresh 板（buddy order12 存量 ~7.5G）+4G 热扩一次成功；3.4 天 uptime 脏板 +2G 都卡（drop_caches 后仍有 6.3G free，但 buddy order≥9 连续 2MB 块存量仅 ~800MB，三轮吃光即停） |

**判据（撞顶就停手，别硬磨）**：每轮只涨几十页甚至 1-2 页、`MemAvailable` 趋零（<300MB）、
swap 缓慢增长 —— 满足这三条就转「持久化 + 干净重启」路径，让开机早期的
systemd-sysctl 在干净内存下一气呵成。

**动手前先看 `HugePages_Free` 是否等于 `Total`**，再决定要不要上三板斧：
```bash
grep -E "HugePages_Total|HugePages_Free" /proc/meminfo   # Free==Total → 可一轮直达
```

### 关键结论 2：扩池只在"池内空闲不够时"才有意义

实测 42G 池跑 131072 ctx + parallel 1：
```
推理实际占用大页: ~27.6G（模型 19.6G 权重 + KV cache）
池内空闲:         ~14.4G
```
池内还剩 14.4G 时，**扩池无收益**。只有换更大模型、更大上下文或多实例 parallel
导致池内吃满时，扩池才有意义。

### 关键结论 3：扩容的真正瓶颈在"非大页侧"

扩池是从 page cache 里挤内存：
- 42G → 46G：挤 4G，`MemAvailable` 降到 ~7.5G → **可行**
- 42G → 48G：挤 6G，`MemAvailable` ~5.5G → **偏紧**（还要留内核/SSH/监控）

**盯 `MemAvailable` 而非 `free`**：
```
free 很低（<1G）是常态——池刚性划走 + 剩下多是可回收 page cache
available 才是真实余量
```

### 关键结论 4：池**只能扩、不能缩**（2026-09-16 A/B 反证）

常规内存被池压到 ~7.5G 时容易让人想"缩池换内存"——**实测该做法会让模型直接加载失败**：

| 池设置 | MemAvailable | 模型加载结果 |
|---|---:|---|
| 2048 页（4 GB） | 51.9 GB | ❌ `llama_model_load: error loading model: unable to allocate CUDA0 buffer` |
| 10240 页（20 GB） | 33.5 GB | ❌ 加载中途进程静默消失（日志截断、无错误行 = 被 SIGKILL） |
| **23552 页（46 GB）** | 7.5 GB | ✅ **ready in 14s** |

⇒ **本平台的 CUDA（unified）分配路径依赖大页池**：池不够大时加载即失败/被杀，
**缩池不是"腾内存"的正解**。常规内存偏紧时的正确做法是：
① 减少并发（**单实例纪律**）② 用 `MemAvailable` 而非 `free` 判断余量
③ 必要时按 §三 的流程**扩**池，而不是缩。

### 关键结论 5（新增）：先开 swap 再扩池（我们的固定口径）

swap 在这条工作流里干两件事，都不靠"变出物理内存"：

1. **解锁 compaction**：无 swap 时匿名页（anon）不可迁移，内存碎片搬不动，
   大页凑不出连续 2MB 块——swap 上线后 anon 可退让，热扩天花板显著抬高。
2. **运行期护住 `MemAvailable`**：极端档位把系统侧压到 1.5-2G 时，swap 兜底。

**边界（别期待错）**：swap 只能把**可换出的 anon** 变成余量。实测 anon 峰值仅 ~1.1G
（fresh 状态 32MB）→ **4-6G swap 足够，12G 是浪费**；56G 极端档配 8G。
介质是板载 vblkdev 分区，写速实测 ~300-900 MB/s，对换页绰绰有余；
健康判据 = bench 期 `vmstat 1 3` 的 si/so ≈ 0。

**swap 自身也要固化**（缺一次干净重启就没 swap，进而扩池退化）：
```bash
# fstab 行（x-systemd.requires 保证先挂数据分区再激活；nofail 保证文件缺失不挡开机）
<数据分区>/swap/swapfile none swap sw,x-systemd.requires=<数据分区>.mount,nofail 0 0
```
- swapfile 放**持久分区**（断电不丢），建法：`dd` → `chmod 600` → `mkswap` → `swapon`
- **恢复链要三重兜底**：① 恢复快照把 `etc/fstab` 收进包里 ② 恢复脚本 ⑤ 段开头
  **幂等补写** fstab 行（`grep -q || echo >> /etc/fstab`）③ 恢复脚本同时执行
  `swapon -a` **和**按路径直挂 `swapon <路径>`（双保险，不赌 fstab 还在）
- 少任何一重的后果是实证过的：路径直挂只救当次，**丢掉的 fstab 行让下一次干净重启没 swap**

---

## 三、五源头固化方法论（防重启漂移；v1"三层"的修正版）

⚠️ **这是新手翻车重灾区**。`sysctl -w` 只改运行时，不落盘；overlay 改动不 sync 会丢。

**v1 写的是"三层"（配置 + overlay 落盘 + 恢复包），实测不够**：2026-09-26 断电恢复后
池被打回旧值，追出**两处隐藏旧源**——恢复主源是**备份树**（旧快照 `cp -a` 拉回，
改过的 live 配置被覆盖），且恢复脚本里**硬编码了旧页数**（循环里 `echo <旧值>`）。

### 五个源头（缺一即回退）

| # | 源头 | 位置 | 角色 |
|---|---|---|---|
| 1 | live 配置 | `/etc/sysctl.d/99-hugepages.conf` + `99-zz-hugepages.conf`（**两份同值**；`/etc/sysctl.conf` 不得有 `nr_hugepages` 残留） | 干净重启真源 |
| 2 | **备份树** | `<数据分区>/account_backup/overlay_restore_tree/etc/` | **断电恢复主源（cp -a 路径）** |
| 3 | 恢复包 | `<数据分区>/account_backup/overlay_restore_bundle.tar.gz` | 恢复回退源（tar 路径） |
| 4 | 恢复脚本硬编码 | `<数据分区>/restore_overlay.sh`（页池循环的目标值 + swap 激活步骤） | 断电恢复执行体 |
| 5 | 板外母本 | 管理主机上的恢复脚本原件 | 板外真源，供下发与 md5 比对 |

**终验法**：五处 grep 全等于目标值 + 重跑 systemd-sysctl 保持 + 改过脚本必 `bash -n`
+ 恢复包解开核对包内值（包时间戳要刷新——停在旧日期 = 没重打）。

### 固化步骤（以 live 生效为目标）

```bash
# 1. 写配置：两份同键文件都改（可能有多份，见 §四 配置漂移陷阱）
echo 'vm.nr_hugepages=27648' | sudo tee /etc/sysctl.d/99-hugepages.conf
echo 'vm.nr_hugepages=27648' | sudo tee /etc/sysctl.d/99-zz-hugepages.conf

# 2. 热生效（在线扩；先按 §二 判据看能不能一轮直达）
sudo sysctl -w vm.nr_hugepages=27648

# 3. 验证
grep -E "HugePages_Total|MemAvailable" /proc/meminfo

# 4. 落盘 + sync，然后走 §三 五个源头逐一对齐 + 重打恢复包

# 5. 大跳变走重启验证（干净内存下一气呵成）
sudo shutdown -r now    # 重启后再查 HugePages_Total 确认
```

---

## 四、配置漂移陷阱（我们实际踩到的）

**症状**：池大小在不同重启后"漂移"，时而是 42G 时而是别的值。

**根因**：hugepages 配置**散落在多处**，且值不一致：

| 位置 | 当时的值 | 问题 |
|---|---|---|
| `/etc/sysctl.d/99-hugepages.conf` | 21504 | 正确 |
| `/etc/sysctl.d/99-zz-hugepages.conf` | 21504 | **重复文件**（无害但脏） |
| `/etc/sysctl.conf` | **10240** | **残留旧值** |

`sysctl.d/` 优先级高于 `sysctl.conf`（本平台三块板实测：sysctl.conf 留着旧值时，
重启后生效的仍是 sysctl.d 的目标值）——但残留旧值就是定时炸弹。

**解法**：扩容时**全部同键文件统一改**，并**删掉旧值**消除漂移源：
```bash
grep -rn nr_hugepages /etc/      # 改前先找全所有位置（三处起步）
```

**v2 补充（2026-09-27 实测）**：删掉的残留**会随断电恢复回来**——恢复主源是备份树，
备份树里的 `sysctl.conf` 还留着旧值，恢复完 grep 又能命中。所以清理残留必须
**连备份树一起改**（§三 五源头的第 2 项），否则只是这一次干净而已。

---

## 五、档位健康判据（每个档位的"验收线"不一样）

58.68 GiB 板，按档位分两组判据【2026-09-26 三板实测】：

| 档位 | 系统侧（MemAvailable） | 验收线 |
|---|---|---|
| ≤ 54 GiB（27648） | ~3.7-4.0G | `MemAvailable ≥ 3G` + `vmstat si/so ≈ 0` + swap active |
| 56 GiB（28672） | **~1.6G** | **别拿 3G 卡它**——更硬的判据 = **swap 0 占用 + `vmstat si/so = 0` + 服务 active**（实测 1.61G 下三板服务全程正常） |

### ★ 56G 档 + 常驻推理服务 = 不可行（实测撞墙）

同一个硬件、同一个 56G 池，跑没跑常驻 LLM 是两种命运【2026-09-26 实测】：

| | 56G 池（28672） | 54G 池（27648） |
|---|---|---|
| 宿主普通内存 | **1.70G** | **3.96G** |
| 常驻 27B llama-server | ❌ warmup 即崩 `CUDA error: out of memory`（`cudaHostAlloc` 路径），服务死循环重启 | ✅ 15s 加载就绪、推理冒烟通过 |
| 非常驻（按需起、用完停） | ✅ 可行（两块板这么用着） | ✅ 可行 |

**机理**：CUDA 加载模型要在宿主普通内存里做 **pinned（锁页）分配，swap 帮不上、
池也帮不上**。56G 池把宿主压到 1.6-1.7G——OS 自己够用，但 27B 模型 warmup 一申请
pinned 内存就 OOM。

⇒ **常驻推理服务的安全上限 = 54 GiB（27648）**；要 56G 只能非常驻 / 降 KV。
要让原本"按需起"的板变常驻，先把池降到 54G 档（一行 echo + 三处持久化同步）。

---

## 六、容量核算（怎么定池大小）

```
池大小 ≥ 权重 + KV cache + 余量
```

以 42G 池为例：
```
HIGHEST 档权重 22.1G
+ 16K 上下文 KV ~4G
+ 余量 6G
= 32G（42G 池足够，还有富余）
```

**KV cache 量化**（长上下文省显存）：
- FP16 KV：64K ≈ 32G（42G 池装不下 64K——19.65G 权重 + 32G KV 超限）
- **q8_0 KV**：64K ≈ 16G ✅（我们最终用这个）
- 128K q8_0 ≈ 8.3G，192K ≈ 12.5G

**结论**：长上下文场景**必须用 KV 量化**（`-ctk q8_0 -ctv q8_0`），
否则池再大也装不下。

---

## 七、给后来人的要点

1. **统一内存板上，GPU 池 = hugepages**，扩池是安全路线，别动设备树
2. **碎片天花板看池占用与开机新鲜度，不看目标档位**：池全空闲 + swap 先开 → live 一轮直达；
   池被占用/长 uptime → 撞顶（每轮只涨几页）就停手转干净重启，别硬磨
3. **先开 swap 再扩池**（固定口径）；swap 是给 compaction 和 MemAvailable 兜底的，
   不是"多出 N G 物理内存"（4-6G 足够，极端档 8G）
4. **固化 = 五源头**，v1 的三层会被断电恢复打回；改残留要连备份树一起改
5. **盯 `MemAvailable` 不盯 `free`**；**验收线按档位分**（≤54G 看 ≥3G，56G 看 swap/si-so/服务活）
6. **常驻推理服务上限 54 GiB**：56G 池 + 常驻 = CUDA pinned OOM（实测），非常驻才可用 56G
7. **扩池只在池内吃满时才有意义**，先算清目标模型/上下文的真实需求
8. **长上下文必须 KV 量化**，否则池再大也装不下

---

## 相关文档

- [hardware-archive.md](../01-hardware-recon/hardware-archive.md) §3.3 — 扩容历程原始数据
- [storage-memory-recon.md](../01-hardware-recon/storage-memory-recon.md) — 58G vs 20G 显存真相
- [TROUBLESHOOTING.md](../../TROUBLESHOOTING.md) A3 — GPU 显存远小于物理内存

---

[整理者注] 本文档由内部工作笔记脱敏整理（v2，2026-09-27）：板载数据分区路径统一用
**`/brand_data/` 代称**（该分区在板上为独立非易失数据分区、板载 vblkdev，原路径名含
车辆品牌字样，为保持品牌中立统一写作 `/brand_data/`；读者在自己板卡上 `ls /`
即可看到真实分区名），fstab 示例中的 `<数据分区>` 同指。板卡序列号、内网 IP、
账号名、管理主机路径均未收录；板卡以 Thor01 / Thor03 / Thor04 泛称（仓库既有用法）。
**技术内容 100% 保留**：全部页数与档位换算（10240/21504/23552/26624/27648/28672/29184）、
plateau 读数（27529 / 26836）、宿主内存读数（1.70G / 3.96G）、验收线数值、
A/B 反证表、报错原文均为原始实测数据；文中「【实测】/【测算】」标注沿用内部记录口径。
