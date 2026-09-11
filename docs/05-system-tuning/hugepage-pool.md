# GPU 大页池（hugepages）扩容与固化

> One-liner: How we grew the GPU unified-memory pool on a DRIVE Thor board from
> the factory 20 GiB to 42 GiB (and later 46 GiB) via hugepages — including the
> fragmentation ceiling, the three-layer persistence method, and the config-drift
> trap that silently changes pool size across reboots.
>
> 适用：DRIVE Thor（p3960-0010 / Tegra264 / sm_101a），DriveOS 7.0.3
> 本仓库版：脱敏整理

---

## 一、为什么需要大页池

Thor 是**统一内存**架构（无独立显存），GPU 能用的内存由内核 carveout + hugepages 决定：

- 物理内存 58 GiB
- 出厂 GPU 可见池 **20 GiB**（hugepages 10240 × 2MB）
- 差额 ~38 GiB 是设备树级内核 carveout（GPU/SATA/CAM/安全岛），**用户态动不了**

要跑 27B 级模型（权重 15-23G + KV cache），20G 池不够。
**安全路线是扩 hugepages 池，不动设备树**（改设备树风险极高，不推荐）。

---

## 二、扩容历程（实测）

| 阶段 | 池大小 | 操作 | 结果 |
|---|---|---|---|
| 出厂 | 20 GiB | sysctl vm.nr_hugepages=10240 | 基准 |
| 运行时 +2G | 22 GiB | `sysctl -w`（在线生效） | ✅ |
| 运行时 +4G | 24 GiB | 同上 | ✅ |
| 运行时 +12G | **32 GiB** | 同上 | ✅ **碎片化前上限** |
| 冷启动 +22G | **42 GiB** | 21504 页，冷启动全量分配 | ✅ |
| 冷启动 +4G | **46 GiB** | 23552 页 | ✅（后续） |

### 关键结论 1：运行时扩容有碎片化上限

在线 `sysctl -w` 加页，只能加到 **~32 GiB** 就失败——因为 2MB 大页需要**连续物理内存**，
运行一段时间后内存碎片化，凑不出连续块。

**要突破 32G 必须冷启动**：重启后内存未碎片化，一次性全量分配成功。
所以 42G/46G 都是**冷启动分配**的。

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
free 很低（<1G）是常态——42G 被大页池刚性划走 + ~11G 是可回收 page cache
available 才是真实余量
```

---

## 三、三层固化方法论（防重启漂移）

⚠️ **这是新手翻车重灾区**。`sysctl -w` 只改运行时，不落盘；overlay 改动不 sync 会丢。
**"固化"= 写配置 + 落非易失区 + sync，三者缺一不可。**

### 三层是什么

| 层 | 位置 | 作用 |
|---|---|---|
| 1. sysctl 配置 | `/etc/sysctl.d/99-hugepages.conf` | 启动时读取，决定池大小 |
| 2. overlay 落盘 | rw_overlay 非易失区 | 断电不丢（DriveOS 根只读，/etc 在 overlay） |
| 3. 恢复 bundle | 非易失备份包 | 断电丢 overlay 时可恢复 |

### 固化步骤

```bash
# 1. 写配置（注意：可能有多处，见下"配置漂移陷阱"）
echo 'vm.nr_hugepages=23552' | sudo tee /etc/sysctl.d/99-hugepages.conf

# 2. 热生效（在线试）
sudo sysctl -w vm.nr_hugepages=23552

# 3. 验证
grep HugePages_Total /proc/meminfo   # 应 = 23552
free -m                              # available 应 ≥ 7G

# 4. 落盘 + sync（关键！）
sudo sync

# 5. 重启验证（冷启动全量分配）
sudo reboot
# 重启后再查 HugePages_Total 确认
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

`sysctl.d/` 优先级高于 `sysctl.conf`，所以旧值被覆盖、暂时无害——
但**启动顺序不同可能导致池大小漂移**。

**解法**：扩容时**三处统一改**，并**删掉旧值**消除漂移源：
```bash
# 统一为 23552
echo 'vm.nr_hugepages=23552' | sudo tee /etc/sysctl.d/99-hugepages.conf
echo 'vm.nr_hugepages=23552' | sudo tee /etc/sysctl.d/99-zz-hugepages.conf
# 删除 sysctl.conf 里的旧值 10240（消除漂移源）
sudo sed -i '/vm.nr_hugepages/d' /etc/sysctl.conf
sudo sync
```

**教训**：改系统级配置前，先 `grep -r nr_hugepages /etc/` 找全所有位置，
确认没有互相打架的旧值。

---

## 五、容量核算（怎么定池大小）

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

## 六、给后来人的要点

1. **统一内存板上，GPU 池 = hugepages**，扩池是安全路线，别动设备树
2. **运行时扩容有碎片化上限（~32G）**，突破必须冷启动
3. **固化 = 写配置 + 落非易失区 + sync**，`sysctl -w` 不算固化
4. **配置漂移陷阱**：hugepages 值散落多处，改前 `grep -r` 找全，删旧值
5. **盯 `MemAvailable` 不盯 `free`**（大页池刚性划走后 free 恒低）
6. **扩池只在池内吃满时才有意义**，先算清目标模型/上下文的真实需求
7. **长上下文必须 KV 量化**，否则池再大也装不下

---

## 相关文档

- [hardware-archive.md](../01-hardware-recon/hardware-archive.md) §3.3 — 扩容历程原始数据
- [storage-memory-recon.md](../01-hardware-recon/storage-memory-recon.md) — 58G vs 20G 显存真相
- [TROUBLESHOOTING.md](../../TROUBLESHOOTING.md) A3 — GPU 显存远小于物理内存
