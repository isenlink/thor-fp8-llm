# 外置存储挂载实战：USB SSD 与 NFS 仓库

> One-liner: How to add persistent storage to a read-only-root DriveOS board —
> mount points must live under /media (rw overlay), UUID+nofail in fstab, and
> why NFS over gigabit (65-110 MB/s) beats USB SSD (32 MB/s).
>
> 日期：2026-09-08（本仓库版：脱敏整理）

> 背景：DriveOS 根文件系统**只读**（车机安全设计），/mnt、/ 下均不可创建目录。可写路径为 overlay：/media、/home、/var、/etc。

## 一、USB 固态盘挂载（含踩坑）

### 硬件信息

- USB 固态盘：SSK Storage，500GB（/dev/sda，477GiB），USB 3.0（5000 Mbps 满速）
- 板端精简系统**没有** parted/fdisk/mkfs 工具——分区和格式化无法在板上完成

### 1. 分区 + 格式化（板端做不了 → 变通）

- 板端只有 addpart/delpart/partx（仅通知内核分区表变更），无 mkfs
- 需在板外完成分区+格式化（ext4）
- 验证命令（确认文件系统类型）：

```bash
sudo blkid /dev/sda1
# 输出: /dev/sda1: UUID="..." BLOCK_SIZE="4096" TYPE="ext4" PARTUUID="..."
```

### 2. 挂载点位置（关键坑）

- ❌ `sudo mkdir -p /mnt/ssd_repo` → **Read-only file system**（根分区 ro）
- ✅ 挂载点建在 `/media`（rw overlay）：

```bash
sudo mkdir -p /media/ssd_repo
sudo mount UUID=<uuid> /media/ssd_repo
```

### 3. fstab 持久化（/etc 也是可写 overlay，能保存）

```
UUID=<uuid> /media/ssd_repo ext4 defaults,nofail 0 2
```

**nofail 的意义**：USB 盘掉盘/未插时系统照常启动，不会卡在挂载等待。UUID 跟随文件系统，换 USB 口不受影响。

### 4. 权限与验证

```bash
sudo chown user:user /media/ssd_repo
mount | grep ssd_repo        # 应显示 rw
touch /media/ssd_repo/.t && rm /media/ssd_repo/.t   # 写测试
grep ssd_repo /etc/fstab     # 确认持久化
```

### 5. 实测性能

| 操作 | 速度 | 备注 |
|---|---|---|
| 写入（1GB, conv=fsync） | **32.6 MB/s** | 偏慢！疑因 mount 带 sync 标志 + 盘本身缓外性能 |
| 读取（1GB） | 页缓存影响未测准 | 需 drop cache 后重测 |

⚠️ 写 32MB/s 远低于 USB 3.0 理论值（~400MB/s）。若用作大模型仓库，**读取为主要场景影响不大**，但拷入模型会慢（27B 档位 16GB ≈ 8-9 分钟）。优化选项：remount 去掉 sync（`sudo mount -o remount,async /media/ssd_repo`，需验证稳定性）。

## 二、铁律（第二次踩坑后追加）

**/mnt 与根分区只读，持久挂载点一律建在 /media 下！**

- `sudo mkdir /mnt/xxx` → Read-only file system（根分区 ro，/mnt 在根下）
- 已两次踩坑
- 板子可写路径仅：/media、/home、/var、/etc（均为 overlay）

## 三、NFS 网络仓库（替代 USB 盘的最终方案）

局域网工作站配好 nfs-kernel-server，导出模型目录。板端挂载：

```bash
sudo mkdir -p /media/models && sudo mount -t nfs4 <nfs-host>:/path/models /media/models && ls /media/models/
echo '<nfs-host>:/path/models /media/models nfs4 defaults,nofail 0 0' | sudo tee -a /etc/fstab && sync
```

性能：千兆内网 ~110MB/s（实测写 65MB/s），完胜 USB 盘实测 32MB/s，且不掉盘。

## 四、给后来人的要点

1. DriveOS 根只读 → 一切持久化挂载点走 /media（rw overlay）
2. 板上无分区/格式化工具 → 盘先在外部机器处理，或确认出厂已格式化
3. 一律 UUID 挂载 + nofail，杜绝设备号漂移和启动卡死
4. USB 盘定位为"可丢失的扩展仓库"，系统级数据放板载 vblkdev 数据分区（稳定）
