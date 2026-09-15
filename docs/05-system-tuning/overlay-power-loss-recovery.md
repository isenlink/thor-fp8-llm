# 断电后 overlay 被吞：为什么板内自愈做不到，以及可行的自动恢复架构

> One-liner: On a DRIVE Thor board the root filesystem is a hypervisor-backed
> read-only virtual block device and `/etc` lives on a *formatted-on-dirty-power-loss*
> overlay — so **no on-board boot hook can survive a dirty power cut**. This note
> documents the exhaustive dead ends we hit trying to prove otherwise, the exact
> boot chain that wipes the overlay, and the external-host watchdog architecture that
> actually works (auto-recovers in ~3 minutes, unattended).
>
> 适用：DRIVE Thor（p3960-0010 / Tegra264）, DriveOS 7.0.3
> 本仓库版：脱敏整理

---

## 一、问题：为什么断电重启后配置全没了

现象很规律：**物理断电 / 脏复位后，板子能 boot 起来，但**

- 静态 IP 丢失（回落到出厂 `10.0.0.2`）
- hugepages 池回到出厂值（46G → 20G）
- 自建账号消失（`no such user`）
- `sudo` 免密失效
- 任何往 `/etc` 里塞的东西——包括 systemd unit、cron、rc.local——全部不见

根因链（后面第二节会给证据）：**`/etc` 是 overlay，上层分区脏断电后被 fsck 判错、固件直接格式化**。

于是自然想法是：*"那把恢复脚本放非易失分区，再想办法让系统每次自己执行它不就行了？"* —— 这正是我们尝试的方向，也是这篇笔记的核心。

---

## 二、板内自愈：我们试过的全部路径（**全部失败**）

先说结论：**在这块板子上，纯板内软件自愈不可行。** 以下每一条都实测过，不是推测。

| 尝试路径 | 结果 | 阻止原因 |
|---|---|---|
| 往 `/etc/systemd/system/` 放 unit | ❌ | overlay 上层，脏断电被格式化 |
| `mount -o remount,rw /` | ❌ | `write-protected` |
| `blockdev --setrw /dev/<rootdev>` | ❌ | 命令报成功，但 `/sys/block/<dev>/ro` 仍为 `1` |
| `mv` / `cp` 到 `/usr/local/lib/systemd/system` | ❌ | 跨设备 move 失败 + 只读根 |
| `cp` 覆盖 `/usr/lib/systemd/system/*` | ❌ | `Read-only file system` |
| `debugfs -w` 写根设备 | ❌ | **`Operation not permitted`** |
| `dd` 写根设备 | ❌ | **`Operation not permitted`** |
| 改 systemd generators | ❌ | 在只读根上 |
| 改 bootloader / kernel cmdline | ❌ | 无 extlinux、无 `/boot` 内容，配置不落盘 |
| 用 `/usr/lib/systemd/system/rc-local.service.d/` drop-in | ❌ | 在只读根上 |

### 为什么全部堵死：根分区是 hypervisor 虚拟块设备

```
$ lsblk -o NAME,RO,SIZE /dev/<rootdev>
NAME       RO SIZE
<rootdev>   1   4G          ← RO=1
$ mount | grep ' / '
/dev/<rootdev> on / type ext4 (ro,noatime,seclabel,norecovery)

$ udevadm info -q property -n /dev/<rootdev> | grep DEVPATH
DEVPATH=/devices/platform/tegra_virt_storage94/block/<rootdev>   ← 虚拟存储后端
```

两块证据：

1. `verity=0`（内核 cmdline）——**没有 dm-verity 完整性校验**，所以一度以为 `remount rw` 可行。
2. 但根设备挂在 `tegra_virt_storageNN`（**hypervisor 虚拟存储后端**）上，块设备的写保护在 **hypervisor 层**。所以：
   - `blockdev --setrw` 会"成功"返回（Linux 侧标志改了），
   - 但任何实际写入（`dd` / `debugfs -w` / `mount -o remount,rw`）都返回 `Operation not permitted` / `write-protected`。

**这不是 Linux 权限问题，是虚拟化层硬保护。** 板内无解。

> 对比：可写数据分区（`vblkdev12` 等）同样走 `tegra_virt_storage`，但它们的写保护标志是关的，所以能读写。写保护是**逐分区**配置的，根分区这一份被 hypervisor 锁死了。

---

## 三、启动链解剖：overlay 到底在哪一步、被谁格式化

搞清楚机制才能设计可靠方案。以下是实测（`cat /proc/cmdline` + 逐个读只读根里的厂商脚本）还原的链路：

```
内核 cmdline:  ... rw_overlay=/dev/vblkdev30:/mnt/rw_overlay rootfstype=ext4 ro ...
        │
        ▼
nv_init.service            (只读根 /usr/lib/systemd/system/)
        │  Type=oneshot, Before=sysvinit.target
        ▼
/usr/sbin/nv_init.sh
        │  "Setting up persistent partitions..."
        ▼
/usr/sbin/driveos-persistence.sh
        │  source /usr/sbin/driveos-persistence-helper.sh
        ▼
mount_private_partitions()
        ├─ format_part()               ★ 格式化入口
        │     fsck 返回码含 (4 | 8)  →  "ready to format"  →  格式化该分区
        │     fsck 返回 0             →  "[ext4] partition no errors"
        │     fsck 返回 1 | 2         →  "errors repair success"
        │
        └─ mount_overlay_filesystems()
              mount -t overlay -o lowerdir=/etc, \
                    upperdir=$PERS_MDATA_MNT_DIR/etc, \
                    workdir=$PERS_MDATA_MNT_DIR/tmp/etc  overlay  /etc
        │
        ▼
systemctl daemon-reload    ← 挂完 overlay 后立刻 reload
```

### 三个关键事实

1. **overlay 的 upperdir 是可写分区**（内核 cmdline 的 `rw_overlay=` 指定）。往 `/etc` 写东西 → 实际落在那个分区。
2. **该分区脏断电后会被 fsck 判错 → 固件格式化**。dmesg 铁证：
   ```
   [overlay] partition errors,ready to format      ← 本次 boot 被格式化了
   [overlay] partition no errors                   ← 干净复位，未格式化
   ```
   所以"固化到 /etc"永远扛不住脏断电——**不是刷盘问题，是分区被重建**。
3. **整个启动链（`nv_init.sh` / `driveos-persistence.sh` / helper / generators）都在只读根上**，改不了 → 无法在链上插钩子。

### 附带发现：为什么 `/etc/<某文件>` 有时"看起来能写"

厂商的 persistence 流程会把部分文件复制到 upperdir。这类文件在 overlay 视角下**权限可能很宽松（甚至 0777）**，于是 `echo >> /etc/...` 会"成功"——但写的是 overlay 上层副本，**断电一样丢**。别被这个现象骗了。

用设备号可以一眼分辨文件在哪一层：
```bash
stat -c '%n dev=%d inode=%i' /etc/passwd /etc/hostname
# dev 号不同 → 分属 overlay 上层 / 只读根下层
```

---

## 四、可行架构：恢复包在非易失分区 + 触发器在外部主机

既然板内插不了钩子，把**触发器移到板外**。唯一的硬要求是：

> **恢复资产必须放在脏断电后依然存活的分区上。**

实测：`/etc`、`/home`、`/media` 都在 overlay 上层（会被格式化），而**独立数据分区在多次脏断电中全部存活**。所以架构是：

```
┌─────────────────────┐        串口(115200 8N1)        ┌──────────────────────┐
│  外部守护主机        │ ◄──────────────────────────► │  DRIVE Thor 板        │
│  (常驻, 24h)        │                              │                      │
│                     │                              │  /etc = overlay      │
│  watchdog.py        │                              │    ├ lower: 只读根    │
│   ├ 按 SN 找串口     │                              │    └ upper: rw分区    │
│   ├ 增量扫描 boot    │                              │        ↑脏断电格式化  │
│   ├ 三重门槛判定     │                              │                      │
│   └ 串口登录跑恢复   │                              │  /brand_data/  ← 非易失 ✅ │
└─────────────────────┘                              │    ├ restore.sh      │
                                                     │    └ backup bundle   │
                                                     └──────────────────────┘
```

### 判定逻辑（三重门槛，防误动作）

```
串口增量扫描检测到 boot marker（login: / multi-user）
        │
        ▼  等待 90s（等网络收敛）
   ping 目标 IP？
        ├─ ✅ 通                     → 健康，不动作
        ├─ ❌ 不通 & 串口叫不醒       → 硬件/假死（非 overlay 问题）→ 只告警
        └─ ❌ 不通 & 串口活着
                └─ ping 连败 3 次    → 判定 overlay 被吞
                        └─ 串口登录 → 执行非易失分区上的 restore 脚本
                                └─ 20s 后 ping 复验 → 报成功 / 需人工
```

**为什么三重门槛**：串口"活着"判断可能因链路抖动误报；只有"串口能登录 **且** 网络确实不通（连续 3 轮）"才动手，健康板最多记一行日志。

### 为什么触发器必须在板外（而不是放板内定时任务）

板内任何钩子（systemd unit / cron / rc.local）都活在 overlay 上，**格式化时一起被抹掉**。这是本笔记开头那十条失败尝试的共同结论。触发器放在常电的外部主机上，不参与被格式化的对象，是**唯一可靠**的位置。

---

## 五、实测结果

两块同型板，各自做了一次完整验证：*清空 overlay 上层（等效格式化）→ MCU 复位 → 观察守护是否自愈。*

| 板 | 判定时刻 | 恢复完成 | 耗时 |
|---|---|---|---|
| 板 A | 14:13:41 | 14:14:55 | **2 分 58 秒** |
| 板 B | 14:27:31 | 14:28:45 | **3 分 00 秒** |

恢复后逐项验证全部通过：自建账号、46 GiB 大页池、静态 IP、NTP 时间同步、`NOPASSWD` sudo。

耗时构成：boot 后固定等 90s（网络收敛） + restore 脚本约 60-90s。

---

## 六、踩坑记录

### 1. MCU 复位端口必须对应本板

板上串口成对出现：**偶数口 = SoC Linux 控制台**，**奇数口 = MCU NvShell**。适配器重插后 `ttyACM` 编号会漂移，**必须按 CH34x 序列号找设备**：

```bash
for d in /dev/ttyACM*; do
  echo -n "$d: "; udevadm info -q property -n $d | grep ID_SERIAL_SHORT
done
```

我们踩的坑：复位脚本里写死了伙伴板的 MCU 端口，结果复位指令发到了**另一块板**上，目标板 `uptime` 纹丝不动（白等 5 分钟）。修法：把 MCU 命令器写成接收端口参数。

### 2. overlay 被吞后 `sudo` 会失效

`/etc/sudoers.d/` 空掉 → `sudo -n` 报 `a password is required`。**恢复必须走串口**：串口登录后 `sudo -i` 用密码提权（串口不依赖 overlay 里的 sudoers 配置生效路径）。

### 3. 守护只在串口有数据时才写日志文件

板子启动完成后串口安静（停在登录提示符），此时日志文件大小不增长是**正常的**，不是守护挂了。boot marker 只在启动过程中产生。

### 4. 串口被守护独占

守护常驻占用 SoC 串口。手动操作前先停服务，否则两个进程抢串口、回显错乱。

### 5. 别用 `mkfs` 模拟格式化

模拟"overlay 被吞"不需要真的格式化：`rm -rf /mnt/<rw_overlay>/{etc,home,media,tmp_ov}` 即可等效（重挂后生效）。真要格式化的命令在自动化环境里通常被安全策略拦掉。

### 6. 别用 `pkill -f <script>.py`

在 agent/自动化终端里这条命令会把**自己**的命令行也匹配上并杀掉。用 `pgrep -f "<script>[.]py"` 拿到 PID 再按 PID kill。

---

## 七、一页速查

```bash
# 认板（按 CH34x 序列号）
for d in /dev/ttyACM*; do echo -n "$d: "; udevadm info -q property -n $d | grep ID_SERIAL_SHORT; done

# 看本次 boot 的 overlay 命运
dmesg | grep -i '\[overlay\]'
#   "partition no errors"      → 干净复位
#   "partition errors,ready to format" → 被格式化，配置已丢

# 看某文件在 overlay 哪一层
stat -c '%n dev=%d' /etc/passwd /etc/hostname

# 确认根设备是虚拟存储 + 只读
mount | grep ' / ' ; cat /sys/block/$(basename $(findmnt -no SOURCE /))/ro

# 崩了之后（串口）：
#   1. 登录 → sudo -i
#   2. bash /brand_data/restore.sh
#   3. 复验: id / cat /proc/sys/vm/nr_hugepages / ip -4 addr
```

---

## 八、结论

1. **板内自愈在这块硬件上不可能**——根分区只读保护在 hypervisor 层，Linux 内部无法解除；overlay 上层脏断电被格式化，板上任何钩子都活不过一次断电。
2. **正确架构**：恢复资产放独立非易失分区（脏断电实测存活），触发器放常电外部主机（串口 + 三重门槛判定）。
3. **效果等同自愈**：实测约 3 分钟无人值守自动恢复，全部关键配置复原。
4. 若真要在板内实现，唯一路径是**改 hypervisor / 固件层**（刷机级工程），不在软件可达范围。

---

[整理者注] 本文档由工作笔记脱敏改写：板载数据分区路径统一改 `/brand_data/` 代称（DriveOS 板上该分区为独立非易失分区、板载 vblkdev、与只读根分区独立，原路径名含车辆品牌字样，为保持品牌中立统一写作 `/brand_data/`，读者在自己板卡上执行 `ls /` 即可看到真实分区名）；内网 IP、板卡序列号、账号名、主机代号均已泛化（板卡以"板 A / 板 B"代称）。**技术内容 100% 保留**：失败路径与报错原文、启动链中厂商脚本的调用关系、判定阈值（90s 收敛窗口 / ping 3 连败）、实测耗时（2分58秒 / 3分钟）均为原始实测数据。
