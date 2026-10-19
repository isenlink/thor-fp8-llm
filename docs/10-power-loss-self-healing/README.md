# 断电自愈：让车规板每次上电自动恢复全部配置

> **效果**：脏断电（直接拔电）后重新上电，板子在 ~70 秒内自动恢复
> 网络、账号、免密 sudo、大页池、本地模型服务，**全程零人工介入**，
> 不依赖外部守护进程，不依赖网络。
>
> 实测平台：NVIDIA DRIVE Thor 域控板（Tegra264 / sm_101a / DriveOS 7.0.x），2026-10 两次真断电验证通过。

---

## 这个目录里有什么

| 文件 | 作用 |
|---|---|
| **[AI-REPRODUCTION-GUIDE.md](AI-REPRODUCTION-GUIDE.md)** | **★ 从这里开始** —— 写给 AI 的分步执行手册（A0→A7），把你的 AI 接上板子就能照着跑 |
| [power-loss-self-healing-via-vendor-hook.md](power-loss-self-healing-via-vendor-hook.md) | 原理详解：为什么这么做、每一步的坑在哪 |
| [scripts/discover_vendor_hook.sh](scripts/discover_vendor_hook.sh) | 只读探测脚本：自动找出你板子上的钩子链位置 |
| [scripts/restore_overlay_template.sh](scripts/restore_overlay_template.sh) | 恢复脚本模板（改参数区即可用） |
| [scripts/hook_launcher_template.sh](scripts/hook_launcher_template.sh) | 开机钩子入口模板 |

---

## 一句话原理

车规板的根文件系统是 **只读 base + 可写 overlay**。断电时 overlay 被格式化，
你在 `/etc` 里做的一切（账号、网络配置、systemd 单元）全部消失。

**但厂商会留一条开机脚本执行链**（供 OEM 注入逻辑），它位于只读 base，
且硬编码了一个指向**持久分区**的钩子路径：

```
usr_app_init.service（只读 base，永远在）
  └→ /etc/.../usr_app_init.sh
       └→ source /vendor/.../services.sh
            └→ if [ -e /persist/.rw_overlay_scripts/<NAME>.sh ]; then
                   /persist/.rw_overlay_scripts/<NAME>.sh     ← 把恢复脚本放这
               fi
```

把恢复脚本放进那个槽位 → **每次开机自动执行**。

---

## 三个最关键的坑

### ① 持久化脚本 ≠ 具备恢复能力

把脚本放持久分区是对的，**但调用它的钩子如果放在 `/etc`，断电后连调用点一起消失**。
脚本活着，永远没人执行。**必须验证调用链整体都在断电不丢的层。**

### ② 文件名必须用原厂硬编码的那个

```bash
if [ -e /persist/.rw_overlay_scripts/do_write_patch.sh ]; then
```

换成别的名字，`-e` 为假，**钩子静默跳过且不报错**。这是最隐蔽的失败形态。

### ③ 清网卡地址用 `flush`，绝不用 `down`

```bash
ip addr flush dev mgbe0     # ✅ 正确
ip link set mgbe0 down      # ❌ 会让驱动持续报错灌满串口，导致登不进去
```

实测：`down` 之后驱动每秒数条 `ethtool: iface not up`，串口控制台完全不可用。
（这是可稳定复现的台架实测，与驱动行为直接相关。）

---

## 快速上手（三步）

```bash
# 1. 探测你板子的钩子链（只读，不改任何东西）
sudo bash scripts/discover_vendor_hook.sh

# 2. 按输出改模板参数，装到持久分区
sudo install -m 755 restore_overlay_template.sh /persist/restore_overlay.sh
#    编辑参数区：NIC / NET_MODE / USERS / LOG

# 3. 手动跑一次，确认 RESTORE_ALL_OK
sudo bash /persist/restore_overlay.sh
```

然后照 [AI-REPRODUCTION-GUIDE.md](AI-REPRODUCTION-GUIDE.md) 的 A5/A6 部署钩子并做真断电验证。

---

## 适用边界

**适用：**
- rootfs 为只读 base + 可写 overlay 结构
- overlay 在掉电时会被格式化（表现为配置文件全丢）
- 厂商留有开机脚本钩子链，且槽位落在持久分区

**不适用：**
- 断电不丢配置的板子（没必要）
- 厂商脚本链已被清理（`vendor脚本` 不存在）→ 改用外部主机 + 串口守护方案
- 分区整体加密/签名校验，无法向持久分区写自定义脚本

**未验证：**
- 不同厂商的开机链结构差异很大，`discover_vendor_hook.sh` 是启发式探测，
  可能在你板子上找不到 —— 那就手动 `grep -r 'ExecStart' /usr/lib/systemd/system/*.service | grep '\.sh'` 顺藤摸瓜

---

## 实测数据

| 项 | 值 |
|---|---|
| 上电 → 钩子调起 | 22 秒 |
| 钩子调起 → 恢复完成 | 约 70 秒 |
| 恢复内容 | 网络(DHCP)、3 个账号、免密 sudo、大页池、26 个车机服务停用、模型服务拉起 |
| 模型加载耗时 | ~1.3 秒（4B Q4_K_M） |
| 验证轮次 | 2 次真断电，均 `RESTORE_ALL_OK` |

⚠️ **注意**：本文档只描述断电自愈本身。**不涉及任何串口控制台异常现象的因果分析** ——
那部分未验证，不在本仓库讨论范围。

---

## 授权

MIT，与仓库其他内容一致。欢迎照做、修改、反馈。
