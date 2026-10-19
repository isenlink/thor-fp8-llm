# 交给 AI 执行的复现手册 / AI-Executable Reproduction Guide

> **这份文档是写给 AI 看的，不是写给人看的。** 如果你的 AI 助手能执行 shell 命令，
> 把这份文档整份交给它，它可以按 A0→A7 的顺序在你的板子上完成全部工作。

---

## 前置条件（人工先确认这些）

| 项 | 判据 |
|---|---|
| 板子能串口登录 | 有 root 或免密 sudo 通道 |
| 板子有**持久分区** | 断电后内容不丢的分区（`lsblk` + 实测） |
| 板子断电后**会丢失** `/etc` 的改动 | 不丢就没必要做这套（确认方法见 A1） |

**如果板子断电不丢配置，本方案没有意义，停止。**

---

## A0 · 先做基线记录（人工做，AI 不要跳过）

断电前必须记录现状，否则恢复后无法判断是否真的恢复成功。

```bash
# AI 执行这段，把输出保存到本地
{
  echo "=== date ===";        date
  echo "=== boot ===";        uptime -s
  echo "=== cmdline ===";     cat /proc/cmdline
  echo "=== net ===";         ip -br addr
  echo "=== route ===";       ip route
  echo "=== interfaces ===";  ls /sys/class/net
  echo "=== users ===";       getent passwd | awk -F: '$3>=1000 && $3<65000 {print $1" "$3" "$6" "$7}'
  echo "=== sudoers ===";     ls -l /etc/sudoers.d/
  echo "=== hugepages ===";   cat /proc/sys/vm/nr_hugepages
  echo "=== network cfgs ==="; ls -l /etc/systemd/network/
  echo "=== services ===";    systemctl list-units --type=service --state=running --no-pager
  echo "=== disks ===";       lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINT
} > /tmp/baseline_before.txt
```

**把 `/tmp/baseline_before.txt` 拷到板子之外保存**（`/tmp` 会丢）。

---

## A1 · 确认哪些分区是持久的

```bash
sudo bash scripts/discover_vendor_hook.sh
```

**脚本会输出：**
- `rw_overlay=` 指向的分区 → **这个断电会被格式化，绝不能放脚本**
- 其它已挂载可写分区 → 持久层候选
- 开机脚本入口 unit + vendor 脚本 + 钩子落点

**AI 需要判断并记录：**
```
PERSIST_DIR = ?          # 持久分区挂载点，例：/persist
OVERLAY_DEV = ?          # overlay 分区，例：/dev/vblkdev30
HOOK_PATH   = ?          # 原厂硬编码的钩子完整路径
HOOK_NAME   = ?          # 原厂硬编码的文件名（一字不能改）
VENDOR_CHAIN_ALIVE = ?   # true/false —— vendor 脚本是否存在
```

**如果 `VENDOR_CHAIN_ALIVE = false`：** 本方案不适用，转外部守护方案（见主文档第 8 节）。

### 验证持久分区是否真的持久（可选但推荐）

```bash
# 在持久分区写个探针
sudo touch $PERSIST_DIR/.persist_probe

# 让用户断电重启，回来后检查
ls -la $PERSIST_DIR/.persist_probe
#   存在 → 确认持久
#   不存在 → 这个分区也不是持久的，换个分区
```

---

## A2 · 确认钩子链是活的（关键一步，别跳过）

```bash
# 1. 确认 vendor 脚本存在
ls -l <A1 找到的 vendor 脚本路径>

# 2. 看它里面怎么调钩子
grep -nE 'rw_overlay_scripts|do_write|\.sh' <vendor 脚本> | head -20

# 3. AI 要提取出确切的判断语句，形如：
#      if [ -e /<persist>/.rw_overlay_scripts/<NAME>.sh ]; then
#    ↓ 记录 NAME（含 .sh），后面一字不改用
```

**自检问题（AI 必须回答）：**
- [ ] 我找到的路径，`-e` 判断的对象**完全一致**吗？（大小写、连字符、扩展名）
- [ ] 这个路径在**持久分区**上吗？（不是 `/etc`、不是 `/tmp`、不是 overlay）
- [ ] vendor 脚本本身在**只读 base** 吗？（`ls -l` 看它在哪个挂载点下）

---

## A3 · 部署恢复脚本

```bash
# 1. 把模板复制到板子（从你下载仓库的位置）
scp scripts/restore_overlay_template.sh user@<board>:/tmp/

# 2. 板子上编辑参数区（AI 按 A0/A1 的实测值填）
#    必须改的：NIC / NET_MODE / USERS / LOG
#    按需改的：STATIC_IP / HUGEPAGES / DISABLE_SERVICES / LLAMA_DIR

# 3. 安装到持久分区
sudo install -m 755 -o root -g root \
     /tmp/restore_overlay_template.sh \
     $PERSIST_DIR/restore_overlay.sh

# 4. 语法检查
bash -n $PERSIST_DIR/restore_overlay.sh
```

**AI 必须确认：**
- [ ] `LOG=` 指向持久分区（不是 `/var/log`、不是 `/tmp`）
- [ ] `NIC=` 是 A0 里 `ip -br addr` 看到的真实网卡名
- [ ] `USERS=` 里的账号在 A0 的 `getent passwd` 里确实存在过（或确实希望创建）

---

## A4 · 手动跑一次恢复脚本（断电前必须做）

```bash
sudo bash $PERSIST_DIR/restore_overlay.sh
echo "exit=$?"
```

**判据：**
| 结果 | 含义 | 处理 |
|---|---|---|
| 输出 `RESTORE_ALL_OK` | 全部成功 | 继续 A5 |
| 输出 `RESTORE_PARTIAL_FAILED:[...]` | 部分失败 | **修好再继续**，别带着已知失败去断电 |
| 中途卡住 | 某步阻塞 | 查日志定位 |

**AI 必须检查日志里这些项：**
```bash
grep -E '网络 OK|hugepages|账|免密|llama|RESTORE_' $PERSIST_DIR/restore_overlay_last.log
```

---

## A5 · 部署钩子启动器

```bash
# 1. 编辑 launcher 的检查参数（与 restore 保持一致）
#    CHECK_USER / CHECK_NIC / EXPECT_IP_PREFIX

# 2. 安装到 A2 找到的确切路径
sudo install -m 755 -o root -g root \
     /tmp/hook_launcher_template.sh \
     "$HOOK_PATH"      # ← 必须用 A2 提取的原厂路径，一字不改

# 3. 确认
ls -l "$HOOK_PATH"
bash -n "$HOOK_PATH"
```

**⚠️ 最常见的失败**：文件名改了。改了之后 `-e` 判断为假，
**钩子静默跳过，不报任何错**，你会以为部署成功但实际没跑。

**自检：**
```bash
# 用 vendor 脚本里的判断语句原文验证
grep -o '/[^ ]*rw_overlay_scripts/[^ ]*' <vendor脚本> | head -1
# 拿它和你部署的路径逐字符对比
```

---

## A6 · 真断电测试（唯一有效的验证）

```
1. 人工：拔电
2. 人工：上电，等 5-10 分钟
3. AI：验收
```

### AI 验收清单

```bash
# ① 钩子是否真的跑了（★ 时间戳必须晚于本轮 boot）
echo "本轮 boot: $(uptime -s)"
grep -E 'hook start|hook end' $PERSIST_DIR/rw_hook_last.log | tail -2

# 判据：hook start 的时间 >> 本轮 boot 时间，且相差几十秒
# 若 hook start 早于 boot → 读到的是上一轮的旧日志，判据不成立

# ② 恢复是否完成
grep -E 'RESTORE_ALL_OK|RESTORE_PARTIAL' $PERSIST_DIR/restore_overlay_last.log | tail -1

# ③ 网络
ip -4 -br addr show $NIC
ping -c 3 <网关>

# ④ 账号与免密
id $CHECK_USER
sudo -n -u $CHECK_USER sudo -n true && echo "免密 OK"

# ⑤ 模型服务（如果配了）
systemctl is-active $LLAMA_UNIT
ss -lnt | grep $LLAMA_PORT
curl -s -X POST http://127.0.0.1:$LLAMA_PORT/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":16}' | head -c 300
```

### 判定

| 现象 | 结论 |
|---|---|
| `hook start` 时间晚于 boot，且有 `RESTORE_ALL_OK` | ✅ 成功 |
| `hook start` 时间早于 boot | ❌ 读的是旧日志，钩子**没跑** |
| 无 `hook start` 记录 | ❌ 钩子未被调起 → 查 A2/A5 的路径与文件名 |
| 有 `hook start` 但无 `restore end` | ❌ 恢复脚本卡住 → 查日志最后一步 |

---

## A7 · 失败排查表

| 症状 | 最可能原因 | 查法 |
|---|---|---|
| 钩子完全没跑 | 路径/文件名与原厂硬编码不符 | `grep -o '/[^ ]*rw_overlay_scripts/[^ ]*' <vendor脚本>` 逐字符对比 |
| 钩子没跑 | vendor 脚本不存在（链断了） | `ls -l <vendor脚本>` |
| 钩子跑了但网络没恢复 | 出厂静态 IP 配置没被改到 | `grep -H '^Address=' /etc/systemd/network/*.network` |
| 网络恢复后断连 | `ip link set down` 把链路关了 | 检查脚本里没有 `down`，只有 `flush` |
| 恢复脚本"成功"但登不进去 | sudoers 语法坏了 | `visudo -cf /etc/sudoers.d/*` |
| 模型服务起不来 | 单元文件没装回 / cgroup 被清理 | `systemctl status`；确认是 systemd 托管不是 nohup |
| 日志是空的 | `exec >>` 路径不在持久分区 | 检查 launcher 第一行 |
| 串口被输出淹没 | launcher 没做重定向 | 确认第一行是 `exec >> ...` |

---

## AI 执行时的纪律

1. **A0 基线不能跳过** —— 没有基线就无法判断恢复是否成功
2. **A2 的路径必须逐字符核对** —— 这是最高频的失败点，且失败时**不报错**
3. **A4 手动跑一次再断电** —— 带着已知错误去断电，会浪费一整轮
4. **A6 必须真断电** —— 模拟测试（删文件+手动跑）验证不了"钩子会不会被调起"
5. **判据看时间戳** —— 读到旧日志是最容易的假绿
6. **不要改原厂的任何文件** —— 只在它指定的槽位放东西
7. **失败时如实报告** —— 不要用"应该是""大概"掩盖未验证的状态
