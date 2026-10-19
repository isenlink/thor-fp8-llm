# 断电自愈：利用原厂钩子链实现零人工介入恢复

> **一句话**：车规域控板断电会丢失 overlay 层全部配置（账号、网络、服务单元）。
> 本文记录如何利用**板内原厂脚本链**（不依赖外部守护、不依赖网络），让板子在每次上电后
> 7 分钟内自动恢复全部配置与本地模型服务。
>
> 实测环境：DRIVE Thor 域控板（Tegra264 / sm_101a / DriveOS 7.0.x），2026-10 实测。

---

## 1. 问题：断电丢失的是什么

车规板的根文件系统是 **只读 base + 可写 overlay** 结构：

```
/            ← overlay 合并视图（可写层在掉电时被格式化）
   ↑
可写层 upper  ← 账号、/etc/systemd、/etc/network、sudoers …… 全在这里
   ↑
只读 base    ← 系统本体，断电不丢
```

`/proc/cmdline` 里有 `rw_overlay=/dev/vblkdev30` 这类参数，指明可写层所在分区。

**脏断电（直接拔电）后，烧录层在下次启动时无条件格式化可写层。** 后果：

| 丢失项 | 现象 |
|---|---|
| `/etc/sudoers.d/*` | 免密 sudo 失效 |
| `/etc/systemd/network/*` | 网卡回到出厂的静态地址，DHCP 不生效 |
| `/etc/systemd/system/*.service` | 自定义服务单元消失，`enabled` 状态一并丢失 |
| `/home/*` | 用户家目录、SSH 公钥消失 |
| `/etc/passwd` 增补 | 自建账号消失 |

**关键点**：这些都在 overlay 里，所以**恢复脚本不能放在 `/etc`**。

---

## 2. 为什么持久层放脚本还不够

第一反应是把恢复脚本放进持久分区（如 `/persist/`、`/data/`），它确实断电不丢。
**但脚本必须有人调用它。**

我们踩过的坑：把调用逻辑写进 `/etc/systemd/system/xxx.service` 或
`/etc/systemd/scripts/xxx.sh` —— **这些都在 overlay 层，断电后连调用点一起消失**。
脚本本体活着，却永远没人执行。**"持久化"只做对了一半。**

判据升级：
> 恢复能力存在 ≠ 恢复脚本存在。
> **必须同时验证"调用它的钩子"也在断电不丢的层。**

---

## 3. 解法：借用原厂留给厂商的钩子槽位

车规板厂商会留一条**开机脚本执行链**，供 OEM 注入自己的启动逻辑。典型形态：

```
usr_app_init.service          ← 只读 base 里的单元，永远存在
  └→ /etc/systemd/scripts/usr_app_init.sh
       └→ source /厂商路径/platform/scripts/<服务脚本>.sh
            └→ exec_do_write_patch()
                 └→ if [ -e /持久分区/.rw_overlay_scripts/do_write_patch.sh ]; then
                        /持久分区/.rw_overlay_scripts/do_write_patch.sh    ← ★ 落点
                    fi
```

**三个关键性质**：

1. `usr_app_init.service` 在**只读 base**，断电不会消失，每次开机必跑
2. 它 source 的厂商脚本也在只读 base
3. 厂商脚本**硬编码约定**了一个落在**持久分区**的钩子文件名

⇒ 把我们的恢复脚本放进那个路径，**每次开机自动执行**，无需外部守护、无需网络。

### 为什么必须用原厂指定的文件名

调用点是**硬编码**的：

```bash
if [ -e /persist/.rw_overlay_scripts/do_write_patch.sh ]; then
```

换成任何其它名字，`-e` 判断为假，钩子静默跳过、**不报错**。
这是最难排查的失败形态：看起来一切正常，实际什么都没跑。

> **纪律**：动笔前先去原厂脚本里 `grep` 出被引用的确切路径和文件名，
> 一字不改照用。不要凭命名偏好改名。

### 不同板子这条链可能断

同一批板子里，`/厂商路径/platform/` 可能被出厂清理过：

| 板状态 | `<vendor>_services.sh` 是否存在 | 钩子链 |
|---|---|---|
| 厂商脚本完整保留 | ✅ 存在 | **活的**，放入钩子即生效 |
| 被清理过 | ❌ 不存在 | **第一环就断**，此路不通 |

判断方法：`ls -l /厂商路径/platform/scripts/` 看目标脚本在不在。
不存在则改用外部守护方案（见 §8）。

---

## 4. 恢复脚本的七个步骤

恢复脚本自身放在持久分区（例：`/persist/restore_overlay.sh`），由钩子调用，以 root 执行。
建议固定顺序，**前一步失败不应中断后续**（各自独立、幂等）：

| 步 | 内容 | 要点 |
|---|---|---|
| ⓪ | 免密 sudo | 最先做。否则后续所有需要提权的操作都会卡在密码提示 |
| ① | 固件/驱动路径 | 网卡固件可能不在 base 里，需从持久分区复制并设置 `firmware_class.path` |
| ② | 网络 | 见 §5，**这是最容易失败的一步** |
| ③ | 时钟 | NTP 同步；时间错会让证书、日志、TLS 全部异常 |
| ④ | 账号 | **缺才建，幂等**。不能假设"bundle/备份一定在"，必须能凭空 bootstrap |
| ⑤ | 大页池 | 持久化到 `sysctl.d` + 运行时同时写 |
| ⑥ | 收尾 | 停用车机业务服务、清理残留内网 IP |
| ⑦ | 各机自己的环境 | **本地模型服务在这一步拉起**（见 §6） |

### ④ 账号必须能凭空建

常见错误：只从备份包还原账号。
**备份包缺失/损坏时，后面的家目录 chown 和 sudoers 免密全部落空**，
形成"恢复跑完了但登不进去"的静默失败。

正确做法：

```bash
for u in USER_A USER_B; do
  if ! id "$u" >/dev/null 2>&1; then
    useradd -m -s /bin/bash "$u"
    usermod -aG sudo "$u"
  fi
  # sudoers 免密：存在就跳过（幂等）
  [ -f "/etc/sudoers.d/$u" ] || {
    echo "$u ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/$u"
    chmod 440 "/etc/sudoers.d/$u"
    visudo -cf "/etc/sudoers.d/$u"     # 语法校验，失败立刻删，防废掉整个 sudo
  }
done
```

### ⚠️ 最隐蔽的坑：密码文件不存在时静默跳过

很多人（包括我们第一版）会把设密码写成这样：

```bash
if [ -f /persist/account_backup/passwords.txt ]; then
    while IFS=: read -r u p; do
        echo "$u:$p" | chpasswd && echo "[4] 密码已设: $u"
    done < /persist/account_backup/passwords.txt
fi
```

**逻辑完全正确，但它有一个致命特性：文件不存在时，整段直接跳过，不报错、不告警。**

表现是：

- 恢复脚本输出 `RESTORE_ALL_OK`，一切看起来正常
- 账号建出来了、sudo 免密也配了、能 `sudo -n` 免密提权
- **但账号没有密码 → SSH 密码登录失败，只能靠密钥或串口进去**

而且因为 `sudo -n` 能过，很容易误判成"账号恢复成功"。

#### 必须做的三件事

**1. 凭据文件必须提前建好，且放持久分区**

```bash
# 格式：每行 用户名:密码
sudo mkdir -p /persist/account_backup
sudo chmod 700 /persist/account_backup
# 写入时用 stdin，别让密码进命令行（会留在 shell history / ps 输出里）
printf '%s\n' "$PAYLOAD" | sudo tee /persist/account_backup/passwords.txt >/dev/null
sudo chmod 600 /persist/account_backup/passwords.txt
```

**2. 文件缺失时报警，而不是静默跳过**

```bash
PWFILE="/persist/account_backup/passwords.txt"
if [ -f "$PWFILE" ]; then
    while IFS=: read -r u p; do
        [ -z "$u" ] && continue
        printf '%s:%s\n' "$u" "$p" | chpasswd 2>/dev/null \
            && echo "[4] 密码已设: $u" \
            || echo "[4] ✗ 密码设置失败: $u"
    done < "$PWFILE"
else
    echo "[4] ⚠️ 凭据文件缺失: $PWFILE —— 账号已建但无密码，SSH 密码登录会失败"
fi
```

**3. 验收时必须真的测一次密码登录**

不要只看 `sudo -n whoami` 通了就认为账号好了 —— 那验证的是 sudoers，不是密码。

```bash
# 用密码真的登一次（在板外执行）
sshpass -p "$PASS" ssh "$USER@$BOARD" 'whoami'
#   返回用户名 → 密码 OK
#   Permission denied → 密码没设上
```

或者板上直接查 shadow 字段：

```bash
# 空 / ! / * 都代表无可用密码
sudo awk -F: '$1=="USER_A"||$1=="USER_B"{print $1"="(($2==""||$2=="!")?"NOPW":"OK")}' /etc/shadow
# 期望输出：USER_A=OK  USER_B=OK
```

#### 我们的实测记录

| 轮次 | 密码状态 | 说明 |
|---|---|---|
| 初始 | `USER_A`/`USER_B` 均 `!`（锁定） | 板上预置账号，出厂无密码 |
| 修后 | 两账号 `OK` + `sudo -n → root` | 设密码 + 已有免密 sudo |
| 模拟验证 | 先 `usermod -p '!'` 锁死，再跑恢复 | 脚本把密码设回来了 → 逻辑正确 |
| **真断电** | 钩子报 `[sudoers缺失][非DHCP][无IP]` | overlay 真被吞 |
| **真断电后** | `USER_A=OK  USER_B=OK` | **凭据文件生效，密码自动恢复成功** |

**关键教训**：第一版恢复脚本里这段代码一直都在，但 `/persist/account_backup/passwords.txt`
从未被创建。所以"密码恢复"这个功能**写了但从未生效**，直到发现并补上文件才真正工作。
如果不是专门去验证 shadow 字段，这个漏洞会一直潜伏 —— 因为所有表面指标都是绿的。

---

## 5. 网络：最容易失败的一步

### 两个独立问题

**问题 A：出厂静态 IP 覆盖了 DHCP**

`/etc/systemd/network/99-vendor_default.network` 可能写死了：

```ini
[Match]
Name=enx*
[Network]
Address=10.0.0.2/24        ← 出厂调试用静态地址
```

**恢复脚本必须把它改成 `DHCP=ipv4`**，否则网卡永远拿到出厂地址。

**问题 B：车载网卡残留 IP 干扰路由**

车载以太网卡（`mgbe*` 之类）可能带着大量内网地址和 VLAN 接口，
产生多个默认路由，导致出网流量走错口。

**处置：只 `flush` 地址，不要 `ip link set down`。**

> ⚠️ **实测坑**：`ip link set mgbe* down` 会触发驱动持续输出
> `ether_get_ethtool_stats: iface not up` 到串口控制台，
> **每秒数条，把 console 完全灌满，导致无法登录串口**。
>
> 正确做法：
> ```bash
> ip addr flush dev mgbe0     # 只清地址
> ip neigh flush dev mgbe0    # 清 ARP
> # 不 down 链路
> ```

### 恢复脚本末尾不要重启 networkd 后就撒手

`systemctl restart systemd-networkd` 在 overlay 刚格式化、networkd 状态异常时，
**不保证能干净地重新配置网卡**。建议加显式兜底：

```bash
ip addr add $IP/24 dev $NIC 2>/dev/null
ip route add default via $GW 2>/dev/null
```

并且**判断成功时以"IP 是否真的在"为准，不以命令退出码为准**。

---

## 6. 第⑦步：把本地模型服务接进恢复链

这是本文最实用的部分 —— **断电后模型服务也要自己回来**。

### 核心设计：母本放持久层，单元文件放 overlay

```
/persist/aispace/llama-server              ← 二进制，持久
/persist/aispace/models/xxx.gguf           ← 权重，持久
/persist/aispace/llama.service             ← ★ 单元文件母本，持久
        ↓ 恢复脚本第⑦步安装
/etc/systemd/system/llama.service          ← overlay，断电会丢
        ↓ systemctl enable + start
llama.service: active, port 8080
```

**为什么不用 `setsid nohup` 直接起进程**：

恢复脚本本身由 `usr_app_init.service` 的钩子调起，**运行在该服务的 cgroup 内**。
直接 `setsid nohup` 起的子进程仍在同一个 cgroup，
**父服务退出时可能被连带清理** —— 表现为"脚本报告启动成功，几秒后进程消失"。

正确做法是**托管给 systemd**，让进程脱离调用者的生命周期。

### 第⑦步代码

```bash
LLAMA_UNIT=llama.service
LLAMA_DIR=/persist/aispace
LLAMA_UNIT_SRC=$LLAMA_DIR/$LLAMA_UNIT
LLAMA_PORT=8080

if [ -x "$LLAMA_DIR/llama-server" ] && [ -f "$LLAMA_DIR/models/xxx.gguf" ]; then
    # ① 单元文件缺失或内容过期 → 从持久层母本装回
    if [ -f "$LLAMA_UNIT_SRC" ]; then
        if ! cmp -s "$LLAMA_UNIT_SRC" "/etc/systemd/system/$LLAMA_UNIT" 2>/dev/null; then
            install -m 644 -o root -g root "$LLAMA_UNIT_SRC" "/etc/systemd/system/$LLAMA_UNIT"
            systemctl daemon-reload
            echo "[7] llama: 单元文件已(重)安装"
        fi
    fi
    # ② enable + start，都是幂等操作
    systemctl enable "$LLAMA_UNIT" >/dev/null 2>&1
    systemctl start "$LLAMA_UNIT" 2>/dev/null
    # ③ 轮询等端口就绪（最多 90s，4B 模型实测 ~1.3s 加载完）
    for _i in $(seq 1 45); do
        if ss -lnt 2>/dev/null | grep -q ":$LLAMA_PORT " || \
           grep -q "listening on" "$LLAMA_DIR/llama-server.log" 2>/dev/null; then
            break
        fi
        sleep 2
    done
    # ④ 以 systemd 状态为准报告结果
    if systemctl is-active --quiet "$LLAMA_UNIT"; then
        echo "[7] llama: OK active pid=$(systemctl show -p MainPID --value $LLAMA_UNIT)"
    else
        echo "[7] llama: 失败 state=$(systemctl is-active $LLAMA_UNIT)，日志尾部："
        tail -10 "$LLAMA_DIR/llama-server.log" 2>/dev/null
    fi
fi
```

### 单元文件模板

```ini
[Unit]
Description=Local LLM Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=user
WorkingDirectory=/persist/aispace
ExecStart=/persist/aispace/llama-server \
    --model /persist/aispace/models/xxx.gguf \
    --host 0.0.0.0 --port 8080 \
    --ctx-size 8192 --n-gpu-layers 99 \
    --threads 8 --parallel 1
Restart=on-failure
RestartSec=10
SuccessExitStatus=143
StandardOutput=append:/persist/aispace/llama-server.log
StandardError=append:/persist/aispace/llama-server.log

[Install]
WantedBy=multi-user.target
```

**注意**：日志要写到持久分区。写到 `/var/log` 会随 overlay 一起丢，
下次恢复时无从排查。

---

## 7. 验证：必须做真断电，不能只做模拟

### 模拟测试（快速回归，不够）

删掉 `/etc/systemd/system/*.service` + `disable` + 杀进程，跑恢复脚本，看能否装回。
**这能验证第⑦步逻辑本身，但验证不了"钩子会不会被调起"。**

### 真断电测试（必须做一次）

```
1. 记录基线：uptime -s / IP / 服务状态
2. 直接拔电（不要 graceful shutdown —— 那样 overlay 可能不被格式化，测不出来）
3. 上电，等 7 分钟
4. 验收：
   - 钩子日志有本轮时间戳（不是上一轮遗留）
   - IP 恢复
   - 服务 active + 端口 LISTEN
   - 实际发一个推理请求，确认返回真实 token
```

### 判据：日志时间戳必须对得上本轮 boot

最容易的假绿：**读到上一轮遗留的日志**，以为恢复成功。

```bash
# 错：只看日志文件存在
ls /persist/restore_last.log

# 对：日志里的时间戳必须 > 本轮 boot 时间
uptime -s                      # 本轮启动时刻
grep 'restore start' /persist/restore_last.log | tail -1   # 必须晚于上面
```

```bash
# 更严的判据：boot 时刻 vs 钩子执行时刻
grep -E 'boot:|hook start' /persist/hook_last.log | tail -2
# boot: 2026-10-19 21:00:48   now: 2026-10-19 21:01:10   ← 相差 22 秒 = 本轮真实执行
```

### 一个反直觉的观察

**开机钩子跑起来后，串口会持续安静**（脚本内部 `exec > $LOG 2>&1` 把输出全重定向了）。
串口安静 ≠ 卡死。**判断进展只能读日志文件，不能等串口输出。**

---

## 8. 兜底：外部守护（钩子链断了才需要）

如果 §3 的原厂链在某块板上不完整，可以退回到**外部主机 + 串口守护**方案：

```
外部主机定时 ping 板子
  ├─ 通 → 健康，不动作
  └─ 不通 → 串口登录 → 跑恢复脚本 → 验收
```

这种方案能工作，但代价明显：

| 维度 | 板内钩子 | 外部守护 |
|---|---|---|
| 依赖 | 无（板子自己） | 外部主机 7×24 开机 + 串口线连接 |
| 触发延迟 | 开机即跑 | 取决于轮询间隔 |
| 单点故障 | 无 | 外部主机挂了就全挂 |
| 适用 | 原厂链完整 | 原厂链断了 |

**优先做板内钩子**，外部守护只作第二道保险。

---

## 9. 检查清单

部署新板时逐条核对：

- [ ] 原厂钩子链是否存在（`ls` 那个被 `source` 的脚本）
- [ ] 钩子落点目录是否在**持久分区**（不是 `/etc`、不是 `/tmp`）
- [ ] 钩子文件名是否与原厂硬编码**完全一致**
- [ ] 恢复脚本是否放在持久分区、root 所有、755
- [ ] 恢复脚本能否**凭空 bootstrap**（不依赖备份包）
- [ ] 网络修复是否覆盖：出厂静态 IP → DHCP、车载网卡 flush
- [ ] 模型单元母本是否在持久分区
- [ ] 模型服务是否交给 systemd 托管（不是 `nohup`）
- [ ] 日志是否写在持久分区
- [ ] **做了一次真断电，且日志时间戳对得上本轮 boot**

---

## 附：整理者注

- 本文所有路径、账号、主机名均为占位符（`/persist/`、`/厂商路径/`、`USER_A`、`enx*`）。
  读者应用本文方法时，需在自己板子上用 `ls` / `grep` 定位真实路径。
- 数据为实测入档。断电测试在真实硬件上完成，恢复耗时约 70 秒（不含开机）。
- 平台描述（DRIVE Thor / Tegra264 / sm_101a）为通用技术信息，保留原样。
