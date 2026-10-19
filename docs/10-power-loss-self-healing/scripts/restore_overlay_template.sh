#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  restore_overlay_template.sh — 断电后自动恢复全部配置
#
#  ★ 使用方式
#    1. 复制到持久分区（不是 /etc！overlay 断电会被格式化）：
#         sudo install -m 755 -o root -g root restore_overlay_template.sh \
#              /persist/restore_overlay.sh
#    2. 修改下面「参数区」的每一项，改成你板子的实际值
#    3. 放到钩子槽位（路径/文件名必须与原厂脚本硬编码的一致）：
#         sudo install -m 755 -o root -g root \
#              hook_launcher_template.sh \
#              /persist/.rw_overlay_scripts/<原厂文件名>.sh
#    4. 先手动跑一次验证：sudo bash /persist/restore_overlay.sh
#    5. 再做一次真断电验证（README 第 7 节）
#
#  ★ 设计原则
#    - 每一步独立、幂等，某步失败不中断后续
#    - 不依赖任何备份包：账号/配置都能凭空 bootstrap
#    - 日志写持久分区（overlay 里的日志会随断电丢失）
#    - 全程不 reboot（避免恢复循环）
# ═══════════════════════════════════════════════════════════════════════════

set -u

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                          参 数 区（必须改）                            ║
# ╚═══════════════════════════════════════════════════════════════════════╝

# ★ 持久分区挂载点（断电不丢的分区；overlay 分区绝不能填这里）
#   探测方法见 discover_vendor_hook.sh，或 lsblk 看哪个分区断电后内容还在
PERSIST_DIR="/persist"

# ── 网络 ────────────────────────────────────────────────────────────────
NIC="enx000000000000"            # 网卡名：ip -br link 查看
NET_MODE="dhcp"                  # "dhcp" 或 "static"
STATIC_IP="192.0.2.100"        # NET_MODE=static 时生效
STATIC_GW="192.0.2.1"
STATIC_DNS="192.0.2.1"

# 车载/其它要清理的网卡前缀（只 flush 地址，不要 down 链路！见 README 第5节）
VEHICLE_NIC_PREFIX="mgbe"

# ── 账号（缺才建，幂等）──────────────────────────────────────────────────
USERS="user"                     # 空格分隔；每个都会建 + 加 sudo + 免密
# 密码来源：$PERSIST_DIR/account_backup/passwords.txt
#   格式：每行 "用户名:密码"（明文，文件权限由脚本设为 600）
#   ★ 必须先建好这个文件，否则账号建出来但没有密码（脚本会告警）
#   建立方法：
#     sudo mkdir -p /persist/account_backup && sudo chmod 700 /persist/account_backup
#     printf 'user:YourPass\n' | sudo tee /persist/account_backup/passwords.txt >/dev/null
#     sudo chmod 600 /persist/account_backup/passwords.txt

# ── 大页池（可选，不需要就设 0）─────────────────────────────────────────
HUGEPAGES=0                      # 0 = 跳过

# ── 要停用的车机业务服务（可选，用 | 分隔）──────────────────────────────
DISABLE_SERVICES=""              # 例："svc_a|svc_b|svc_c"

# ── 本地模型服务（可选，不需要就留空）───────────────────────────────────
LLAMA_DIR=""                     # 例：/persist/aispace
LLAMA_UNIT="llama.service"
LLAMA_PORT=8080

# ── 日志 ────────────────────────────────────────────────────────────────
LOG="/persist/restore_overlay_last.log"   # ★ 必须在持久分区

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                       以下不用改（按需调整）                           ║
# ╚═══════════════════════════════════════════════════════════════════════╝

exec > >(tee -a "$LOG") 2>&1
echo ""
echo "════════ restore start $(date) ════════"
echo "boot: $(uptime -s)  now: $(date)"

# 结果汇总（最后统一判定）
FAILED=""

fail() { FAILED="$FAILED [$1]"; echo "  ✗ $1: $2"; }

# ═══════════════════════════════════════════════════════════════════════
# ⓪ 免密 sudo —— 必须最先做，否则后续全部卡密码
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "───── ⓪ 免密 sudo ─────"
mkdir -p /etc/sudoers.d
chmod 750 /etc/sudoers.d

for u in $USERS; do
    if [ -f "/etc/sudoers.d/90-$u" ]; then
        echo "  $u: 已存在，跳过"
        continue
    fi
    echo "$u ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/90-$u"
    chmod 440 "/etc/sudoers.d/90-$u"
    # ★ 语法校验，失败立刻删——写坏 sudoers 会锁死整个系统
    if ! visudo -cf "/etc/sudoers.d/90-$u" >/dev/null 2>&1; then
        rm -f "/etc/sudoers.d/90-$u"
        fail "sudoers-$u" "语法校验失败，已删除"
    else
        echo "  $u: 免密已配置"
    fi
done

# ═══════════════════════════════════════════════════════════════════════
# ① 网卡固件路径（若固件不在 base 里，需从持久分区复制）
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "───── ① 固件 ─────"
FW_DIR="/persist/firmware"
if [ -d "$FW_DIR" ] && [ -n "$(ls -A "$FW_DIR" 2>/dev/null)" ]; then
    mkdir -p /lib/firmware
    cp -an "$FW_DIR"/* /lib/firmware/ 2>/dev/null
    echo "  已从持久分区复制固件"
else
    echo "  无自定义固件（跳过）"
fi

# ═══════════════════════════════════════════════════════════════════════
# ② 网络 —— 最容易失败的一步
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "───── ② 网络 ─────"

mkdir -p /etc/systemd/network

# 2a. 把厂商写死静态地址的配置改成 DHCP
#     ★ 只匹配 "Name=<NIC> 或 enx*" 且带 Address= 的配置
for f in /etc/systemd/network/*.network; do
    [ -f "$f" ] || continue
    # 该文件是否匹配我们的网卡
    mname=$(awk -F= '/^Name=/{print $2; exit}' "$f" 2>/dev/null)
    case "$mname" in
        "$NIC"|enx*|eth*)
            if grep -q '^Address=' "$f" 2>/dev/null; then
                cp -n "$f" "$f.bak-restore" 2>/dev/null
                sed -i 's/^Address=/#Address=/' "$f"
                if [ "$NET_MODE" = "dhcp" ]; then
                    grep -q '^DHCP=' "$f" || echo "DHCP=ipv4" >> "$f"
                    sed -i 's/^DHCP=.*/DHCP=ipv4/' "$f"
                fi
                echo "  已改 $f（原静态地址注释掉）"
            fi
            ;;
    esac
done

# 2b. 屏蔽车载网卡的 IPv6/RA 干扰（只加配置，不 down 链路）
if [ -n "$VEHICLE_NIC_PREFIX" ]; then
    SHIELD="/etc/systemd/network/10-lab-shield-vehicle.network"
    if [ ! -f "$SHIELD" ]; then
        cat > "$SHIELD" <<EOF
[Match]
Name=${VEHICLE_NIC_PREFIX}*

[Network]
DHCP=no
LinkLocalAddressing=no
IPv6AcceptRA=no
EOF
        echo "  已创建 $SHIELD（屏蔽 ${VEHICLE_NIC_PREFIX}*）"
    fi
fi

systemctl restart systemd-networkd 2>/dev/null
sleep 3

# 2c. ★ flush 车载网卡地址 —— 只 flush，绝不 down！
#     down 链路会让驱动持续输出 ethtool 报错灌满串口
for n in /sys/class/net/${VEHICLE_NIC_PREFIX}*; do
    [ -e "$n" ] || continue
    dev=$(basename "$n")
    ip addr flush dev "$dev" 2>/dev/null
    ip neigh flush dev "$dev" 2>/dev/null
    echo "  已 flush $dev（未 down）"
done

# 2d. 静态模式下显式配 IP
if [ "$NET_MODE" = "static" ]; then
    ip addr add "$STATIC_IP/24" dev "$NIC" 2>/dev/null
    ip route add default via "$STATIC_GW" 2>/dev/null
    echo "  已设静态 $STATIC_IP"
fi

# 2e. ★ 显式兜底：networkd 有时不重建配置，以"IP 是否真的在"为准
sleep 3
CUR_IP=$(ip -4 -br addr show "$NIC" 2>/dev/null | awk '{print $3}')
if [ -z "$CUR_IP" ]; then
    echo "  networkd 未生效，走 DHCP 客户端兜底"
    (dhclient -1 -v "$NIC" >/dev/null 2>&1 || udhcpc -i "$NIC" >/dev/null 2>&1) || true
    sleep 3
    CUR_IP=$(ip -4 -br addr show "$NIC" 2>/dev/null | awk '{print $3}')
fi

if [ -n "$CUR_IP" ]; then
    echo "  网络 OK: $NIC = $CUR_IP"
else
    fail "网络" "$NIC 无 IPv4 地址"
fi

# DNS
if [ -z "$(grep -E '^nameserver' /etc/resolv.conf 2>/dev/null)" ]; then
    mkdir -p /etc
    echo "nameserver ${STATIC_DNS}" > /etc/resolv.conf
    echo "  已写 DNS ${STATIC_DNS}"
fi

# ═══════════════════════════════════════════════════════════════════════
# ③ 时钟
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "───── ③ 时钟 ─────"
systemctl enable --now systemd-timesyncd >/dev/null 2>&1
systemctl restart systemd-timesyncd >/dev/null 2>&1
sleep 2
echo "  NTP: $(timedatectl show -p NTPSynchronized --value 2>/dev/null || echo unknown)"

# ═══════════════════════════════════════════════════════════════════════
# ④ 账号 —— 缺才建，不依赖备份包
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "───── ④ 账号 ─────"

# 凭据文件：每行 用户名:密码，放持久分区
PWFILE="$PERSIST_DIR/account_backup/passwords.txt"

for u in $USERS; do
    if ! id "$u" >/dev/null 2>&1; then
        useradd -m -s /bin/bash "$u" 2>/dev/null && echo "  已建 $u" || fail "账号-$u" "创建失败"
    else
        echo "  $u: 已存在"
    fi
    # 确保在 sudo 组
    id -nG "$u" 2>/dev/null | grep -qw sudo || usermod -aG sudo "$u" 2>/dev/null
    # 确保家目录存在且属主正确（useradd -m 偶尔因 overlay 异常失败）
    if [ -d "/home/$u" ]; then
        chown "$u:$u" "/home/$u" 2>/dev/null
        chmod 750 "/home/$u" 2>/dev/null
    else
        mkdir -p "/home/$u" && chown "$u:$u" "/home/$u" && chmod 750 "/home/$u" \
            && echo "  补建家目录 /home/$u"
    fi
done

# ★ 密码：文件缺失必须报警，不能静默跳过
#   （静默跳过是本方案最隐蔽的失败形态：sudo -n 能过，看起来账号好了，
#     实际密码为空，SSH 密码登录失败）
if [ -f "$PWFILE" ]; then
    PW_OK=0; PW_FAIL=0
    while IFS=: read -r u p; do
        [ -z "$u" ] && continue
        case "$u" in \#*) continue ;; esac
        if printf '%s:%s\n' "$u" "$p" | chpasswd 2>/dev/null; then
            echo "  密码已设: $u"
            PW_OK=$((PW_OK+1))
        else
            fail "密码-$u" "chpasswd 失败"
            PW_FAIL=$((PW_FAIL+1))
        fi
    done < "$PWFILE"
    echo "  密码设置结果: 成功 $PW_OK 个，失败 $PW_FAIL 个"
else
    # 不算硬失败（可能本来就不用密码），但必须显式告警
    echo "  ⚠️  凭据文件缺失: $PWFILE"
    echo "      账号已建但无密码 —— SSH 密码登录会失败（密钥/串口仍可用）"
    echo "      修复：mkdir -p $(dirname "$PWFILE") && 写入 '用户名:密码' 每行一条，chmod 600"
fi

# ★ 收紧权限：凭据文件含明文密码
[ -f "$PWFILE" ] && chmod 600 "$PWFILE" 2>/dev/null

# ★ 自检：明确报告是否有可用密码（不要只看 sudo 能过就以为好了）
echo "  ── 密码状态自检 ──"
for u in $USERS; do
    st=$(awk -F: -v u="$u" '$1==u{print ($2==""||$2=="!"||$2=="*")?"NOPW":"OK"}' /etc/shadow 2>/dev/null)
    echo "     $u = ${st:-未找到}"
done

# ═══════════════════════════════════════════════════════════════════════
# ⑤ 大页池
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "───── ⑤ 大页池 ─────"
if [ "$HUGEPAGES" -gt 0 ] 2>/dev/null; then
    # 运行时写
    echo "$HUGEPAGES" > /proc/sys/vm/nr_hugepages 2>/dev/null
    # 持久化到 sysctl.d（overlay 会丢，所以每次恢复都重写）
    mkdir -p /etc/sysctl.d
    echo "vm.nr_hugepages = $HUGEPAGES" > /etc/sysctl.d/99-hugepages.conf
    CUR=$(cat /proc/sys/vm/nr_hugepages 2>/dev/null)
    echo "  hugepages: 期望=$HUGEPAGES 实际=$CUR"
    [ "$CUR" = "$HUGEPAGES" ] || fail "大页池" "实际 $CUR != 期望 $HUGEPAGES"
else
    echo "  跳过（HUGEPAGES=0）"
fi

# ═══════════════════════════════════════════════════════════════════════
# ⑥ 收尾：停用车机业务服务、清理残留
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "───── ⑥ 收尾 ─────"
if [ -n "$DISABLE_SERVICES" ]; then
    OLDIFS=$IFS; IFS='|'
    CNT=0
    for svc in $DISABLE_SERVICES; do
        systemctl disable --now "$svc" >/dev/null 2>&1 && CNT=$((CNT+1))
    done
    IFS=$OLDIFS
    echo "  已停用 $CNT 个业务服务"
else
    echo "  跳过（DISABLE_SERVICES 为空）"
fi

# 检查是否还有残留的内网业务 IP（除主网卡外）
LEFTOVER=$(ip -4 -br addr show 2>/dev/null | grep -v " $NIC " | grep -vc '^lo')
echo "  其他网卡仍有地址的条数: $LEFTOVER"

# ═══════════════════════════════════════════════════════════════════════
# ⑦ 本地模型服务 —— 交给 systemd，不要 nohup
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "───── ⑦ 本地模型服务 ─────"
if [ -n "$LLAMA_DIR" ] && [ -d "$LLAMA_DIR" ]; then
    UNIT_SRC="$LLAMA_DIR/$LLAMA_UNIT"
    UNIT_DST="/etc/systemd/system/$LLAMA_UNIT"

    # 7a. 单元文件缺失或内容过期 → 从持久层母本装回
    if [ -f "$UNIT_SRC" ]; then
        if ! cmp -s "$UNIT_SRC" "$UNIT_DST" 2>/dev/null; then
            install -m 644 -o root -g root "$UNIT_SRC" "$UNIT_DST"
            systemctl daemon-reload
            echo "  单元文件已(重)安装: $UNIT_DST"
        else
            echo "  单元文件已是最新"
        fi
    else
        fail "模型单元" "$UNIT_SRC 不存在"
    fi

    # 7b. enable + start（都幂等）
    systemctl enable "$LLAMA_UNIT" >/dev/null 2>&1
    systemctl start "$LLAMA_UNIT" 2>/dev/null

    # 7c. 轮询等端口就绪（最多 90s）
    for _i in $(seq 1 45); do
        if ss -lnt 2>/dev/null | grep -q ":$LLAMA_PORT "; then
            break
        fi
        if grep -q "listening on" "$LLAMA_DIR/llama-server.log" 2>/dev/null; then
            break
        fi
        sleep 2
    done

    # 7d. 以 systemd 状态为准
    if systemctl is-active --quiet "$LLAMA_UNIT"; then
        echo "  llama: OK active pid=$(systemctl show -p MainPID --value "$LLAMA_UNIT") port=$LLAMA_PORT"
    else
        fail "模型服务" "state=$(systemctl is-active "$LLAMA_UNIT" 2>/dev/null)"
        tail -10 "$LLAMA_DIR/llama-server.log" 2>/dev/null | sed 's/^/    /'
    fi
else
    echo "  跳过（LLAMA_DIR 未配置）"
fi

# ═══════════════════════════════════════════════════════════════════════
# 总结
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "════════ restore end $(date) ════════"
if [ -z "$FAILED" ]; then
    echo "RESTORE_ALL_OK"
    exit 0
else
    echo "RESTORE_PARTIAL_FAILED:$FAILED"
    exit 1
fi
