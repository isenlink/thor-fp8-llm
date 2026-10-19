#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  hook_launcher_template.sh — 开机钩子入口（放进原厂槽位）
#
#  ★ 部署位置：原厂 vendor 脚本里硬编码的那个路径 + 文件名，一字不改
#      例：/persist/.rw_overlay_scripts/<原厂指定的名字>.sh
#
#  ★ 为什么要单独一个 launcher 而不是直接调 restore：
#    1. 先做健康检查——全好的话不跑恢复，省时间也不折腾系统
#    2. 日志单独落，便于判断"这一轮到底跑没跑"
#    3. 恢复脚本可能在别处也能手动调用，两者职责分开
#
#  ★ 重要：第一行就重定向输出
#    钩子由 usr_app_init.service 调起，它的 stdout 会混进串口控制台。
#    不重定向的话大量输出会干扰串口登录。
# ═══════════════════════════════════════════════════════════════════════════

# ★ 必须第一行：把本脚本全部输出重定向到持久分区日志
exec >> /persist/rw_hook_last.log 2>&1

RESTORE="/persist/restore_overlay.sh"

# ── 需要检查的目标参数（与 restore 脚本保持一致）───────────────────────
CHECK_USER="user"                    # 要检查免密的账号
CHECK_NIC="enx000000000000"          # 要检查网络的网卡
EXPECT_IP_PREFIX="192.0.2."          # 期望 IP 前缀（示例段，改你自己的）
CHECK_LLAMA_UNIT=""                  # 有模型服务就填单元名，如 llama.service
CHECK_LLAMA_DIR=""                   # 模型目录，如 /persist/aispace

echo ""
echo "════ thor hook start $(date) ════"
echo "boot: $(uptime -s)  now: $(date)"

NEED=0
REASONS=""

# ── 检查 1：sudoers 免密文件 ────────────────────────────────────────────
if [ -n "$CHECK_USER" ] && [ ! -f "/etc/sudoers.d/90-$CHECK_USER" ]; then
    NEED=1; REASONS="$REASONS [sudoers缺失]"
fi

# ── 检查 2：网卡配置是否为 DHCP ─────────────────────────────────────────
if [ -n "$CHECK_NIC" ]; then
    NETCONF=""
    for f in /etc/systemd/network/*.network; do
        [ -f "$f" ] || continue
        mname=$(awk -F= '/^Name=/{print $2; exit}' "$f" 2>/dev/null)
        case "$mname" in
            "$CHECK_NIC"|enx*|eth*) NETCONF="$f" ;;
        esac
    done
    if [ -z "$NETCONF" ]; then
        NEED=1; REASONS="$REASONS [网卡配置缺失]"
    elif ! grep -q '^DHCP=ipv4' "$NETCONF" 2>/dev/null; then
        NEED=1; REASONS="$REASONS [网卡配置非DHCP]"
    fi
fi

# ── 检查 3：网卡是否真的拿到期望网段的 IP ───────────────────────────────
if [ -n "$CHECK_NIC" ]; then
    CUR=$(ip -4 -br addr show "$CHECK_NIC" 2>/dev/null | awk '{print $3}')
    if [ -z "$CUR" ]; then
        NEED=1; REASONS="$REASONS [$CHECK_NIC无IP]"
    elif [ -n "$EXPECT_IP_PREFIX" ]; then
        case "$CUR" in
            ${EXPECT_IP_PREFIX}*) : ;;
            *) NEED=1; REASONS="$REASONS [$CHECK_NIC IP=$CUR 不符${EXPECT_IP_PREFIX}*]" ;;
        esac
    fi
    echo "当前 $CHECK_NIC IP: ${CUR:-无}"
fi

# ── 检查 4：模型服务单元是否存在 ────────────────────────────────────────
if [ -n "$CHECK_LLAMA_UNIT" ] && [ -n "$CHECK_LLAMA_DIR" ]; then
    if [ ! -f "/etc/systemd/system/$CHECK_LLAMA_UNIT" ]; then
        NEED=1; REASONS="$REASONS [模型单元缺失]"
    elif ! systemctl is-active --quiet "$CHECK_LLAMA_UNIT" 2>/dev/null; then
        NEED=1; REASONS="$REASONS [模型服务未运行]"
    fi
fi

echo "检测结果: NEED_RESTORE=$NEED 原因:${REASONS:- 全部健康}"

# ── 执行恢复 ────────────────────────────────────────────────────────────
if [ "$NEED" = "1" ]; then
    if [ -x "$RESTORE" ] || [ -f "$RESTORE" ]; then
        echo "→ 调用恢复脚本: $RESTORE"
        bash "$RESTORE"
        RC=$?
        echo "→ 恢复脚本退出码: $RC"
    else
        echo "→ ✗ 恢复脚本不存在: $RESTORE"
        RC=127
    fi
else
    echo "→ 全部健康，跳过恢复"
    RC=0
fi

echo "════ thor hook end $(date) ════"
exit $RC
