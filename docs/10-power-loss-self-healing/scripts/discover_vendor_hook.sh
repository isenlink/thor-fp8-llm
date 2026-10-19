#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  discover_vendor_hook.sh — 自动定位本板的原厂开机钩子链
#
#  用法：  sudo bash discover_vendor_hook.sh
#
#  作用：  只读探测。找出：
#    ① 开机脚本入口 unit（在只读 base，断电不丢）
#    ② 它调用的 vendor 脚本
#    ③ vendor 脚本里硬编码的持久分区落点（钩子该放哪、该叫什么名）
#    ④ 持久分区候选（判断哪个分区断电不丢）
#    ⑤ 本地模型服务相关的既有单元（如果有）
#
#  不做任何修改，输出一份结论报告。
# ═══════════════════════════════════════════════════════════════════════════
set -u

say() { printf '%s\n' "$*"; }
hdr() { say ""; say "════ $* ════"; }

[ "$(id -u)" -ne 0 ] && { say "需要 root：sudo bash $0"; exit 1; }

say "discover_vendor_hook.sh — 只读探测，不改任何配置"
say "时间：$(date)"

# ── ① 找到 overlay 参数，确认哪些分区是持久层 ────────────────────────────
hdr "① 启动参数（overlay 所在）"
CMDLINE=$(cat /proc/cmdline)
say "cmdline: $CMDLINE"
OVERLAY_DEV=$(printf '%s' "$CMDLINE" | tr ' ' '\n' | grep -oE 'rw_overlay=/dev/[^ ]+' | cut -d= -f2 || true)
ROOT_DEV=$(printf '%s' "$CMDLINE" | tr ' ' '\n' | grep -oE 'root=/dev/[^ ]+' | cut -d= -f2 || true)
say "root      : ${ROOT_DEV:-未识别}"
say "rw_overlay: ${OVERLAY_DEV:-未识别}   ← 这个分区断电会被格式化，别放脚本"

hdr "② 挂载的块设备（找持久分区候选）"
lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINT 2>/dev/null | grep -vE '^\s*$' || mount | grep '^/dev'

say ""
say "判据：overlay 分区以外的、已挂载且可写的分区 = 持久层候选。"
say "      常见形态：/persist、/data、/map、/hdmap 之类，或厂商命名分区。"
say "      ⚠️ 只读挂载的分区不能放钩子。"

# ── ② 找开机脚本入口 unit ────────────────────────────────────────────────
hdr "③ 开机脚本入口 unit"
# 供应商开机脚本通常由某个 .service 调起 shell 脚本
CANDIDATES=$(grep -lsE 'ExecStart=.*(\.sh|scripts/)' \
    /usr/lib/systemd/system/*.service /etc/systemd/system/*.service 2>/dev/null \
    | grep -viE 'networkd|resolved|timesync|logrotate|apt-daily|man-db' | head -30)

if [ -z "$CANDIDATES" ]; then
    say "未自动找到候选 unit。手动查："
    say "  grep -r 'ExecStart' /usr/lib/systemd/system/*.service | grep '\.sh'"
else
    say "候选 unit（含 shell 脚本执行）："
    for u in $CANDIDATES; do
        say "  ── $(basename "$u")  [$(systemctl is-enabled "$(basename "$u" .service)" 2>/dev/null || echo '?')]"
        grep -E '^(ExecStart|Description)=' "$u" | sed 's/^/       /'
    done
fi

# ── ④ 顺着脚本找 vendor 链与钩子落点 ────────────────────────────────────
hdr "④ 钩子落点探测（关键）"
say "在所有开机脚本里搜「持久分区路径 + 可执行脚本」的模式："
say ""

# 找出被调用的脚本（source 或直接执行）
SCRIPTS=$(for u in $CANDIDATES; do
    grep -hE '^ExecStart=' "$u" 2>/dev/null | sed 's/^ExecStart=//; s/^-//' | awk '{print $1}'
done | grep -E '^/' | sort -u)

for s in $SCRIPTS; do
    [ -f "$s" ] || continue
    say "── 脚本：$s"
    # 找 source 的 vendor 脚本
    grep -nE '^\s*(source|\.)\s+/' "$s" 2>/dev/null | sed 's/^/     source: /' | head -10
    say ""
done

say "手动续查被判据圈定的 vendor 脚本（上面 source 出来的路径）："
say "  grep -nE 'source|\\.\\s+/' <那个 vendor 脚本>"
say ""
say "在 vendor 脚本里找硬编码的持久分区落点，典型模式："
say "  grep -nE '/[a-z_]+/.*\\.[a-z]+' <vendor脚本> | grep -iE 'patch|hook|init|script'"
say ""
say "★ 你要找的是这种形态（文件名可能不同，路径一定在持久分区）："
say "     if [ -e /<persist>/.rw_overlay_scripts/<some_name>.sh ]; then"
say "         /<persist>/.rw_overlay_scripts/<some_name>.sh"
say "     fi"
say "  找到后：路径和文件名一字不改照用（改名则 -e 为假，静默跳过不报错）。"

# ── ⑤ 现状核对：钩子槽位是否已经存在 ────────────────────────────────────
hdr "⑤ 钩子槽位现状"
for d in /persist /data /map /hdmap /vendor_data /vendor_map $PERSIST_DIR; do
    [ -d "$d" ] || continue
    say "── $d"
    ls -la "$d" 2>/dev/null | head -8 | sed 's/^/     /'
    for sub in .rw_overlay_scripts scripts .scripts; do
        [ -e "$d/$sub" ] && say "     ★ 已存在子目录：$d/$sub" && ls -la "$d/$sub" | sed 's/^/       /'
    done
done

# ── ⑥ 本地模型服务现状 ──────────────────────────────────────────────────
hdr "⑥ 现有模型服务单元（如果有）"
systemctl list-unit-files --no-pager 2>/dev/null \
    | grep -iE 'llama|vllm|sglang|ollama|sd-server' | sed 's/^/  /' || say "  无"

hdr "⑦ 完成"
say "下一步："
say "  1. 把 ④ 找到的路径/文件名记下来"
say "  2. 照 README 的 restore_overlay_template.sh 改写参数"
say "  3. 部署后必须做一次真断电验证（见 README 第 7 节）"
