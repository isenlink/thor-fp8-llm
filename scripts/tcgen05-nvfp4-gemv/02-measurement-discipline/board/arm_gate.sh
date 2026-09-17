#!/bin/bash
# arm_gate.sh — 每臂闸门升级版（规则 B）+ boot 标记（规则 C）。板端只读 + 只写 /tmp/t4work/m2c/logs/。
# 用法:
#   arm_gate.sh pre  <tag>   # 起臂前记基线快照
#   arm_gate.sh post <tag>   # 停臂后复核四项；打印 "[<tag>] GATE4=OK" 或 "[<tag>] GATE4=BAD reason=..."
#                            # 并追加一行到 logs/arm-ledger.tsv（含 boot_id / uptime / tps）
# 退出码: 0=OK 1=BAD（调用方必须按"停手"处理：不跑下一臂、不重跑、通知现场）
set -u
L=/tmp/t4work/m2c/logs; mkdir -p "$L"
MODE=${1:?pre|post}; TAG=${2:?tag}
S="$L/.gate-$TAG.pre"
LED="$L/arm-ledger.tsv"

snap() {   # 只在 pre 用：把 6 项快照写入 $S
  {
    echo "boot=$(cat /proc/sys/kernel/random/boot_id)"
    echo "uptime=$(cut -d' ' -f1 /proc/uptime)"
    echo "err=$(sudo -n dmesg | grep -c '\[ERR\]')"
    echo "grf=$(sudo -n dmesg | grep -cE 'fecs method|gr_init|failed to reset gr|runlist [0-9]+ preempt|timeout gr busy|sm err state')"
    echo "dstate=$(for p in /proc/[0-9]*; do s=$(awk '{print $3}' "$p/stat" 2>/dev/null) || continue; case "$s" in D*) [ -e "$p/exe" ] && printf "%s," "$(cat $p/comm 2>/dev/null)";; esac; done)"
    echo "hpfree=$(awk '/HugePages_Free/{print $2}' /proc/meminfo)"
    echo "p8091=$(ss -ltn 2>/dev/null | grep -c ':8091')"
    echo "p8080=$(ss -ltn 2>/dev/null | grep -c ':8080')"
  } > "$S"
}
get() { awk -F= -v k="$1" '$1==k{sub(/^[^=]*=/,"");print}' "$S"; }

if [ "$MODE" = pre ]; then
  snap
  boot=$(get boot)
  n=$(awk -F'\t' -v b="$boot" '$2==b{c++} END{print c+0}' "$LED" 2>/dev/null || echo 0)
  MAX=${GATE_MAX_ARMS:-2}; EXCL=${GATE_EXCLUSIVE:-0}
  echo "[$TAG] GATE4_PRE boot=$boot hpfree=$(get hpfree) err=$(get err) arms_this_boot=$n/$MAX excl=$EXCL"
  if [ "$EXCL" -eq 1 ] && [ "$n" -gt 0 ]; then
    echo "blocked=1" >> "$S"
    echo "[$TAG] GATE4_PRE=BLOCKED 本臂要求独占 boot（已跑 $n 臂）⇒ 不开臂（规则 D）"; exit 2
  elif [ "$n" -ge "$MAX" ]; then
    echo "blocked=1" >> "$S"
    echo "[$TAG] GATE4_PRE=BLOCKED 本 boot 已跑 $n 臂（预算 $MAX）⇒ 不开臂（规则 D）"; exit 2
  fi
  exit 0
fi

[ -f "$S" ] || { echo "[$TAG] GATE4=BAD reason=c0_无基线快照"; exit 1; }

# c0b：pre 已被 BLOCKED（预算/独占）⇒ 本臂根本不该跑，绝不写"OK"行（2026-09-17 冒烟发现）
if [ "$(get blocked)" = "1" ]; then
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date +%FT%T)" "$(cat /proc/sys/kernel/random/boot_id)" "-" "$TAG" "BAD" \
    "c0b_预检已BLOCKED(本臂不该跑)" "-" "-" "-" >> "$LED"
  echo "[$TAG] GATE4=BAD reason=c0b_预检已BLOCKED(本臂不该跑) ⇒ 停手：不开臂、不重跑"
  exit 1
fi

boot=$(cat /proc/sys/kernel/random/boot_id)
up=$(cut -d' ' -f1 /proc/uptime)
err=$(sudo -n dmesg | grep -c '\[ERR\]')
grf=$(sudo -n dmesg | grep -cE 'fecs method|gr_init|failed to reset gr|runlist [0-9]+ preempt|timeout gr busy|sm err state')
ds=$(for p in /proc/[0-9]*; do s=$(awk '{print $3}' "$p/stat" 2>/dev/null) || continue; case "$s" in D*) [ -e "$p/exe" ] && printf "%s," "$(cat $p/comm 2>/dev/null)";; esac; done)
hp=$(awk '/HugePages_Free/{print $2}' /proc/meminfo)
p8091=$(ss -ltn 2>/dev/null | grep -c ':8091')
p8080=$(ss -ltn 2>/dev/null | grep -c ':8080')
tps=$(grep -ao '[0-9.]* tokens per second' "$L/$TAG.log" 2>/dev/null | tail -1 | awk '{print $1}')

# c6 残留实验进程（2026-09-17 新增；旧写法 `ps|grep llama|grep -c t4` 会把内核线程 ext4-rsv-conver 数进来 ⇒ 假 BAD）
# 2026-09-17 03:0x 补强（B6 事故）：原实现只记 count ⇒ 02:52 出现 count=1 时**无从定性**（没有身份证据）。
#   改为 pgrep -af 取「PID + 完整 cmdline」写入 $L/$TAG.gate.post 与台账 reason，事后可判定是误报还是真残留。
# 2026-09-17 09:1x 根因定位（B6 两臂双双 BAD）：原谓词匹配**整条 cmdline** ⇒ 任何"命令行里提到过这些串"的进程
#   都被算成残留——实测是一块**外来监控探针** `bash -c stdbuf -oL tegrastats --interval 1000 … cat /proc/loadavg …`
#   （周期性出现），于是 02:52 与 08:56 两次假 BAD。修法：**只认 argv[0]（第一个 token）**，即真被执行的程序名。
if command -v pgrep >/dev/null 2>&1; then
  c6_list=$(pgrep -af '^[^ ]*(llama-server|llama-cli|c58_v[0-9]|t4_mmq)' 2>/dev/null)
else
  c6_list=$(ps -eo pid,args 2>/dev/null | awk '{print $1, $2}' | grep -E '(llama-server|llama-cli|c58_v[0-9]|t4_mmq)' | grep -v grep)
fi
lo=$(printf '%s\n' "${c6_list:-}" | grep -c . ); lo=${lo:-0}
c6_show=$(printf '%s' "${c6_list:-}" | tr '\n' '; ' | tr -s ' ' | cut -c1-200)
# c7 接管/生效行（只有 GATE_REQUIRE_TAKEOVER=1 时强制；防「静默不接管」把旧读数当成新臂读数）
ta=$(grep -acE 'ACAP=|T4-MMQ: 命中|ARES_ACTIVE_LINES' "$L/$TAG.log" 2>/dev/null); ta=${ta:-0}

bad=""
[ "$ds"    = "$(get dstate)" ] || bad="$bad c1_D态[$(get dstate)->$ds]"
[ "$err"   = "$(get err)"    ] || bad="$bad c2_nvgpuERR[$(get err)->$err]"
[ "$grf"   = "$(get grf)"    ] || bad="$bad c3_GRfatal[$(get grf)->$grf]"
[ "$hp"    = "$(get hpfree)" ] || bad="$bad c4_池未回收[$(get hpfree)->$hp]"
[ "$p8091" = "0"             ] || bad="$bad c5_8091未释放"
[ "$p8080" = "0"             ] || bad="$bad c5b_生产口被占"
[ "${lo:-0}" -eq 0 ] 2>/dev/null || bad="$bad c6_残留实验进程[$lo:${c6_show:-?}]"
if [ "${GATE_REQUIRE_TAKEOVER:-0}" -eq 1 ] && [ "${ta:-0}" -eq 0 ]; then bad="$bad c7_无接管行(静默未生效)"; fi

printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$(date +%FT%T)" "$boot" "$up" "$TAG" "$([ -z "$bad" ] && echo OK || echo BAD)" \
  "${bad# }" "$err" "$hp" "${tps:-NA}" >> "$LED"
# 取证落盘（可回取）：post 细节含 c6 命中身份，区别于 pre 快照
{ echo "ts=$(date +%FT%T) boot=$boot up=$up"
  echo "c6_count=$lo"; echo "c6_list=$c6_show"
  echo "hp=$hp p8091=$p8091 p8080=$p8080 err=$err grf=$grf dstate=$ds"; } > "$L/$TAG.gate.post"

if [ -z "$bad" ]; then
  echo "[$TAG] GATE4=OK"
else
  echo "[$TAG] GATE4=BAD reason=${bad# }"
  echo "[$TAG] ⇒ 停手：不跑下一臂、不重跑、不 pkill；跑 t4gate.sh triage 取证并通知现场断电"
fi
echo "[$TAG] GATE4_REC $(tail -1 "$LED")"
[ -z "$bad" ] || exit 1
