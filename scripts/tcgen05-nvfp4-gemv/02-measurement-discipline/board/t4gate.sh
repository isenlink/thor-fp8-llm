#!/bin/bash
# t4gate.sh — 规则 A（安灯 triage）/ C（boot 台账·可疑标记）/ D（启停预算）。宿主侧，只读板卡。
# 用法:
#   t4gate.sh triage [note]        # A：收到任何"板上报错"即刻跑一次；出结论 + 取证落盘
#   t4gate.sh wedge  <reason>      # C：登记一次 wedge ⇒ 该 boot 的所有臂标 SUSPECT
#   t4gate.sh preflight [--exclusive-boot] [--max-arms N]   # D：起臂前查预算/状态，rc!=0 ⇒ 不许开
#   t4gate.sh sync                 # 把板端 arm-ledger.tsv 并回宿主台账
#   t4gate.sh suspect [boot_id]    # 列出可疑臂
#   t4gate.sh andon <note>         # A：板上报错即刻拉起安灯（triage + 挂 .halt 停线牌）——不许"等会儿再处理"
#   t4gate.sh andon-clear          # A：现场断电重启并复核洁净后，摘掉停线牌
#   t4gate.sh budget               # D：启停预算总览（今日 boot 数 / 每 boot 臂数 / 下一 boot 的预登记臂）
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)                  # project/
LED="$ROOT/evidence/boot-ledger.tsv"
SSH=${THOR_SSH:-${BOARD_SSH:?set BOARD_SSH to your ssh wrapper}}
TOTAL_HP_EXPECT=23552
bssh() { timeout 90 bash "$SSH" "$1" 2>/dev/null; }

led_probe() {  # 板卡只读探针（一次 ssh 拿全部）
  bssh 'echo "BOOT=$(cat /proc/sys/kernel/random/boot_id)"; echo "UP=$(cut -d" " -f1 /proc/uptime)";
        echo "ERR=$(sudo -n dmesg | grep -c "\[ERR\]")";
        echo "GRF=$(sudo -n dmesg | grep -cE "fecs method|failed to reset gr|timeout gr busy|runlist [0-9]+ preempt|sm err state|gr_init.*fail")";
        echo "DSTATE=$(for p in /proc/[0-9]*; do s=$(awk "{print \$3}" "$p/stat" 2>/dev/null) || continue; case "$s" in D*) [ -e "$p/exe" ] && printf "%s," "$(cat $p/comm 2>/dev/null)";; esac; done)";
        echo "HPFREE=$(awk "/HugePages_Free/{print \$2}" /proc/meminfo)";
        echo "HPTOTAL=$(awk "/HugePages_Total/{print \$2}" /proc/meminfo)";
        echo "P8091=$(ss -ltn 2>/dev/null | grep -c ":8091")"'
}

case "${1:-}" in
triage)
  note=${2:-manual}
  stamp=$(date +%Y%m%d-%H%M)
  dir="$ROOT/evidence/incident-$stamp"; mkdir -p "$dir"
  bssh 'uptime' > "$dir/uptime.txt"
  bssh 'sudo -n dmesg' > "$dir/dmesg-full.txt"
  bssh 'ps -eo pid,ppid,stat,etime,wchan:32,cmd' > "$dir/ps.txt"
  bssh 'cat /proc/meminfo' > "$dir/meminfo.txt"
  bssh 'ls -lt /tmp | head -40' > "$dir/tmp-ls.txt"
  echo "note=$note" > "$dir/summary.txt"
  ( cd "$dir" && sha256sum -b ./* > SHA256SUMS )
  p=$(led_probe)
  BOOT=$(echo "$p" | sed -n 's/^BOOT=//p' | tail -1)
  GRF=$(echo "$p" | sed -n 's/^GRF=//p' | tail -1)
  ERR=$(echo "$p" | sed -n 's/^ERR=//p' | tail -1)
  DS=$(echo "$p" | sed -n 's/^DSTATE=//p' | tail -1)
  HP=$(echo "$p" | sed -n 's/^HPFREE=//p' | tail -1)
  HPT=$(echo "$p" | sed -n 's/^HPTOTAL=//p' | tail -1)
  verdict=OK; reason=""
  if [ "${GRF:-0}" -gt 0 ]; then verdict=WEDGED; reason="GR不可恢复签名($GRF 行)"; fi
  if [ -n "${DS:-}" ]; then verdict=WEDGED; reason="$reason D态:$DS"; fi
  if [ "${HP:-0}" != "${HPT:-0}" ]; then reason="$reason 池未回收($HP/$HPT)"; [ "$verdict" = OK ] && verdict=STUCK; fi
  echo "TRIAGE=$verdict boot=$BOOT err=$ERR grfatal=$GRF dstate=[$DS] hp=$HP/$HPT evidence=$dir"
  if [ "$verdict" != OK ]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date +%FT%T)" "$BOOT" "-" "-" "WEDGE" "$verdict:$reason" "$ERR" "$HP" "$note" >> "$LED"
    echo "⇒ 已登记台账（规则 C）。停手：不 pkill / 不软复位 / 不重跑；通知现场断电（规则 A）。"
    bash "$0" suspect "$BOOT"
  fi
  ;;
wedge)
  reason=${2:-unspecified}
  p=$(led_probe); BOOT=$(echo "$p" | sed -n 's/^BOOT=//p' | tail -1)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date +%FT%T)" "$BOOT" "-" "-" "WEDGE" "$reason" "-" "-" "-" >> "$LED"
  echo "WEDGE_RECORDED boot=$BOOT reason=$reason"
  bash "$0" suspect "$BOOT"
  ;;
preflight)
  max=2; excl=0
  while [ $# -gt 0 ]; do case "$1" in --exclusive-boot) excl=1;; --max-arms) shift; max=$1;; esac; shift; done
  p=$(led_probe); BOOT=$(echo "$p" | sed -n 's/^BOOT=//p' | tail -1)
  DS=$(echo "$p" | sed -n 's/^DSTATE=//p' | tail -1)
  HP=$(echo "$p" | sed -n 's/^HPFREE=//p' | tail -1); HPT=$(echo "$p" | sed -n 's/^HPTOTAL=//p' | tail -1)
  n=$(awk -F'\t' -v b="$BOOT" '$2==b && $5!="WEDGE"{c++} END{print c+0}' "$LED" 2>/dev/null || echo 0)
  w=$(awk -F'\t' -v b="$BOOT" '$2==b && $5=="WEDGE"{c++} END{print c+0}' "$LED" 2>/dev/null || echo 0)
  fail=""
  [ -f "$ROOT/evidence/.halt" ] && fail="$fail 停线牌在架(ANDON 未摘：先 t4gate.sh andon-clear)"
  [ -n "${DS:-}" ] && fail="$fail 有D态进程[$DS]"
  [ "${HP:-0}" != "${HPT:-0}" ] && fail="$fail 池未回收($HP/$HPT)"
  [ "$w" -gt 0 ] && fail="$fail 本boot已登记wedge"
  [ "$n" -ge "$max" ] && fail="$fail 本boot已跑${n}臂(预算${max})"
  [ "$excl" = 1 ] && [ "$n" -gt 0 ] && fail="$fail 本臂要求独占boot(已跑${n}臂)"
  if [ -z "$fail" ]; then echo "PREFLIGHT=OK boot=$BOOT arms_this_boot=$n/${max} hp=$HP/$HPT"
  else echo "PREFLIGHT=BLOCKED boot=$BOOT arms_this_boot=$n/${max} reason=${fail# }"; exit 1; fi
  ;;
andon)
  note=${2:-manual}
  stamp=$(date +%Y%m%d-%H%M)
  HALT="$ROOT/evidence/.halt"
  bash "$0" triage "andon:$note"
  doc="$ROOT/evidence/ANDON-$stamp.md"
  if [ ! -f "$doc" ]; then
    { echo "# ANDON $stamp — 板上报错，停线"; echo;
      echo "- 触发：$note";
      echo "- 快照：$(bash "$SSH" 'echo boot=$(cat /proc/sys/kernel/random/boot_id) up=$(cut -d" " -f1 /proc/uptime)' 2>/dev/null | tail -1)";
      echo;
      echo "## 必填三行（不填完不许摘牌）";
      echo "1. 现象：";
      echo "2. 影响面（哪些臂/读数作废，按规则 C 标 SUSPECT）：";
      echo "3. 解除条件（要哪个新 boot 上的什么读数才算解除）：";
    } > "$doc"
  fi
  : > "$HALT"
  echo "ANDON=RAISED note=$note 停线牌=$HALT 记录=$doc"
  echo "⇒ 停手：不开臂、不重跑、不 pkill、不软复位；跑 triage 取证后通知现场断电重启"
  ;;
andon-clear)
  HALT="$ROOT/evidence/.halt"
  if [ ! -f "$HALT" ]; then echo "ANDON=NOT_RAISED（无停线牌）"; exit 0; fi
  boot=$(led_probe | sed -n 's/^BOOT=//p' | tail -1)
  GRF=$(led_probe | sed -n 's/^GRF=//p' | tail -1)
  if [ "${GRF:-0}" -gt 0 ]; then echo "ANDON_CLEAR=REFUSED 仍见 GR 致命签名($GRF) ⇒ 不许摘牌"; exit 1; fi
  if awk -F'\t' -v b="$boot" '$2==b && $5=="WEDGE"{f=1} END{exit !f}' "$LED" 2>/dev/null; then
    echo "ANDON_CLEAR=REFUSED 当前 boot($boot) 已登记 wedge ⇒ 需现场再断电一次"; exit 1
  fi
  rm -f "$HALT"
  echo "ANDON=CLEARED boot=$boot（下次 preflight 恢复放行）"
  ;;
budget)
  Q="$ROOT/evidence/arm-queue.tsv"
  today=$(date +%F)
  echo "== 启停预算（规则 D：每个 boot 最多 2 臂；每次请求断电前必须已写死下一 boot 的两臂）=="
  echo "-- 今日（$today）boot 与臂数 --"
  awk -F'\t' -v d="$today" 'index($1,d)==1{print $2"\t"$4}' "$LED" 2>/dev/null | sort -u | \
    awk -F'\t' '{a[$1]=a[$1]" "$2} END{for(b in a) printf "  boot %.12s… arms:%s\n", b, a[b]}'
  echo "-- 队列（$Q）--"
  if [ -f "$Q" ]; then cat "$Q"; else echo "  （无队列文件 ⇒ 不许请求断电：先在 $Q 写死下一 boot 的两臂与判据）"; fi
  ;;
sync)
  tmp=$(mktemp)
  bssh 'cat /tmp/t4work/m2c/logs/arm-ledger.tsv' > "$tmp" 2>/dev/null || true
  n=0
  while IFS=$'\t' read -r ts boot up tag gate reason err hp tps; do
    [ -z "${tag:-}" ] && continue
    grep -qP "^[^\t]*\t\Q$boot\E\t[^\t]*\t\Q$tag\E\t" "$LED" 2>/dev/null && continue
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$ts" "$boot" "$up" "$tag" "$gate" "$reason" "$err" "$hp" "$tps" >> "$LED"
    n=$((n+1))
  done < "$tmp"
  rm -f "$tmp"; echo "SYNC_OK new_arms=$n ledger=$LED"
  ;;
suspect)
  boot=${2:-}
  if [ -n "$boot" ]; then awk -F'\t' -v b="$boot" '$2==b{print}' "$LED"; else awk -F'\t' '$5=="WEDGE"{print}' "$LED"; fi
  echo "--- 可疑臂（其 boot 有 wedge 记录 ⇒ 规则 C：需在新 boot 复测后才可用于结论）---"
  awk -F'\t' -v b="$boot" '$4!="-" && $4!=""{print}' "$LED" | while IFS=$'\t' read -r ts bo up tag gate reason err hp tps; do
    if [ -z "$boot" ] || [ "$bo" = "$boot" ]; then
      if awk -F'\t' -v x="$bo" '$2==x && $5=="WEDGE"{f=1} END{exit !f}' "$LED"; then echo "SUSPECT_SAME_BOOT $tag boot=$bo gate=$gate tps=$tps"; fi
    fi
  done
  ;;
*)
  sed -n '2,12p' "$0"; exit 2;;
esac
