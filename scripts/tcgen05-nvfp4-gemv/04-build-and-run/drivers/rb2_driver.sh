#!/bin/bash
# rb2_driver.sh — R-B2 判别臂宿主编排（把规则 A/B/C/D 变成机械门）
#   待解释的缺口：原型 ws32k 257.14 GB/s（250 ns/4608B chunk/SM） vs 生产腿 ~101 GB/s（~627 ns/chunk）
#   假设 H1：主项 = B 拷贝的 **cp.async 粒度**（腿 4 B/次 ⇒ 1152 请求/chunk；原型 16 B/次 ⇒ 288；实测 SASS LDGSTS 9→36）
#   设计：同一源文件、同一宿主驱动、同一臂序、同 boot，唯一变量 = cubin（gran0 / gran1）
#   判据：① g0 复现归档口径（≥249.4 GB/s = 257.14 −3%）⇒ 本 boot 锚点，链子没坏
#         ② g1 ≤140 GB/s ⇒ H1 成立；≥200 GB/s ⇒ H1 证伪；中间 ⇒ 不确定（需 8 B 粒度补一臂）
#   链路：build(基线同一性门) → selftest → 判据 replay → 洁净检查 → preflight(规则 D) → 上传(板端核 sha)
#         → 两臂（各 6 次内核调用，硬超时 300 s）→ 回取 → rb1_metrics.py → verdict.py + 台账 sync
#   BAD/异常 ⇒ 自动 t4gate.sh andon（规则 A：同回合处置）
# 用法: bash t4/m2c/drivers/rb2_driver.sh            # 真跑（会占满本 boot 的 2 臂预算）
#       bash t4/m2c/drivers/rb2_driver.sh --dry-run  # 只跑到 preflight，打印两臂计划，不碰板卡
set -u
ROOT=/path/to/project
SSH=${THOR_SSH:-${BOARD_SSH:?set BOARD_SSH to your ssh wrapper}}
R=/tmp/t4work/m2c
FX=$ROOT/t4/m2c/host/verdict_fixtures
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
if [ "$DRY" = 1 ]; then EV=/tmp/rb2_dryrun; else EV=$ROOT/evidence/$(date +%Y%m%d-%H%M)-rb2-gran; fi
mkdir -p "$EV"
log(){ echo "[$(date '+%F %T')] $*"; }
clean_ok() { [ "${1:-x}" = 0 ] && [ "${2:-x}" = 0 ] && [ -z "${3:-}" ]; }
echo "=== RB2 DRIVER START $(date -Iseconds) dry_run=$DRY evidence=$EV ==="

log "--- 构建门：gran0 必须与归档件 SASS 逐字节同 + gran1 的 LDGSTS 必须 4× ---"
bash "$ROOT/t4/m2c/drivers/rb2_build.sh" > "$EV/00_build.txt" 2>&1 || { echo "BUILD_GATE_FAIL"; tail -8 "$EV/00_build.txt"; exit 9; }
grep -q BASE_IDENTICAL "$EV/00_build.txt" && grep -q GRAN4_EFFECTIVE "$EV/00_build.txt" || { echo "BUILD_GATE_FAIL（缺 BASE_IDENTICAL/GRAN4_EFFECTIVE）"; tail -8 "$EV/00_build.txt"; exit 9; }
sed 's/^/  /' "$EV/00_build.txt"
SHA_G1=$(awk -F'\t' '$1=="g1"{print $2}' "$ROOT/t4/m2c/stage/rb2/expected.tsv")
log "gran1 sha16=$SHA_G1"

log "--- 夹具回归（新逻辑先过夹具；pass/fail 里 fail 必须为 0）---"
bash "$ROOT/t4/m2c/host/selftest.sh" > "$EV/01_selftest.txt" 2>&1 || { echo "SELFTEST FAIL ⇒ 拒绝上板"; tail -5 "$EV/01_selftest.txt"; exit 9; }
SF=$(grep -o 'pass=[0-9]* fail=[0-9]*' "$EV/01_selftest.txt" | tail -1); echo "SELFTEST $SF"
echo "$SF" | grep -q 'fail=0' || { echo "SELFTEST 有 FAIL ⇒ 拒绝上板"; exit 9; }

log "--- 规则 B：三条判据的可用性预检（好/坏样本判定必须不同）---"
rp(){ python3 "$ROOT/t4/m2c/host/verdict.py" --spec "$1" --replay "$2" "$3"; }
{ rp "$FX/rb2_g0.spec"    "$FX/rb2_g0_good.metrics" "$FX/rb2_g0_bad.metrics"
  rp "$FX/rb2_g1_h1.spec" "$FX/rb2_g1_low.metrics"  "$FX/rb2_g1_high.metrics"
  rp "$FX/rb2_g1_h0.spec" "$FX/rb2_g1_high.metrics" "$FX/rb2_g1_low.metrics"; } > "$EV/02_replay.txt" 2>&1
sed 's/^/  /' "$EV/02_replay.txt"
[ "$(grep -c 'REPLAY=OK' "$EV/02_replay.txt")" = 3 ] || { echo "ABORT: 判据无判别力（REPLAY != 3 个 OK）"; exit 9; }

log "--- 洁净检查（ERR/GR 致命/D 态 必须全零）---"
P=$(bash "$SSH" 'echo "BOOT=$(cat /proc/sys/kernel/random/boot_id)"; echo "UP=$(cut -d" " -f1 /proc/uptime)";
   echo "ERR=$(sudo -n dmesg | grep -c "\[ERR\]")";
   echo "GRF=$(sudo -n dmesg | grep -cE "fecs method|failed to reset gr|timeout gr busy|runlist [0-9]+ preempt|sm err state|gr_init.*fail")";
   echo "DSTATE=$(for p in /proc/[0-9]*; do s=$(awk "{print \$3}" "$p/stat" 2>/dev/null) || continue; case "$s" in D*) [ -e "$p/exe" ] && printf "%s," "$(cat $p/comm 2>/dev/null)";; esac; done)"' 2>/dev/null)
printf '%s\n' "$P" > "$EV/03_boot-check.txt"; printf '%s\n' "$P" | sed 's/^/  /'
ER=$(printf '%s\n' "$P" | sed -n 's/^ERR=//p' | tail -1); GF=$(printf '%s\n' "$P" | sed -n 's/^GRF=//p' | tail -1)
DS=$(printf '%s\n' "$P" | sed -n 's/^DSTATE=//p' | tail -1)
clean_ok "${ER:-x}" "${GF:-x}" "$DS" || { echo "ABORT: 板卡不洁净（ERR=$ER GRF=$GF D=[$DS]）⇒ 拉安灯"; bash "$ROOT/t4/m2c/drivers/t4gate.sh" andon "rb2-preclean-ERR${ER}-GRF${GF}" | tail -3; exit 5; }

log "--- 规则 D：预算预检（本 boot 两臂：g0 → g1，须在同一 boot 内配对）---"
PL=$(bash "$ROOT/t4/m2c/drivers/t4gate.sh" preflight --max-arms 2 2>&1); PRC=$?
printf '%s\n' "$PL" > "$EV/04_preflight.txt"; printf '%s\n' "$PL" | tail -1
[ "$PRC" -eq 0 ] || { echo "PREFLIGHT=BLOCKED ⇒ 不开臂（等新 boot；这是规则 D 的机械拦截，不是建议）"; exit 4; }

if [ "$DRY" = 1 ]; then
  echo "--- DRY-RUN：以下两臂会依次执行（各 6 次内核调用：smoke/main/vary5/copy/grid7/grid1）---"
  echo "  1) bash rb2_arm.sh rb2-g0 g0   # 16 B/次，期望 MAIN_GBPS ≥249.4（锚点）"
  echo "  2) bash rb2_arm.sh rb2-g1 g1   #  4 B/次，判据 ≤140 ⇒ H1 成立 / ≥200 ⇒ H1 证伪"
  echo "DRY_RUN=DONE（未上传、未开臂）"; exit 0
fi

log "--- 上传（板端核 sha：g0/g1/host 三件）---"
bash "$ROOT/t4/m2c/drivers/upload_drivers.sh" | tail -2
UP=$(bash "$ROOT/t4/m2c/drivers/rb2_upload.sh" 2>&1); printf '%s\n' "$UP" > "$EV/05_upload.txt"; printf '%s\n' "$UP" | tail -4
grep -q 'RB2_UPLOAD=OK' "$EV/05_upload.txt" || { echo "ABORT: 工件上传/板端 sha 校验失败"; exit 3; }
bash "$SSH" "for f in arm_gate.sh rb2_arm.sh rb1_metrics.py; do [ -f $R/\$f ] && echo \"  OK \$f\" || echo \"  MISSING \$f\"; done" 2>/dev/null | tee "$EV/06_board-files.txt"
grep -q MISSING "$EV/06_board-files.txt" && { echo "ABORT: 板端缺件"; exit 3; }

arm_one() {   # <variant>  产出 $EV/rb2-<v>.{out,arm.done,metrics.tsv}
  local v=$1 tag=rb2-$1
  log "=== ARM $tag（6 次内核调用）$(date -Iseconds) ==="
  local out; out=$(timeout 2700 bash "$SSH" "bash $R/rb2_arm.sh $tag $v" 2>&1)
  printf '%s\n' "$out" > "$EV/$tag.arm.out"
  printf '%s\n' "$out" | sed 's/^/ARM /' | tail -12
  local mark; mark=$(bash "$SSH" "cat $R/logs/$tag.arm.done 2>/dev/null" 2>/dev/null | tail -1)
  echo "ARM $tag MARK=${mark:-<none>}"
  for f in $tag.out $tag.arm.done $tag.gate.pre $tag.gate.post; do
    bash -c 'scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "user@${BOARD_ADDR:?set BOARD_ADDR to your board IP}:'"$R"'/logs/'"$f"'" "'"$EV"'/"' 2>/dev/null || echo "  (缺 $f)"
  done
  bash "$ROOT/t4/m2c/drivers/t4gate.sh" sync | tail -1
  python3 "$ROOT/t4/m2c/host/rb1_metrics.py" "$EV/$tag.out" > "$EV/$tag.metrics.tsv" 2>&1; local rc=$?
  sed 's/^/  /' "$EV/$tag.metrics.tsv"
  [ "$rc" -eq 0 ] || { echo "ABORT: $tag 无读数 ⇒ 不判决；拉安灯等现场断电"; bash "$ROOT/t4/m2c/drivers/t4gate.sh" andon "rb2-$v-no-metrics" | tail -3; return 3; }
  case "${mark:-}" in
    OK) return 0 ;;
    ABORT*) echo "ABORT: $tag pre 闸门 BLOCKED ⇒ 无读数"; return 2 ;;
    EFFMISS*) echo "BAD: $tag 生效串缺失 ⇒ 跑的不是这版内核；拉安灯"; bash "$ROOT/t4/m2c/drivers/t4gate.sh" andon "rb2-$v-effmiss" | tail -3; return 1 ;;
    *) echo "ARM $tag GATE=BAD ⇒ 停手：不重跑、不 pkill；拉安灯等现场断电"; bash "$ROOT/t4/m2c/drivers/t4gate.sh" andon "rb2-$tag-gate-not-ok" | tail -3; return 1 ;;
  esac
}

# cred 行只由**承重判据**写：g0 锚点 + g1 的 H0 侧（两者都要求读数本身达标）。
#   对立判据（h1）**不得**写 cred：两支必须恰有一个 FAIL，若把 FAIL 那支记成 V4，台账就会把**有效读数**标成不可用
#   （2026-09-17 11:49 实测踩到，已重判）。cred 描述的是「读数能不能用」，不是「假设成不成立」。
V0(){ python3 "$ROOT/t4/m2c/host/verdict.py" --spec "$1" --metrics "$2" --label "$3" --cred-out "$ROOT/evidence/data-credibility.tsv"; }
VJ(){ python3 "$ROOT/t4/m2c/host/verdict.py" --spec "$1" --metrics "$2" --label "$3"; }

arm_one g0 || exit $?
log "--- 判据 1：g0 锚点（复现归档口径 ≥249.4 GB/s）---"
V0 "$FX/rb2_g0.spec" "$EV/rb2-g0.metrics.tsv" rb2-g0 | tee "$EV/rb2-g0.VERDICT.txt"; G0RC=${PIPESTATUS[0]}
[ "$G0RC" -eq 0 ] || { echo "ANCHOR_FAIL：gran0 没复现归档口径 ⇒ 本 boot 读数不可用于 H1 判决（构建/板卡有问题），不开第 2 臂"; exit 6; }

arm_one g1 || exit $?
log "--- 判据 2：g1 判别（H1: ≤140 GB/s 成立 / H0: ≥200 GB/s 证伪）---"
VJ "$FX/rb2_g1_h1.spec" "$EV/rb2-g1.metrics.tsv" rb2-g1-h1 > "$EV/rb2-g1.h1.txt"; H1RC=$?
V0 "$FX/rb2_g1_h0.spec" "$EV/rb2-g1.metrics.tsv" rb2-g1 > "$EV/rb2-g1.h0.txt"; H0RC=$?
sed 's/^/  /' "$EV/rb2-g1.h1.txt"; sed 's/^/  /' "$EV/rb2-g1.h0.txt"
GB=$(awk -F'\t' '$1=="RB1_MAIN_GBPS"{print $2}' "$EV/rb2-g1.metrics.tsv")
if [ "$H1RC" -eq 0 ]; then R=H1_CONFIRMED; elif [ "$H0RC" -eq 0 ]; then R=H1_REFUTED; else R=INDETERMINATE; fi
echo "RB2_RESULT=$R gran1_main_gbps=$GB g0_main_gbps=$(awk -F'\t' '$1=="RB1_MAIN_GBPS"{print $2}' "$EV/rb2-g0.metrics.tsv")" | tee "$EV/RESULT.txt"
( cd "$EV" && sha256sum -b ./* > SHA256SUMS 2>/dev/null )
echo "=== RB2 DONE $(date -Iseconds) result=$R evidence=$EV ==="
[ "$R" != INDETERMINATE ] || exit 7
exit 0
