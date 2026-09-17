#!/bin/bash
# rb2_mem_driver.sh — R-B2「H-mem 生产寻址」判别臂宿主编排（A 安灯 / B 每臂闸门 / C 可信度 / D 启停预算）
#   待判的问题：原型 ws32k（4 B/次、线性地址）257 GB/s vs 生产腿 136.5 GB/s 的 1.66× 缺口，
#   是不是**真实权重寻址**（32 行 × row_stride 2880、每 64-K 记录 36 B）本身造成的？
#   设计：同一份源码第三个变体（T4_V9F_PRODADDR=1，结构计数与 gran1 全等 ⇒ 单变量 = 取数寻址）
#         + 新宿主 binary（stream / prodlay 两模式）
#   判据（上板前冻结，见 host/verdict_fixtures/rb2_mem_{anchor,h1,h0,integrity}.spec）：
#     臂1 anchor：stream + gran1 ⇒ MAIN_GBPS ≥ 220（复现 226.85；不达标 ⇒ 本 boot 读数不得用于判决）
#     臂2 prod  ：prodlay + prod ⇒ ≤ 170 ⇒ H-mem **成立**（目标不可达、R-B 停线）
#                                 ≥ 220 ⇒ H-mem **否证**（布局无罪）；170–220 ⇒ INDETERMINATE
#   链路：build(5 个门) → selftest → 判据 replay ×4 → 洁净 → preflight(D) → 上传(板端核 5 键 sha)
#         → 两臂（各 6 次内核调用，硬超时 300 s）→ 回取 → rb1_metrics.py → verdict + 台账 sync
#   BAD/异常 ⇒ 自动 t4gate.sh andon（规则 A：同回合处置）
# 用法: bash t4/m2c/drivers/rb2_mem_driver.sh [--dry-run]
set -u
ROOT=/path/to/project
SSH=${THOR_SSH:-${BOARD_SSH:?set BOARD_SSH to your ssh wrapper}}
R=/tmp/t4work/m2c
FX=$ROOT/t4/m2c/host/verdict_fixtures
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
if [ "$DRY" = 1 ]; then EV=/tmp/rb2_mem_dryrun; else EV=$ROOT/evidence/$(date +%Y%m%d-%H%M)-rb2-mem; fi
mkdir -p "$EV"
log(){ echo "[$(date '+%F %T')] $*"; }
clean_ok() { [ "${1:-x}" = 0 ] && [ "${2:-x}" = 0 ] && [ -z "${3:-}" ]; }
echo "=== RB2-MEM DRIVER START $(date -Iseconds) dry_run=$DRY evidence=$EV ==="

log "--- 构建门：5 条（gran0 同一性 / gran1 粒度生效 / prod 结构全等 / 生产布局一致 / GGUF 行步长）---"
bash "$ROOT/t4/m2c/drivers/rb2_build.sh" > "$EV/00_build.txt" 2>&1 || { echo "BUILD_GATE_FAIL"; tail -8 "$EV/00_build.txt"; exit 9; }
for k in BASE_IDENTICAL GRAN4_EFFECTIVE PRODADDR_STRUCTURAL_PARITY OK\ LAYOUT_IDENTICAL OK\ ROWSTRIDE_EXPECT; do
  grep -q "$k" "$EV/00_build.txt" || { echo "BUILD_GATE_FAIL（缺 $k）"; tail -12 "$EV/00_build.txt"; exit 9; }
done
python3 "$ROOT/t4/m2c/host/check_prodaddr_fill.py" > "$EV/00b_fill_identity.txt" 2>&1 || { echo "BUILD_GATE_FAIL：宿主↔内核填充恒等式不成立"; cat "$EV/00b_fill_identity.txt"; exit 9; }
sed 's/^/  /' "$EV/00b_fill_identity.txt"
sed 's/^/  /' "$EV/00_build.txt" | tail -16

log "--- 夹具回归（新逻辑先过夹具；fail 必须为 0）---"
bash "$ROOT/t4/m2c/host/selftest.sh" > "$EV/01_selftest.txt" 2>&1 || { echo "SELFTEST FAIL ⇒ 拒绝上板"; tail -5 "$EV/01_selftest.txt"; exit 9; }
SF=$(grep -o 'pass=[0-9]* fail=[0-9]*' "$EV/01_selftest.txt" | tail -1); echo "SELFTEST $SF"
echo "$SF" | grep -q 'fail=0' || { echo "SELFTEST 有 FAIL ⇒ 拒绝上板"; exit 9; }

log "--- 规则 B：四条判据的可用性预检（好/坏样本判定必须不同）---"
rp(){ python3 "$ROOT/t4/m2c/host/verdict.py" --spec "$1" --replay "$2" "$3"; }
{ rp "$FX/rb2_mem_anchor.spec"    "$FX/rb2_mem_prod_high.metrics" "$FX/rb2_mem_prod_low.metrics"
  rp "$FX/rb2_mem_h1.spec"        "$FX/rb2_mem_prod_low.metrics"  "$FX/rb2_mem_prod_high.metrics"
  rp "$FX/rb2_mem_h0.spec"        "$FX/rb2_mem_prod_high.metrics" "$FX/rb2_mem_prod_low.metrics"
  rp "$FX/rb2_mem_integrity.spec" "$FX/rb2_mem_prod_high.metrics" "$FX/rb2_mem_prod_dm.metrics"; } > "$EV/02_replay.txt" 2>&1
sed 's/^/  /' "$EV/02_replay.txt"
[ "$(grep -c 'REPLAY=OK' "$EV/02_replay.txt")" = 4 ] || { echo "ABORT: 判据无判别力（REPLAY != 4 个 OK）"; exit 9; }

log "--- 洁净检查（ERR/GR 致命/D 态 必须全零）---"
P=$(bash "$SSH" 'echo "BOOT=$(cat /proc/sys/kernel/random/boot_id)"; echo "UP=$(cut -d" " -f1 /proc/uptime)";
   echo "ERR=$(sudo -n dmesg | grep -c "\[ERR\]")";
   echo "GRF=$(sudo -n dmesg | grep -cE "fecs method|failed to reset gr|timeout gr busy|runlist [0-9]+ preempt|sm err state|gr_init.*fail")";
   echo "DSTATE=$(for p in /proc/[0-9]*; do s=$(awk "{print \$3}" "$p/stat" 2>/dev/null) || continue; case "$s" in D*) [ -e "$p/exe" ] && printf "%s," "$(cat $p/comm 2>/dev/null)";; esac; done)"' 2>/dev/null)
printf '%s\n' "$P" > "$EV/03_boot-check.txt"; printf '%s\n' "$P" | sed 's/^/  /'
ER=$(printf '%s\n' "$P" | sed -n 's/^ERR=//p' | tail -1); GF=$(printf '%s\n' "$P" | sed -n 's/^GRF=//p' | tail -1)
DS=$(printf '%s\n' "$P" | sed -n 's/^DSTATE=//p' | tail -1)
# 板卡**不可达**（断电/SSH 不通）与"不洁净"是两回事：不可达 ⇒ 只停手，**不拉安灯**（否则会给自己挂停线牌）
if [ -z "$(printf '%s' "$P" | tr -d '[:space:]')" ]; then
  echo "BOARD_UNREACHABLE：SSH 无任何输出（板子断电或链路不通）⇒ 不开臂、不拉安灯"; exit 5
fi
clean_ok "${ER:-x}" "${GF:-x}" "$DS" || { echo "ABORT: 板卡不洁净（ERR=$ER GRF=$GF D=[$DS]）⇒ 拉安灯"; bash "$ROOT/t4/m2c/drivers/t4gate.sh" andon "rb2mem-preclean-ERR${ER}-GRF${GF}" | tail -3; exit 5; }

log "--- 规则 D：预算预检（本 boot 两臂：anchor → prod，须在同一 boot 内配对）---"
PL=$(bash "$ROOT/t4/m2c/drivers/t4gate.sh" preflight --max-arms 2 2>&1); PRC=$?
printf '%s\n' "$PL" > "$EV/04_preflight.txt"; printf '%s\n' "$PL" | tail -1
[ "$PRC" -eq 0 ] || { echo "PREFLIGHT=BLOCKED ⇒ 不开臂（等新 boot；规则 D 的机械拦截）"; exit 4; }

if [ "$DRY" = 1 ]; then
  echo "--- DRY-RUN：以下两臂会依次执行（各 6 次内核调用：smoke/main/vary5/copy/grid7/grid1）---"
  echo "  1) bash rb2_mem_arm.sh rb2mem-anchor anchor  # stream  + gran1 ⇒ 期望 MAIN_GBPS ≥220"
  echo "  2) bash rb2_mem_arm.sh rb2mem-prod   prod    # prodlay + prod  ⇒ ≤170 成立 / ≥220 否证"
  echo "DRY_RUN=DONE（未上传、未开臂）"; exit 0
fi

log "--- 上传（板端核 5 键 sha：g0/g1/prod/host/host2）---"
bash "$ROOT/t4/m2c/drivers/upload_drivers.sh" | tail -2
UP=$(bash "$ROOT/t4/m2c/drivers/rb2_upload.sh" 2>&1); printf '%s\n' "$UP" > "$EV/05_upload.txt"; printf '%s\n' "$UP" | tail -7
grep -q 'RB2_UPLOAD=OK' "$EV/05_upload.txt" || { echo "ABORT: 工件上传/板端 sha 校验失败"; exit 3; }
bash "$SSH" "for f in arm_gate.sh rb2_arm.sh rb2_mem_arm.sh rb1_metrics.py; do [ -f $R/\$f ] && echo \"  OK \$f\" || echo \"  MISSING \$f\"; done" 2>/dev/null | tee "$EV/06_board-files.txt"
grep -q MISSING "$EV/06_board-files.txt" && { echo "ABORT: 板端缺件"; exit 3; }

arm_one() {   # <variant: anchor|prod>  产出 $EV/rb2mem-<v>.{out,arm.done,metrics.tsv}
  local v=$1 tag=rb2mem-$1
  log "=== ARM $tag（6 次内核调用）$(date -Iseconds) ==="
  local out; out=$(timeout 2700 bash "$SSH" "bash $R/rb2_mem_arm.sh $tag $v" 2>&1)
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
  [ "$rc" -eq 0 ] || { echo "ABORT: $tag 无读数 ⇒ 不判决；拉安灯等现场断电"; bash "$ROOT/t4/m2c/drivers/t4gate.sh" andon "rb2mem-$v-no-metrics" | tail -3; return 3; }
  case "${mark:-}" in
    OK) return 0 ;;
    ABORT*) echo "ABORT: $tag pre 闸门 BLOCKED ⇒ 无读数"; return 2 ;;
    EFFMISS*) echo "BAD: $tag 生效串缺失 ⇒ 跑的不是这版内核；拉安灯"; bash "$ROOT/t4/m2c/drivers/t4gate.sh" andon "rb2mem-$v-effmiss" | tail -3; return 1 ;;
    *) echo "ARM $tag GATE=BAD ⇒ 停手：不重跑、不 pkill；拉安灯等现场断电"; bash "$ROOT/t4/m2c/drivers/t4gate.sh" andon "rb2mem-$tag-gate-not-ok" | tail -3; return 1 ;;
  esac
}

arm_one anchor || exit $?
log "--- 判据 1：anchor（新宿主 stream 模式必须复现 gran1 的 226.85，band −3%）---"
python3 "$ROOT/t4/m2c/host/verdict.py" --spec "$FX/rb2_mem_anchor.spec" --metrics "$EV/rb2mem-anchor.metrics.tsv" \
  --label rb2-mem-anchor --cred-out "$ROOT/evidence/data-credibility.tsv" | tee "$EV/rb2mem-anchor.VERDICT.txt"; ARC=${PIPESTATUS[0]}
[ "$ARC" -eq 0 ] || { echo "ANCHOR_FAIL：新宿主/新臂脚本没复现出 gran1 的读数 ⇒ 本 boot 的 prod 臂不可用于判决（不开第 2 臂）"; exit 6; }

arm_one prod || exit $?
log "--- 判据 2：prod（H-mem：≤170 成立 / ≥220 否证）---"
# 读数可用性只认 integrity 判据；对立判据（h1/h0）**不写 cred 行**（DISCIPLINE §10）
python3 "$ROOT/t4/m2c/host/verdict.py" --spec "$FX/rb2_mem_integrity.spec" --metrics "$EV/rb2mem-prod.metrics.tsv" \
  --label rb2-mem-prod --cred-out "$ROOT/evidence/data-credibility.tsv" > "$EV/rb2mem-prod.integrity.txt" 2>&1
sed 's/^/  /' "$EV/rb2mem-prod.integrity.txt"
python3 "$ROOT/t4/m2c/host/verdict.py" --spec "$FX/rb2_mem_h1.spec" --metrics "$EV/rb2mem-prod.metrics.tsv" --label rb2-mem-h1 > "$EV/rb2mem-prod.h1.txt" 2>&1; H1RC=$?
python3 "$ROOT/t4/m2c/host/verdict.py" --spec "$FX/rb2_mem_h0.spec" --metrics "$EV/rb2mem-prod.metrics.tsv" --label rb2-mem-h0 > "$EV/rb2mem-prod.h0.txt" 2>&1; H0RC=$?
sed 's/^/  /' "$EV/rb2mem-prod.h1.txt"; sed 's/^/  /' "$EV/rb2mem-prod.h0.txt"
GB=$(awk -F'\t' '$1=="RB1_MAIN_GBPS"{print $2}' "$EV/rb2mem-prod.metrics.tsv")
GA=$(awk -F'\t' '$1=="RB1_MAIN_GBPS"{print $2}' "$EV/rb2mem-anchor.metrics.tsv")
if [ "$H1RC" -eq 0 ]; then R=HMEM_CONFIRMED; elif [ "$H0RC" -eq 0 ]; then R=HMEM_REFUTED; else R=INDETERMINATE; fi
echo "RB2_MEM_RESULT=$R anchor_main_gbps=$GA prod_main_gbps=$GB" | tee "$EV/RESULT.txt"
( cd "$EV" && sha256sum -b ./* > SHA256SUMS 2>/dev/null )
echo "=== RB2-MEM DONE $(date -Iseconds) result=$R evidence=$EV ==="
[ "$R" != INDETERMINATE ] || exit 7
exit 0
