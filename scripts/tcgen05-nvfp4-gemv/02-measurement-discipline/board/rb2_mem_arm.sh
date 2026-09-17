#!/bin/bash
# rb2_mem_arm.sh — R-B2「H-mem 生产寻址」判别臂（板端）：**新宿主 binary** 的两个模式
#   anchor : mode=stream   + gran1 cubin  ⇒ 必须复现 gran1 的 226.85（证明新宿主/新臂脚本没引入偏差）
#   prod   : mode=prodlay  + prod cubin  ⇒ 生产寻址（32 行 × row_stride 2880、36 B 记录、4 B/次）
#   变体→(mode, cubin, sha) 映射只从 /tmp/t4work/rb2/expected.tsv 读（不硬编码 sha）
#   臂序与 gran 臂逐条相同（smoke/main/vary5/copy/grid7/grid1）⇒ 读数可与 257.69 / 226.85 并列
# 用法: rb2_mem_arm.sh <tag> <anchor|prod>   产物: logs/<tag>.out logs/<tag>.arm.done(OK|BAD|ABORT)
set -u
R=/tmp/t4work/m2c
D=/tmp/t4work/rb2
L=$R/logs; mkdir -p "$L"
TAG=${1:-rb2mem}
V=${2:-anchor}
OUT=$L/$TAG.out; DONE=$L/$TAG.arm.done
: > "$DONE"
abort() { echo "ABORT: $1"; echo "ABORT $2" >> "$DONE"; exit 2; }
case "$V" in
  anchor) MODE=stream;  CK=g1;   HK=host2 ;;
  prod)   MODE=prodlay; CK=prod; HK=host2 ;;
  *) abort "未知变体 $V（只认 anchor|prod）" "bad_variant" ;;
esac
HOSTBIN=c58_v9f_host_prod
for f in expected.tsv $HOSTBIN; do [ -f "$D/$f" ] || abort "缺 $D/$f（新 boot 后 /tmp 清空 ⇒ 先 rb2_upload.sh）" "missing_$f"; done
row=$(awk -F'\t' -v v="$CK" '$1==v{print $2"\t"$3}' "$D/expected.tsv")
[ -n "$row" ] || abort "expected.tsv 里没有 cubin 变体 $CK" "no_variant_$CK"
EXP_SHA=${row%%	*}; CUBIN=${row##*	}
HOST_SHA=$(awk -F'\t' -v v="$HK" '$1==v{print $2}' "$D/expected.tsv")
[ -n "$HOST_SHA" ] || abort "expected.tsv 里没有宿主变体 $HK" "no_host_$HK"
[ -f "$D/$CUBIN" ] || abort "缺 $D/$CUBIN" "missing_cubin"
sc=$(sha256sum "$D/$CUBIN" | cut -c1-16); sh=$(sha256sum "$D/$HOSTBIN" | cut -c1-16)
[ "$sc" = "$EXP_SHA" ] || abort "$CUBIN sha=$sc != $EXP_SHA" "cubin_sha"
[ "$sh" = "$HOST_SHA" ] || abort "$HOSTBIN sha=$sh != $HOST_SHA" "host_sha"
chmod +x "$D/$HOSTBIN"
[ -f "$R/arm_gate.sh" ] || abort "缺 $R/arm_gate.sh（门脚本不在 ⇒ 不许开臂）" "missing_gate"
PREOUT=$(bash $R/arm_gate.sh pre "$TAG" 2>&1); printf '%s\n' "$PREOUT" > "$L/$TAG.gate.pre"
if printf '%s\n' "$PREOUT" | grep -q 'GATE4_PRE=BLOCKED'; then
  echo "RB2_MEM_ARM_ABORT tag=$TAG reason=pre_blocked"; echo "ABORT pre_blocked" >> "$DONE"; exit 3
fi
if ! printf '%s\n' "$PREOUT" | grep -q 'GATE4_PRE boot='; then
  echo "RB2_MEM_ARM_ABORT tag=$TAG reason=pre_gate_not_run (输出: $(printf '%s' "$PREOUT" | head -2 | tr '\n' ' '))"
  echo "ABORT pre_gate_not_run" >> "$DONE"; exit 3
fi
echo "[$TAG] PRE $(tail -1 "$L/$TAG.gate.pre") mode=$MODE cubin=$CUBIN sha=$sc host_sha=$sh"
run() {  # <arm> <chunks> <grid> <flags>
  local a=$1; shift
  echo "ARM_BEGIN $a"
  ( cd "$D" && timeout 300 ./$HOSTBIN "$MODE" "$CUBIN" "$@" )
  echo "ARM_END $a rc=$?"
}
{
echo "# rb2_mem_arm.sh tag=$TAG variant=$V mode=$MODE cubin=$CUBIN cubin_sha=$sc host_sha=$sh $(date -Iseconds)"
run smoke  64   1  30000 3
run main   4096 14 30000 1
run vary5  4096 14 30000 5
run copy   4096 14 30000 0
run grid7  4096 7  30000 1
run grid1  4096 1  30000 1
} >> "$OUT" 2>&1
POST=$(bash $R/arm_gate.sh post "$TAG"); echo "$POST"
grep -a '"tag":"env".*"smem_required":114752' "$OUT" > /dev/null || {
  echo "RB2_MEM_ARM_EFFMISS tag=$TAG 生效串缺失（smem_required 非 114752 ⇒ 跑的不是这版内核）"
  echo "EFFMISS smem" >> "$DONE"; exit 5; }
if echo "$POST" | grep -q 'GATE4=OK'; then
  echo "OK" >> "$DONE"; echo "RB2_MEM_ARM_DONE tag=$TAG gate=OK variant=$V mode=$MODE"; exit 0
fi
echo "BAD" >> "$DONE"; echo "RB2_MEM_ARM_DONE tag=$TAG gate=BAD variant=$V"; exit 1
