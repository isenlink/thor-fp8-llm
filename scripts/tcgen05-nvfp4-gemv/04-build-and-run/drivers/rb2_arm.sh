#!/bin/bash
# rb2_arm.sh — R-B2 判别臂（板端）：同一宿主驱动 + 同一臂序，唯一变量 = cubin
#   gran0 = 归档 ws32k 口径（16 B/次 cp.async，288 请求/4608B chunk）
#   gran1 = 生产腿口径（4 B/次 cp.async，1152 请求/chunk；SASS 已核 LDGSTS 9→36）
#   臂序与 rb1 逐条相同（smoke/main/vary5/copy/grid7/grid1）⇒ g0 与 g1 严格配对
#   变体→(cubin,sha) 的映射只从 /tmp/t4work/rb2/expected.tsv 读（不硬编码）
# 用法: rb2_arm.sh <tag> <g0|g1>      产物: logs/<tag>.out logs/<tag>.arm.done(OK|BAD|ABORT)
set -u
R=/tmp/t4work/m2c
D=/tmp/t4work/rb2
L=$R/logs; mkdir -p "$L"
TAG=${1:-rb2}
V=${2:-g0}
OUT=$L/$TAG.out; DONE=$L/$TAG.arm.done
: > "$DONE"
abort() { echo "ABORT: $1"; echo "ABORT $2" >> "$DONE"; exit 2; }
for f in expected.tsv c58_v9f_host; do [ -f "$D/$f" ] || abort "缺 $D/$f（新 boot 后 /tmp 清空 ⇒ 先 rb2_upload.sh）" "missing_$f"; done
row=$(awk -F'\t' -v v="$V" '$1==v{print $2"\t"$3}' "$D/expected.tsv")
[ -n "$row" ] || abort "expected.tsv 里没有变体 $V" "no_variant_$V"
EXP_SHA=${row%%	*}; CUBIN=${row##*	}
HOST_SHA=$(awk -F'\t' '$1=="host"{print $2}' "$D/expected.tsv")
[ -f "$D/$CUBIN" ] || abort "缺 $D/$CUBIN" "missing_cubin"
sc=$(sha256sum "$D/$CUBIN" | cut -c1-16); sh=$(sha256sum "$D/c58_v9f_host" | cut -c1-16)
[ "$sc" = "$EXP_SHA" ] || abort "$CUBIN sha=$sc != $EXP_SHA" "cubin_sha"
[ "$sh" = "$HOST_SHA" ] || abort "host sha=$sh != $HOST_SHA" "host_sha"
chmod +x "$D/c58_v9f_host"
# pre 闸门：**只认两种明确结局**（BLOCKED ⇒ 停；出现 GATE4_PRE 行 ⇒ 放行）。
#  旧写法 `[ $? -ne 0 ] && grep -q BLOCKED` 有静默放行洞：门脚本缺失/崩溃时既非 0 也 grep 不到 BLOCKED ⇒ 照样开臂。
[ -f "$R/arm_gate.sh" ] || abort "缺 $R/arm_gate.sh（门脚本不在 ⇒ 不许开臂）" "missing_gate"
PREOUT=$(bash $R/arm_gate.sh pre "$TAG" 2>&1); printf '%s\n' "$PREOUT" > "$L/$TAG.gate.pre"
if printf '%s\n' "$PREOUT" | grep -q 'GATE4_PRE=BLOCKED'; then
  echo "RB2_ARM_ABORT tag=$TAG reason=pre_blocked"; echo "ABORT pre_blocked" >> "$DONE"; exit 3
fi
if ! printf '%s\n' "$PREOUT" | grep -q 'GATE4_PRE boot='; then
  echo "RB2_ARM_ABORT tag=$TAG reason=pre_gate_not_run (输出: $(printf '%s' "$PREOUT" | head -2 | tr '\n' ' '))"
  echo "ABORT pre_gate_not_run" >> "$DONE"; exit 3
fi
echo "[$TAG] PRE $(tail -1 "$L/$TAG.gate.pre") variant=$V cubin=$CUBIN sha=$sc"
run() {  # <arm> <chunks> <grid> <flags>
  local a=$1; shift
  echo "ARM_BEGIN $a"
  ( cd "$D" && timeout 300 ./c58_v9f_host stream "$CUBIN" "$@" )   # 输出由外层块统一落 $OUT，别在这里再 tee
  echo "ARM_END $a rc=$?"
}
{
echo "# rb2_arm.sh tag=$TAG variant=$V cubin=$CUBIN cubin_sha=$sc host_sha=$sh $(date -Iseconds)"
run smoke  64   1  30000 3
run main   4096 14 30000 1
run vary5  4096 14 30000 5
run copy   4096 14 30000 0
run grid7  4096 7  30000 1
run grid1  4096 1  30000 1
} >> "$OUT" 2>&1
POST=$(bash $R/arm_gate.sh post "$TAG"); echo "$POST"
grep -a '"tag":"env".*"smem_required":114752' "$OUT" > /dev/null || {
  echo "RB2_ARM_EFFMISS tag=$TAG 生效串缺失（smem_required 非 114752 ⇒ 跑的不是这版内核）"
  echo "EFFMISS smem" >> "$DONE"; exit 5; }
if echo "$POST" | grep -q 'GATE4=OK'; then
  echo "OK" >> "$DONE"; echo "RB2_ARM_DONE tag=$TAG gate=OK variant=$V"; exit 0
fi
echo "BAD" >> "$DONE"; echo "RB2_ARM_DONE tag=$TAG gate=BAD variant=$V"; exit 1
