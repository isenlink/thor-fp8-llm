#!/usr/bin/env bash
# rb2_upload.sh — 传 R-B2 判别臂工件到 board1:/tmp/t4work/rb2（新 boot 后 /tmp 清空 ⇒ 每次重传）
#   sha 期望值**只从 stage/rb2/expected.tsv 读**（构建脚本产出；本脚本不硬编码任何 sha）
set -u
cd "$(dirname "$0")"
ROOT=$(cd ../../.. && pwd)
S=$ROOT/t4/m2c/stage/rb2
EV=$ROOT/evidence/20260916-76-t4-3-b9f-ws32k/host
SSH=${THOR_SSH:-${BOARD_SSH:?set BOARD_SSH to your ssh wrapper}}
D=/tmp/t4work/rb2
R=/tmp/t4work/m2c
THOR_ADDR=${BOARD_ADDR:?set BOARD_ADDR to your board IP}
scp1() { scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$1" "${THOR_USER:-user}@$THOR_ADDR:$2" || { echo "SCP_FAIL $2"; return 3; }; }
for f in "$S/gran0.cubin" "$S/gran1.cubin" "$S/prod.cubin" "$S/expected.tsv" "$EV/c58_v9f_host.stream1" "$S/c58_v9f_host_prod" rb2_arm.sh rb2_mem_arm.sh "$ROOT/t4/m2c/host/rb1_metrics.py"; do
  [ -f "$f" ] || { echo "MISSING $f"; exit 3; }
done
bash "$SSH" "mkdir -p $D $R/logs" || exit 3
scp1 "$S/gran0.cubin"        "$D/gran0.cubin"      || exit 3
scp1 "$S/gran1.cubin"        "$D/gran1.cubin"      || exit 3
scp1 "$S/prod.cubin"         "$D/prod.cubin"       || exit 3
scp1 "$S/expected.tsv"       "$D/expected.tsv"     || exit 3
scp1 "$EV/c58_v9f_host.stream1" "$D/c58_v9f_host"  || exit 3
scp1 "$S/c58_v9f_host_prod"  "$D/c58_v9f_host_prod" || exit 3
scp1 rb2_arm.sh              "$R/"                 || exit 3
scp1 rb2_mem_arm.sh          "$R/"                 || exit 3
scp1 "$ROOT/t4/m2c/host/rb1_metrics.py" "$R/"      || exit 3
bash "$SSH" '
 D=/tmp/t4work/rb2; R=/tmp/t4work/m2c
 chmod +x $D/c58_v9f_host $D/c58_v9f_host_prod $R/rb2_arm.sh $R/rb2_mem_arm.sh
 bad=0
 for k in g0 g1 prod host host2; do
   exp=$(awk -F"\t" -v k=$k "\$1==k{print \$2\" \"\$3}" $D/expected.tsv)
   sha=${exp% *}; fn=${exp#* }
   got=$(sha256sum $D/$fn 2>/dev/null | cut -c1-16)
   [ "$got" = "$sha" ] && echo "  SHA_OK  $k $fn=$got" || { echo "  SHA_BAD $k $fn=$got != $sha"; bad=1; }
 done
 [ "$bad" = 0 ] && echo RB2_UPLOAD=OK || echo RB2_UPLOAD=SHA_MISMATCH'
