#!/usr/bin/env bash
# upload_drivers.sh — 把板端闸门/驱动上传到 board1:/tmp/t4work/m2c（断电后 /tmp 会丢，需要重传）
# 用法: bash t4/m2c/drivers/upload_drivers.sh
set -u
cd "$(dirname "$0")"
R=/tmp/t4work/m2c
FILES=(arm_gate.sh m2c_run.sh ring_cfg_sweep.sh ares_seq.sh p1_arm.sh p2_oracle.sh p3_arm.sh m2c_serve.sh logp_run.sh logp_probe.py stop8091.sh found1_arm.sh b6_arm.sh step_metrics.py)
rm -f MANIFEST.sha256
for f in "${FILES[@]}"; do
  [ -f "$f" ] || { echo "MISSING $f"; exit 3; }
  sha256sum -b "$f" >> MANIFEST.sha256
done
bash ${THOR_SSH:-${BOARD_SSH:?set BOARD_SSH to your ssh wrapper}} "mkdir -p $R/logs"
bash -c 'scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null MANIFEST.sha256 '"${FILES[*]}"' "${THOR_USER}@${BOARD_ADDR:?set BOARD_ADDR to your board IP}":"/tmp/t4work/m2c/"' || exit 3
bash ${THOR_SSH:-${BOARD_SSH:?set BOARD_SSH to your ssh wrapper}} \
  "cd $R && chmod +x *.sh && sha256sum -c MANIFEST.sha256 2>&1 | grep -v ': OK\$' ; echo VERIFY_DONE; grep -c GATE4 arm_gate.sh; grep -c GATE_MAX_ARMS arm_gate.sh; grep -c 'SHA256\|sha_bin\|3dafb508' p1_arm.sh"
