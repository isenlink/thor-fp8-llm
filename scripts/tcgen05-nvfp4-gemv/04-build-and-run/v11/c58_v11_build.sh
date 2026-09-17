#!/usr/bin/env bash
# c58_v11_build.sh — v11 构建（宿主侧：nvcc/ptxas/SASS + aarch64 交叉编译，零板端动作）
set -u
SRC=/path/to/project/t4/v11/src
V5=/path/to/project/evidence/20260916-67-t4-3-b5-nvfp4
T4C=/path/to/project/t4/common
OUT=${OUT:-/tmp/c58v11work}
CPS=${CPS:-}
DT_CPS=""
[ -n "$CPS" ] && DT_CPS="-DT4_V10R_CPS=$CPS"
EXTRA=${EXTRA:-}
CUDA=/usr/local/cuda-12.8/bin
CUT=/tmp/cutlass-main
GXX=$(command -v aarch64-linux-gnu-g++-14 || command -v aarch64-linux-gnu-g++)
mkdir -p "$OUT"
log(){ echo "[$(date '+%F %T')] $*"; }
ABORTED=0
log "=== C58 v11 (GGML pure-copy real tensor) BUILD START (GXX=$GXX) ==="
$CUDA/nvcc -ptx -arch=sm_101a -O2 -std=c++17 $DT_CPS $EXTRA -I"$CUT/include" -I"$V5/src" -I"$T4C" "$SRC/c58_v11_ggml.cu" -o "$OUT/c58_v11.ptx" > "$OUT/01_nvcc_ptx.log" 2>&1
rc=$?; log "NVCC_PTX_RC=$rc"
if [ $rc -ne 0 ]; then ABORTED=1; tail -40 "$OUT/01_nvcc_ptx.log"; fi
if [ $rc -eq 0 ]; then
  $CUDA/ptxas -arch=sm_101a -O2 "$OUT/c58_v11.ptx" -o "$OUT/c58_v11.cubin" > "$OUT/02_ptxas.log" 2>&1
  rc=$?; log "PTXAS_RC=$rc"
  if [ $rc -ne 0 ]; then ABORTED=1; tail -40 "$OUT/02_ptxas.log"; fi
  [ $rc -eq 0 ] && log "CUBIN_BYTES=$(stat -c%s "$OUT/c58_v11.cubin")"
fi
if [ $rc -eq 0 ]; then
  $CUDA/cuobjdump -sass "$OUT/c58_v11.cubin" > "$OUT/03_sass.txt" 2>&1
  $CUDA/cuobjdump -res-usage "$OUT/c58_v11.cubin" > "$OUT/04_res_usage.txt" 2>&1
  log "SASS: LDGSTS=$(grep -c 'LDGSTS' "$OUT/03_sass.txt") UTCMMA=$(grep -c 'UTCMMA\|UTCOMMA' "$OUT/03_sass.txt") UTCST=$(grep -c 'UTCST' "$OUT/03_sass.txt") BAR=$(grep -c 'BAR.SYNC' "$OUT/03_sass.txt")"
  grep -c 'LDGSTS.E.BYPASS.128' "$OUT/03_sass.txt" | sed 's/^/  SASS LDGSTS.128 = /'
  grep -c 'LDGSTS.E.BYPASS.32'  "$OUT/03_sass.txt" | sed 's/^/  SASS LDGSTS.32  = /'
  cat "$OUT/04_res_usage.txt" | sed 's/^/  RES /'
fi
"$GXX" -std=c++17 -O2 $DT_CPS $EXTRA -I/usr/local/cuda-12.8/include -I"$T4C" "$SRC/c58_v11_host.cpp" -o "$OUT/c58_v11_host" -ldl -lstdc++ > "$OUT/05_gxx_aarch64.log" 2>&1
rc6=$?; log "GXX_AARCH64_RC=$rc6"
if [ $rc6 -ne 0 ]; then ABORTED=1; tail -30 "$OUT/05_gxx_aarch64.log"; else
  log "HOST_BIN_BYTES=$(stat -c%s "$OUT/c58_v11_host")"
fi
log "SHA: CUBIN=$(sha256sum "$OUT/c58_v11.cubin" 2>/dev/null|cut -c1-16) HOST=$(sha256sum "$OUT/c58_v11_host" 2>/dev/null|cut -c1-16)"
log "=== C58 v11 BUILD DONE (ABORTED=$ABORTED) ==="
exit $ABORTED
