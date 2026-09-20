#!/bin/bash
# 结构自检：架构 / cubin / 依赖 / 起草器路径是否都在
# 用法: verify.sh <llama-server 路径>
set -uo pipefail
BIN=${1:?用法: verify.sh <llama-server 路径>}
[ -f "$BIN" ] || { echo "VERIFY=FAIL reason=no_such_file path=$BIN"; exit 1; }
fail=0
say(){ printf '%-24s %s\n' "$1" "$2"; }

if file "$BIN" | grep -q 'ARM aarch64'; then say arch aarch64; else say arch "BAD: $(file -b "$BIN" | cut -c1-40)"; fail=1; fi

n=$(strings -n 4 "$BIN" | grep -c 'sm_101a' || true)
if [ "${n:-0}" -gt 0 ]; then say sm_101a_cubin "$n hits"; else say sm_101a_cubin BAD; fail=1; fi

n=$(strings -n 5 "$BIN" | grep -c 'draft-dflash' || true)
if [ "${n:-0}" -gt 0 ]; then say draft_dflash_path yes; else say draft_dflash_path "BAD: 该 build 不支持 DFlash2 起草器"; fail=1; fi

missing=0
for lib in libgomp.so.1 libcudart.so.12 libcublas.so.12 libcublasLt.so.12 libcuda.so.1 libstdc++.so.6 libgcc_s.so.1 libc.so.6; do
  readelf -d "$BIN" 2>/dev/null | grep -q "$lib" || { say "needed:$lib" MISSING; missing=$((missing+1)); }
done
if [ "$missing" = 0 ]; then say needed_libs ok; else fail=1; fi

if ldd "$BIN" >/dev/null 2>&1; then
  n=$(ldd "$BIN" 2>/dev/null | grep -c 'not found' || true)
  if [ "${n:-0}" = 0 ]; then say ldd_resolved ok; else say ldd_resolved "$n not found"; fail=1; fi
else
  say ldd_resolved "skip（宿主架构不同，需在板子上跑）"
fi

if [ "$fail" = 0 ]; then echo "VERIFY=OK"; else echo "VERIFY=FAIL"; fi
exit $fail
