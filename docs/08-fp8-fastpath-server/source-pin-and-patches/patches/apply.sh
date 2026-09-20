#!/bin/bash
# 把本包补丁集应用到 pin 之后的 llama.cpp 源码树
#   apply.sh <llama.cpp 源码目录> --check    # 只 dry-run（推荐先跑）
#   apply.sh <llama.cpp 源码目录>            # 真应用
# 退出码: 0=全部成功 1=有补丁失败 2=用法错
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
SRC=${1:?用法: apply.sh <llama.cpp 源码目录> [--check]}
MODE=${2:-apply}
[ -d "$SRC/ggml/src/ggml-cuda" ] || { echo "ABORT: $SRC 看起来不是 llama.cpp 源码树"; exit 2; }
shopt -s nullglob
diffs=("$here"/01-as-built/*.diff)
[ ${#diffs[@]} -gt 0 ] || { echo "ABORT: 没有补丁文件（01-as-built/*.diff）"; exit 2; }

use_git=0; [ -d "$SRC/.git" ] && use_git=1
fail=0
for d in "${diffs[@]}"; do
  n=$(basename "$d")
  if [ "$MODE" = "--check" ]; then
    if [ "$use_git" = 1 ]; then git -C "$SRC" apply --check "$d" 2>&1 && echo "CHECK_OK   $n" || { echo "CHECK_FAIL $n"; fail=1; }
    else patch -p1 -d "$SRC" --dry-run <"$d" >/dev/null 2>&1 && echo "CHECK_OK   $n" || { echo "CHECK_FAIL $n"; fail=1; }; fi
  else
    if [ "$use_git" = 1 ]; then git -C "$SRC" apply "$d" 2>&1 && echo "APPLIED    $n" || { echo "APPLY_FAIL $n"; fail=1; }
    else patch -p1 -d "$SRC" <"$d" >/dev/null 2>&1 && echo "APPLIED    $n" || { echo "APPLY_FAIL $n"; fail=1; }; fi
  fi
done
# 新增源文件（不覆盖已存在文件，除非 APPLY_NEW_OVERWRITE=1）
new="$here/01-as-built/new-files"
if [ -d "$new" ]; then
  ( cd "$new" && find . -type f -print0 | while IFS= read -r -d '' f; do
      dst="$SRC/${f#./}"
      if [ -e "$dst" ] && [ "${APPLY_NEW_OVERWRITE:-0}" != "1" ]; then echo "SKIP_EXIST $(basename "$dst")"; continue; fi
      mkdir -p "$(dirname "$dst")" && cp -f "$f" "$dst" && echo "$([ "$MODE" = "--check" ] && echo NEW_WOULD_COPY || echo NEW_COPIED)   ${f#./}"
    done )
fi
if [ "$fail" = 0 ]; then echo "RESULT=OK"; else echo "RESULT=FAIL"; fi
exit $fail
