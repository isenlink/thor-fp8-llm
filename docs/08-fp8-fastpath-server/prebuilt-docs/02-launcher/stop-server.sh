#!/bin/bash
# 干净停服（按件名精确匹配，不要用 pkill -f llama-server 这种宽匹配）
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
name=$(basename "${BIN:-$here/../01-binary/llama-server-aarch64-sm101a-nvfp4}")
pids=$(pgrep -f "$name" || true)
if [ -z "$pids" ]; then echo "NO_PROC ($name)"; exit 0; fi
echo "stopping: $(echo "$pids" | tr '\n' ' ')"
# shellcheck disable=SC2086
kill $pids 2>/dev/null || true
for _ in $(seq 1 20); do
  sleep 1
  pgrep -f "$name" >/dev/null || { echo "STOPPED"; exit 0; }
done
echo "仍在跑，手工确认: pgrep -af $name"
exit 1
