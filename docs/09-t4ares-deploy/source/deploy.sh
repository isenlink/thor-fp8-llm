#!/usr/bin/env bash
set -euo pipefail

DEPLOY_DIR=$(cd "$(dirname "$0")" && pwd)
PERSIST_BIN=$DEPLOY_DIR/bin/llama-server-m2c-t4
BIN=/tmp/c58t4/m2c/llama-server-m2c-t4
BIN_SHA=2066e6814171bd19e0ac473d1dd1986c229386c2c32e2aa605117d712b9c2358
MODEL=/brand_data/ai_workspace/models/RadixArk-F8attn-v2.gguf
DRAFT=/brand_data/ai_workspace/ai02/models/Qwen3.8-27B-DFlash2-BF16.gguf
MMPROJ=/brand_data/ai_workspace/models/mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf
LOGDIR=/tmp/c58t4/m2c/logs
PIDFILE=$LOGDIR/radixark-dflash-t4ares-mmproj-262k-8080.pid
HOST=0.0.0.0
PORT=8080

need_files() {
  local missing=0
  for f in "$PERSIST_BIN" "$MODEL" "$DRAFT" "$MMPROJ"; do
    if [ ! -f "$f" ]; then
      echo "MISSING $f"
      missing=1
    fi
  done
  [ "$missing" = 0 ]
}

sha_of() {
  sha256sum "$1" | awk '{print $1}'
}

ensure_runtime_bin() {
  local psha rsha
  psha=$(sha_of "$PERSIST_BIN")
  if [ "$psha" != "$BIN_SHA" ]; then
    echo "BAD_PERSIST_BIN_SHA expected=$BIN_SHA got=$psha path=$PERSIST_BIN"
    return 6
  fi

  if [ -f "$BIN" ]; then
    rsha=$(sha_of "$BIN")
    [ "$rsha" = "$BIN_SHA" ] && return 0
    echo "RUNTIME_BIN_SHA_MISMATCH restoring path=$BIN got=$rsha"
  else
    echo "RUNTIME_BIN_MISSING restoring path=$BIN"
  fi

  mkdir -p "$(dirname "$BIN")"
  cp -p "$PERSIST_BIN" "$BIN"
  chmod 0755 "$BIN"
  echo "RUNTIME_BIN_READY path=$BIN sha=$BIN_SHA"
}

port_pids() {
  for p in /proc/[0-9]*; do
    [ -r "$p/cmdline" ] || continue
    cmd=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null || true)
    case "$cmd" in
      *llama-server*'--port 8080'*|*llama-server*' 8080 '*)
        echo "${p#/proc/}"
        ;;
    esac
  done
}

status() {
  echo "--- process ---"
  local any=0
  for pid in $(port_pids); do
    any=1
    echo "PID=$pid"
    tr '\0' ' ' < "/proc/$pid/cmdline"
    echo
    echo "--- env $pid ---"
    tr '\0' '\n' < "/proc/$pid/environ" | grep -E '^(T4|GGML|LLAMA)' | sort || true
  done
  [ "$any" = 1 ] || echo "NO_8080_LLAMA"
  echo "--- health ---"
  if command -v curl >/dev/null 2>&1; then
    curl --noproxy '*' -sS "http://127.0.0.1:$PORT/health" || true
    echo
  else
    echo "curl not found on board; check health from another machine"
  fi
  echo "--- memory ---"
  grep -E 'MemAvailable|SwapFree|HugePages_Total|HugePages_Free|HugePages_Rsvd' /proc/meminfo
  echo "--- binary ---"
  [ -f "$PERSIST_BIN" ] && sha256sum "$PERSIST_BIN" || echo "MISSING $PERSIST_BIN"
  [ -f "$BIN" ] && sha256sum "$BIN" || echo "MISSING $BIN"
}

stop_service() {
  local pids
  pids=$(port_pids || true)
  if [ -z "$pids" ]; then
    echo "STOP no 8080 llama-server"
    return 0
  fi
  for pid in $pids; do
    echo "TERM $pid"
    kill "$pid" 2>/dev/null || true
  done
  for _ in $(seq 1 80); do
    pids=$(port_pids || true)
    [ -z "$pids" ] && break
    sleep 0.5
  done
  pids=$(port_pids || true)
  if [ -n "$pids" ]; then
    echo "STOP_FAILED pids=$pids"
    return 2
  fi
  echo "STOP OK"
}

start_service() {
  need_files
  ensure_runtime_bin
  mkdir -p "$LOGDIR"
  if [ -n "$(port_pids || true)" ]; then
    echo "START_REFUSED: 8080 already has llama-server; use restart"
    return 3
  fi

  export GGML_CUDA_GRAPH_OPT=1
  export GGML_MMVQ_MAX=2
  export GGML_NVFP4_WIDE_MAX=0
  export LLAMA_SPEC_TIMING=1
  export T4_MMQ=1
  export T4_CPS=7
  export T4_STAGES=2
  export T4_ARES=1

  local stamp log pid
  stamp=$(date +%Y%m%d-%H%M%S)
  log=$LOGDIR/radixark-dflash-t4ares-mmproj-262k-8080-$stamp.log
  nohup "$BIN" \
    -m "$MODEL" \
    -md "$DRAFT" \
    --mmproj "$MMPROJ" --mmproj-offload \
    --alias qwen3.8-27b --reasoning-effort medium \
    -ngl 99 -c 262144 -fa on \
    --cache-type-k f16 --cache-type-v f16 \
    --parallel 1 --port "$PORT" --host "$HOST" \
    --spec-type draft-dflash --spec-draft-n-max 7 \
    > "$log" 2>&1 &
  pid=$!
  echo "$pid" > "$PIDFILE"
  echo "STARTED pid=$pid log=$log"

  for _ in $(seq 1 120); do
    if grep -q 'listening on http://0.0.0.0:8080' "$log" 2>/dev/null; then
      echo "LISTENING"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "START_FAILED process exited"
      tail -n 80 "$log" || true
      return 4
    fi
    sleep 1
  done
  echo "START_TIMEOUT log=$log"
  tail -n 80 "$log" || true
  return 5
}

case "${1:-status}" in
  status) status ;;
  start) start_service ;;
  stop) stop_service ;;
  restart)
    stop_service
    start_service
    ;;
  *)
    echo "usage: $0 {status|start|stop|restart}"
    exit 2
    ;;
esac
