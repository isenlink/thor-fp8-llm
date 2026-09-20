#!/bin/bash
# NVIDIA DRIVE Thor (sm_101a) · NVFP4 target + 起草器 · 生产启动脚本
#
# 用法:
#   ① DFlash2（推荐）:  MODEL=/path/to/target.gguf DRAFT=/path/to/dflash2.gguf bash start-server.sh
#   ② MTP 头回退:       MODEL=/path/to/target.gguf SPEC=mtp bash start-server.sh     # 不需要 DRAFT
#   ③ 对照臂:           MODEL=/path/to/target.gguf SPEC=none CTX=8192 bash start-server.sh
# 可覆盖: SPEC(dflash2|mtp|none) CTX(131072) SPEC_N(7) MTP_N(12) MTP_PMIN(0.5) KV(f16|q8_0) PORT(8080) PARALLEL(1) LOG
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
BIN=${BIN:-$here/../01-binary/llama-server-aarch64-sm101a-nvfp4}
MODEL=${MODEL:?set MODEL=/path/to/target-nvfp4.gguf}
SPEC=${SPEC:-dflash2}
CTX=${CTX:-131072}
SPEC_N=${SPEC_N:-7}
MTP_N=${MTP_N:-12}
MTP_PMIN=${MTP_PMIN:-0.5}
KV=${KV:-f16}
PORT=${PORT:-8080}
PARALLEL=${PARALLEL:-1}
LOG=${LOG:-/tmp/llama-server-$PORT.log}

[ -x "$BIN" ]   || { echo "ABORT: 没有可执行件 $BIN"; exit 2; }
[ -f "$MODEL" ] || { echo "ABORT: 没有 target 模型 $MODEL"; exit 2; }
case "$KV" in f16|q8_0) ;; *) echo "ABORT: KV 只能是 f16 或 q8_0"; exit 2;; esac
case "$CTX$SPEC_N$PORT$PARALLEL" in *[!0-9]*) echo "ABORT: CTX/SPEC_N/PORT/PARALLEL 必须是数字"; exit 2;; esac

name=$(basename "$BIN")
if pgrep -f "$name" >/dev/null; then echo "注意: 已有同件进程在跑（先跑 stop-server.sh）"; fi

# 本件的三条环境变量：缺一条会掉到 9-13 t/s 量级（见 README §6）
export GGML_CUDA_GRAPH_OPT=1 GGML_MMVQ_MAX=2 GGML_NVFP4_WIDE_MAX=0

args=(-m "$MODEL" -ngl 99 -c "$CTX" -fa on --cache-type-k "$KV" --cache-type-v "$KV" \
      --parallel "$PARALLEL" --port "$PORT")
case "$SPEC" in
  dflash2)
    DRAFT=${DRAFT:?set DRAFT=/path/to/dflash2-draft.gguf（或 SPEC=mtp 走 MTP 头回退）}
    [ -f "$DRAFT" ] || { echo "ABORT: 没有起草器 $DRAFT（见 README §5）"; exit 2; }
    args+=(-md "$DRAFT" --spec-type draft-dflash --spec-draft-n-max "$SPEC_N") ;;
  mtp)
    [ -n "${DRAFT:-}" ] && echo "提示: SPEC=mtp 不使用 DRAFT（MTP 头在 target 文件里）"
    args+=(--spec-type draft-mtp --spec-draft-n-max "$MTP_N" --spec-draft-p-min "$MTP_PMIN") ;;
  none) ;;
  *) echo "ABORT: SPEC 只能是 dflash2|mtp|none"; exit 2 ;;
esac

echo "cmdline: $BIN ${args[*]}"
echo "env: GGML_CUDA_GRAPH_OPT=$GGML_CUDA_GRAPH_OPT GGML_MMVQ_MAX=$GGML_MMVQ_MAX GGML_NVFP4_WIDE_MAX=$GGML_NVFP4_WIDE_MAX"
setsid "$BIN" "${args[@]}" >"$LOG" 2>&1 </dev/null &
sleep 3
pid=$(pgrep -f "$name" | head -1 || true)
echo "started pid=${pid:-unknown} log=$LOG"
echo "探活: curl -s localhost:$PORT/v1/models | head -c 200"
echo "看数: grep -E 'draft accept|eval time|n_gpu_layers|flash_attn|wrong number of tensors' $LOG"
