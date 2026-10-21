#!/usr/bin/env bash
set -euo pipefail
# Candidate only: deploy after sequential + concurrent quality/performance gates.
THOR_MAX_NUM_SEQS="${THOR_MAX_NUM_SEQS:-2}"
case "$THOR_MAX_NUM_SEQS" in 1|2|3|4) ;; *) echo 'max-num-seqs must be 1..4' >&2; exit 2;; esac

THOR_MTP="${THOR_MTP:-4}"
case "$THOR_MTP" in 4|5|6|7) ;; *) echo 'MTP must be 4..7' >&2; exit 2;; esac
THOR_VERIFY=$((THOR_MTP + 1))
case "$THOR_MAX_NUM_SEQS" in
  1) THOR_GRAPH_SIZES="[1,$THOR_VERIFY]" ;;
  2) THOR_GRAPH_SIZES="[1,$THOR_VERIFY,$((2 * THOR_VERIFY))]" ;;
  3) THOR_GRAPH_SIZES="[1,$THOR_VERIFY,$((2 * THOR_VERIFY)),$((3 * THOR_VERIFY))]" ;;
  4) THOR_GRAPH_SIZES="[1,$THOR_VERIFY,$((2 * THOR_VERIFY)),$((4 * THOR_VERIFY))]" ;;
esac
THOR_GRAPH_CONFIG="{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":$THOR_GRAPH_SIZES}"
THOR_SPEC_CONFIG="{\"method\":\"mtp\",\"num_speculative_tokens\":$THOR_MTP}"
ROOT=/zeekr_data/ai_workspace/vllm032-clean
MODEL="${THOR_MODEL:-$ROOT/models/qwen38-mixed}"
if [[ -e "$MODEL/INCOMPLETE-DO-NOT-LOAD" || ! -f "$MODEL/VERIFIED-MANIFEST.json" ]]; then
  echo "Model staging is not verified" >&2; exit 3
fi
export LD_LIBRARY_PATH="$ROOT/runtime-lib:$ROOT/env/lib/python3.12/site-packages/torch/lib:/usr/local/cuda-12.8/targets/aarch64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TMPDIR="$ROOT/qualification-20261021/tmp"
export XDG_CACHE_HOME="$ROOT/qualification-20261021/cache"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$HF_HOME"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export CC="$ROOT/board-cc-wrapper.sh"
export TRITON_PTXAS_BLACKWELL_PATH="$ROOT/ptxas-12.8"

export PYTHONPATH="$ROOT/qualification-20261021/opt-mixed-qualification:$ROOT/qualification-20261021/opt-hybrid-qualification${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_GDN_DECODE_KERNEL=triton

exec "$ROOT/env/bin/vllm" serve "$MODEL" \
  --served-model-name "${THOR_SERVED_MODEL:-qwen3.8-27b}" \
  --safetensors-load-strategy lazy \
  --host "${THOR_HOST:-127.0.0.1}" \
  --port "${THOR_PORT:-8081}" \
  --max-model-len 262144 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs "$THOR_MAX_NUM_SEQS" \
  --gpu-memory-utilization 0.90 \
  --speculative-config "$THOR_SPEC_CONFIG" \
  --override-generation-config '{"temperature":0.3}' \
  --default-chat-template-kwargs '{"enable_thinking":true,"reasoning_effort":"medium"}' \
  --reasoning-parser deepseek_r1 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --enable-prefix-caching \
  --kernel-config '{"enable_jit_warmup":false}' \
  --compilation-config "$THOR_GRAPH_CONFIG" \
  --attention-backend FLASH_ATTN \
  --attention-config '{"flash_attn_version":2}'
