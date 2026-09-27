#!/bin/bash
# vLLM 0.11.2 (sm_101) multimodal serve — Qwen3.8-27B with grafted Qwen3VL vision tower
# Parameterized template. See docs/PLAN-A-MULTIMODAL.md for the full story.
#
# Key points:
#   - vision tower and text tower CANNOT run as two separate serves (GPU memory exclusive)
#   - 200K ctx + MTP + seqs 3 verified; NOTE: KV = 78,208 tokens at util 0.94,
#     so 200K is SINGLE-REQUEST only for the multimodal variant (text-only variant fits 2x)
#   - --reasoning-parser deepseek_r1 for <think> splitting (see start-serve-text.sh notes)

WORKSPACE=${WORKSPACE:-/opt/vllm-p3}
MODEL_DIR=${MODEL_DIR:-/data/models/p3-vl-27b}   # nested qwen3_vl config + weights
MTP_DRAFT=${MTP_DRAFT:-/data/models/mtp-draft}
PORT=${PORT:-8996}
UTIL=${UTIL:-0.94}
CTX=${CTX:-200000}
SEQS=${SEQS:-3}
GCC_ROOT=${GCC_ROOT:-$WORKSPACE/gcc-root/root}

cd "$WORKSPACE"
setsid env \
  VLLM_CACHE_ROOT=$WORKSPACE/vllm-cache \
  R=$GCC_ROOT \
  LD_LIBRARY_PATH=$GCC_ROOT/usr/lib/aarch64-linux-gnu:$GCC_ROOT/lib/aarch64-linux-gnu:$WORKSPACE/venv/lib:/usr/local/cuda-12.8/lib64 \
  CC=$GCC_ROOT/usr/bin/gcc \
  LIBRARY_PATH=$GCC_ROOT/usr/lib/aarch64-linux-gnu:$GCC_ROOT/lib/aarch64-linux-gnu \
  CPATH=$GCC_ROOT/usr/include:$GCC_ROOT/usr/include/aarch64-linux-gnu:$GCC_ROOT/usr/include/python3.12 \
  PATH=$GCC_ROOT/usr/bin:/usr/bin:/bin \
  venv/bin/vllm serve "$MODEL_DIR" \
    --served-model-name vl27 \
    --max-model-len $CTX \
    --max-num-seqs $SEQS \
    --gpu-memory-utilization $UTIL \
    --no-enable-prefix-caching \
    --reasoning-parser deepseek_r1 \
    --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":1,\"model\":\"$MTP_DRAFT\"}" \
    --port $PORT > serve-vl.log 2>&1 < /dev/null &
echo $! > serve-vl.pid
echo "VL SERVE STARTED $(cat serve-vl.pid) port=$PORT"
