#!/bin/bash
# vLLM 0.11.2 (sm_101) production text serve — parameterized template
# Usage: edit the variables below to match your board layout, then run.
# Origin: Qwen3.8-27B NVFP4 text-only serve on DRIVE OS 7.0.3 / 60GB unified memory board.
# Key points:
#   - hugepage pool sized for GPU (54G here) -> --gpu-memory-utilization 0.94 is REQUIRED
#     (0.90 locks KV at 23.7G / 200K-concurrency 1.82x; 0.94 gives 27.95G / 2.14x)
#   - MTP speculative decoding with the transplanted qwen3_next MTP head
#   - --reasoning-parser deepseek_r1 (NOT qwen3: qwen3 parser requires <think> in output,
#     but chat template already injects it into the prompt -> split fails)

WORKSPACE=${WORKSPACE:-/opt/vllm-p3}          # your venv + scripts dir
MODEL_DIR=${MODEL_DIR:-/data/models/p3-text}  # model dir (config.json + model.safetensors)
MTP_DRAFT=${MTP_DRAFT:-/data/models/mtp-draft}
PORT=${PORT:-8998}
UTIL=${UTIL:-0.94}
CTX=${CTX:-200000}
SEQS=${SEQS:-3}
GCC_ROOT=${GCC_ROOT:-$WORKSPACE/gcc-root/root}  # offline gcc for torch.compile

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
    --served-model-name qwen \
    --max-model-len $CTX \
    --max-num-seqs $SEQS \
    --gpu-memory-utilization $UTIL \
    --no-enable-prefix-caching \
    --reasoning-parser deepseek_r1 \
    --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":1,\"model\":\"$MTP_DRAFT\"}" \
    --port $PORT > serve-text.log 2>&1 < /dev/null &
echo $! > serve.pid
echo "TEXT SERVE STARTED $(cat serve.pid) port=$PORT"
