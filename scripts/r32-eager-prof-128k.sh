#!/bin/bash
# R32: 128K + 全 eager（GGML_CUDA_DISABLE_GRAPHS=1）+ 滚动 op_prof
# 目的：CUDA graph 回放对 op_prof 是盲区 → 关 graph 让全部 forward 走 eager，
#       拿 perj + K12/p0.5 的 MTP step 真实墙钟构成（draft 链/verify/attn/MLP）
set -u
B=/ai_workspace/bench
T=/ai_workspace/ai-assistant
SERVER=$T/llama-server-ai-assistant-perj

stop_server() {
  for pid in $(pgrep -f 'llama-server' 2>/dev/null); do
    exe="$(readlink -f /proc/$pid/exe 2>/dev/null || true)"
    cmd="$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null || true)"
    if [ "$exe" = "$SERVER" ] || echo "$cmd" | grep -Fq "$SERVER"; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  sleep 3
}

stop_server
export GGML_CUDA_GRAPH_OPT=1
export GGML_MMVQ_MAX=2
export GGML_OP_PROF=1
export GGML_CUDA_DISABLE_GRAPHS=1
setsid "$SERVER" \
  -m /ai_workspace/models/RadixArk-F8attn-v2.gguf \
  -ngl 99 -c 131072 -fa on \
  --cache-type-k f16 --cache-type-v f16 \
  --spec-type draft-mtp --spec-draft-n-max 12 --spec-draft-p-min 0.5 \
  --parallel 1 --port 8080 > $B/llama-r32-eager-prof-128k-perj-k12p05.log 2>&1 < /dev/null &
SRV=$!
echo "server pid $SRV"

for i in $(seq 1 180); do
  if python3 -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=2)" 2>/dev/null; then
    echo "ready after ${i}0s"; break
  fi
  sleep 10
done
python3 -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=2)" 2>/dev/null || { echo "SERVER NOT READY"; tail -20 $B/llama-r32-eager-prof-128k-perj-k12p05.log; exit 1; }

python3 $B/a4-bench-8080.py 124000 R32-perj-k12p05-eager
cp $B/op_prof.txt $B/op_prof-r32-decode-tail-perj-k12p05.txt
stop_server
echo "=== R32 done ==="
tail -6 $B/llama-r32-eager-prof-128k-perj-k12p05.log
echo "=== decode 尾段（全 eager，100% 覆盖）==="
cat $B/op_prof-r32-decode-tail-perj-k12p05.txt
