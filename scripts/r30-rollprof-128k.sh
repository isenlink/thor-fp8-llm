#!/bin/bash
# R30: 128K decode 滚动窗口拆账（op_prof v3：每 20000 ops dump+清零）
# 目的：perj + K12/p0.5 后重新拆 128K decode 尾段账本，刷新过期的 R30/R32 结论
set -u
B=/brand_data/ai_workspace/bench
T=/brand_data/ai_workspace/ai-assistant
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
setsid "$SERVER" \
  -m /brand_data/ai_workspace/models/RadixArk-F8attn-v2.gguf \
  -ngl 99 -c 131072 -fa on \
  --cache-type-k f16 --cache-type-v f16 \
  --spec-type draft-mtp --spec-draft-n-max 12 --spec-draft-p-min 0.5 \
  --parallel 1 --port 8080 > $B/llama-r30-rollprof-128k-perj-k12p05.log 2>&1 < /dev/null &
SRV=$!
echo "server pid $SRV"

for i in $(seq 1 180); do
  if python3 -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=2)" 2>/dev/null; then
    echo "ready after ${i}0s"; break
  fi
  sleep 10
done
python3 -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=2)" 2>/dev/null || { echo "SERVER NOT READY"; tail -20 $B/llama-r30-rollprof-128k-perj-k12p05.log; exit 1; }

# 跑基准；decode 段（最后 24s）里 op_prof 会滚动 dump 纯 decode 账本
python3 $B/a4-bench-8080.py 124000 R30-perj-k12p05

# decode 结束后立刻抓一份滚动账本（此时文件=最后20000 ops=decode 尾段）
cp $B/op_prof.txt $B/op_prof-r30-decode-tail-perj-k12p05.txt
stop_server
echo "=== R30 done ==="
tail -6 $B/llama-r30-rollprof-128k-perj-k12p05.log
echo "=== decode 尾段滚动账本 ==="
cat $B/op_prof-r30-decode-tail-perj-k12p05.txt
