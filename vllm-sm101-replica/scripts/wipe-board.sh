#!/bin/bash
# Wipe board back to blank test state. Only touches paths this package created.
# Run ON THE BOARD. Root needed for /brand_hdmap entries.
set -x
# 1. stop serves (by PID file, never pkill -f with broad strings)
for f in /brand_data/ai_workspace/vllm-p3/serve*.pid; do
  [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null
done
sleep 5
# engine cores often survive; kill exact remaining processes
ps -eo pid,args | grep -E "VLLM::EngineCore|vllm serve" | grep -v grep | awk '{print $1}' | xargs -r kill -9 2>/dev/null
sleep 3
grep HugePages_Free /proc/meminfo   # expect: full pool free
# 2. remove our artifacts
rm -rf /brand_data/ai_workspace/vllm-p3
rm -rf /brand_data/ai_workspace/vllm-cache
rm -rf /brand_data/ai_workspace/gcc-root
sudo rm -rf /brand_hdmap/p3-models /brand_hdmap/p3-vl-27b /brand_hdmap/p3-build \
            /brand_hdmap/p3-venv /brand_hdmap/p3-wheels
rm -rf /brand_data/ai_workspace/tmp
df -h /brand_data/ /brand_hdmap
echo "board is blank (other pre-existing model/tool dirs untouched — not ours)"
