#!/bin/bash
# 开机后跑一次：GPU 大页池 + carveout（需要 root/sudo）
# 关键点：大页池必须**冷启动全量分配**；在线加页会碎片化（实测加到 ~32 GiB 就上不去）
set -euo pipefail
POOL_PAGES=${POOL_PAGES:-23552}    # 23552 × 2 MiB = 46 GiB（我们的读数就是这个池）
CARVEOUT=${CARVEOUT:-40}           # GiB 划给 GPU
CARVEOUT_BIN=${CARVEOUT_BIN:-/usr/local/bin/gpu-carveout.sh}

echo "[1/3] 当前:"; grep -E 'HugePages_Total|HugePages_Free|Hugepagesize' /proc/meminfo
echo "[2/3] 设置大页池 = $POOL_PAGES 页"
sysctl -w vm.nr_hugepages="$POOL_PAGES"
echo "[3/3] carveout ${CARVEOUT} GiB"
if [ -x "$CARVEOUT_BIN" ]; then
  "$CARVEOUT_BIN" -g "$CARVEOUT"
else
  echo "跳过: $CARVEOUT_BIN 不存在（按你平台的方式给 GPU 划内存）"
fi
grep -E 'HugePages_Total|HugePages_Free' /proc/meminfo
cat <<'TIP'
自检: HugePages_Free 应 ≈ HugePages_Total。
若 Free 明显小于 Total：重启一次（冷启动才能全量拿到）；别指望在线补页。
持久化: 把 vm.nr_hugepages=<页数> 写进 /etc/sysctl.d/*.conf。
TIP
