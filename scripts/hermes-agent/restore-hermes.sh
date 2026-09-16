#!/bin/bash
# Hermes Agent 恢复脚本（幂等）—— 母本放非易失分区，overlay 被吞后可重跑
#
# 调用方（二选一，可并存）：
#   A) 板端既有恢复脚本的末尾钩子（见 docs/07-hermes-agent/persistence-autostart-*.md）
#   B) 板外常电主机看护进程的恢复命令表
# 说明：overlay 侧（/etc）会在脏断电后被格式化，本脚本负责把薄壳重建出来并拉起服务。
set -u
set -u

HROOT=/brand_data/hermes
LOG=$HROOT/logs/restore-hermes.log
mkdir -p "$HROOT/logs" && touch "$LOG"
exec >> "$LOG" 2>&1
echo "===== restore-hermes $(date -Is) ====="

say() { echo "[$(date +%H:%M:%S)] $*"; }

# root 或免密 sudo（串口恢复时本就是 root；SSH 下用 sudo -n）
SUDO=""
if [ "$(id -u)" != "0" ]; then
  if sudo -n true 2>/dev/null; then SUDO="sudo -n"; else say "⚠️ 非 root 且无免密 sudo：单元无法安装"; fi
fi

# 1) 目录与属主（数据分区，理论上不缺，兜底）
mkdir -p "$HROOT/home" "$HROOT/bin" "$HROOT/logs"
chown -R user:user "$HROOT" 2>/dev/null || true

# 2) 启动器（落到数据分区，不依赖 overlay）
cat > "$HROOT/bin/hermes" <<'EOF'
#!/bin/bash
export HERMES_HOME=/brand_data/hermes/home
export PATH=/brand_data/hermes/bin:/usr/local/bin:/usr/bin:/bin
exec /brand_data/hermes/venv/bin/hermes "$@"
EOF
chmod 755 "$HROOT/bin/hermes"

# 2b) 环境变量：**只读根上 /usr/local/bin 不可写（实测连 sudo 都不行）**，
#     所以 shim 不能用符号链接，改走 /etc/profile.d（overlay 可写，由本脚本每次重建）。
cat > "$HROOT/env.sh" <<'EOF'
export HERMES_HOME=/brand_data/hermes/home
export PATH=/brand_data/hermes/bin:$PATH
EOF
chmod 644 "$HROOT/env.sh"
$SUDO mkdir -p /etc/profile.d
printf '# Hermes Agent（持久区在 /brand_data/hermes；本文件由 restore-hermes.sh 重建）\n. /brand_data/hermes/env.sh\n' \
  | $SUDO tee /etc/profile.d/99-hermes.sh >/dev/null && say "已安装 /etc/profile.d/99-hermes.sh"

# 3) systemd 单元（overlay 会被格式化 ⇒ 每次恢复时从数据分区母本重新装回）
install_unit() {
  local name=$1 src=$HROOT/systemd/$1.service dst=/etc/systemd/system/$1.service
  [ -f "$src" ] || { say "跳过 $name（母本不存在）"; return; }
  if ! cmp -s "$src" "$dst" 2>/dev/null; then
    $SUDO install -m 644 "$src" "$dst"
    say "已安装单元 $name.service"
  else
    say "单元 $name.service 无变化"
  fi
}
install_unit hermes
install_unit llama-server

# 4a) 大页池兜底：模型加载依赖 46G 池（脏断电后会回落到出厂 20G）
CUR=$(cat /proc/sys/vm/nr_hugepages 2>/dev/null || echo 0)
if [ "$CUR" -lt 23552 ]; then
  $SUDO sysctl -w vm.nr_hugepages=23552 >/dev/null 2>&1 || say "⚠️ 大页池设置失败"
  echo "vm.nr_hugepages=23552" > /etc/sysctl.d/99-hugepages-hermes.conf 2>/dev/null || true
  say "大页池 $CUR -> $(cat /proc/sys/vm/nr_hugepages)"
fi

# 4b) DNS 兜底：/etc/resolv.conf 原为车端 TACP 遗留的 198.18.x.x（已失效）
#     ⇒ 会让一切域名解析挂死（实测拖到 180s 超时）。必须每次恢复时重建。
#     机理补充（对方实测）：该遗留地址落在车载内网路由（mgbe 网口）上，默认路由若不指向车载内网，
#     它永远不可用 ⇒ 解析器要跟随默认路由（网关 + 一个公共解析器），并加 timeout/attempts 防挂死。
#     ★ 读者按自己环境替换 <LAN-GATEWAY>（默认路由网关，通常也是可用的内网 DNS）。
if ! grep -q "223.5.5.5" /etc/resolv.conf 2>/dev/null || ! grep -q "options timeout" /etc/resolv.conf 2>/dev/null; then
  printf "nameserver 223.5.5.5\nnameserver <LAN-GATEWAY>\noptions timeout:2 attempts:2\n" | $SUDO tee /etc/resolv.conf >/dev/null
  $SUDO mkdir -p /etc/systemd/resolved.conf.d
  printf "[Resolve]\nDNS=223.5.5.5 <LAN-GATEWAY>\nFallbackDNS=114.114.114.114\n" \
    | $SUDO tee /etc/systemd/resolved.conf.d/99-thor-dns.conf >/dev/null
  $SUDO systemctl restart systemd-resolved 2>/dev/null || true
  say "DNS 已修正（原为车端 TACP 遗留 198.18.x.x）"
fi

# 4) 时间（overlay 丢掉 sudoers 之后可能没对时；服务依赖正确时钟）
if ! timedatectl is-synchronized >/dev/null 2>&1; then
  systemctl restart systemd-timesyncd >/dev/null 2>&1 || true
fi

# 5) 起服务（先模型、后 Agent）
$SUDO systemctl daemon-reload
$SUDO systemctl enable hermes.service llama-server.service >/dev/null 2>&1 || true
$SUDO systemctl restart llama-server.service >/dev/null 2>&1 || say "⚠️ llama-server 启动失败"
$SUDO systemctl restart hermes.service       >/dev/null 2>&1 || say "⚠️ hermes 启动失败"

# 6) 复验
sleep 6
say "llama-server: $(systemctl is-active llama-server.service 2>/dev/null)"
say "hermes:       $(systemctl is-active hermes.service 2>/dev/null)"
say "HERMES_HOME:  $(ls -d $HROOT/home)"
say "DNS:          $(grep -m1 nameserver /etc/resolv.conf 2>/dev/null)"
say "profile.d:    $(ls /etc/profile.d/99-hermes.sh 2>/dev/null || echo 缺失)"
say "DeepSeek 可达: $(python3 $HROOT/bin/check-ds.py 2>/dev/null || echo 未知)"
say "完成"
sync
