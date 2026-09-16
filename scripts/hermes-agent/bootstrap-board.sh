#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Hermes Agent — 板端镜像自举安装（适用于无 pip / 无 ensurepip / 无 apt / 无编译器的车规板）
#
# 适用前提：板能访问任一 PyPI 镜像（先修 DNS，见 docs/07-hermes-agent/）
# 实测平台：aarch64 / Python 3.12.3 / DriveOS 7.0.3
#
# 用法：sudo bash hermes-board-bootstrap.sh [镜像URL] [包名==版本]
# ═══════════════════════════════════════════════════════════════
set -e

MIRROR="${1:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PKG="${2:-hermes-agent==0.19.0}"
ROOT=/brand_data/hermes
VENV="$ROOT/venv"
WORK="$ROOT/.bootstrap"

echo "=== [1/5] 目录 ==="
mkdir -p "$WORK" "$ROOT/logs" "$ROOT/home"
chown -R "$(id -un):$(id -gn)" "$ROOT" 2>/dev/null || true

echo "=== [2/5] 取 pip wheel（★ 必须按版本号数值排序）==="
cd "$WORK"
python3 - <<PYEOF
import urllib.request, urllib.parse, re, os
idx = "$MIRROR/pip/"
h = urllib.request.urlopen(idx, timeout=20).read().decode("utf-8", "ignore")
whls = re.findall(r'href="([^"]+\.whl[^"]*)"', h)

def vkey(name):
    """★ 关键：按版本号数值排序。
    纯字符串排序会选到 pip-9.0.3（'9' > '2'），那个版本在现代 Python 上连自己都装不上：
      ModuleNotFoundError: No module named 'pip._vendor.urllib3.packages.six.moves'
    """
    m = re.search(r'pip-([0-9][^-]*?)-py', name)
    v = m.group(1) if m else "0"
    return [int(x) if x.isdigit() else 0 for x in re.split(r'[.\-]', v)]

names = sorted({w.split("/")[-1].split("#")[0] for w in whls if "py3-none-any" in w}, key=vkey)
if not names:
    raise SystemExit("镜像索引里没有 py3-none-any wheel")
name = names[-1]
url = next(w for w in whls if w.split("/")[-1].split("#")[0] == name)
url = urllib.parse.urljoin(idx, url)   # 索引里是相对路径，用 urljoin 正确解析（勿手工拼主机名）
print("  选中:", name)
open(name, "wb").write(urllib.request.urlopen(url, timeout=90).read())
print("  已下载 %.0f KB" % (os.path.getsize(name) / 1024))
PYEOF

PIPWHL=$(ls -1 pip-*.whl | tail -1)

echo "=== [3/5] 建 venv（--without-pip：板上无 ensurepip）==="
rm -rf "$VENV"
python3 -m venv --without-pip "$VENV"
echo "  $($VENV/bin/python --version)"

echo "=== [4/5] 自举 pip ==="
"$VENV/bin/python" "$WORK/$PIPWHL/pip" install --no-index --find-links "$WORK" "$WORK/$PIPWHL" 2>&1 | tail -2
"$VENV/bin/pip" --version

cat > "$VENV/pip.conf" <<EOF
[global]
index-url = $MIRROR
timeout = 60
retries = 3
EOF

echo "=== [5/5] 装包：$PKG ==="
"$VENV/bin/pip" install "$PKG" 2>&1 | tail -3

echo
echo "=== 完成 ==="
echo "  验证: HERMES_HOME=$ROOT/home $VENV/bin/hermes --version"
echo "  下一步：把 $ROOT/home 设为 HERMES_HOME（见 hermes-persist-apply.sh 的 profile.d 母本）"
