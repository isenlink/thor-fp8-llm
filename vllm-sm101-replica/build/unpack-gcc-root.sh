#!/bin/bash
# Rebuild offline gcc root on the board (board image has no gcc).
# Run ON THE BOARD, from the directory containing gcc-debs/.
# Runtime torch.compile needs a real gcc; serve scripts expect it at $GCC_ROOT.
set -e
DEST=${1:-/brand_data/ai_workspace/gcc-root/root}
mkdir -p "$DEST"
for deb in gcc-debs/*.deb; do
  echo "unpacking $deb"
  dpkg-deb -x "$deb" "$DEST"
done
echo "=== sanity check:"
"$DEST/usr/bin/gcc" --version | head -1
echo "OK. For serve scripts: export GCC_ROOT=$(dirname "$DEST")"
