#!/usr/bin/env bash
# rb2_build.sh — R-B2 判别臂构建：**同一源文件**出两个 cubin，唯一变量 = -DT4_V9F_GRAN4
#   gran0(GRAN4=0) = 归档 ws32k 口径（16 B/次 cp.async）；gran1(GRAN4=1) = 生产腿口径（4 B/次）。
#   基线同一性判据**不是 cubin sha**：nvcc 把源文件的 internal 符号名里编进了源路径（.strtab/.symtab 变），
#   改名或换目录就会换 sha 而机器码一字不改。2026-09-17 实测：SASS diff=0 行、三段 .text md5 全同、
#   9 个 internal 符号名不同 ⇒ cubin sha 差异 100% 是命名伪影。（旧版按 sha 比对 ⇒ 结构性误报，已废）
#   ⇒ 判据：① gran0 与归档件 SASS 逐字节同一 ② gran1 的 LDGSTS 计数 = 4 × gran0（4 B/次 vs 16 B/次）
#   2026-09-17 追加（H-mem 臂）：③ prod 变体（T4_V9F_PRODADDR=1）与 gran1 的**结构计数全等**
#     （LDGSTS / UTCOMMA / UTCBAR / SYNCS.ARRIVE / BAR.SYNC —— 请求数、MMA、握手、线程屏障一处不改），
#     只有取数地址算术不同 ⇒ 单变量可机械核；④ 参数单一来源与生产头 t4_mmq_canon.h 的 b_smem_off 全表一致
#     （t4_prodaddr_check.cpp 的 static_assert）；⑤ 真实模型行步长与 CPR 对齐（GGUF 头直读，非文档）
#   产出: stage/rb2/{gran0,gran1}.cubin + expected.tsv（驱动与上传脚本读这个，不硬编码 sha）
set -u
R=/path/to/project
SRC=$R/t4/m2c/src/rb2/c58_v9f_gran4.cu
V5=$R/evidence/20260916-67-t4-3-b5-nvfp4/src
T4C=$R/t4/common
CUT=/tmp/cutlass-main
CUDA=/usr/local/cuda-12.8/bin
OUT=$R/t4/m2c/stage/rb2
ARCH=$R/evidence/20260916-76-t4-3-b9f-ws32k/host/c58_v9f.cubin
ARCH_HOST=$R/evidence/20260916-76-t4-3-b9f-ws32k/host/c58_v9f_host.stream1
HOST_SRC=$R/t4/m2c/src/rb2/c58_v9f_host_prod.cpp
LAYOUT_CHECK=$R/t4/m2c/src/rb2/t4_prodaddr_check.cpp
GGUF=${T4_PROD_MODEL:-~/work/thor-driveos/models/model-nvfp4.gguf}
CANON_DIR=${T4_CANON_DIR:-~/work/thor-driveos/xbuild/llama-cpp-latest/ggml/src/ggml-cuda/t4}
GXX=$(command -v aarch64-linux-gnu-g++-14 || command -v aarch64-linux-gnu-g++)
SHA_ARCH_EXPECT=5af5d5c2bcdc2ce9
SHA_HOST_EXPECT=1b37ea20b534d8f0
[ -d "$CUT/include" ] || { echo "ABORT: 缺 $CUT/include（易失路径 ⇒ 先恢复 cutlass 头再谈上板）"; exit 4; }
[ -f "$CANON_DIR/t4_mmq_canon.h" ] || { echo "ABORT: 缺 $CANON_DIR/t4_mmq_canon.h（生产布局单一来源）"; exit 4; }
[ -f "$GGUF" ] || { echo "ABORT: 缺生产模型 $GGUF（行步长事实来源）"; exit 4; }
[ -n "$GXX" ] || { echo "ABORT: 缺 aarch64-linux-gnu-g++（宿主交叉构建）"; exit 4; }
a=$(sha256sum "$ARCH" | cut -c1-16)
[ "$a" = "$SHA_ARCH_EXPECT" ] || { echo "ABORT: 归档 cubin sha=$a ≠ $SHA_ARCH_EXPECT（归档本人被动过）"; exit 4; }
h=$(sha256sum "$ARCH_HOST" | cut -c1-16)
[ "$h" = "$SHA_HOST_EXPECT" ] || { echo "ABORT: 归档 host sha=$h ≠ $SHA_HOST_EXPECT"; exit 4; }
mkdir -p "$OUT"
# ---- H-mem 臂的两个"参数不是猜的"门（先建，再谈编内核）----
g++ -std=c++17 -O2 -I"$R/t4/m2c/src/rb2" -I"$CANON_DIR" "$LAYOUT_CHECK" -o "$OUT/00_layout_check" > "$OUT/00_layout_check.log" 2>&1 \
  || { echo "ABORT: 生产寻址参数与生产布局头不一致（编译期 static_assert 失败）"; tail -8 "$OUT/00_layout_check.log"; exit 4; }
"$OUT/00_layout_check" > "$OUT/00_layout_check.out" 2>&1
grep -q OK LAYOUT_IDENTICAL "$OUT/00_layout_check.out" || { echo "ABORT: layout check 未通过"; cat "$OUT/00_layout_check.out"; exit 4; }
sed 's/^/  /' "$OUT/00_layout_check.out"
CPR=$(sed -n 's/.*CPR=\([0-9][0-9]*\).*/\1/p' "$OUT/00_layout_check.out")
RS=$((CPR * 144))
echo "  参数事实来源：GGUF 行步长（期望 type40 K=5120 ⇒ row_stride=$RS = $CPR x 144）"
python3 "$R/t4/m2c/host/gguf_rowstride.py" "$GGUF" --type 40 --expect "40:5120:$RS" > "$OUT/00_gguf_rowstride.log" 2>&1 \
  || { echo "ABORT: GGUF 行步长与 CPR=$CPR 不一致（见 $OUT/00_gguf_rowstride.log）"; tail -6 "$OUT/00_gguf_rowstride.log"; exit 4; }
sed 's/^/  /' "$OUT/00_gguf_rowstride.log"
for g in 0 1; do
  $CUDA/nvcc -ptx -arch=sm_101a -O2 -std=c++17 -DT4_V9F_GRAN4=$g -I"$CUT/include" -I"$V5" -I"$T4C" \
      "$SRC" -o "$OUT/gran$g.ptx" > "$OUT/01_nvcc_gran$g.log" 2>&1 || { echo "NVCC_FAIL gran$g"; tail -20 "$OUT/01_nvcc_gran$g.log"; exit 3; }
  $CUDA/ptxas -arch=sm_101a -O2 "$OUT/gran$g.ptx" -o "$OUT/gran$g.cubin" > "$OUT/02_ptxas_gran$g.log" 2>&1 || { echo "PTXAS_FAIL gran$g"; tail -20 "$OUT/02_ptxas_gran$g.log"; exit 3; }
done
$CUDA/cuobjdump -sass "$ARCH"                 > "$OUT/03_sass_arch.txt"  2>&1
$CUDA/cuobjdump -sass "$OUT/gran0.cubin"      > "$OUT/03_sass_gran0.txt" 2>&1
$CUDA/cuobjdump -sass "$OUT/gran1.cubin"      > "$OUT/03_sass_gran1.txt" 2>&1
if diff -q "$OUT/03_sass_arch.txt" "$OUT/03_sass_gran0.txt" > /dev/null; then
  echo "BASE_IDENTICAL gran0 == 归档件（SASS $(wc -l < "$OUT/03_sass_gran0.txt") 行逐字节同）"
else
  echo "BASE_DRIFT gran0 与归档件 SASS 不同 ⇒ 停手（不许上板）"
  diff "$OUT/03_sass_arch.txt" "$OUT/03_sass_gran0.txt" | head -20; exit 4
fi
l0=$(grep -c LDGSTS "$OUT/03_sass_gran0.txt"); l1=$(grep -c LDGSTS "$OUT/03_sass_gran1.txt")
[ "$l1" -eq $((l0 * 4)) ] || { echo "GRAN4_NOT_EFFECTIVE LDGSTS gran0=$l0 gran1=$l1（期望 4×）⇒ 停手"; exit 4; }
echo "GRAN4_EFFECTIVE LDGSTS gran0=$l0 gran1=$l1（4 B/次 vs 16 B/次）"
# ---- H-mem 臂：同一份源码的第三个变体（GRAN4=1 + PRODADDR=1），只换取数寻址 ----
$CUDA/nvcc -ptx -arch=sm_101a -O2 -std=c++17 -DT4_V9F_GRAN4=1 -DT4_V9F_PRODADDR=1 -I"$CUT/include" -I"$V5" -I"$T4C" \
    "$SRC" -o "$OUT/prod.ptx" > "$OUT/01_nvcc_prod.log" 2>&1 || { echo "NVCC_FAIL prod"; tail -20 "$OUT/01_nvcc_prod.log"; exit 3; }
$CUDA/ptxas -arch=sm_101a -O2 "$OUT/prod.ptx" -o "$OUT/prod.cubin" > "$OUT/02_ptxas_prod.log" 2>&1 || { echo "PTXAS_FAIL prod"; tail -20 "$OUT/02_ptxas_prod.log"; exit 3; }
$CUDA/cuobjdump -sass "$OUT/prod.cubin" > "$OUT/03_sass_prod.txt" 2>&1
bad=0
for op in LDGSTS UTCOMMA UTCBAR "SYNCS.ARRIVE" "BAR.SYNC"; do
  c1=$(grep -c "$op" "$OUT/03_sass_gran1.txt"); c2=$(grep -c "$op" "$OUT/03_sass_prod.txt")
  printf '  %-14s gran1=%-4s prod=%-4s %s\n' "$op" "$c1" "$c2" "$([ "$c1" = "$c2" ] && echo SAME || echo DIFF)"
  [ "$c1" = "$c2" ] || bad=1
done
[ "$bad" = 0 ] || { echo "PRODADDR_STRUCT_DRIFT 结构计数与 gran1 不同 ⇒ 不是单变量，停手"; exit 4; }
echo "PRODADDR_STRUCTURAL_PARITY 请求数/MMA/握手/屏障计数与 gran1 全等（唯一变量 = 取数寻址）"
g0=$(sha256sum "$OUT/gran0.cubin" | cut -c1-16); g1=$(sha256sum "$OUT/gran1.cubin" | cut -c1-16)
pr=$(sha256sum "$OUT/prod.cubin" | cut -c1-16)
# 宿主：H-mem 臂用的副本（mode=stream 时与归档宿主逐行同 ⇒ 计划里先用它复现 gran1 作锚点）
$GXX -std=c++17 -O2 -I/usr/local/cuda-12.8/include -I"$T4C" -I"$R/t4/m2c/src/rb2" "$HOST_SRC" \
    -o "$OUT/c58_v9f_host_prod" -ldl -lstdc++ > "$OUT/05_gxx_host_prod.log" 2>&1 \
  || { echo "GXX_FAIL host_prod"; tail -20 "$OUT/05_gxx_host_prod.log"; exit 3; }
h2=$(sha256sum "$OUT/c58_v9f_host_prod" | cut -c1-16)
printf 'g0\t%s\tgran0.cubin\t归档口径 16 B cp.async\n' "$g0" > "$OUT/expected.tsv"
printf 'g1\t%s\tgran1.cubin\t生产腿口径 4 B cp.async\n' "$g1" >> "$OUT/expected.tsv"
printf 'prod\t%s\tprod.cubin\t生产寻址 4 B cp.async（H-mem 臂；结构计数与 g1 全等）\n' "$pr" >> "$OUT/expected.tsv"
printf 'host\t%s\tc58_v9f_host\t宿主驱动件（与 rb1 同一件）\n' "$SHA_HOST_EXPECT" >> "$OUT/expected.tsv"
printf 'host2\t%s\tc58_v9f_host_prod\tH-mem 臂宿主（stream=线性 / prodlay=生产寻址）\n' "$h2" >> "$OUT/expected.tsv"
echo "SHA g0=$g0 g1=$g1 prod=$pr host=$SHA_HOST_EXPECT host2=$h2"
echo "WROTE $OUT/expected.tsv"
