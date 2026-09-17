#!/bin/bash
# selftest_rb2_mem_arm.sh — rb2_mem_arm.sh 的**接缝回归**（宿主侧，不碰板卡）
#   本臂新增的接缝 = 「模式(mode) + cubin」两件一起选：写错任一（例如 anchor 却传 prodlay）读数仍会出来，
#   但**比的不是同一样东西**。所以这里用假宿主把 argv 记下来，逐条对照已知答案：
#     A 门脚本缺失 ⇒ ABORT missing_gate      B pre 门 BLOCKED ⇒ ABORT pre_blocked
#     C prod.cubin sha 不符 ⇒ ABORT cubin_sha D host2 sha 不符 ⇒ ABORT host_sha
#     E 变体键缺失 ⇒ ABORT bad_variant       F **模式/cubin 接线**：anchor⇒(stream,gran1) / prod⇒(prodlay,prod.cubin)
#     G 全链路 ⇒ done=OK + 6 个 ARM 块 + 生效串 + 读数可解析
#   用法: bash t4/m2c/host/selftest_rb2_mem_arm.sh
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
FIX=$ROOT/t4/m2c/stage/rb2
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
mkdir -p "$W/rb2" "$W/m2c/logs"
pass=0; fail=0
chk() { if [ "$2" = "$3" ]; then echo "  PASS  $1"; pass=$((pass+1)); else echo "  FAIL  $1（期望 $2 实得 $3）"; fail=$((fail+1)); fi; }
chks() { if printf '%s' "$3" | grep -qF -- "$2"; then echo "  PASS  $1"; pass=$((pass+1)); else echo "  FAIL  $1（期望含 $2 实得 $3）"; fail=$((fail+1)); fi; }
for f in expected.tsv gran0.cubin gran1.cubin prod.cubin; do cp "$FIX/$f" "$W/rb2/" || exit 3; done
cp "$ROOT/t4/m2c/drivers/rb2_mem_arm.sh" "$W/m2c/rb2_mem_arm.sh"
sed -i "s#^R=/tmp/t4work/m2c#R=$W/m2c#; s#^D=/tmp/t4work/rb2#D=$W/rb2#" "$W/m2c/rb2_mem_arm.sh"
# 假宿主：把 argv 记到文件（用于核对接线），再出与真宿主同字段的两行
cat > "$W/rb2/c58_v9f_host_prod" <<'FAKEHOST_EOF'
#!/bin/bash
echo "$1 $2" >> "${FAKE_ARGV:-/dev/null}"
echo '{"tag":"env","mode":"stream","cc":"10.1","cuda_ver":12080,"shape":"M=128,N=32,K=256","sm_count":14,"chunks":4096,"grid":14,"chunk_bytes":4096,"threads":288,"smem_required":114752}'
echo '{"tag":"stream","ok":true,"grid":14,"chunks":4096,"bytes":234881024,"gbps":226.85,"per_sm_gbps":17.06,"d_mismatch":0,"first_bad_idx":-1,"mbar_ok":1,"prod_fail":0}'
FAKEHOST_EOF
H=$(sha256sum "$W/rb2/c58_v9f_host_prod" | cut -c1-16)
cp "$W/rb2/c58_v9f_host_prod" "$W/fakehost.orig"     # D 步破坏后从这里恢复（不要从 stage 恢复真件）
sed -i "s/^host2\t[0-9a-f]*\t/host2\t$H\t/" "$W/rb2/expected.tsv"
export FAKE_ARGV="$W/argv.txt"
gate_ok() { cat > "$W/m2c/arm_gate.sh" <<'GATESTUB_EOF'
#!/bin/bash
if [ "$1" = pre ]; then echo "[$2] GATE4_PRE boot=stub hpfree=23552 err=0 arms_this_boot=0/2 excl=0"; exit 0; fi
echo "[$2] GATE4=OK"; exit 0
GATESTUB_EOF
}

echo "== rb2_mem_arm.sh 接缝回归 =="
echo "--- A 门脚本缺失 ---"
gate_ok; mv "$W/m2c/arm_gate.sh" "$W/m2c/arm_gate.off"
o=$(bash "$W/m2c/rb2_mem_arm.sh" tA prod 2>&1 | tail -1)
chk "门缺失必须停" "ABORT: 缺 $W/m2c/arm_gate.sh（门脚本不在 ⇒ 不许开臂）" "$o"
chk "门缺失的 done 标记" "ABORT missing_gate" "$(cat "$W/m2c/logs/tA.arm.done")"
mv "$W/m2c/arm_gate.off" "$W/m2c/arm_gate.sh"

echo "--- B pre 门 BLOCKED ---"
cat > "$W/m2c/arm_gate.sh" <<'BLOCKSTUB_EOF'
#!/bin/bash
if [ "$1" = pre ]; then echo "[$2] GATE4_PRE boot=stub"; echo "[$2] GATE4_PRE=BLOCKED 沙盒"; exit 2; fi
echo "[$2] GATE4=OK"; exit 0
BLOCKSTUB_EOF
bash "$W/m2c/rb2_mem_arm.sh" tB prod > /dev/null 2>&1
chk "pre BLOCKED 必须停" "ABORT pre_blocked" "$(cat "$W/m2c/logs/tB.arm.done")"

echo "--- C prod.cubin sha 不符 ---"
gate_ok; printf x >> "$W/rb2/prod.cubin"
bash "$W/m2c/rb2_mem_arm.sh" tC prod > /dev/null 2>&1
chk "cubin sha 不符必须停" "ABORT cubin_sha" "$(cat "$W/m2c/logs/tC.arm.done")"
cp "$FIX/prod.cubin" "$W/rb2/prod.cubin"

echo "--- D host2 sha 不符 ---"
printf x >> "$W/rb2/c58_v9f_host_prod"
bash "$W/m2c/rb2_mem_arm.sh" tD prod > /dev/null 2>&1
chk "host sha 不符必须停" "ABORT host_sha" "$(cat "$W/m2c/logs/tD.arm.done")"
cp "$W/fakehost.orig" "$W/rb2/c58_v9f_host_prod"; chmod +x "$W/rb2/c58_v9f_host_prod"

echo "--- E 变体键不存在 ---"
bash "$W/m2c/rb2_mem_arm.sh" tE zz > /dev/null 2>&1
chk "变体缺失必须停" "ABORT bad_variant" "$(cat "$W/m2c/logs/tE.arm.done")"

echo "--- F 模式/cubin 接线（anchor 必须 stream+gran1；prod 必须 prodlay+prod.cubin）---"
: > "$W/argv.txt"
bash "$W/m2c/rb2_mem_arm.sh" tF-anchor anchor > /dev/null 2>&1
bash "$W/m2c/rb2_mem_arm.sh" tF-prod prod > /dev/null 2>&1
chk "anchor 接线 6/6（stream+gran1）" "6" "$(grep -c '^stream gran1\.cubin$' "$W/argv.txt")"
chk "prod 接线 6/6（prodlay+prod.cubin）" "6" "$(grep -c '^prodlay prod\.cubin$' "$W/argv.txt")"
chk "两臂共 12 次调用且无第三种组合" "12 2" "$(wc -l < "$W/argv.txt" | tr -d ' ') $(sort -u "$W/argv.txt" | wc -l | tr -d ' ')"
chks "anchor 日志证据链" "variant=anchor mode=stream cubin=gran1.cubin" "$(head -1 "$W/m2c/logs/tF-anchor.out")"
chks "prod 日志证据链" "variant=prod mode=prodlay cubin=prod.cubin" "$(head -1 "$W/m2c/logs/tF-prod.out")"

echo "--- G 全链路 ---"
chk "全链路 done=OK" "OK" "$(cat "$W/m2c/logs/tF-prod.arm.done")"
chk "6 个 ARM 块" "6" "$(grep -c '^ARM_BEGIN' "$W/m2c/logs/tF-prod.out")"
chk "读数可解析（MAIN_GBPS）" $'RB1_MAIN_GBPS\t226.85' "$(python3 "$ROOT/t4/m2c/host/rb1_metrics.py" "$W/m2c/logs/tF-prod.out" | grep MAIN_GBPS)"

echo "RB2_MEM_ARM_SEAM pass=$pass fail=$fail"
[ "$fail" -eq 0 ] || exit 1
