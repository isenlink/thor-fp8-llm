#!/bin/bash
# selftest_rb2_arm.sh — rb2_arm.sh 的**接缝回归**（宿主侧，不碰板卡）
#   为什么需要：2026-09-17 的小错全部出在"接缝"（两个部件各自看着对、接缝没人验）。本脚本把
#   rb2_arm.sh 的路径与门脚本**替换到沙盒**、宿主换成**假宿主**（0.1 s 出数），把四条分支逐一跑出已知答案：
#     A 门脚本缺失 ⇒ 必须 ABORT missing_gate（旧写法会**静默开臂**——这就是被堵掉的洞）
#     B pre 门 BLOCKED ⇒ 必须 ABORT pre_blocked
#     C gran1 sha 不对 ⇒ 必须 ABORT cubin_sha
#     D 变体键不存在 ⇒ 必须 ABORT no_variant
#     E 全链路（正常门 + 假宿主）⇒ done=OK + 6 个 ARM 块 + 生效串 + 读数可解析
#   用法: bash t4/m2c/host/selftest_rb2_arm.sh
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
FIX=$ROOT/t4/m2c/stage/rb2
HOSTBIN=$ROOT/evidence/20260916-76-t4-3-b9f-ws32k/host/c58_v9f_host.stream1
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
mkdir -p "$W/rb2" "$W/m2c/logs"
pass=0; fail=0
chk() { if [ "$2" = "$3" ]; then echo "  PASS  $1"; pass=$((pass+1)); else echo "  FAIL  $1（期望 $2 实得 $3）"; fail=$((fail+1)); fi; }
chks() { if printf '%s' "$3" | grep -qF -- "$2"; then echo "  PASS  $1"; pass=$((pass+1)); else echo "  FAIL  $1（期望含 $2 实得 $3）"; fail=$((fail+1)); fi; }
for f in expected.tsv gran0.cubin gran1.cubin; do cp "$FIX/$f" "$W/rb2/" || exit 3; done
cp "$ROOT/t4/m2c/drivers/rb2_arm.sh" "$W/m2c/rb2_arm.sh"
sed -i "s#^R=/tmp/t4work/m2c#R=$W/m2c#; s#^D=/tmp/t4work/rb2#D=$W/rb2#" "$W/m2c/rb2_arm.sh"
# 假宿主：一次调用出 env + stream 两行（与真宿主同字段），0.1 s 完成，无需 GPU
cat > "$W/rb2/c58_v9f_host" <<'EOF'
#!/bin/bash
echo '{"tag":"env","mode":"stream","cc":"10.1","cuda_ver":12080,"shape":"M=128,N=32,K=256","sm_count":14,"chunks":4096,"grid":14,"chunk_bytes":4096,"threads":288,"smem_required":114752}'
echo '{"tag":"stream","ok":true,"grid":14,"chunks":4096,"bytes":234881024,"gbps":257.14,"per_sm_gbps":18.51,"d_mismatch":0,"first_bad_idx":-1,"mbar_ok":1,"prod_fail":0}'
EOF
H=$(sha256sum "$W/rb2/c58_v9f_host" | cut -c1-16); sed -i "s/^host\t[0-9a-f]*\t/host\t$H\t/" "$W/rb2/expected.tsv"
gate_ok() { cat > "$W/m2c/arm_gate.sh" <<'EOF'
#!/bin/bash
if [ "$1" = pre ]; then echo "[$2] GATE4_PRE boot=stub hpfree=23552 err=0 arms_this_boot=0/2 excl=0"; exit 0; fi
echo "[$2] GATE4=OK"; exit 0
EOF
}

echo "== rb2_arm.sh 接缝回归 =="
echo "--- A 门脚本缺失 ---"
gate_ok; mv "$W/m2c/arm_gate.sh" "$W/m2c/arm_gate.off"
o=$(bash "$W/m2c/rb2_arm.sh" tA g1 2>&1 | tail -1)
chk "门缺失必须停" "ABORT: 缺 $W/m2c/arm_gate.sh（门脚本不在 ⇒ 不许开臂）" "$o"
chk "门缺失的 done 标记" "ABORT missing_gate" "$(cat "$W/m2c/logs/tA.arm.done")"
mv "$W/m2c/arm_gate.off" "$W/m2c/arm_gate.sh"

echo "--- B pre 门 BLOCKED ---"
cat > "$W/m2c/arm_gate.sh" <<'EOF'
#!/bin/bash
if [ "$1" = pre ]; then echo "[$2] GATE4_PRE boot=stub"; echo "[$2] GATE4_PRE=BLOCKED 沙盒"; exit 2; fi
echo "[$2] GATE4=OK"; exit 0
EOF
bash "$W/m2c/rb2_arm.sh" tB g1 > /dev/null 2>&1
chk "pre BLOCKED 必须停" "ABORT pre_blocked" "$(cat "$W/m2c/logs/tB.arm.done")"

echo "--- C gran1 sha 不对 ---"
gate_ok; printf x >> "$W/rb2/gran1.cubin"
bash "$W/m2c/rb2_arm.sh" tC g1 > /dev/null 2>&1
chk "sha 不符必须停" "ABORT cubin_sha" "$(cat "$W/m2c/logs/tC.arm.done")"
cp "$FIX/gran1.cubin" "$W/rb2/gran1.cubin"

echo "--- D 变体键不存在 ---"
bash "$W/m2c/rb2_arm.sh" tD g9 > /dev/null 2>&1
chk "变体缺失必须停" "ABORT no_variant_g9" "$(cat "$W/m2c/logs/tD.arm.done")"

echo "--- E 全链路（正常门 + 假宿主）---"
bash "$W/m2c/rb2_arm.sh" tE g1 > /dev/null 2>&1
chk "全链路 done=OK" "OK" "$(cat "$W/m2c/logs/tE.arm.done")"
chk "6 个 ARM 块" "6" "$(grep -c '^ARM_BEGIN' "$W/m2c/logs/tE.out")"
chk "读数可解析（MAIN_GBPS）" $'RB1_MAIN_GBPS\t257.14' "$(python3 "$ROOT/t4/m2c/host/rb1_metrics.py" "$W/m2c/logs/tE.out" | grep MAIN_GBPS)"
chks "日志含变体与 cubin sha（证据链）" "variant=g1 cubin=gran1.cubin cubin_sha=aea2753b48fe16e7" "$(head -1 "$W/m2c/logs/tE.out")"

echo "RB2_ARM_SEAM pass=$pass fail=$fail"
[ "$fail" -eq 0 ] || exit 1
