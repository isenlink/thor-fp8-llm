#!/bin/bash
# selftest.sh — 我方工具链的**夹具回归**（上板前必跑）：任何新写的判定/解析/换算，先用「已知答案」的数据跑一遍。
# 用法: bash t4/m2c/host/selftest.sh        # 全部 PASS 才允许把相关逻辑用于板端
# 由来：2026-09-17 一夜 8 处小错的共同根因是「新逻辑没在用之前验证过」；本脚本把该验证固定下来。
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
H=$ROOT/t4/m2c/host
D=$ROOT/t4/m2c/drivers
F113=$ROOT/evidence/20260916-113-t4-m2c-logp/local
LOC=$ROOT/t4/m2c/local
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0; skip=0
chk() { # <名称> <期望子串> <实际输出>
  if printf '%s' "$3" | grep -qF -- "$2"; then echo "  PASS  $1"; pass=$((pass+1));
  else echo "  FAIL  $1"; echo "        期望含: $2"; printf '%s\n' "$3" | sed 's/^/        实际: /' | head -6; fail=$((fail+1)); fi
}

echo "== 1. leg_rate.py 口径（对 113 归档，期望 85.4 / 100.9 GB/s）=="
o=$(python3 "$H/leg_rate.py" "$F113/op_prof-prof8koff.txt" --label off 2>&1); chk "leg_rate off=85.4" "=> 85.4 GB/s" "$o"
o=$(python3 "$H/leg_rate.py" "$F113/op_prof-prof8kon.txt" --label on 2>&1);  chk "leg_rate on=100.9" "=> 100.9 GB/s" "$o"
o=$(python3 "$H/leg_rate.py" /nonexistent 2>&1); chk "leg_rate 缺文件友好报错" "NO_PROF_FILE" "$o"

echo "== 2. p2_cmp.py 主判据=逐位 token（合成夹具：第 2 位起翻转 ⇒ 2/4）=="
python3 - "$TMP" <<'PY'
import json, sys
d=sys.argv[1]
def mk(p, flip=None):
    toks=["ab","cd","ef","gh","ij"]; cp=[]
    for i in range(4):
        t=toks[i] if (flip is None or i<flip) else toks[i+1]
        cp.append({"id":i,"token":t,"logprob":-0.1,"top_logprobs":[{"id":i,"token":t,"logprob":-0.1}]})
    open(p,"w").write(json.dumps({"stage":"start","prompt_tokens":10})+"\n"+json.dumps({"stage":"result","completion_probabilities":cp})+"\n")
mk(d+"/a.jsonl"); mk(d+"/b.jsonl", flip=2)
PY
o=$(python3 "$D/p2_cmp.py" "$TMP/a.jsonl" "$TMP/b.jsonl" 2>&1); chk "合成翻转 idx=2 检出" "首次分歧 idx=2" "$o"
chk "合成一致率 2/4" "TOKEN_IDENTICAL=2/4" "$o"

echo "== 3. p2_cmp.py 对真实归档（本地 T4 off/on 128 token）=="
if [ -f "$LOC/m2c-nspoff128.logp.jsonl" ]; then
  o=$(python3 "$D/p2_cmp.py" "$LOC/m2c-nspoff128.logp.jsonl" "$LOC/m2c-nspon128.logp.jsonl" 2>&1)
  chk "真实 128 token 全一致" "TOKEN_IDENTICAL=128/128" "$o"
else echo "  SKIP  (缺 $LOC/m2c-nspoff128.logp.jsonl)"; fi

echo "== 4. 空值不得计成一致（参照只在首 token 给 top_logprobs 的形态）=="
python3 - "$TMP" <<'PY'
import json, sys
d=sys.argv[1]
def mk(p, e):
    cp=[{"id":1,"token":"A","logprob":0.0,"top_logprobs":[{"id":1,"token":"A","logprob":0.0}],"probs":[{"tok_str":"A","prob":0.9}]},
        {"id":2,"token":e,"logprob":0.0,"top_logprobs":[],"probs":[]}]
    open(p,"w").write(json.dumps({"stage":"start","prompt_tokens":5})+"\n"+json.dumps({"stage":"result","completion_probabilities":cp})+"\n")
mk(d+"/e1.jsonl","B"); mk(d+"/e2.jsonl","C")   # 第 2 位不同但两侧 top_logprobs 都空 ⇒ 必须判为分歧
PY
o=$(python3 "$D/p2_cmp.py" "$TMP/e1.jsonl" "$TMP/e2.jsonl" 2>&1); chk "空 top_logprobs 仍按 token 判分歧" "TOKEN_IDENTICAL=1/2" "$o"

echo "== 5. 板端/宿主驱动清单完整性（缺件即失败）=="
missing=0
for f in arm_gate.sh m2c_run.sh ring_cfg_sweep.sh ares_seq.sh p1_arm.sh p2_oracle.sh p3_arm.sh m2c_serve.sh logp_run.sh logp_probe.py stop8091.sh found1_arm.sh; do
  [ -f "$D/$f" ] || { echo "  FAIL  清单文件缺失: $f"; missing=1; }
done
grep -q "p3_arm.sh" "$D/upload_drivers.sh" || { echo "  FAIL  upload_drivers.sh 未包含 p3_arm.sh"; missing=1; }
grep -q "logp_run.sh" "$D/upload_drivers.sh" || { echo "  FAIL  upload_drivers.sh 未包含 logp_run.sh"; missing=1; }
[ "$missing" -eq 0 ] && { echo "  PASS  板端清单与上传清单一致"; pass=$((pass+1)); } || fail=$((fail+1))

echo "== 6. 驱动级判据解析（BAD 必须停手）=="
for f in p1_driver.sh p2_driver.sh p3_driver.sh; do
  if grep -qE "GATE=OK|GATE4=OK" "$D/$f" && grep -qE "停手|STOP|ABORT|exit 1" "$D/$f"; then
    echo "  PASS  $f 解析闸门并含停手分支"; pass=$((pass+1))
  else echo "  FAIL  $f 缺少闸门解析或停手分支"; fail=$((fail+1)); fi
done

echo "== 7. R-B 第 1 步：权重离线重排（对真实生产权重夹具）=="
if command -v gcc >/dev/null 2>&1; then
  S=$ROOT/t4/relayout
  if gcc -O2 -o "$TMP/t4_1_equiv" "$S/test_relayout_equiv.c" "$S/nvfp4_relayout.c" -I"$S" -lm 2>/dev/null \
  && gcc -O2 -o "$TMP/t4_1_real"  "$S/test_relayout_real.c"  "$S/nvfp4_relayout.c" -I"$S" -lm 2>/dev/null; then
    chk "relayout 等价性 mismatch=0" '"mismatch":0' "$("$TMP/t4_1_equiv" 2>&1)"
    chk "relayout 真实权重 256x17408 mismatch=0" '"mismatch":0' "$("$TMP/t4_1_real" "$S/raw.bin" "$S/meta.txt" 2>&1)"
    chk "relayout ok:true（真实权重臂）" '"ok":true' "$("$TMP/t4_1_real" "$S/raw.bin" "$S/meta.txt" 2>&1 | tail -1)"
  else echo "  FAIL  relayout 测试编译失败"; fail=$((fail+1)); fi
else echo "  SKIP  无 gcc"; fi

echo
echo "== 8. R-B 第 3 步：mxf4nvf4 描述符（idesc/smem 原子/SF 记录语义）=="
# 需要 CUTLASS 头 + 与 stageA 证据同版本的 sha256；缺件 = 显式 SKIP（不算通过）
o=$(bash "$H/desc_mxf4_check.sh" 2>&1); rc=$?
if [ $rc -eq 3 ]; then
  echo "  SKIP  $o"; skip=$((skip+1))
else
  chk "desc_mxf4 全部对拍 fail=0" "fail=0" "$o"
fi

echo
echo "== 9. R-B 第 2 步：激活侧量化 + canonical 打包（对 ggml 参考口径 + 已验 B 侧）=="
if command -v gcc >/dev/null 2>&1; then
  S=$ROOT/t4/relayout
  if gcc -O2 -o "$TMP/t4_act" "$S/test_act_quant.c" "$S/nvfp4_act_quant.c" "$S/nvfp4_relayout.c" -I"$S" -lm 2>"$TMP/act_build.log"; then
    chk "act_quant 全部夹具 fail=0" "ACT_QUANT fail=0" "$("$TMP/t4_act" 2>&1)"
  else echo "  FAIL  act_quant 测试编译失败"; tail -5 "$TMP/act_build.log" | sed 's/^/        /'; fail=$((fail+1)); fi
else echo "  SKIP  无 gcc"; skip=$((skip+1)); fi

echo
echo "== 10. verdict.py 判据判定 + 判据可用性预检（机制 B；对夹具的已知答案）=="
FX=$ROOT/t4/m2c/host/verdict_fixtures
o=$(python3 "$H/verdict.py" --spec "$FX/found1.spec" --metrics "$FX/found1_good.metrics" --label fix-good 2>&1); rc=$?
chk "verdict 好样本 PASS" "VERDICT=PASS" "$o"; [ "$rc" -eq 0 ] || { echo "  FAIL  期望 rc=0 实得 $rc"; fail=$((fail+1)); }
o=$(python3 "$H/verdict.py" --spec "$FX/found1.spec" --metrics "$FX/found1_bad.metrics" 2>&1); rc=$?
chk "verdict 坏样本 FAIL" "VERDICT=FAIL" "$o"; [ "$rc" -eq 1 ] || { echo "  FAIL  期望 rc=1 实得 $rc"; fail=$((fail+1)); }
o=$(python3 "$H/verdict.py" --spec "$FX/noKey.spec" --metrics "$FX/noKey.metrics" 2>&1); rc=$?
chk "verdict 缺量 SPEC_DEFECT" "VERDICT=SPEC_DEFECT" "$o"; [ "$rc" -eq 2 ] || { echo "  FAIL  期望 rc=2 实得 $rc"; fail=$((fail+1)); }
o=$(python3 "$H/verdict.py" --spec "$FX/found1.spec" --replay "$FX/found1_good.metrics" "$FX/found1_bad.metrics" 2>&1)
chk "replay 有判别力" "REPLAY=OK" "$o"
o=$(python3 "$H/verdict.py" --spec "$FX/dematch.spec" --replay "$FX/dematch_good.metrics" "$FX/dematch_bad.metrics" 2>&1)
chk "replay 判据无判别力必须拦下（ISO-1 教训）" "REPLAY=SPEC_UNSATISFIED" "$o"
o=$(bash "$H/found1_metrics.sh" "$F113/op_prof-prof8koff.txt" "$F113/op_prof-prof8kon.txt" /dev/stdout 2>/dev/null)
chk "found1_metrics 归档值 off=85.4" "LEG_RATE_OFF	85.4" "$o"
chk "found1_metrics 归档值 on=100.9" "LEG_RATE_ON	100.9" "$o"

echo
echo "== 11. found1_driver.sh 编排判据谓词（GATE4 解析：BAD/BLOCKED 不得判 OK）=="
o=$(bash "$D/found1_driver.sh" --selftest 2>&1)
chk "编排谓词夹具 fail=0" "FOUND1_DRIVER_SELFTEST pass=4 fail=0" "$o"

echo
echo "== 12. found1_arm.sh 完成谓词（无 DONE 行必须判 INCOMPLETE）=="
o=$(bash "$D/found1_arm.sh" --selftest 2>&1)
chk "臂完成+生效谓词夹具 fail=0" "FOUND1_ARM_SELFTEST pass=7 fail=0" "$o"

echo
echo "== 13. B5 环配置臂的判据与单臂指标口径 =="
FX=$ROOT/t4/m2c/host/verdict_fixtures
o=$(python3 "$H/verdict.py" --spec "$FX/cps7.spec" --replay "$FX/cps7_pass.metrics" "$FX/cps7_fail.metrics" 2>&1)
chk "cps7 判据有判别力" "REPLAY=OK" "$o"
o=$(bash "$H/found1_metrics.sh" "$ROOT/evidence/20260917-0220-found1-leg-rate/found1b.op_prof.txt" - 2>/dev/null)
chk "单臂模式出 LEG_RATE" "LEG_RATE	101.0" "$o"
o=$(python3 "$H/verdict.py" --spec "$FX/cps7.spec" --metrics "$FX/cps7_pass.metrics" 2>&1)
chk "cps7 达标样本 PASS" "VERDICT=PASS" "$o"

echo
echo "== 14. B6 生产口径步耗解析（step_metrics.py 对夹具的已知答案）+ 判据 =="
o=$(python3 "$D/step_metrics.py" "$FX/steplog_8k_fixture.log" "$FX/stepbench_8k_fixture.jsonl" 2>&1)
chk "step_metrics 8K 夹具 step=194.231" $'STEP_MS\t194.231' "$o"
chk "step_metrics 8K 夹具 mean_len=6.330" $'MEAN_LEN\t6.330' "$o"
o=$(python3 "$D/step_metrics.py" "$FX/steplog_200k_fixture.log" "$FX/stepbench_200k_fixture.jsonl" 2>&1)
chk "step_metrics 200K 夹具 step=254.140（= 116 §2 的 254.120）" $'STEP_MS\t254.140' "$o"
chk "step_metrics 200K 夹具 prompt_tokens=199979" $'PROMPT_TOKENS\t199979' "$o"
o=$(python3 "$D/step_metrics.py" "$FX/steplog_bad.log" "$FX/stepbench_bad_fixture.jsonl" 2>&1)
chk "step_metrics 缺量必须 NO_STEP_METRICS" "NO_STEP_METRICS" "$o"
o=$(python3 "$H/verdict.py" --spec "$FX/b6_200k.spec" --replay "$FX/b6_200k_pass.metrics" "$FX/b6_200k_fail.metrics" "$FX/b6_200k_wrongctx.metrics" 2>&1)
chk "b6 判据有判别力（含口径守卫）" "REPLAY=OK" "$o"
o=$(python3 "$H/verdict.py" --spec "$FX/b6_200k.spec" --metrics "$FX/b6_200k_wrongctx.metrics" 2>&1)
chk "b6 口径不对（8K prompt）必须 FAIL" "VERDICT=FAIL" "$o"
o=$(python3 "$H/verdict.py" --spec "$FX/b6_200k.spec" --metrics "$FX/b6_200k_pass.metrics" 2>&1)
chk "b6 达标样本 PASS" "VERDICT=PASS" "$o"
o=$(bash "$D/b6_arm.sh" --selftest 2>&1)
chk "b6 臂完成+生效谓词夹具 fail=0" "B6_ARM_SELFTEST pass=6 fail=0" "$o"
o=$(bash "$D/b6_driver.sh" --selftest 2>&1)
chk "b6 编排谓词夹具 fail=0" "B6_DRIVER_SELFTEST pass=10 fail=0" "$o"
o=$(python3 "$H/rb1_metrics.py" "$FX/rb1_good.out" 2>&1)
chk "rb1 解析：主臂 gbps=259.15" $'RB1_MAIN_GBPS\t259.15' "$o"
chk "rb1 解析：对照臂（纯拷贝）gbps=261.47" $'RB1_COPY_GBPS\t261.47' "$o"
chk "rb1 解析：冒烟 smem=114752" $'RB1_SMOKE_SMEM\t114752' "$o"
chk "rb1 解析：扩展性臂 grid1=23.58（真夹具，不拼期望值）" $'RB1_GRID1_GBPS\t23.58' "$(python3 "$H/rb1_metrics.py" "$FX/rb1_good.out" 2>&1)"
chk "rb1 解析：扩展性臂 grid7=167.15" $'RB1_GRID7_GBPS\t167.15' "$(python3 "$H/rb1_metrics.py" "$FX/rb1_good.out" 2>&1)"
o=$(python3 "$H/verdict.py" --spec "$FX/rb1.spec" --replay "$FX/rb1_good.out.metrics" "$FX/rb1_bad.out.metrics" 2>&1)
chk "rb1 判据有判别力" "REPLAY=OK" "$o"
o=$(python3 "$H/verdict.py" --spec "$FX/rb1.spec" --metrics "$FX/rb1_good.out.metrics" 2>&1)
chk "rb1 board4 好样本 PASS" "VERDICT=PASS" "$o"
o=$(python3 "$H/verdict.py" --spec "$FX/rb1.spec" --metrics "$FX/rb1_bad.out.metrics" 2>&1)
chk "rb1 坏样本 FAIL" "VERDICT=FAIL" "$o"
o=$(python3 "$H/verdict.py" --spec "$FX/b6_200k_off.spec" --replay "$FX/b6_200k_off_pass.metrics" "$FX/b6_200k_off_fail.metrics" 2>&1)
chk "b6-off 校准判据有判别力（±5% 带）" "REPLAY=OK" "$o"
o=$(python3 "$H/verdict.py" --spec "$FX/b6_200k_off.spec" --metrics "$FX/b6_200k_off_pass.metrics" 2>&1)
chk "b6-off 复现样本 PASS" "VERDICT=PASS" "$o"
o=$(python3 "$H/verdict.py" --spec "$FX/b6_200k_off.spec" --metrics "$FX/b6_200k_off_fail.metrics" 2>&1)
chk "b6-off 偏离样本 FAIL" "VERDICT=FAIL" "$o"

echo
echo "== 14. 夹具来源审计（防「自洽的假夹具」：spec 键名与手敲夹具同错同源 ⇒ REPLAY 照过）=="
o=$(python3 "$H/check_fixture_provenance.py" 2>&1); rc=$?
chk "夹具来源审计 fail=0" "PROVENANCE fail=0" "$o"
[ "$rc" -eq 0 ] || { echo "  FAIL  期望 rc=0 实得 $rc"; fail=$((fail+1)); }
cp -r "$FX" "$TMP/fxneg" 2>/dev/null
printf 'STEP_MS\t1\n' > "$TMP/fxneg/unregistered_by_selftest.metrics"
o=$(python3 "$H/check_fixture_provenance.py" "$TMP/fxneg" 2>&1)
chk "未登记夹具必须被拦下（负例）" "UNREGISTERED unregistered_by_selftest.metrics" "$o"

echo
echo "== 15. R-B2 判别臂判据组（gran0 锚点 / gran1 的 H1 成立与证伪）=="
o=$(python3 "$H/verdict.py" --spec "$FX/rb2_g0.spec" --replay "$FX/rb2_g0_good.metrics" "$FX/rb2_g0_bad.metrics" 2>&1)
chk "rb2 g0 锚点判据有判别力" "REPLAY=OK" "$o"
o=$(python3 "$H/verdict.py" --spec "$FX/rb2_g1_h1.spec" --replay "$FX/rb2_g1_low.metrics" "$FX/rb2_g1_high.metrics" 2>&1)
chk "rb2 H1 成立判据有判别力" "REPLAY=OK" "$o"
o=$(python3 "$H/verdict.py" --spec "$FX/rb2_g1_h0.spec" --replay "$FX/rb2_g1_high.metrics" "$FX/rb2_g1_low.metrics" 2>&1)
chk "rb2 H1 证伪判据有判别力" "REPLAY=OK" "$o"
o=$(python3 "$H/verdict.py" --spec "$FX/rb2_g0.spec" --metrics "$FX/rb2_g0_good.metrics" 2>&1)
chk "rb2 g0 对真臂读数（257.14）PASS" "VERDICT=PASS" "$o"
o=$(python3 "$H/verdict.py" --spec "$FX/rb2_g1_h1.spec" --metrics "$FX/rb2_g1_high.metrics" 2>&1)
chk "gran1 若停在 257 ⇒ H1 侧 FAIL" "VERDICT=FAIL" "$o"
o=$(python3 "$H/verdict.py" --spec "$FX/rb2_g1_h0.spec" --metrics "$FX/rb2_g1_high.metrics" 2>&1)
chk "gran1 若停在 257 ⇒ H0 侧 PASS" "VERDICT=PASS" "$o"

echo
echo "== 16. rb2_arm.sh 接缝回归（宿主侧沙盒 + 假宿主：门缺失/门BLOCKED/sha不符/变体缺失/全链路）=="
o=$(bash "$H/selftest_rb2_arm.sh" 2>&1)
chk "rb2_arm 接缝 fail=0" "RB2_ARM_SEAM pass=9 fail=0" "$o"
o=$(grep -c '^  FAIL' <<<"$(bash "$H/selftest_rb2_arm.sh" 2>&1)")
chk "rb2_arm 接缝无 FAIL 行" "0" "$o"

echo
echo "== 17. R-B2 H-mem 生产寻址臂（判据组 + rb2_mem_arm.sh 接缝）=="
o=$(python3 "$H/verdict.py" --spec "$FX/rb2_mem_anchor.spec" --replay "$FX/rb2_mem_prod_high.metrics" "$FX/rb2_mem_prod_low.metrics" 2>&1)
chk "rb2_mem anchor 判据有判别力" "REPLAY=OK" "$o"
o=$(python3 "$H/verdict.py" --spec "$FX/rb2_mem_h1.spec" --replay "$FX/rb2_mem_prod_low.metrics" "$FX/rb2_mem_prod_high.metrics" 2>&1)
chk "H-mem 成立判据有判别力" "REPLAY=OK" "$o"
o=$(python3 "$H/verdict.py" --spec "$FX/rb2_mem_h0.spec" --replay "$FX/rb2_mem_prod_high.metrics" "$FX/rb2_mem_prod_low.metrics" 2>&1)
chk "H-mem 否证判据有判别力" "REPLAY=OK" "$o"
o=$(python3 "$H/verdict.py" --spec "$FX/rb2_mem_integrity.spec" --replay "$FX/rb2_mem_prod_high.metrics" "$FX/rb2_mem_prod_dm.metrics" 2>&1)
chk "读数可用性判据有判别力（完整性行真的有牙）" "REPLAY=OK" "$o"
o=$(python3 "$H/verdict.py" --spec "$FX/rb2_mem_h1.spec" --metrics "$FX/rb2_mem_prod_low.metrics" 2>&1)
chk "prod 停在 140 ⇒ 成立侧 PASS" "VERDICT=PASS" "$o"
o=$(python3 "$H/verdict.py" --spec "$FX/rb2_mem_h0.spec" --metrics "$FX/rb2_mem_prod_low.metrics" 2>&1)
chk "prod 停在 140 ⇒ 否证侧 FAIL" "VERDICT=FAIL" "$o"
o=$(python3 "$H/verdict.py" --spec "$FX/rb2_mem_h0.spec" --metrics "$FX/rb2_mem_prod_high.metrics" 2>&1)
chk "prod 停在 226.85 ⇒ 否证侧 PASS" "VERDICT=PASS" "$o"
o=$(python3 "$H/verdict.py" --spec "$FX/rb2_mem_integrity.spec" --metrics "$FX/rb2_mem_prod_dm.metrics" 2>&1)
chk "d_mismatch=5 ⇒ 读数判为不可用" "VERDICT=FAIL" "$o"
o=$(python3 "$H/verdict.py" --spec "$FX/rb2_mem_anchor.spec" --metrics "$FX/rb2_mem_prod_high.metrics" 2>&1)
chk "anchor 对真臂读数（226.85）PASS" "VERDICT=PASS" "$o"
o=$(python3 "$H/check_prodaddr_fill.py" 2>&1)
chk "宿主填字节↔内核生产寻址 = canonical（逐字节恒等，CPR=20）" "OK PRODADDR_FILL_IDENTICAL" "$o"
o=$(python3 "$H/check_prodaddr_fill.py" --cpr 68 2>&1)
chk "同一恒等式在第二真实形状（K=17408, CPR=68）也成立" "OK PRODADDR_FILL_IDENTICAL" "$o"
o=$(printf 'x' | python3 "$H/check_prodaddr_fill.py" 2>&1; echo "rc=$?")
chk "恒等夹具不得因输入退化而假过" "rc=0" "$o"
o=$(bash "$H/selftest_rb2_mem_arm.sh" 2>&1)
chk "rb2_mem_arm 接缝 fail=0" "RB2_MEM_ARM_SEAM pass=14 fail=0" "$o"
o=$(grep -c '^  FAIL' <<<"$(bash "$H/selftest_rb2_mem_arm.sh" 2>&1)")
chk "rb2_mem_arm 接缝无 FAIL 行" "0" "$o"

echo
echo "SELFTEST pass=$pass fail=$fail skip=$skip"
[ "$fail" -eq 0 ] || exit 1
