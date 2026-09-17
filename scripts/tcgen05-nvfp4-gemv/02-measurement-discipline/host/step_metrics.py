#!/usr/bin/env python3
"""step_metrics.py — 生产口径步耗解析器（口径 = results/20260916-116 §0，与 p1_arm.sh 内联解析逐字同式）

    ms_per_step = 1000 × mean_len / tps
      mean_len : 服务端 `draft acceptance = ... mean len = X` 行
      tps      : bench jsonl 的 timings.predicted_per_second（回退 predicted_per_second）
      prompt_tokens : bench jsonl 的 start 行（口径守卫：ctx 不对 ⇒ 判据应 FAIL）

用法: step_metrics.py <server.log> <bench.jsonl>     # 输出 TSV: KEY<TAB>VALUE（verdict.py 可直接吃）
缺量时只打印 NO_STEP_METRICS 并 exit 1（让 verdict.py 判 SPEC_DEFECT，而不是猜一个数）。
"""
import json
import re
import sys

R_ACC = re.compile(r"draft acceptance = ([0-9.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s*([0-9.]+)")
R_EV = re.compile(r"\beval time =\s*([0-9.]+) ms /\s*([0-9]+) tokens \(\s*([0-9.]+) ms per token")


def parse_log(fn):
    acc = ev = None
    try:
        fh = open(fn, errors='replace')
    except OSError:
        return acc, ev
    for ln in fh:
        m = R_ACC.search(ln)
        if m:
            acc = (float(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4)))
        m = R_EV.search(ln)
        if m:
            ev = (float(m.group(1)), int(m.group(2)), float(m.group(3)))
    return acc, ev


def parse_bench(fn):
    tps = pn = ptok = None
    try:
        fh = open(fn, errors='replace')
    except OSError:
        return tps, pn, ptok
    for ln in fh:
        ln = ln.strip()
        if not ln.startswith('{'):
            continue
        try:
            d = json.loads(ln)
        except ValueError:
            continue
        if d.get('stage') == 'start' and d.get('prompt_tokens') is not None:
            ptok = int(d['prompt_tokens'])
        t = d.get('timings') or {}
        v = t.get('predicted_per_second') or d.get('predicted_per_second')
        if v:
            tps = float(v)
            pn = t.get('predicted_n') or d.get('tokens_predicted') or pn
    return tps, pn, ptok


def main(argv):
    if len(argv) < 3:
        print("USAGE: step_metrics.py <server.log> <bench.jsonl>")
        return 2
    acc, ev = parse_log(argv[1])
    tps, pn, ptok = parse_bench(argv[2])
    if not acc or not tps:
        print("NO_STEP_METRICS acc=%s tps=%s" % (bool(acc), tps))
        return 1
    mean_len = acc[3]
    step_ms = 1000.0 * mean_len / tps
    print("STEP_MS\t%.3f" % step_ms)
    print("MEAN_LEN\t%.3f" % mean_len)
    print("ACC_RATE\t%.5f" % acc[0])
    print("TPS\t%.3f" % tps)
    if ptok is not None:
        print("PROMPT_TOKENS\t%d" % ptok)
    if pn:
        print("PRED_N\t%d" % int(pn))
        print("STEPS\t%.1f" % (int(pn) / mean_len))
    if ev:
        print("EVAL_MS_PER_TOK\t%.3f" % ev[2])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
