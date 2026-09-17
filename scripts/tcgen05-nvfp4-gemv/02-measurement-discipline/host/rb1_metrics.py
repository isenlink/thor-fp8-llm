#!/usr/bin/env python3
"""rb1_metrics.py — 解析 R-B 第 1 臂（ws32k 复现）的臂输出 → verdict.py 可吃的 TSV

输入是 rb1_arm.sh 的输出（每条 JSON 读数为一行，臂之间用显式标记分隔）：
    ARM_BEGIN <arm>            ← 由运行脚本打印，避免"靠注释行猜是哪条臂"
    { ... "gbps":259.15 ... }  ← c58_v9f_host 的 JSON 读数
    ARM_END <arm> rc=<rc>
输出（每个臂取该臂内最后一条含 gbps 的 JSON）：
    RB1_<ARM>_GBPS / _DMISMATCH / _MBAR_OK / _PROD_FAIL / _RC / _SMEM
缺量 ⇒ 打印 NO_RB1_METRICS 并 exit 1（让 verdict.py 判 SPEC_DEFECT，不猜数）。
"""
import json
import re
import sys

R_BEGIN = re.compile(r"^ARM_BEGIN\s+(\S+)")
R_END = re.compile(r"^ARM_END\s+(\S+)\s+rc=(\d+)")


def main(argv):
    if len(argv) < 2:
        print("USAGE: rb1_metrics.py <arm_out>")
        return 2
    cur = None
    arms = {}
    for ln in open(argv[1], errors='replace'):
        ln = ln.strip()
        m = R_BEGIN.match(ln)
        if m:
            cur = m.group(1)
            arms.setdefault(cur, {})
            continue
        m = R_END.match(ln)
        if m:
            arms.setdefault(m.group(1), {})['RC'] = int(m.group(2))
            cur = None
            continue
        if not ln.startswith('{'):
            continue
        try:
            d = json.loads(ln)
        except ValueError:
            continue
        a = arms.setdefault(cur or 'UNKNOWN', {})
        if 'smem_required' in d:                      # env 行没有 gbps，但 smem 是生效校验量
            a['SMEM'] = float(d['smem_required'])
        if 'gbps' not in d:
            continue
        a['GBPS'] = float(d['gbps'])
        for k, src in (('DMISMATCH', 'd_mismatch'), ('MBAR_OK', 'mbar_ok'),
                       ('PROD_FAIL', 'prod_fail')):
            if src in d:
                a[k] = float(d[src])
    if not arms:
        print("NO_RB1_METRICS 无臂标记或无可解析读数")
        return 1
    for name in sorted(arms):
        for k in ('GBPS', 'DMISMATCH', 'MBAR_OK', 'PROD_FAIL', 'SMEM', 'RC'):
            if k in arms[name]:
                v = arms[name][k]
                print("RB1_%s_%s\t%s" % (name.upper(), k, ('%g' % v) if isinstance(v, float) else v))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
