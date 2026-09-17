#!/usr/bin/env python3
"""verdict.py — 把臂卡里**冻结的判据**变成机器可执行的判定（机制 B：每臂闸门升级 / C：数据可信度标记）。

由来：2026-09-17 T4-ISO-1 白烧一个 boot —— 卡里冻结的判据（`d_mismatch=0`）在本调用口径下**根本不可满足**，
而这一点在打臂前就能发现（对一个"已知好"的归档臂跑同一条判据，它也会 FAIL）。
本工具把「判据必须在已知好/已知坏样本上给出不同判定」变成机械步骤。

用法:
  verdict.py --spec S --metrics M [--label TAG] [--cred-out TSV]
  verdict.py --spec S --replay M1 M2 ...        # 判据可用性预检（好/坏样本判定必须不同）
格式:
  spec 行:    KEY <TAB> OP <TAB> VAL      OP ∈ MIN|MAX|EQ|NE     （`#` 开头为注释）
  metrics 行: KEY <TAB> VALUE
退出码: 0=PASS  1=FAIL  2=SPEC_DEFECT  3=用法/文件错
规则（机械，无例外）:
  · metrics 里缺 spec 要求的 KEY ⇒ **SPEC_DEFECT**（判据引用了测不到的量）
  · replay 若对所有样本给出同一判定 ⇒ **SPEC_UNSATISFIED**（判据没有判别力 ⇒ 不许用于上板）
  · 判定写进 --cred-out 时，cred 取值只有 V1_VERIFIED / V4_UNUSABLE 两种（由判据决定，不由叙述决定）
"""
import sys, os

def die(msg, code=3):
    print(msg); sys.exit(code)

def parse(fn, kind):
    rows = []
    for i, ln in enumerate(open(fn, encoding='utf-8', errors='replace'), 1):
        ln = ln.rstrip('\n')
        if not ln.strip() or ln.lstrip().startswith('#'):
            continue
        f = ln.split('\t') if '\t' in ln else ln.split()
        if kind == 'spec':
            if len(f) < 3: die('SPEC_BAD_LINE %s:%d %r' % (fn, i, ln))
            op = f[1].strip().upper()
            if op not in ('MIN', 'MAX', 'EQ', 'NE'): die('SPEC_BAD_OP %s:%d %s' % (fn, i, op))
            rows.append((f[0].strip(), op, float(f[2])))
        else:
            if len(f) < 2: die('METRICS_BAD_LINE %s:%d %r' % (fn, i, ln))
            try: v = float(f[1])
            except ValueError: v = float('nan')
            rows.append((f[0].strip(), v))
    return rows

def check(key, op, val, m):
    if key not in m: return ('SPEC_DEFECT', 'NO_METRIC')
    v = m[key]
    if v != v: return ('SPEC_DEFECT', 'METRIC_NAN')
    ok = {'MIN': v >= val, 'MAX': v <= val, 'EQ': v == val, 'NE': v != val}[op]
    return ('PASS' if ok else 'FAIL', '%g %s %g' % (v, op, val))

def main(argv):
    spec = metrics = label = cred_out = None; replay = []
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == '--spec': spec = argv[i+1]; i += 2
        elif a == '--metrics': metrics = argv[i+1]; i += 2
        elif a == '--label': label = argv[i+1]; i += 2
        elif a == '--cred-out': cred_out = argv[i+1]; i += 2
        elif a == '--replay': replay = argv[i+1:]; i = len(argv)
        else: die('USAGE: unknown arg %s' % a)
    if not spec or not os.path.exists(spec): die('SPEC_MISSING %s' % spec)
    sp = parse(spec, 'spec')
    if not sp: die('SPEC_EMPTY %s' % spec)

    if replay:
        verdicts = {}
        for fn in replay:
            m = dict(parse(fn, 'metrics'))
            res = [check(k, o, v, m)[0] for k, o, v in sp]
            v = 'SPEC_DEFECT' if 'SPEC_DEFECT' in res else ('FAIL' if 'FAIL' in res else 'PASS')
            verdicts.setdefault(v, []).append(os.path.basename(fn))
            print('  %-28s => %s' % (os.path.basename(fn), v))
        if len(replay) < 2:
            print('REPLAY=INSUFFICIENT 样本数 %d <2 ⇒ 无法确认判别力（至少给一个好样本+一个坏样本）' % len(replay))
            return 2
        if len(verdicts) < 2:
            print('REPLAY=SPEC_UNSATISFIED 全部样本同判(%s) ⇒ 判据无判别力，禁止用于上板（规则 B）'
                  % ','.join(verdicts))
            return 2
        print('REPLAY=OK 判别力确认（%s）' % ' / '.join('%s=%d' % (k, len(v)) for k, v in sorted(verdicts.items())))
        return 0

    if not metrics or not os.path.exists(metrics): die('METRICS_MISSING %s' % metrics)
    m = dict(parse(metrics, 'metrics'))
    res = [check(k, o, v, m)[0] for k, o, v in sp]
    overall = 'SPEC_DEFECT' if 'SPEC_DEFECT' in res else ('FAIL' if 'FAIL' in res else 'PASS')
    for (k, o, v), r in zip(sp, res):
        st, why = check(k, o, v, m)
        print('  %-24s spec=%-10s %-11s %s' % (k, '%s %g' % (o, v), st, why))
    tag = label or os.path.basename(metrics)
    print('VERDICT=%s label=%s items=%d' % (overall, tag, len(sp)))
    if cred_out:
        cred = 'V1_VERIFIED' if overall == 'PASS' else ('V4_UNUSABLE' if overall == 'FAIL' else 'SPEC_DEFECT')
        import time
        with open(cred_out, 'a', encoding='utf-8') as fh:
            fh.write('%s\t%s\t%s\tverdict.py\t%s\n' % (time.strftime('%FT%T'), tag, cred, spec))
        print('CRED_RECORDED %s %s -> %s' % (tag, cred, cred_out))
    return {'PASS': 0, 'FAIL': 1, 'SPEC_DEFECT': 2}[overall]

if __name__ == '__main__':
    sys.exit(main(sys.argv))
