#!/usr/bin/env python3
"""check_fixture_provenance.py — 夹具来源审计（防「自洽的假夹具」）

由来：2026-09-17 11:0x SPEC_DEFECT —— spec 的键名写错，而三个 fixture 也是照**同一个错键名**手敲的
⇒ 自洽的假工件让 `REPLAY=OK` 照过，「判据先过夹具」这条纪律被架空（DISCIPLINE §5.1）。
本工具把「夹具必须来自真工件」变成机械检查：
  · 每个 *.metrics 必须在 verdict_fixtures/PROVENANCE.tsv 里登记（未登记 ⇒ FAIL）
  · 登记为真工件来源的 ⇒ 源文件必须存在且 sha16 一致
  · 与源文件内容不同的 ⇒ 只许**恰好一行**不同，且该行的键必须等于登记的被改列
  · SYNTHETIC（手敲/无真工件来源）⇒ 不 FAIL，但必须打印出来：它是**可见的弱证据**，不当成已验证
用法: check_fixture_provenance.py [fixtures_dir]
退出: 0=全过 1=有 FAIL
"""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))      # project/
FX = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'verdict_fixtures')
PROV = os.path.join(FX, 'PROVENANCE.tsv')


def sha16(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(65536), b''):
            h.update(b)
    return h.hexdigest()[:16]


def lines(p):
    with open(p, encoding='utf-8', errors='replace') as fh:
        return fh.read().splitlines()


def key_of(ln):
    return (ln.split('\t')[0] if '\t' in ln else ln.split()[0]).strip()


def main():
    if not os.path.exists(PROV):
        print('NO_PROVENANCE %s（没有登记表 ⇒ 夹具来源无从审计）' % PROV)
        print('PROVENANCE fail=1')
        return 1
    rows, fails, synth = {}, [], []
    for i, ln in enumerate(open(PROV, encoding='utf-8'), 1):
        ln = ln.rstrip('\n')
        if not ln.strip() or ln.lstrip().startswith('#'):
            continue
        f = ln.split('\t')
        if len(f) < 4:
            fails.append('PROV_BAD_LINE %s:%d' % (os.path.basename(PROV), i))
            continue
        rows[f[0]] = dict(src=f[1], sha=f[2].strip(), col=f[3].strip(),
                          note=(f[4] if len(f) > 4 else ''))

    have = sorted(x for x in os.listdir(FX) if x.endswith('.metrics'))
    for name in have:
        r = rows.get(name)
        if r is None:
            fails.append('UNREGISTERED %s（必须在 PROVENANCE.tsv 登记来源）' % name)
            continue
        fp = os.path.join(FX, name)
        if r['src'] == 'SYNTHETIC':
            synth.append(name)
            continue
        sp = os.path.join(ROOT, r['src'])
        if not os.path.exists(sp):
            fails.append('SRC_MISSING %s <- %s' % (name, r['src']))
            continue
        got = sha16(sp)
        if got != r['sha']:
            fails.append('SRC_SHA %s <- %s（%s != 登记 %s）' % (name, r['src'], got, r['sha']))
            continue
        if open(sp, 'rb').read() == open(fp, 'rb').read():
            continue                                     # 原件：逐字节同
        a, b = lines(sp), lines(fp)
        if len(a) != len(b):
            fails.append('DERIVE_LINECOUNT %s（源 %d 行 vs 夹具 %d 行）' % (name, len(a), len(b)))
            continue
        ch = [k for k in range(len(a)) if a[k] != b[k]]
        if len(ch) != 1:
            fails.append('DERIVE_COLUMNS %s（改了 %d 行；派生夹具只许改 1 行）' % (name, len(ch)))
            continue
        if key_of(b[ch[0]]) != r['col']:
            fails.append('DERIVE_KEY %s（改的是 %s，登记的是 %s）' % (name, key_of(b[ch[0]]), r['col']))
    for name in sorted(rows):
        if not os.path.exists(os.path.join(FX, name)):
            fails.append('PROV_DANGLING %s（登记了但文件不存在）' % name)

    if synth:
        print('WARN_SYNTHETIC %d/%d 个夹具无真工件来源（手敲，弱证据）: %s'
              % (len(synth), len(have), ' '.join(synth)))
    for f in fails:
        print('FAIL ' + f)
    print('PROVENANCE checked=%d real_source=%d synthetic=%d' % (len(have), len(have) - len(synth), len(synth)))
    print('PROVENANCE fail=%d' % len(fails))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
