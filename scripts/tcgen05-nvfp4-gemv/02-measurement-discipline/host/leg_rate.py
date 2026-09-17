#!/usr/bin/env python3
"""leg_rate.py — 从 GGML_OP_PROF dump 算 NVFP4「腿」的有效速率（GB/s），口径与 results/20260916-113 §6 一致。

用法: leg_rate.py <op_prof.txt> [--bytes-per-eval 11.01e9] [--label 名字]
算法（显式，不猜）：
  · bucket 行格式: <name> <calls> <total_ms> <avg_us> <pct>% <tflops>
  · 腿 = 所有名字里带 `NVFP4` 的 `MUL_MAT*` bucket 合计 total_ms（注意 gate+up 桶名是 `MUL_MAT:ffn:NVFP4:N17408`）
  · evals/window = calls(`:NVFP4:N17408` 桶) / 128   （模型 64 层 × gate/up 两张 = 128 calls/eval）
     交叉校验：calls(`MUL_MAT:F8:N5120`) / 64 应给出同一个数（不一致就打印告警）
  · GB/s = bytes_per_eval / (腿 ms per eval)
默认 bytes_per_eval = 11.01e9（`results/20260916-113 §6` 实测 call 计数推出的整腿字节/eval，冻结值）
"""
import re, sys

path = sys.argv[1]
args = sys.argv[2:]
BPE = 11.01e9
label = path
for i, a in enumerate(args):
    if a == '--bytes-per-eval': BPE = float(args[i+1])
    if a == '--label': label = args[i+1]

rows = []
try:
    fh = open(path, encoding='utf-8', errors='replace')
except FileNotFoundError:
    print('NO_PROF_FILE %s（本臂未产出 op_prof dump）' % path); sys.exit(1)
for ln in fh:
    m = re.match(r'^(\S.*?)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%\s+([\d.]+)\s*$', ln.rstrip())
    if m:
        rows.append(dict(bucket=m.group(1).strip(), calls=int(m.group(2)),
                         total_ms=float(m.group(3)), avg_us=float(m.group(4)),
                         pct=float(m.group(5)), tflops=float(m.group(6))))
leg = [r for r in rows if r['bucket'].startswith('MUL_MAT') and 'NVFP4' in r['bucket']]
if not leg:
    print('NO_LEG_BUCKETS in %s (rows=%d)' % (path, len(rows))); sys.exit(1)
n17408 = sum(r['calls'] for r in leg if r['bucket'].endswith('N17408'))
f8 = sum(r['calls'] for r in rows if r['bucket'].startswith('MUL_MAT:F8:N5120'))
ev_a = n17408 / 128.0 if n17408 else 0.0
ev_b = f8 / 64.0 if f8 else 0.0
ev = ev_a or ev_b
leg_ms = sum(r['total_ms'] for r in leg)
for r in sorted(leg, key=lambda r: -r['total_ms']):
    print('  %-46s calls=%-6d avg_us=%-9.1f total_ms=%.1f GB/s/call=%.1f'
          % (r['bucket'], r['calls'], r['avg_us'], r['total_ms'],
             1e-3*BPE*(r['calls']/ev)/r['total_ms'] if ev else 0))
print('LEG %s: calls_total=%d  leg_ms=%.1f  evals=%.2f (N17408/128=%.2f F8/64=%.2f)'
      % (label, sum(r['calls'] for r in leg), leg_ms, ev, ev_a, ev_b))
if ev and ev_a and ev_b and abs(ev_a - ev_b) / ev_a > 0.02:
    print('WARN evals 两条推导不一致（%.2f vs %.2f）⇒ 窗口口径可疑' % (ev_a, ev_b))
if ev:
    ms_eval = leg_ms / ev
    print('LEG_RATE %s: leg_ms_per_eval=%.2f leg_GB_per_eval=%.3f => %.1f GB/s'
          % (label, ms_eval, BPE/1e9, (BPE/1e9) / (ms_eval/1000.0)))
