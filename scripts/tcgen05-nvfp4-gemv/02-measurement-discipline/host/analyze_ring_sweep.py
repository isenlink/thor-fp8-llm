#!/usr/bin/env python3
"""analyze_ring_sweep.py — 把 ring_cfg_sweep 四臂的读数算成同一张表（宿主，零板端动作）。

口径（与 results/20260916-113 §2 完全一致，必须同 trial 才可配对）：
  每步 ms = eval_time_ms / steps，steps = 日志里 'spec timing: verify_sample' 行数（= verify 次数）
  draft 段 = 'draft_call=' 行的平均值；verify 段 = 每步 − draft 段
  腿时间 = verify 段（113 §5：verify 段 ≈ NVFP4 腿 + 少量 attention/elementwise）
用法: python3 analyze_ring_sweep.py <tag>=<log> [<tag>=<log> ...]
"""
import re
import sys


def parse(path):
    steps = 0
    draft_ms = []
    verify_ms = []
    ev_ms = ev_tok = None
    acc = accn = gen = None
    for ln in open(path, errors="replace"):
        if "spec timing: draft_call=" in ln:
            m = re.search(r"draft_call=([\d.]+) ms", ln)
            if m:
                draft_ms.append(float(m.group(1)))
        if "spec timing: verify_sample=" in ln:
            steps += 1
            m = re.search(r"verify_sample=([\d.]+) ms", ln)
            if m:
                verify_ms.append(float(m.group(1)))
        if "eval time =" in ln:
            m = re.search(r"eval time =\s*([\d.]+) ms /\s*(\d+) tokens", ln)
            if m:
                ev_ms, ev_tok = float(m.group(1)), int(m.group(2))
        if "draft acceptance =" in ln:
            m = re.search(r"draft acceptance = ([\d.]+) \(\s*(\d+) accepted /\s*(\d+) generated\)", ln)
            if m:
                acc, accn, gen = float(m.group(1)), int(m.group(2)), int(m.group(3))
    return dict(steps=steps, draft=draft_ms, verify=verify_ms, ev_ms=ev_ms, ev_tok=ev_tok,
                acc=acc, accn=accn, gen=gen)


def main():
    print(f"{'arm':10s} {'steps':>6s} {'每步ms':>9s} {'draft段':>8s} {'verify段':>9s} "
          f"{'t/s':>7s} {'acc':>6s} {'verify段GB/s':>11s}")
    NVFP4_GB = 11.01          # 实测口径（results/20260916-113 §5：每 eval 11.01 GB）
    for a in sys.argv[1:]:
        tag, _, path = a.partition("=")
        r = parse(path)
        if not r["steps"] or r["ev_ms"] is None:
            print(f"{tag:10s} —— 读数不全（steps={r['steps']} eval={r['ev_ms']}）")
            continue
        step = r["ev_ms"] / r["steps"]
        draft = sum(r["draft"]) / len(r["draft"]) if r["draft"] else float("nan")
        verify = step - draft
        tps = r["ev_tok"] / (r["ev_ms"] / 1000.0)
        leg = NVFP4_GB / (verify / 1000.0) if verify > 0 else float("nan")
        print(f"{tag:10s} {r['steps']:6d} {step:9.2f} {draft:8.2f} {verify:9.2f} "
              f"{tps:7.2f} {r['acc'] if r['acc'] is not None else float('nan'):6.3f} {leg:10.1f}")


if __name__ == "__main__":
    main()
