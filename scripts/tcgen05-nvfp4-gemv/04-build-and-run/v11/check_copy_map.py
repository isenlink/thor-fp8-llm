#!/usr/bin/env python3
"""check_copy_map.py — v11「纯拷贝」判据（宿主，零板端动作）

设计规则（results/20260916-109 §2b）：内核算 Σ_slot A[slot]·B[slot]（slot = 硬件固定位置），
⇒ D = 真值 ⟺ **A 与 B 用同一元素落位约定**（点积对索引置换不变）。
v11 走纯拷贝：B 侧把真实 GGML 张量字节原样搬进 canonical slot，A 侧改用 ggml 落位。
判据：
  ① 落位约定一致性 e_A(g,bb,h) == e_B(g,bb,h)（逐 slot）
  ①b 每个 16-slot 组映射到连续 16 元素组（缩放关联前提）
  ② A 字节槽位算式 == 纯拷贝目的槽位算式（off_fp4 一致）
  ③ 端到端 D 仿真（单 K64 块，M=128×N=32）：Σ_slot 与真值 Σ_e 逐位比
  ④ 搬运映射：源/目的字节一一覆盖 0..4095（无重无漏）
用法: python3 check_copy_map.py
"""
import sys

ROOT = '/path/to/project/t4/stream/'
ROW_STRIDE = 2880                    # (K_IN/64)*36, K_IN=5120
CHUNK = 4608
E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]


def sfv(b):
    e = (b >> 3) & 0xF
    m = b & 7
    if e == 0:
        return 0.0
    v = (1.0 + m / 8.0) * (1 << (e - 7))
    return -v if (b >> 7) & 1 else v


def genA(m, k):
    return 4 if m == 64 else (2 if k % 2 == 0 else 4)


def genSFA(m, k):
    return 0x40 if (m == 0 and (k % 256) < 16) else 0x38


def off_fp4(m, q):
    """v5 头文件的槽位算式（q = K64 块内元素序号 0..63）。"""
    return (m % 8) * 16 + (q // 32) * 128 + (m // 8) * 256 + (q % 32) // 2


def v11_items(t, row_base, c):
    """与内核生产者逐字同式（4 B 粒度）。返回 [(src, dst, size), ...] 本 chunk 该线程发的全部 cp.async。

    内核：u_ = t&7, row_ = (t>>3)&31；qs 每线程 4 个 4B（r = K64 子步）；
          sf 由 warp0 独占（row_ = t&31，r 循环 4 个子步）。
    """
    out = []
    if t < 256:
        u_ = t & 7
        row_ = (t >> 3) & 31
        src0 = row_base + row_ * ROW_STRIDE + (4 * c) * 36 + 4 + 4 * u_
        dst0 = (row_ & 7) * 16 + (128 if u_ >= 4 else 0) + (row_ >> 3) * 256 + (u_ & 3) * 4
        for r in range(4):
            out.append((src0 + r * 36, dst0 + r * 1024, 4))
    if t < 32:
        row_ = t & 31
        for r in range(4):
            out.append((row_base + row_ * ROW_STRIDE + (4 * c + r) * 36,
                        4096 + r * 128 + row_ * 4, 4))
    return out


def e_A(g, bb, h):                   # v11 的 A 循环落的元素（ggml 约定）
    return 32 * g + bb + (8 if bb >= 8 else 0) + 8 * h


def e_B(g, bb, h):                   # 纯拷贝搬过去的那个字节承载的元素（ggml 约定）
    return 16 * (2 * g + bb // 8) + (bb % 8) + 8 * h


def main():
    raw = open(ROOT + 'work/raw.bin', 'rb').read()
    fails = []

    bad1 = [(g, bb, h) for g in range(2) for bb in range(16) for h in range(2)
            if e_A(g, bb, h) != e_B(g, bb, h)]
    print('① e_A==e_B：不一致 %d/64（应 0）%s' % (len(bad1), bad1[:4]))
    if bad1:
        fails.append('①')

    bad1b = []
    for gi in range(4):
        g, half = gi // 2, gi % 2
        els = sorted(e_A(g, half * 8 + j, h) for j in range(8) for h in range(2))
        if els != list(range(16 * gi, 16 * gi + 16)):
            bad1b.append((gi, els))
    print('①b 16 元素组连续：异常 %d/4（应 0）%s' % (len(bad1b), bad1b[:2]))
    if bad1b:
        fails.append('①b')

    bad2 = 0
    for g in range(2):
        for bb in range(16):
            for h in range(2):
                q = 2 * (16 * g + bb)        # A 循环里写的字节下标（= off_fp4 的 q）
                if off_fp4(0, q) != (0 % 8) * 16 + g * 128 + (0 // 8) * 256 + bb:
                    bad2 += 1
    print('② A 槽位 == 拷贝目的槽位：不一致 %d/64（应 0）' % bad2)
    if bad2:
        fails.append('②')

    # ④/⑤ 搬运映射 + **对齐/越界判据**（事故志「A 类：非对齐/越界访存」的机械化拦截）
    cover = [0] * CHUNK
    bad4 = bad5 = bad6 = 0
    align_hist = {}
    for c in (0, 7, 19):
        row_base = 0
        for t in range(256):
            for src, dst, sz in v11_items(t, row_base, c):
                if src % sz != 0:
                    bad5 += 1                      # 非对齐（v11 首臂就是栽在这条：16 B 访问落在 +4/+20）
                if dst % sz != 0:
                    bad5 += 1
                if not (0 <= src and src + sz <= 32 * ROW_STRIDE):
                    bad6 += 1                      # 越界（源必须在 tile 内）
                if not (0 <= dst and dst + sz <= CHUNK):
                    bad6 += 1
                if dst < 4096:
                    for b in range(sz):
                        cover[dst + b] += 1
                else:
                    for b in range(sz):
                        cover[dst + b] += 1
                if dst < 4096:
                    align_hist[(src % 16)] = align_hist.get((src % 16), 0) + 1
    print('④ 覆盖：每字节命中次数 min=%d max=%d（3 个 chunk ⇒ 应 3,3）' % (min(cover), max(cover)))
    print('⑤ 对齐：违规 %d（应 0）' % bad5)
    print('⑥ 越界：违规 %d（应 0）' % bad6)
    print('   ℹ granule 源地址 mod 16 分布 = %s ⇒ 16 B 访问必然非对齐（%d/%d 落在 4/12）'
          % (sorted(align_hist.items()), sum(v for k, v in align_hist.items() if k != 0), sum(align_hist.values())))
    if bad4 or min(cover) != 3 or max(cover) != 3:
        fails.append('④')
    if bad5:
        fails.append('⑤')
    if bad6:
        fails.append('⑥')

    # ③ 端到端 D 仿真：单 K64 块（c=0, s=0, tile 0）
    c = s0 = 0
    Dv11 = [[0.0] * 32 for _ in range(128)]
    Dtrue = [[0.0] * 32 for _ in range(128)]
    for n in range(32):
        base = (0 * 32 + n) * ROW_STRIDE + (4 * c + s0) * 36
        sb = raw[base]
        sfbv = sfv(sb) if sb else 0.0
        for m in range(128):
            t_acc = 0.0
            for e in range(64):
                byte = raw[base + 4 + (e // 16) * 8 + (e % 16) % 8]
                nib = (byte & 0xF) if (e % 16) < 8 else (byte >> 4)
                t_acc += (E2M1[genA(m, (c * 256 + s0 * 64 + e) % 256)]
                          * sfv(genSFA(m, c * 256 + s0 * 64 + e)) * E2M1[nib] * sfbv)
            Dtrue[m][n] = t_acc
            v_acc = 0.0
            for g in range(2):
                for bb in range(16):
                    byte = raw[base + 4 + 16 * g + bb]
                    for h, nib in ((0, byte & 0xF), (1, byte >> 4)):
                        ee = e_A(g, bb, h)
                        v_acc += (E2M1[genA(m, (c * 256 + s0 * 64 + ee) % 256)]
                                  * sfv(genSFA(m, c * 256 + s0 * 64 + ee)) * E2M1[nib] * sfbv)
            Dv11[m][n] = v_acc
    bad3 = 0
    mx = 0.0
    for m in range(128):
        for n in range(32):
            d = abs(Dv11[m][n] - Dtrue[m][n])
            mx = max(mx, d)
            if d != 0.0:
                bad3 += 1
    print('③ D 仿真（K64 单块，4096 点）v11 vs 真值：不等 %d maxabs=%.6g（应 0/0）' % (bad3, mx))
    if bad3:
        fails.append('③')

    print('=== %s ===' % ('ALL PASS' if not fails else 'FAIL ' + ','.join(fails)))
    return 0 if not fails else 1


if __name__ == '__main__':
    sys.exit(main())
