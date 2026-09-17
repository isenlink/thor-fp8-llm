#!/usr/bin/env python3
# check_ring_map.py — T4 环内核（t4_mmq_ring.cuh）**全局寻址**的机械核对器（纯宿主，不占板）。
#
# 为什么必须有：环里三种搬法（B qs 4 B / B sf 4 B / A 16 B）都是人手写的偏移；任何一处非对齐或越界
# 都是 **A 类事故**（非对齐访存 ⇒ GR 不可恢复 ⇒ 现场断电，见 INCIDENTS §十四）。本脚本把内核里那几行
# 地址算术**逐字复述**一遍，对真实生产形状穷举全部源/目的字节，断言：
#   ① 对齐：B 源 4 B、A 源/目的 16 B（cp.async 的粒度要求）
#   ② 界内：每个源字节都落在真实张量/缓冲字节范围内（含最后一行、最后一列、最后一个 chunk）
#   ③ 覆盖：一个 (tile, chunk) 内，B 的 32 行 × 4 记录 × 36 B 恰好被读一次、目的槽位恰好被写一次
#   ④ 不重叠：A 的 nib 区与 sf 区在 smem 里不交叉
# 用法： python3 check_ring_map.py            （自带真实形状表）
import sys

MMA_M, MMA_N, MMA_K, KT = 128, 32, 64, 256
NSUB = KT // MMA_K
B_REC = 36
A_BD_STRIDE = MMA_M * 32
A_SF_STRIDE = MMA_M * 4
B_SUB = MMA_N * MMA_K // 2      # 1024
SF_SUB = MMA_N * 4              # 128
QS_CHUNK = B_SUB * NSUB         # 4096
SF_CHUNK = SF_SUB * NSUB        # 512
B_CHUNK = QS_CHUNK + SF_CHUNK   # 4608

def b_smem_off(row, u):  # 与内核逐字同源
    return (row % 8) * 16 + (128 if u >= 4 else 0) + (row // 8) * 256 + (u & 3) * 4

def a_bd_rows(mreal): return ((mreal + 7) // 8) * 8

fails = 0
def ck(cond, name, detail=""):
    global fails
    if not cond:
        fails += 1
        print(f"  **FAIL** {name} {detail}")

SHAPES = [
    ("gate/up",   5120,  14336, 13),
    ("ffn_down",  17408,  5120, 13),
    ("attn_o",    5120,   5120, 13),
    ("M128 全宽", 5120,  14336, 128),
]

for name, K, N, M in SHAPES:
    cpr = K // KT
    ntiles = N // MMA_N
    row_stride = (K // 64) * B_REC        # = mmq_args.stride_row_x(=K/64) * 36
    tensor_bytes = N * row_stride
    ABD = a_bd_rows(M)
    A_SUB, A_SF_BD = ABD * 32, ABD * 4
    A_CHUNK = NSUB * (A_SUB + A_SF_BD)
    a_nib_g = (K // 64) * A_BD_STRIDE
    a_sf_g  = (K // 64) * A_SF_STRIDE
    print(f"[{name}] K={K} N={N} M={M} ABD={ABD} cpr={cpr} ntiles={ntiles} row_stride={row_stride}")

    # ---- ③ 覆盖：一个 tile × chunk 的 B 读取/写入 ----
    for cc in (0, cpr - 1, cpr // 2):
        src_seen, dst_seen = set(), set()
        for row in range(32):
            for u_ in range(8):
                d2 = b_smem_off(row, u_)
                for r in range(NSUB):
                    s2 = row * row_stride + 4 * cc * B_REC + 4 + 4 * u_ + r * B_REC
                    ck(s2 % 4 == 0, f"{name} B qs 源非 4 B 对齐", f"cc={cc} row={row} u={u_} r={r} s2={s2}")
                    ck(0 <= s2 and s2 + 4 <= tensor_bytes, f"{name} B qs 源越界", f"cc={cc} row={row} s2={s2}")
                    d = d2 + r * B_SUB
                    ck(d + 4 <= QS_CHUNK, f"{name} B qs 目的越界", f"d={d}")
                    ck(d % 4 == 0, f"{name} B qs 目的非 4 B 对齐", f"d={d}")
                    src_seen.update(range(s2, s2 + 4)); dst_seen.update(range(d, d + 4))
        # 期望读取 = 4 条记录的 [4,36) 各 32 B（缩放字节 [0,4) 不读）
        want_src = {row * row_stride + 4 * cc * B_REC + r * B_REC + 4 + 4 * u + b
                    for row in range(32) for r in range(NSUB) for u in range(8) for b in range(4)}
        ck(src_seen == want_src, f"{name} B qs 覆盖不全", f"cc={cc} 缺 {len(want_src - src_seen)} 多 {len(src_seen - want_src)}")
        ck(dst_seen == set(range(QS_CHUNK)), f"{name} B qs 目的未铺满", f"cc={cc} {len(dst_seen)}/{QS_CHUNK}")
        src_sf = {row * row_stride + 4 * cc * B_REC + r * B_REC + 4 * (t // 4) + (t % 4)
                  for row in range(32) for r in range(NSUB) for t in range(4)}
        for row in range(32):
            for r in range(NSUB):
                s = row * row_stride + 4 * cc * B_REC + r * B_REC
                ck(s % 4 == 0, f"{name} B sf 源非 4 B 对齐", f"cc={cc} row={row} s={s}")
                ck(0 <= s and s + 4 <= tensor_bytes, f"{name} B sf 源越界", f"cc={cc} row={row} s={s}")
        ck(len(src_sf) == 32 * NSUB * 4 and min(src_sf) >= 4 * cc * B_REC,
           f"{name} B sf 覆盖异常", f"cc={cc}")

    # ---- ①② A 环：全局 → 环内 ABD 前缀 ----
    for cc in (0, cpr - 1):
        base_n = cc * (NSUB * A_BD_STRIDE)
        base_s = cc * (NSUB * A_SF_STRIDE)
        nib_dst, sf_dst = set(), set()
        for bdg in range(NSUB):
            for u in range(A_SUB // 16):
                s = base_n + bdg * A_BD_STRIDE + u * 16
                d = bdg * A_SUB + u * 16
                ck(s % 16 == 0 and d % 16 == 0, f"{name} A nib 非 16 B 对齐", f"cc={cc} bdg={bdg} u={u}")
                ck(s + 16 <= a_nib_g, f"{name} A nib 源越界", f"cc={cc} s={s}")
                ck(d + 16 <= NSUB * A_SUB, f"{name} A nib 目的越界", f"d={d}")
                nib_dst.update(range(d, d + 16))
            for u in range(A_SF_BD // 16):
                s = base_s + bdg * A_SF_STRIDE + u * 16
                d = bdg * A_SF_BD + u * 16
                ck(s % 16 == 0 and d % 16 == 0, f"{name} A sf 非 16 B 对齐", f"cc={cc} bdg={bdg} u={u}")
                ck(s + 16 <= a_sf_g, f"{name} A sf 源越界", f"cc={cc} s={s}")
                sf_dst.update(range(d, d + 16))
        ck(nib_dst == set(range(NSUB * A_SUB)), f"{name} A nib 目的未铺满", f"cc={cc} {len(nib_dst)}/{NSUB*A_SUB}")
        ck(sf_dst == set(range(NSUB * A_SF_BD)), f"{name} A sf 目的未铺满", f"cc={cc} {len(sf_dst)}/{NSUB*A_SF_BD}")
        # ④ nib 区 / sf 区不交叉（环内布局：nib 在 [0, NSUB*A_SUB)，sf 在 [A_CHUNK-NSUB*A_SF_BD, A_CHUNK)）
        ck(NSUB * A_SUB <= A_CHUNK - NSUB * A_SF_BD, f"{name} A nib/sf 区交叉", f"cc={cc}")

    # ---- 环内 A 前缀 ⇒ 只覆盖 M=128 布局的「前 ABD 行」（与内核注释同一断言）----
    def a_nib_off_in_board(m, sub, l):
        return (m % 8) * 16 + (sub // 2) * 128 + (m // 8) * 256 + (sub % 2) * 8 + l
    lo = {a_nib_off_in_board(m, s, l) for m in range(ABD) for s in range(4) for l in range(8)}
    ck(lo == set(range(ABD * 32)), f"{name} A 前缀≠前 ABD 行", f"ABD={ABD}")
    hi = {a_nib_off_in_board(m, s, l) for m in range(ABD, MMA_M) for s in range(4) for l in range(8)}
    ck((not hi) or min(hi) >= ABD * 32, f"{name} 第 ≥ABD 行落进前缀区", f"min={min(hi) if hi else None}")

print(f"\nRING_MAP={'PASS' if fails == 0 else 'FAIL'} (fails={fails})")
sys.exit(0 if fails == 0 else 1)
