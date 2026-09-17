#!/usr/bin/env python3
# check_aq_vs_archive.py — 把「生产 A 量化器」与「归档 canonical（evidence-90）」的 nib 做**元素级**对账。
#
# 背景：两边的**打包约定不同**（同一个 4bit 值放哪个半字节/哪个字节），所以逐字节比会大面积"失配"：
#   归档（流约定）：byte = (m%8)*16 + (k'/32)*128 + (m/8)*256 + (k'%32)/2，低半字节 = 偶数 k'
#   生产（ggml 约定，与权重 B 同源）：byte = (m%8)*16 + (sub/2)*128 + (m/8)*256 + (sub%2)*8 + l
#                                   sub = e/16，l = e%8（低半字节 = l，高半字节 = l+8）
# ⇒ 正确判据是**元素级**：(m, k') 上的 4bit 值必须逐元素相同（这才是 MMA 数值等价的条件）。
# 用法: python3 check_aq_vs_archive.py <归档dir> <生产dumpdir> <M> <K>
import sys

def ours_off(m, e_in_board):                     # t4_mmq_canon.h: a_nib_off_in_board
    sub, l = e_in_board // 16, e_in_board % 16
    byte = (m % 8) * 16 + (sub // 2) * 128 + (m // 8) * 256 + (sub % 2) * 8 + (l % 8)
    return byte, 0 if l < 8 else 1               # 0=低半字节 1=高半字节

def arch_off(m, kp):                             # aq_layout.h: aq_a_nib_off
    byte = (m % 8) * 16 + (kp // 32) * 128 + (m // 8) * 256 + (kp % 32) // 2
    return byte, 0 if kp % 2 == 0 else 1

adir, bdir, M, K = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
nib_arch = open(f"{adir}/canon_nib.bin", "rb").read()
nib_ours = open(f"{bdir}/out_nib.bin", "rb").read()
assert len(nib_arch) == len(nib_ours) == (K // 64) * 4096, (len(nib_arch), len(nib_ours))

def get(buf, m, board, byte):
    return buf[board * 4096 + byte]

mism = 0
first = None
for m in range(M):
    for board in range(K // 64):
        base_a, base_o = board * 4096, board * 4096
        for kp in range(64):
            ba, sa = arch_off(m, kp)
            bo, so = ours_off(m, kp)
            va = (nib_arch[base_a + ba] >> (4 * sa)) & 0xF
            vo = (nib_ours[base_o + bo] >> (4 * so)) & 0xF
            if va != vo:
                if first is None:
                    first = (m, board, kp, va, vo)
                mism += 1
tot = M * (K // 64) * 64
print(f"元素级对账：total={tot} mism={mism} first={first}")
# 反证：按归档自己的约定读我方 dump，应当**完全对不上**（证明两边确实是不同打包）
mism_pack = 0
for m in range(M):
    for board in range(K // 64):
        for kp in range(64):
            ba, sa = arch_off(m, kp)
            va = (nib_arch[board * 4096 + ba] >> (4 * sa)) & 0xF
            vo = (nib_ours[board * 4096 + ba] >> (4 * sa)) & 0xF
            if va != vo:
                mism_pack += 1
print(f"（口径对照）按归档字节位置直接读：mism={mism_pack}  ← 应远大于 0，说明只是打包不同")
print("AQ_ELEM_MATCH=" + ("PASS" if mism == 0 else "FAIL"))
sys.exit(0 if mism == 0 else 1)
