#!/usr/bin/env python3
"""check_prodaddr_fill.py — H-mem 臂的**宿主↔内核接缝恒等式**（CPU 夹具，零板端动作）

为什么需要：本臂把 B 的取数源换成生产布局后，正确性依赖三处约定同时成立：
  ① 宿主把 canonical 的 4 B 单元放到 (row, cc, r, u) 对应的**文件式**偏移；
  ② 内核按 (t→u=t&7,row=(t>>3)&31) 从同样的偏移读 4 B；
  ③ 落到 smem 的 b_smem_off(row,u) + r*1024。
任一处错 ⇒ 板上表现为 d_mismatch≠0（白烧一个 boot）。这里用纯 CPU 把三步串起来，逐字节对照 canonical。

判据：对全部 cc ∈ [0, CPR)、全部 (row,u,r)，重建出的 4096 B chunk 必须**逐字节等于** canonical stage。
用法: check_prodaddr_fill.py [--cpr 20]     退出 0 = 恒等成立
"""
import sys

B_SUB, NSUB, ROWS, REC = 1024, 4, 32, 36


def b_smem_off(row, u):
    return (row % 8) * 16 + (128 if u >= 4 else 0) + (row // 8) * 256 + (u & 3) * 4


def genB_nib(n, k):
    return 4 if n == 24 else 2


def canonical_stage():
    """与 c58_v9f_host.cpp:fill_stage 同一张表（B_CHUNK=4096）"""
    stage = bytearray(4096)
    for s in range(4):
        for n in range(ROWS):
            for k in range(64):
                byte = (n % 8) * 16 + (k // 32) * 128 + (n // 8) * 256 + (k % 32) // 2
                nib = genB_nib(n, 64 * s + k)
                idx = s * B_SUB + byte
                stage[idx] = (stage[idx] & 0xF0) | (nib & 15) if k % 2 == 0 else (stage[idx] & 0x0F) | (nib << 4)
    return bytes(stage)


def build_tile(stage, cpr):
    """与 c58_v9f_host_prod.cpp 的 prodlay 填法逐行同构（行步长 = cpr*144，sf 4 B 留 0）"""
    row_stride = cpr * 4 * REC
    tile = bytearray(ROWS * row_stride)
    for row in range(ROWS):
        for u in range(8):
            for r in range(NSUB):
                src = stage[r * B_SUB + b_smem_off(row, u): r * B_SUB + b_smem_off(row, u) + 4]
                for cc in range(cpr):
                    off = row * row_stride + cc * (4 * REC) + r * REC + 4 + 4 * u
                    tile[off:off + 4] = src
    return bytes(tile), row_stride


def kernel_read_chunk(tile, row_stride, cpr, cc, t_range=range(256)):
    """与 c58_v9f_gran4.cu:cp_chunk_prod 同构：t → (u=t&7,row=(t>>3)&31)，读 4 B×4 子块"""
    smem = bytearray(4096)
    for t in t_range:
        u, row = t & 7, (t >> 3) & 31
        s2 = row * row_stride + cc * (4 * REC) + 4 + 4 * u
        for r in range(NSUB):
            chunk = tile[s2 + r * REC: s2 + r * REC + 4]
            d = r * B_SUB + b_smem_off(row, u)
            smem[d:d + 4] = chunk
    return bytes(smem)


def main(argv):
    cpr = 20
    if len(argv) > 2 and argv[1] == '--cpr':
        cpr = int(argv[2])
    stage = canonical_stage()
    tile, row_stride = build_tile(stage, cpr)
    bad = 0
    for cc in range(cpr):
        got = kernel_read_chunk(tile, row_stride, cpr, cc)
        if got != stage:
            diff = [i for i in range(4096) if got[i] != stage[i]]
            print("MISMATCH cc=%d n=%d first=%s" % (cc, len(diff), diff[:6]))
            bad += 1
    # 覆盖率自检：canonical 里非零字节必须都被 tile 承载（防止"恒等"是靠全 0 蒙过去）
    nz_stage = sum(1 for b in stage if b)
    placed = set()
    for row in range(ROWS):
        for u in range(8):
            for r in range(NSUB):
                for cc in range(cpr):
                    off = row * row_stride + cc * (4 * REC) + r * REC + 4 + 4 * u
                    if any(tile[off:off + 4]):
                        placed.add((r, b_smem_off(row, u)))
    print("canonical 非零字节=%d；被 tile 承载且非零的 (r,off) 组合=%d" % (nz_stage, len(placed)))
    if nz_stage == 0:
        print("FAIL 夹具退化（canonical 全 0 ⇒ 恒等式无意义）")
        return 1
    if bad == 0 and len(placed) == NSUB * ROWS * 8:   # 4 子块 × 32 行 × 8 个 4 B 单元 = 1024
        print("OK PRODADDR_FILL_IDENTICAL cpr=%d row_stride=%d 全部 %d 个 cc 逐字节还原 canonical"
              % (cpr, row_stride, cpr))
        return 0
    print("FAIL PRODADDR_FILL bad_cc=%d placed=%d" % (bad, len(placed)))
    return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
