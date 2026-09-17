#!/usr/bin/env python3
"""check_canonical_equiv.py — **生产形态寻址** ↔ **定版件 canonical 流** 的逐位等价检查（宿主，不占板）。

口径（来源：results/20260916-108 / 109 / 110 / 112）：
- 真实权重张量（GGML NVFP4，N 行主序）：每行每 64-K 块 36 B = [4 B 缩放码][32 B nibble]；行字节步长 = nb01。
- canonical 流（定版件 `evidence/20260916-77` 的输入）：每 chunk 4608 B = 4×1024 B qs + 4×128 B sf。
- 生产内核要执行的映射（= v11 已验证公式的「张量首址 + 行步长」形参化；base = n-tile 起点）：
    qs: dst = (row&7)*16 + ((u>=4)?128:0) + (row>>3)*256 + (u&3)*4      [子步 r 的 1024 B 板内]
        源  = base + row*nb01 + (4c+r)*36 + 4 + 4u                        (u=0..7)
    sf: dst = 4096 + r*128 + row*4 ;  源 = base + row*nb01 + (4c+r)*36
- 两侧 nibble 归属不同（canonical：byte i ↔ 元素 2i/2i+1；ggml：byte j ↔ j/j+8，见 109 §2b）
  ⇒ 比对前用**经验置换 σ**（本脚本现场标定，不写死）对齐，再逐 nibble 对账。
- 本脚本在多个 nb01 上跑（真实 2880 / 补零 2944 / 3072），验证映射**不依赖**「步长恰好 = K*9/16」。

用法: check_canonical_equiv.py <tensor.bin> <stream.bin> <chunks> <ntiles> <nb01> [nb01...]
"""
import sys

CH, BOARD, SF, ROW_LEN = 4608, 1024, 128, 36


def coff(n, kk):
    return (n % 8) * 16 + (kk // 32) * 128 + (n // 8) * 256 + (kk % 32) // 2


def main():
    tensor, stream, chunks, ntiles = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    strides = [int(x) for x in sys.argv[5:]]
    raw = open(tensor, 'rb').read()
    st = open(stream, 'rb').read()
    s0 = (chunks_src := None)  # 原生行步长 = 本张量实际的 (K/64)*36；由 chunks 反推
    # 由流尺寸与 chunks 反推 K：chunk 内每行 144 B
    src_row = chunks * 144
    assert len(st) >= ntiles * chunks * CH, "流文件太小"
    fails = 0
    for nb01 in strides:
        assert nb01 >= src_row, "nb01=%d 装不下 %d B 行" % (nb01, src_row)
        # 造（可能补零的）张量视图：行内容按 nb01 重排
        nrow = ntiles * 32
        buf = bytearray(max(nrow * nb01, len(raw)))
        for r in range(nrow):
            buf[r * nb01: r * nb01 + src_row] = raw[r * src_row: (r + 1) * src_row]
        # —— σ 现场标定（用 tile0 chunk0 的 (s,n) 全组合唯一确定）
        sigma = {}
        for j in range(32):
            for half in (0, 1):
                hit = []
                for kk in range(64):
                    ok = True
                    for s in range(4):
                        for n in range(32):
                            rb = buf[(n * nb01 + (0 * 4 + s) * 36) + 4 + j]
                            rv = (rb & 0xF) if half == 0 else (rb >> 4)
                            cb = st[0 * CH + s * BOARD + coff(n, kk)]
                            cv = (cb & 0xF) if (kk % 2 == 0) else (cb >> 4)
                            if rv != cv:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        hit.append(kk)
                sigma[(j, half)] = hit
        amb = [k for k, v in sigma.items() if len(v) != 1]
        if amb:
            print('# nb01=%d: σ 标定失败（歧义 %d 个）⇒ FAIL' % (nb01, len(amb)))
            fails += 1
            continue
        S = {k: v[0] for k, v in sigma.items()}
        qbad = sbad = 0
        for t in range(ntiles):
            base = t * 32 * nb01
            for c in range(chunks):
                for s in range(4):
                    for n in range(32):
                        sf_src = buf[base + n * nb01 + (4 * c + s) * 36: base + n * nb01 + (4 * c + s) * 36 + 4]
                        if sf_src != st[t * chunks * CH + c * CH + 4 * BOARD + s * SF + n * 4:
                                        t * chunks * CH + c * CH + 4 * BOARD + s * SF + n * 4 + 4]:
                            sbad += 1
                        for j in range(32):
                            rb = buf[base + n * nb01 + (4 * c + s) * 36 + 4 + j]
                            for half, rv in ((0, rb & 0xF), (1, rb >> 4)):
                                cb = st[t * chunks * CH + c * CH + s * BOARD + coff(n, S[(j, half)])]
                                cv = (cb & 0xF) if (S[(j, half)] % 2 == 0) else (cb >> 4)
                                if rv != cv:
                                    qbad += 1
        tot = ntiles * chunks * 4 * 32 * 2
        verdict = 'PASS' if (qbad == 0 and sbad == 0) else 'FAIL'
        print('# nb01=%-5d qs 失配 %d/%d   sf 失配 %d/%d   ⇒ %s' % (nb01, qbad, tot, sbad, ntiles * chunks * 4 * 32, verdict))
        if verdict == 'FAIL':
            fails += 1
    print('# 几何恒等式: CH=%d == 4*%d+4*%d ✓, 每 chunk 每行 = 4*36 = 144 B ✓, 行内容 = chunks*144 = %d B' %
          (CH, BOARD, SF, src_row))
    print('CANONICAL_EQUIV=%s' % ('PASS' if fails == 0 else 'FAIL'))
    return 0 if fails == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
