#!/usr/bin/env python3
"""verify_v11_d.py — v11 D 判据（口径修正版）

背景：内核 `epilogue_store_d` 按 **N 主序** 落盘（`D[blk*M*N + n*M + m]`，见 c58_v11_ggml.cu ★83b），
而 `mk_stream` / `verify_t4_4.py` 的参考是 **M 主序**（`ref[m*N+n]`）。二者必须显式对账，否则会误判 97% 失配。

参考：直接复用 verify_t4_4.py 的 `ref_row`（canonical 流同源，逐点精确）。
用法: python3 verify_v11_d.py <stream.bin> <oc> <dump.bin> [ntiles]
  dump.bin 允许是「D + ref 拼接」的宿主落盘（前 ntiles*4096 = D，其后是宿主的 ref，忽略即可）。
"""
import importlib.util, struct, sys

M, N = 128, 32
spec = importlib.util.spec_from_file_location(
    'v45', '/path/to/project/evidence/20260916-77-t4-4-real-stream/host/verify_t4_4.py')
v45 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v45)


def load(p, n=None):
    b = open(p, 'rb').read()
    k = len(b) // 4
    if n is not None:
        k = min(k, n)
    return list(struct.unpack('<%df' % k, b[:4 * k]))


def main():
    stream, oc, dump = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    ntiles = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    raw = open(stream, 'rb').read()
    D = load(dump, ntiles * M * N)
    if len(D) < ntiles * M * N:
        print('# ERROR: dump 只有 %d floats，需要 %d' % (len(D), ntiles * M * N))
        return 2
    bad_all = 0
    for blk in range(ntiles):
        part = raw[blk * oc * v45.CH:(blk + 1) * oc * v45.CH]
        bad = []
        for m in range(M):
            R = v45.ref_row(part, oc, m)
            for n in range(N):
                got = D[blk * M * N + n * M + m]      # ★ N 主序读
                if got != R[n]:
                    bad.append((m, n, got, R[n]))
        mx = max((abs(b[2] - b[3]) for b in bad), default=0.0)
        lanes = sorted(set(b[0] // 32 for b in bad))
        print('# blk=%d out_chunks=%d mism=%d/%d maxabs=%.6g lanes=%s'
              % (blk, oc, len(bad), M * N, mx, lanes))
        for b in bad[:6]:
            print('#   D[%d][%d]=%.6g ref=%.6g' % b)
        bad_all += len(bad)
    print('# TOTAL mism=%d / %d' % (bad_all, ntiles * M * N))
    return 0 if bad_all == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
