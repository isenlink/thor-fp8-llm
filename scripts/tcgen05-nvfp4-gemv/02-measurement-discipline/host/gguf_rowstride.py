#!/usr/bin/env python3
"""gguf_rowstride.py — 从 GGUF 头直读**真实行步长**（H-mem 判别臂的参数事实来源，不引用文档）

做法：GGUF 的 tensor_info 给出每个张量的 offset；张量在文件里顺序紧排 ⇒
    row_stride = (next_offset - offset) / ne[1]   （比按 ggml 规则"推"更硬：直接用文件布局）
用法：
    gguf_rowstride.py <model.gguf> [--expect type:K:row_stride ...] [--type 40]
判据：--expect 全部命中 ⇒ 打印 OK ROWSTRIDE_EXPECT 并 exit 0；否则 exit 1（构建门据此 ABORT）。
"""
import struct
import sys
import collections

SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 8: None, 9: None, 10: 8, 11: 1, 12: 8}


def read_header(path):
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'GGUF':
            raise SystemExit("NOT_GGUF %s (magic=%r)" % (path, magic))
        ver = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]

        def rd_str():
            n = struct.unpack('<Q', f.read(8))[0]
            return f.read(n).decode('utf-8', 'replace')

        def skip(t):
            if t == 8:
                rd_str()
            elif t == 9:
                et = struct.unpack('<I', f.read(4))[0]
                n = struct.unpack('<Q', f.read(8))[0]
                for _ in range(n):
                    if et == 8:
                        rd_str()
                    else:
                        f.read(SCALAR[et])
            else:
                f.read(SCALAR[t])

        for _ in range(n_kv):
            rd_str()
            skip(struct.unpack('<I', f.read(4))[0])
        tensors = []
        for _ in range(n_tensors):
            name = rd_str()
            nd = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(nd)]
            ttype = struct.unpack('<I', f.read(4))[0]
            off = struct.unpack('<Q', f.read(8))[0]
            tensors.append((name, dims, ttype, off))
        return ver, tensors


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    path = argv[1]
    want_type = 40
    expects = []
    i = 2
    while i < len(argv):
        if argv[i] == '--type':
            want_type = int(argv[i + 1]); i += 2
        elif argv[i] == '--expect':
            t, k, rs = argv[i + 1].split(':'); expects.append((int(t), int(k), int(rs))); i += 2
        else:
            raise SystemExit("BAD_ARG %s" % argv[i])
    ver, tensors = read_header(path)
    order = sorted(tensors, key=lambda x: x[3])
    agg = collections.defaultdict(lambda: [0, 0])          # (type,K,N) -> [bytes, count]
    for idx, (name, dims, ttype, off) in enumerate(order):
        if idx + 1 == len(order):
            break
        span = order[idx + 1][3] - off
        k = dims[0]
        n = dims[1] if len(dims) > 1 else 1
        agg[(ttype, k, n)][0] += span
        agg[(ttype, k, n)][1] += 1
    got = {}
    total = 0
    for (ttype, k, n), (nb, cnt) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        if ttype != want_type:
            continue
        row_stride = nb // cnt // n
        got[(ttype, k)] = row_stride
        total += nb
        print("  type=%d K=%-6d N=%-7d tensors=%-4d %8.1f MiB row_stride=%d (=%s)"
              % (ttype, k, n, cnt, nb / 2**20, row_stride, "x".join(str(x) for x in (row_stride // 144, 144)) if row_stride % 144 == 0 else "non-144-multiple"))
    print("GGUF %s ver=%d type%d total=%.1f MiB" % (path.split('/')[-1], ver, want_type, total / 2**20))
    bad = 0
    for (ttype, k, rs) in expects:
        have = got.get((ttype, k))
        ok = (have == rs)
        print("  EXPECT type=%d K=%d row_stride=%d -> %s %s" % (ttype, k, rs, have, "OK" if ok else "MISMATCH"))
        if not ok:
            bad += 1
    if expects and not bad:
        print("OK ROWSTRIDE_EXPECT n=%d" % len(expects))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
