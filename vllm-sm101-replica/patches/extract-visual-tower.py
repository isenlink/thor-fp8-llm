import json
import struct
import os

SRC = '/data/models/Qwen3.8-27B-NVFP4-HF/model.safetensors'
DST = './visual-tower.safetensors'  # output next to this script

with open(SRC, 'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]
    header = json.loads(f.read(n))
    base = 8 + n

vis = {k: header[k] for k in header if k.startswith('model.visual.')}
# 重写 header: 去掉 __metadata__ 保留, offsets 重排
new_header = {'__metadata__': {'format': 'pt'}}
out_buf = bytearray()
cursor = 0
with open(SRC, 'rb') as f:
    for k in sorted(vis):
        info = vis[k]
        s, e = info['data_offsets']
        length = e - s
        f.seek(base + s)
        data = f.read(length)
        new_header[k] = {'dtype': info['dtype'], 'shape': info['shape'],
                         'data_offsets': [cursor, cursor + length]}
        out_buf += data
        cursor += length

hdr_bytes = json.dumps(new_header).encode()
pad = (8 - (len(hdr_bytes) % 8)) % 8
hdr_bytes += b' ' * pad
with open(DST, 'wb') as f:
    f.write(struct.pack('<Q', len(hdr_bytes)))
    f.write(hdr_bytes)
    f.write(out_buf)

print('written', DST, os.path.getsize(DST) / 2**20, 'MiB', len(vis), 'tensors')
