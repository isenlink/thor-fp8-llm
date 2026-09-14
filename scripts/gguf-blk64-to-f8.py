#!/usr/bin/env python3
# Author: AI assistant
# RadixArk-F8attn-v2.gguf 的 blk.64 (MTP draft 层) BF16 权重 → F8_E4M3 + per-tensor .scale。
# 只改 8 个二维权重（eh_proj / attn_q / attn_k / attn_v / attn_output / ffn_gate / ffn_up / ffn_down），
# 其余张量与全部元数据原样拷贝。运行时走既有 F8 cuBLASLt 路径，scale 由 build_lora_mm 的 ggml_mul 施加。
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.environ.get('LLAMA_CPP_DIR', os.path.expanduser('~/work/llama-cpp-latest')), 'gguf-py'))

import ml_dtypes  # noqa: F401  (注册 bfloat16/float8 dtype)
import numpy as np
import gguf

SRC = Path(sys.argv[1])
DST = Path(sys.argv[2])

TARGETS = {
    'blk.64.nextn.eh_proj.weight',
    'blk.64.attn_q.weight',
    'blk.64.attn_k.weight',
    'blk.64.attn_v.weight',
    'blk.64.attn_output.weight',
    'blk.64.ffn_gate.weight',
    'blk.64.ffn_up.weight',
    'blk.64.ffn_down.weight',
}
F8 = gguf.GGMLQuantizationType.F8_E4M3

reader = gguf.GGUFReader(str(SRC))
arch_field = reader.get_field('general.architecture')
arch = arch_field.contents()
writer = gguf.GGUFWriter(str(DST), arch=arch, endianess=reader.endianess)

align_field = reader.get_field(gguf.Keys.General.ALIGNMENT)
if align_field is not None:
    writer.data_alignment = align_field.contents()

for field in reader.fields.values():
    if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
        continue
    val_type = field.types[0]
    sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
    writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

def bf16_to_f32(raw_u8: np.ndarray) -> np.ndarray:
    # gguf-py 对 BF16 返回 raw uint8（末维按字节）；还原成 float32 元素视图
    return raw_u8.view(ml_dtypes.bfloat16).astype(np.float32)

converted = {}   # name -> (f8_bytes uint8 np.ndarray, scale float)
scales = {}      # scale tensor name -> np.float32[1]

for tensor in reader.tensors:
    if tensor.name not in TARGETS:
        continue
    assert tensor.tensor_type == gguf.GGMLQuantizationType.BF16, (tensor.name, tensor.tensor_type)
    w = bf16_to_f32(tensor.data)
    amax = float(np.abs(w).max())
    s = amax / 448.0 if amax > 0 else 1.0
    wq = np.clip(w / s, -448.0, 448.0)
    f8 = wq.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    converted[tensor.name] = (f8, s)
    scales[tensor.name.removesuffix('.weight') + '.scale'] = np.asarray([s], dtype=np.float32)
    print(f'{tensor.name}: amax={amax:.4f} scale={s:.6g} {w.shape} {tensor.n_bytes/1e6:.1f}MB -> {f8.nbytes/1e6:.1f}MB', flush=True)

missing = TARGETS - converted.keys()
assert not missing, f'targets not found: {missing}'

# 第一遍：登记 tensor info（保持原顺序，scale 紧跟其 weight）
for tensor in reader.tensors:
    if tensor.name in converted:
        f8, _ = converted[tensor.name]
        writer.add_tensor_info(tensor.name, f8.shape, f8.dtype, f8.nbytes, F8)
        sc = scales[tensor.name.removesuffix('.weight') + '.scale']
        writer.add_tensor_info(tensor.name.removesuffix('.weight') + '.scale', sc.shape, sc.dtype, sc.nbytes, gguf.GGMLQuantizationType.F32)
    else:
        writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype, tensor.data.nbytes, tensor.tensor_type)

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_ti_data_to_file()

for tensor in reader.tensors:
    if tensor.name in converted:
        f8, _ = converted[tensor.name]
        writer.write_tensor_data(f8, tensor_endianess=reader.endianess)
        writer.write_tensor_data(scales[tensor.name.removesuffix('.weight') + '.scale'], tensor_endianess=reader.endianess)
        print(f'wrote {tensor.name} (F8) + .scale', flush=True)
    else:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)

writer.close()
print('done ->', DST, flush=True)
