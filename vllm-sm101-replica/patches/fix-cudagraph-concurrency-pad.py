p = "${VENV:-/opt/vllm-p3/venv}/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_next.py"
src = open(p).read()

old = """        elif attn_metadata.num_decodes > 0:
            mixed_qkv_non_spec = causal_conv1d_update(
                mixed_qkv_non_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[
                    : attn_metadata.num_actual_tokens
                ],
                validate_data=True,
            )"""
new = """        elif attn_metadata.num_decodes > 0:
            # FIX(banshan): CUDA-graph pads the model input x to the capture
            # bucket size while metadata.num_actual_tokens stays unpadded, so
            # slicing the indices to num_actual_tokens broke the
            # (batch,) == conv_state_indices.shape contract. Pad the indices
            # to x's batch dim with PAD_SLOT_ID instead (kernel skips pads).
            _cs_indices = non_spec_state_indices_tensor[
                : attn_metadata.num_actual_tokens
            ]
            _x_batch = mixed_qkv_non_spec.shape[0]
            if _cs_indices.shape[0] < _x_batch:
                import torch as _torch
                from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
                    PAD_SLOT_ID,
                )

                _cs_indices = _torch.nn.functional.pad(
                    _cs_indices,
                    (0, _x_batch - _cs_indices.shape[0]),
                    value=PAD_SLOT_ID,
                )
            mixed_qkv_non_spec = causal_conv1d_update(
                mixed_qkv_non_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=_cs_indices,
                validate_data=True,
            )"""
assert old in src, "decode branch pattern not found"
open(p, "w").write(src.replace(old, new))
print("PATCHED qwen3_next.py decode branch")
