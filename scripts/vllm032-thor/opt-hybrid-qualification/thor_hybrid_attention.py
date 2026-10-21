"""Thor05 Qwen3.8 BF16: FA2 prefill and existing Triton decode kernels."""
import torch

_scratch = None
_seen = set()

def decode_route(p):
    q, k, v = p['q'], p['k'], p['v']
    return (
        p.get('fa_version', 2) == 2 and p.get('block_table') is not None
        and 0 < q.shape[0] <= 6 and q.shape[1:] == (24, 256)
        and k.shape[1:] in ((432, 4, 256), (816, 4, 256)) and v.shape == k.shape
        and q.dtype == k.dtype == v.dtype == torch.bfloat16
        and p['cu_seqlens_q'].numel() == 2
        and not p.get('return_softmax_lse', False)
        and p.get('softcap', 0) == 0
        and tuple(p.get('window_size') or (-1, -1)) == (-1, -1)
        and all(p.get(x) is None for x in (
            's_aux', 'alibi_slopes',
            'mask_mod', 'aux_tensors', 'dynamic_causal', 'output_scale'))
    )

def hybrid_attention(**p):
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    if not decode_route(p):
        if 'prefill' not in _seen:
            print('THOR_HYBRID_FA2_HIT', flush=True)
            _seen.add('prefill')
        return flash_attn_varlen_func(**p)
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    global _scratch
    q = p['q']
    if _scratch is None:
        _scratch = (
            torch.empty((6, 24, 16, 256), dtype=torch.float32, device=q.device),
            torch.empty((6, 24, 16), dtype=torch.float32, device=q.device),
            torch.empty((6, 24, 16), dtype=torch.float32, device=q.device),
        )
    out = p.get('out')
    if out is None:
        out = torch.empty_like(q)
    unified_attention(
        q=q, k=p['k'], v=p['v'], out=out,
        cu_seqlens_q=p['cu_seqlens_q'], max_seqlen_q=p['max_seqlen_q'],
        seqused_k=p['seqused_k'], max_seqlen_k=p['max_seqlen_k'],
        softmax_scale=p.get('softmax_scale') or q.shape[-1] ** -0.5,
        causal=p.get('causal', False), window_size=(-1, -1),
        block_table=p['block_table'], softcap=0,
        q_descale=None, k_descale=None, v_descale=None,
        seq_threshold_3D=6, num_par_softmax_segments=16,
        softmax_segm_output=_scratch[0], softmax_segm_max=_scratch[1],
        softmax_segm_expsum=_scratch[2],
    )
    if 'decode' not in _seen:
        print('THOR_HYBRID_TRITON_HIT', flush=True)
        _seen.add('decode')
    return out

def register():
    from vllm.v1.attention.backends import flash_attn
    flash_attn._FA4_DENSE_ATTENTION_KERNEL = hybrid_attention
    print('THOR_HYBRID_PLUGIN_LOADED', flush=True)
