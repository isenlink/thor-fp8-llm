# 03 — Text-only production deploy (Qwen3.8-27B NVFP4 + MTP)

## Model directory layout

```
/data/models/p3-text/
  config.json            # flat qwen3_next config (model_type: qwen3_next,
                         # num_experts: 0, full_attention_interval: 4,
                         # max_position_embeddings: 262144)
  model.safetensors      # NVFP4 text weights (~22G)
  tokenizer.json / tokenizer_config.json / generation_config.json
```

The production weights came from a weight-surgery pass (MTP head integrated,
lm_head handling, GDN projection layout `in_proj_qkvz` 16 groups
[q128|k128|v384|z384]). If you start from a stock HF checkpoint, the
quantization_config in `../configs/` shows the required ignore patterns
(`linear_attn` projections are NOT NVFP4-quantized — regex
`re:model\.layers\.\d+\.linear_attn.*`).

## MTP draft

The qwen3_next MTP head lives in a separate directory (`MTP_DRAFT` in the
serve template) with its own config.json (`architectures: ["Qwen3NextMTP"]`)
and draft weights. vLLM routes it via `--speculative-config` with
`"method":"mtp"`.

**Hidden requirement**: `hf_config_override` in vllm/config/speculative.py
rewrites `model_type: qwen3_next → qwen3_next_mtp` ONLY when the draft config's
model_type is `qwen3_next`. If you point at a directory with any other
model_type, the draft silently routes to the wrong architecture.

## Serve

Edit variables in `../scripts/start-serve-text.sh` (WORKSPACE, MODEL_DIR,
MTP_DRAFT, GCC_ROOT), then run it. Success checklist is in
`00-QUICKSTART.md`.

## Why these exact flags

| flag | value | reason |
|---|---|---|
| --gpu-memory-utilization | 0.94 | 0.90 → KV locked at 23.7G / 200K concurrency 1.82x. 0.94 → 27.95G / 2.14x. Board has unified memory; be aggressive. |
| --max-model-len | 200000 | 262K dual-request is physically impossible (2×262K×64KB/tok > pool). 200K dual = 2.14x, verified 797s both-complete. |
| --max-num-seqs | 3 | concurrency 3 is the sweet spot on 54G pool |
| --no-enable-prefix-caching | | qwen3_next asserts with it on; mamba_block_size alignment handles paging |
| --reasoning-parser | deepseek_r1 | NOT qwen3 — see KNOWN-ISSUES #4 |
| --speculative-config | mtp spec=1 | spec≥2 + CG concurrency crashes (KNOWN-ISSUES #2) |

## Validation matrix (all passing)

| test | expectation |
|---|---|
| 1+1=2, Paris, 9.9>9.11 | correct (9.11 self-corrects via reasoning) |
| single stream 400 tok | ~18.1 tok/s |
| 3× concurrent 300 tok | ~41.8 tok/s total |
| 200K × 2 concurrent | both complete ~797s |
| >200K request | HTTP 400 clean reject (not deadlock) |
| reasoning split | content='2', reasoning_content=thinking text |

## Ops: switching between text and multimodal serves

They cannot run simultaneously (GPU memory exclusive at util 0.94). Stop one,
verify memory freed (`pkill -9 -f "VLLM::EngineCor[e]"` + memory check), start
the other. Templates emit pidfiles; kill by pidfile, never by pattern.
