# Benchmark methodology & full numbers

All numbers measured on the target board (sm_101, DRIVE OS 7.0.3, 60G unified
memory, hugepage GPU pool 54G). llama.cpp comparison numbers from the same board
class: llama.cpp-tcgen05 (NVFP4 kernels), Qwen3.8-27B-NVFP4-Q8 GGUF + DFlash2
draft (n=5), temp 0, ctx 262144.

## Text-only production config (this repo)

`start-serve-text.sh`: util 0.94, max-model-len 200000, max-num-seqs 3,
MTP spec=1, CUDA graphs, no prefix caching, reasoning-parser deepseek_r1.

### Decode speed

| Scenario | vLLM this repo | llama.cpp-tcgen05 |
|---|---|---|
| Single stream, short Chinese prompt | **18.1 tok/s** (3-run mean) | 12.6 tok/s (256-tok gen) |
| Short JSON (temp 0, full-output timing) | 19.4 tok/s | **27.5 tok/s** |
| Short code generation (temp 0) | 18.7 tok/s | **30.3 tok/s** |
| Concurrent 3 × short | 41.8 tok/s total (13.9-14.8 each) | n/a (single-slot) |
| MTP mean acceptance length | 1.71–2.00 | 0.44–0.68 (draft rate) |

### Long context

| Prompt size | prefill | decode (after prefill) | total |
|---|---|---|---|
| 40K × 2 concurrent | — | 2.6–2.8 tok/s each | **76s** |
| 150K × 1 | — | — | 245s (incl. 200 tok gen) |
| **200K × 2 concurrent** | — | — | **797s, both complete, 2.14x KV headroom** |
| 150K (multimodal config) | **9,500 tok/s (~16s)** | ~2 tok/s | 121s (150 tok gen) |

llama.cpp same-class board at 150K: prefill 200 tok/s (~12.5 min), decode
18.3 tok/s (54.6ms/token, cache-hit precise timing). 32K: 21.9, 75K: 24.0.

### KV budget arithmetic (memorize this before filing issues)

- KV price: 64KB/token (16 full-attention layers × 2 × 4 heads × 256 dim × 2 bytes fp16)
- Mamba (GDN) state: ~76MB per request
- Text-only config: KV pool 27.95G = 107,328 tokens → 200K single = 0.54x,
  200K dual = 1.07x of one request... wait: 2 × 200K needs 2.03x of a 200K-capable
  pool → verified working at 2.14x headroom
- Multimodal config: vision tower eats ~2G → KV 78,208 tokens → 200K single only

## Multimodal (grafted vision tower)

| Scenario | Result |
|---|---|
| Image understanding (describe) | correct, Chinese output |
| OCR | reads bottle-label text correctly |
| Image token cost | ~1000 tokens per image (1070 tok prompt incl. text) |
| Short generation w/ vision | 16.1–18.7 tok/s, fr=stop |
| Reasoning/content split | clean (deepseek_r1 parser) |

### Thinking separation gotcha

`--reasoning-parser qwen3` DOES NOT WORK with this chat template (template injects
`<think>` into prompt; parser requires it in output). Use `deepseek_r1`.

## Draft-depth sensitivity (DFlash2 n=2/3/5 on llama.cpp, for reference)

At 150K context: 747 / 748 / 751-753s — no difference (<1%). Prefill dominates.
Draft depth only matters for short-context generation.

## How to reproduce

1. Follow `docs/00-QUICKSTART.md` through `docs/03-TEXT-DEPLOY.md`
2. Run `scripts/bench_short_gen.py`, `scripts/bench_json_code.py`,
   `scripts/bench_longctx.py` against the running serve
3. Compare against the tables above; ±5% run-to-run noise is normal
