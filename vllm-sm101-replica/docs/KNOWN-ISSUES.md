# KNOWN ISSUES / Open Problems

Honest list. If you hit these, it's not you — it's us.

## 1. Long-context decode degradation (OPEN)

vLLM 0.11.2 + Qwen3-Next GDN hybrid: decode speed drops hard at 150K+ context.
Measured: 150K prompt → ~2 tok/s decode (multimodal config). Short context = 17-19 tok/s.
llama.cpp on the same class of board does 18.3 tok/s at 150K.

Suspected: GDN state handling cost at depth / mamba page alignment. Untested fixes:
larger `--max-num-batched-tokens`, decode-side chunking, or a newer vLLM with
hybrid KV improvements. **If you solve this, please publish.**

## 2. spec=2+ MTP with CUDA graphs crashes (PATCHED AROUND)

`num_speculative_tokens >= 2` + concurrent requests → illegal memory access in
rejection sampler. Production config uses spec=1 (verified stable).
Fix attempt `patches/fix-cudagraph-concurrency-pad.py` addresses the CG padding
half; the sampler half is untouched.

## 3. Multimodal KV budget (CONSTRAINT, not bug)

Vision tower + 200K ctx at util 0.94 → KV 78,208 tokens → 200K works for ONE
request only (text-only variant fits 2 concurrent). Physical memory limit
(60G unified, no discrete VRAM). Don't file "dual 200K fails" — it's arithmetic.

## 4. Thinking-tag splitting: use deepseek_r1 parser, NOT qwen3 (GOTCHA)

`--reasoning-parser qwen3` silently fails to split because its strict mode
requires `<think>` in the model OUTPUT, but the chat template already injects
`<think>\n` into the PROMPT. Model output starts with bare reasoning and ends
with `</think>` → qwen3 parser returns (None, everything).

Use `--reasoning-parser deepseek_r1` (splits on `</think>` alone).
Verified: content/reasoning_content cleanly separated, no speed regression.

## 5. NVML stub is required (environment quirk)

Board has no NVML library. FP8 triton kernel config lookup calls
`nvmlDeviceGetName()` — a naive stub returning bytes crashes it
(`get_device_name().replace(" ", "_")` needs str). Patches included:
`patches/patch-pynvml-stub.py`, `patches/fix-pynvml-nvml-name.py`.

## 6. Prefix caching must stay OFF

qwen3_next family asserts with prefix caching enabled. Also mamba_block_size
is only settable when prefix caching is on — the hybrid alignment logic handles
page sizing instead. Just don't enable it.

## 7. MTP > 120K context: probabilistic early EOS (MODEL BEHAVIOR)

~1/3 of long generations (120K+) end with immediate EOS regardless of engine.
Present on llama.cpp too → model-level, not a porting bug. Mitigation: prompt
constraints or presence_penalty.

## 8. 262K single-request fits, 262K dual-request is physically impossible

2×262K = ~34G KV > what 60G unified memory can give after weights. We capped
max-model-len at 200K (dual-request verified at 2.14x). Extending the hugepage
pool cannot fix arithmetic.

## 9. EngineCore zombie after kill (OPERATIONS)

Killing the APIServer pid leaves `VLLM::EngineCore` holding 48G of GPU memory.
Kill pattern: `pkill -9 -f "VLLM::EngineCor[e]"` (bracket trick avoids self-match).
Always verify with nvidia-smi / free before restarting.
