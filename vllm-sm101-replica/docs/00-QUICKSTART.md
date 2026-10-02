# 00 — Quickstart: minimal path to a running serve

Assumes: DRIVE OS 7.0.3 board (sm_101), CUDA 12.8, Python 3.12 aarch64,
60G unified memory (hugepage pool configurable), host x86_64 for cross-compile.

Total time budget: ~1 day (cross-compile is the long pole).

## 0. What you need

- [ ] Host x86_64 with CUDA 12.8 aarch64 cross toolchain (see `01-CROSS-COMPILE.md`)
- [ ] Board with root/ssh, ≥40G free persistent storage
- [ ] Qwen3.8-27B NVFP4-HF checkpoint (HuggingFace safetensors)
- [ ] Offline gcc for the board (torch.compile needs it at runtime) — see `02-BOARD-PREP.md`

## 1. Cross-compile torch + vLLM on the host

Follow `01-CROSS-COMPILE.md`. Outputs:
- `torch-2.9.0-cp312-cp312-linux_aarch64.whl`
- `vllm-0.11.2-cp312-cp312-linux_aarch64.whl`

Both install into a board-side venv. (Prebuilt wheels are NOT in this repo;
the recipe is deterministic.)

## 2. Prepare the board

Follow `02-BOARD-PREP.md`. In short:
- hugepage pool: target 52-54G (`/proc/meminfo HugePages_Total`)
- offline gcc unpacked at a fixed path
- NVML stub patches applied to the venv (board has no NVML)
- deps: the wheel set in `scripts/board-deps-list.txt`

## 3. Text-only serve

Follow `03-TEXT-DEPLOY.md` (model dir layout, weight surgery for MTP head,
config). ⚠️ **MTP depth must be 1** — `num_speculative_tokens ≥ 2` crashes under
concurrent load on vLLM 0.11.2 + qwen3_next (confirmed 2026-10-02, see
KNOWN-ISSUES #2). Then:

```bash
WORKSPACE=/opt/vllm-p3 MODEL_DIR=/data/models/p3-text bash scripts/start-serve-text.sh
# wait for "startup complete" in serve-text.log (~3-5 min)
curl -s localhost:8998/health   # {"status":"ok"}
```

Expect: KV ~27.95G / 107K tokens / "Maximum concurrency for 200000: 2.14x".

## 4. Multimodal (vision) serve

Follow `docs/PLAN-A-MULTIMODAL.md` — this is the advanced path:
extract the vision tower, build the nested qwen3_vl config, install the grafted
model class, apply the 13 documented patches. Then:

```bash
WORKSPACE=/opt/vllm-p3 MODEL_DIR=/data/models/p3-vl-27b bash scripts/start-serve-multimodal.sh
# KV will show ~78K tokens; 200K is single-request (see KNOWN-ISSUES #3)
```

## 5. Verify

```bash
python3 scripts/smoke_vl.py            # text + image + OCR checks
python3 scripts/check_reasoning_split.py   # content vs reasoning_content
python3 scripts/bench_short_gen.py     # expect 16-19 tok/s
python3 scripts/bench_json_code.py     # expect ~19 tok/s (fr=stop)
```

## Startup success checklist (memorize)

In serve log you MUST see:
1. `Setting attention block size to 400 tokens...` (hybrid mamba alignment fired)
2. `Padding mamba page size by ...%` (same)
3. `Available KV cache memory: ...`
4. `GPU KV cache size: ...` + `Maximum concurrency for ...`
5. `startup complete`

No line 1-2 → IsHybrid mixin not effective → will crash later at KV profile.
See `docs/PLAN-A-MULTIMODAL.md` §四.1.

## Operations

Kill properly: APIServer pid from serve.pid, then
`pkill -9 -f "VLLM::EngineCor[e]"` (bracket trick), verify GPU memory freed
BEFORE restarting. Zombie EngineCore holds ~48G.
