# 04 — Restore Thor01 from a clean board (tested recovery path)

The board was wiped on 2026-09-27 (0 models, 0 venv, 0 processes). Everything
below rebuilds the full stack from this package. Order matters; each step has a
verifiable exit condition. Do not parallelize steps 3-6.

> Paths: `<WS>` = `/brand_data/ai_workspace`, `<HD>` = `/brand_hdmap`.
> Get this package onto the board via SMB share or scp from the packaging host.

## Step 0 — prerequisites on board

```bash
python3 --version        # 3.12.3
ls /usr/local/cuda-12.8/ # exists
grep MemTotal /proc/meminfo
grep HugePages_Total /proc/meminfo   # note the pool size
df -h /brand_data/        # need ~35G free for everything
```

## Step 1 — hugepage pool (if not already sized)

54G pool = 27648 x 2MB pages. Verify with `grep HugePages_Total /proc/meminfo`.
If your pool is smaller, see docs/02-BOARD-PREP.md (boot-time config).

Exit condition: `HugePages_Total: 27648` (or your chosen size).

## Step 2 — offline gcc root (runtime torch.compile dependency)

```bash
cd <package>/build
bash unpack-gcc-root.sh /brand_data/ai_workspace/gcc-root/root
# exit: prints "gcc (Ubuntu 12.3.0-...)" version line
```

## Step 3 — venv + wheels

```bash
python3 -m venv /brand_data/ai_workspace/vllm-p3/venv
V=/brand_data/ai_workspace/vllm-p3/venv
$V/bin/pip install ../binaries/torch-2.9.0-cp312-cp312-linux_aarch64-fixed.whl
$V/bin/pip install ../binaries/vllm-0.11.2-cp312-cp312-linux_aarch64.whl \
    --no-deps --find-links ../build/   # frozendict/compressed_tensors are local
$V/bin/pip install fastapi uvicorn openai hvloop 2>/dev/null || \
    $V/bin/pip install fastapi uvicorn openai   # serve deps
# exit: `$V/bin/python -c "import vllm; print(vllm.__version__)"` -> 0.11.2
```

## Step 4 — apply the 13 patches (in this order)

Run each `patches/NN-*.py` **on the board** against the installed vllm tree,
`$V/lib/python3.12/site-packages/vllm/`. Order + purpose table in
patches/README.md. After each, the patch script self-verifies.

Critical ones (skip = serve will not start):
- NVML stub (no libnvidia-ml on DRIVE OS)
- qwen3_next GDN (BOARD_PATCHED model class)
- MTP draft route + mamba whitelist
- CUDA graph capture fix

## Step 5 — model weights

Text (Qwen3.8-27B NVFP4 v4): copy to `<HD>/p3-models/` —
`model.safetensors` (23,839,051,704 B), `model_mtp.safetensors` (849,400,392 B),
`config.json` (flat qwen3_next), `tokenizer.json`, `tokenizer_config.json`,
`vocab.json`, `chat_template.jinja`, `generation_config.json`.
Weights are NOT in this package (size); source them from your training/packaging
pipeline. Verify byte sizes exactly — wrong v4 tensor will crash at load.

Multimodal add-on: `configs/vl27b-config.json` + `visual-tower.safetensors`
(879M, md5 31f61f97...) go into the model dir; see docs/PLAN-A-MULTIMODAL.md.

## Step 6 — serve

```bash
bash scripts/start-serve-text.sh        # port 8080, 54G pool budget
# exit conditions (watch log until ALL true):
#   "attention block size 416" (or 400 for VL)
#   KV cache tokens ≈ 107,328
#   /health returns ok
#   smoke: 1+1=2, no </think> leak in content
```

Multimodal: `bash scripts/start-serve-multimodal.sh` (port 8996, KV ≈ 78,208).
Reasoning split uses `--reasoning-parser deepseek_r1` (NOT qwen3 — strict mode
rejects template-injected `<think>`; see BENCHMARKS.md notes).

## Verification checklist (marks a faithful restore)

| Check | Expected |
|---|---|
| short text speed | 16-18 tok/s |
| JSON / code (temp 0) | ~19 / ~18.5 tok/s |
| 100K prefill | ~9,500 tok/s |
| MTP acceptance | 1.7-2.0 |
| KV tokens (text) | ≈107K |
| reasoning split | content clean, reasoning_content filled |

Full methodology: docs/BENCHMARKS.md. If numbers are far off, check hugepage
pool size and gpu_memory_utilization first — they dominate everything.

## Cleanup (wipe back to blank)

`scripts/wipe-board.sh` (kill serve by PID file, rm venv/cache/gcc-root/model
dirs — only the ones this package created).
