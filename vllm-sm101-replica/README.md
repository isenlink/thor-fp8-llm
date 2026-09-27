# vLLM 0.11.2 on DRIVE OS Thor (sm_101) — Full Reproduction Package

> **Unofficial community effort** · scripts & docs **MIT** · not affiliated with
> vLLM, NVIDIA, or Qwen (upstream projects keep their own licenses).

> Community edition. Everything here was proven on real hardware: text-only and
> multimodal inference, 200K-token contexts, MTP speculative decoding, on a board
> the official vLLM wheels don't support.
>
> **Goal**: a stranger with the same board should be able to reproduce every result
> by following docs in order. If something is missing, that's a bug in this package —
> file an issue.

## What this is

Qwen3.8-27B (hybrid GDN/attention architecture, NVFP4 quantized) running on
vLLM 0.11.2, cross-compiled for sm_101 (DRIVE OS 7.0.3, CUDA 12.8), on a 60GB
unified-memory board with no discrete VRAM. Includes:

- **Text-only serve** — 18.1 tok/s single stream, 200K-token dual-request verified
- **Multimodal serve** — Qwen3VL vision tower grafted onto the Qwen3-Next text tower
  (this architecture doesn't exist upstream in vLLM 0.11.2; we built it, see
  `docs/PLAN-A-MULTIMODAL.md`)
- **MTP speculative decoding** — 1.7–2.0 mean acceptance length
- **Complete pitfall list** — 13 documented crash points with exact fixes

## Benchmark summary (measured, reproducible)

| Scenario | This stack | llama.cpp (same board class) |
|---|---|---|
| Short Chinese generation | **17–18 tok/s** | 12.6 tok/s |
| Short JSON / code | 18.7–19.4 tok/s | **27.5–30.3 tok/s** |
| 150K-token prefill | **9,500 tok/s (~16s)** | 200 tok/s (~12.5 min) |
| 150K-token decode | 2–18 tok/s (see below) | 18.3 tok/s |
| 200K dual request | ✅ both complete (~797s) | engine defect: 21% empty answers |
| Multimodal (image + text) | ✅ works | ✅ works (mmproj) |

Honest notes:
- llama.cpp wins on short structured output and long-context decode (KV attention
  cost grows with context — physics, not implementation).
- vLLM wins on prefill by ~47x and is the only one of the two that survives 200K
  dual requests.
- Long-context decode on vLLM/GDN degrades hard at 150K+ (2 tok/s in our
  multimodal config). Open problem, see `docs/KNOWN-ISSUES.md`.

## Directory layout

```
docs/
  00-QUICKSTART.md           start here: minimal path to a running serve
  01-CROSS-COMPILE.md        torch 2.9.0 + vLLM 0.11.2 sm_101 cross-compile recipe
  02-BOARD-PREP.md           hugepage pool, offline gcc, NVML stub, python env
  03-TEXT-DEPLOY.md          text-only production deploy (54G pool, MTP, CG)
  04-RESTORE-THOR01.md       6-step restore from a blank board (each step has an exit condition)
  PLAN-A-MULTIMODAL.md       ★ the vision-tower graft: architecture, 13 pitfalls, fixes
  KNOWN-ISSUES.md            what's broken / open, with reproduction cases
  BENCHMARKS.md              full methodology + numbers, so you can compare fairly
build/                       cross-compile host scripts (paths are placeholders — set your own)
patches/                     every source modification, named by purpose
model-classes/               qwen38_vl.py (the grafted model class) + MTP patch
configs/                     vl27b-config.json (nested qwen3_vl config example)
scripts/                     serve templates + all benchmark/smoke scripts
MANIFEST.txt                 file list of the full distribution tarball (incl. binaries)
```

## Hardware this was proven on

- Board: DRIVE AGX Thor-class (sm_101), DRIVE OS 7.0.3, CUDA 12.8, Python 3.12 aarch64
- Memory: 60GB unified (LPDDR), hugepage GPU pool 52–54G
- Host for cross-compiling: x86_64 Linux with NVCC for aarch64 + CUDA 12.8

## Model assets you need

| Asset | Source | Size |
|---|---|---|
| Qwen3.8-27B NVFP4-HF (text tower) | HuggingFace | ~22G |
| Vision tower (333 tensors, BF16) | extracted from the same checkpoint — `patches/extract-visual-tower.py` | 879MiB |
| MTP draft head | included in checkpoint or separate | ~850M |

Weights are NOT redistributed here. Scripts assume HF-format safetensors.

## License / attribution

Scripts and docs: MIT. Upstream vLLM/torch keep their licenses.
This is an unofficial community effort — not affiliated with vLLM, NVIDIA, or Qwen.


## Binary distribution (not stored in git)

The full distribution tarball and all wheels/debs stay **out of the git tree**
(git is for source + docs). Everything below was hash-measured from the package.

| Asset | Size (bytes) | sha256 |
|---|---:|---|
| `vllm-sm101-replica-20260927.tar.gz` (complete package: docs + patches + wheels + offline gcc debs) | 767,571,390 | `dde6cbdccf0a0d2aa2644667b845a8ec2a41716f95a2bd8a49e023c18d30a002` |
| `torch-2.9.0-cp312-cp312-linux_aarch64-fixed.whl` (aarch64 cross-build of upstream torch 2.9.0, pytorch.org) | 272,523,042 | `3479cc4c05eadca33b2bd30a77190ca91d5c0547060b1e935d1c15e3e76cddaa` |
| `vllm-0.11.2-cp312-cp312-linux_aarch64.whl` (aarch64 cross-build of upstream vLLM 0.11.2, github.com/vllm-project/vllm) | 459,708,107 | `a30627de81ca1f1fad4911be4d0a3556ae462bd15c27db9c47924706ca471cc9` |
| `compressed_tensors-0.9.0-py3-none-any.whl` | 96,438 | `c4a0bccf2fd180a18a4bf0646f7746df4ea8ea537c06061f4b571662debf4424` |
| `frozendict-2.4.7-py3-none-any.whl` | 16,264 | `972af65924ea25cf5b4d9326d549e69a9a4918d8a76a9d3a7cd174d98b237550` |
| `build/gcc-debs/` (offline gcc-12 toolchain debs, in tarball only) | 46,222,672 | (in tarball) |

🔗 **下载：链接待补**（发布时填入百度网盘直链 + 提取码；或见本仓 Releases）
— download the tarball, then `sha256sum -c` against the table above.

## Path aliases (brand-neutral naming)

Board data-partition paths are written with aliases to keep this repo
brand-neutral (the repo-wide convention):

| Alias in this package | Meaning |
|---|---|
| `/brand_data/` | the board's large persistent data partition (`ls /` on your board shows its real name) |
| `/brand_hdmap/` | the board's map/data side partition (holds `p3-models/` etc. in our layout) |

`MANIFEST.txt` is the file list of the **distribution tarball** — it includes the
binaries above, which are intentionally not committed to git.

## About this branch

This content lives on the `vllm-sm101-replica` branch of a repository whose `main`
branch documents the **llama.cpp** stack for the same board class. The two
frameworks are independent lines: vLLM conclusions do not update `main`, and
`main` benchmark numbers do not apply here (see the benchmark comparison table
above for the head-to-head).
