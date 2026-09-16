# NVIDIA DRIVE Thor (Tegra264) LLM Deployment Notes

> Grassroots field notes: deploying and optimizing llama.cpp GPU inference on an
> NVIDIA DRIVE Thor domain controller (p3960-0010 / Tegra264, sm_101a,
> DriveOS 7.0.3, CUDA 12.8) — from cross-compilation to NVFP4 quantization.
> Every number here was measured on real hardware.
>
> [中文版本 / Chinese](README.md)

## Why this repo exists

Community documentation for running local LLMs on a **DRIVE Thor** is essentially
nonexistent. Official material only covers the DriveOS SDK perspective, and forum
discussion stops at "can it run at all." We spent about a week in 2026-09 working
the full chain from scratch, and this repo records the pitfalls, the failed
experiments, and the measured data — to save the next person the same groping.

> **🔧 Hit a problem? Start with [TROUBLESHOOTING.md](TROUBLESHOOTING.md)** — a
> symptom-indexed pitfall reference (30 entries, English error strings preserved
> verbatim for searchability) covering cross-compilation, model conversion,
> inference tuning, and system/mounting.

## Hardware

| Item | Spec |
|------|------|
| SoC | NVIDIA Tegra264 (DRIVE Thor, Blackwell) |
| GPU | compute capability 10.1 (sm_101a), 14 SM @ 1530 MHz, L2 24 MiB |
| GPU-visible memory | 20 GiB (unified-memory partition; 58 GiB physical, the gap is device-tree kernel carveout) |
| System | DriveOS 7.0.3, aarch64, full CUDA 12.8 runtime but **no compiler on board** |

## Key results

| Item | Result |
|------|--------|
| llama.cpp cross-compile | 0.4.0-dev, all 91 binaries incl. CUDA sm_101a backend ✅ |
| Qwen3-4B Q4_K_M baseline | pp128 1516 tok/s / tg64 40.2 tok/s (ngl=99, full GPU) |
| Qwen3.8-27B NVFP4 | decode **26.48 tok/s** (beat the 25.89 community target), acceptance 85.9% |
| 128K long-context decode | 16.4±0.7 tok/s (production baseline, MTP K7 + F8 attn + NVFP4 MLP) |
| B3 kernel optimization (standalone) | NVFP4 MMVQ gemv 144→207 GB/s (+44%), root cause = repeated L2 reads of the y vector (see 04/) |
| B3 final (perj fix + MTP K12 p0.5) | 2K decode **31.02 tok/s** (+20.6%), 128K decode **19.61 tok/s** (+16.2%), output byte-identical to baseline (see 04/ b3-final-results) |
| Post-9/13 optimization review | B4/B5 verify-kernel work, vocab crop, and draft F8 were rejected by paired A/B tests; 19.61 tok/s remains the production best (see 04/ takeover, draft-levers, incidents) |
| 9/16 DFlash2 verdict revised | The earlier "DFlash2 rejected" call is **retracted**: the collapsed acceptance came from a numerically damaged *target model*, not from DFlash2. Paired re-run (6 content types × 5 prompts): DFlash2 beats built-in MTP across the board (median +8.1%, code +28.3%), and quantizing the draft to 560 MB yields **byte-identical output** (see 04/ `dflash2-revalidation-2026-09-16.md`) |
| GPU hugepage pool | 20G → 42G (later expanded to 46G), persisted |
| Full-load temperature | 72–74 °C (passive cooling, stable) |

## Repo structure

```
TROUBLESHOOTING.md        Pitfall reference (symptom index, verbatim errors ← read this first)
docs/
  01-hardware-recon/      Board environment recon: memory truth, carveout, tmpfs, storage layout
  02-cross-compile/       x86 host cross-compiling aarch64 + sm_101a full toolchain (11 pitfalls)
  03-model-conversion/    FP8 → GGUF conversion, three-layer obstacles + model file ledger
  04-nvfp4-optimization/  NVFP4 quantization + speculative-decoding tuning (incl. failed MTP K7, B3 kernel-level optimization decision chain, microbench breakdown, correctness-incident fix chain, final results, rejected follow-up paths, GPU deadlock discipline, multi-board parallel testing readiness)
  05-system-tuning/       Hugepage pool expansion & persistence, overlay, power-loss recovery, temperature
  06-benchmarks/          Per-stage benchmarks + community comparison
scripts/                  Board/host helper scripts (UART probe, GPU pool check, B3 kernel microbench suite)
```

## The headline optimization (TL;DR)

The first major win was **speculative-decoding tuning**, followed by the B3
NVFP4 MMVQ kernel fix. On this bandwidth-constrained board, the stable
production recipe became:

```
NVFP4 MMVQ y-reuse fix (perj)
+ deep drafting (--spec-draft-n-max 12)
+ high-confidence gating (--spec-draft-p-min 0.5)
+ flash attention (-fa on)
+ single stream (--parallel 1)
```

This took the 128K production target from **16.88 → 19.61 tok/s**, while the
2K short-context benchmark reached **31.02 tok/s**. Later attempts to reach
30 tok/s at 128K are documented as rejected paths in
[docs/04-nvfp4-optimization/takeover-2026-09-13.md](docs/04-nvfp4-optimization/takeover-2026-09-13.md).

## What makes DRIVE Thor different (and hard)

1. **No compiler on board** — DriveOS philosophy is "compile on host, run on board."
   You cross-compile aarch64 + sm_101a on an x86 machine (QEMU-based ARM64 nvcc
   wrapper; see [docs/02-cross-compile/](docs/02-cross-compile/)).
2. **Read-only root** — `/mnt`, `/sbin`, `/usr/local` are all read-only.
   Persistent mounts/files go under the writable overlay: `/media`, `/home`,
   `/var`, `/etc`. We hit this three times.
3. **58 GiB physical ≠ 20 GiB GPU** — the gap is kernel carveout you cannot
   reclaim from userspace. Grow the GPU pool via hugepages instead
   ([docs/05-system-tuning/hugepage-pool.md](docs/05-system-tuning/hugepage-pool.md)).
4. **sm_101a, not sm_120** — Thor *can* use tcgen05 (5th-gen Tensor Core PTX);
   consumer RTX 50 series (sm_120) *cannot*. This is why the community calls it
   "Blackwell-specific." See [docs/04-nvfp4-optimization/tcgen05-research.md](docs/04-nvfp4-optimization/tcgen05-research.md).

## Status

🚧 Content is being organized (source docs are being sanitized and restructured).
The private phase is the content-review window; a final sensitive-info pass is
done before going public.

## License

MIT (TBD)
