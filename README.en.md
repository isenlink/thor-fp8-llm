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
| CPU | ARM 12 cores [measured /proc/cpuinfo: implementer 0x41 part 0xd83], no SMT, 6 cpufreq domains, 54 MHz–2.6 GHz (marketing core name not exposed on-board; external sources say Neoverse V3AE class [inferred]) |
| GPU | compute capability 10.1 (sm_101a), 14 SM @ 1530 MHz, L2 24 MiB (Linux-domain visible; board runs under a hypervisor) ⚠️ NOT the same chip as Jetson AGX Thor (sm_110/20SM) |
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
| 9/16 Drafting recipes | Without confidence gating: built-in MTP sweet spot **n_max=2-3**, DFlash2 sweet spot **n_max=5** with a flat n4-n7 plateau (±2%; hard cap = the draft head's trained block size, 8); external MTP is **exactly equivalent** to built-in (20.31 vs 20.32 tok/s, wasting 1.37 GB); **`p-min` gating loses at every depth tried** (leave it off); **`draft-dspark` is the same code path as `draft-dflash`** (no benefit); the gain is **strongly content-type dependent** (see 04/ `speculative-drafting-recipes`) |
| 9/16 KV budget & 256K | Only **16 of 65 layers are full-attention** ⇒ KV costs just **64 KB/token**, so 256K in f16 needs 16.8 GB and **runs end-to-end**; TTFT 67 s (30K) → 25.6 min (256K) while decode drops only −28%; follow-up questions on the same document: **TTFT 5.9 s** (prefix cache); **KV-type ablation: q8_0 saves 6.5 GB but costs 34-36% of decode** (acceptance unchanged ⇒ pure dequant overhead), so **f16 is the fixed choice at 256K** (see 05/ `kv-budget-and-256k`) |
| 9/16 Runtime memory growth **solved** | +626 MB per **novel prompt** ⇒ OOM after ~12 prompts. Cause: the host-side prompt cache defaults to **8192 MiB, more than the ~7.5 GB of movable RAM left** by the 46 GiB hugepage pool. Confirmed by a dose-response sweep (`--cache-ram` 0/512/default → plateau +0.7G/+1.2G/linear-until-OOM). **Fix: `--cache-ram 512`** (see 05/ `runtime-memory-growth`) |
| 9/16 Speculative-decoding matrix (round 1) | NVFP4-Q8 target + DFlash2 draft (n=7): **28.39 t/s avg / 29.85 peak vs 11.38 with no drafting (+149%)**; **NVFP4 only pays off with the tcgen05 kernel path** — a K-quant build 27% smaller is 16% *slower* (K-quant must dequantize before it reaches the tensor cores); DFlash2 beats built-in MTP everywhere (+149% vs +71%); **depth is content-dependent**: n=7 for code/JSON, n=3 for Chinese prose (at n=7 Chinese drops to 8.5 t/s, below the no-drafting baseline) → [details](docs/06-benchmarks/thor03-dflash2-speculative-matrix-2026-09-15.md) |
| 9/16 Multi-slot concurrency + KV/prefix reuse | **Set `np` ≥ peak concurrency** (so waves = ⌈concurrency/np⌉ = 1): 4-way/8-way instantaneous aggregate **2.12×/2.67×**, while an `np=1` control arm gets **zero speedup** (all the gain comes from multi-slot continuous batching); **extra slots never slow a single stream** (≤0.06 t/s per request); multi-turn prefix reuse saves **91–97%** of prefill (the last 4 tokens are always recomputed by upstream checkpoint design, source `{4+n_ubatch, 4}`); **cross-request prefix caching is unavailable** (`llama_memory_can_shift()` = false ⇒ `--cache-reuse` is a no-op) ⇒ prefill inflates **3.9×** under concurrency and TTFT degrades; verdict: **no need for vLLM for multi-turn chat**, only for "one fixed long system prompt + many independent short requests" → [details](docs/06-benchmarks/thor04-multi-slot-concurrency-kv-reuse-2026-09-16.md) |
| 9/15 Qwen3.6-35B-A3B (Q4_K_M MoE) | pp512 **739.98 t/s** / tg128 **40.24 t/s** — dense-4B-class speed with 9× the parameters; extremely asymmetric GQA (2 KV heads) ⇒ 256K of f16 KV costs only **20 GB**; 71 °C peak under air cooling → [details](docs/06-benchmarks/thor03-qwen36-35b-a3b-moe-2026-09-15.md) |
| 9/23 Occamy-1.0 at 200K KV (Q4_K_M MoE) | **`-c 200000` (`n_ctx_slot=200192`) verified**: a 35,651-token prompt is fully accepted (the 32768 baseline cannot hold it); prefill 694–705 t/s; short-context decode **50.4 t/s, unchanged from the 32K baseline's 50.11** (6.1× the context, same speed), 43.9 t/s at 34K context (**−12.5%**); ⚠️ negative result: attaching the 27B DFlash2 draft **crashes** in speculative init (`GGML_ASSERT(ggml_can_repeat)`), a `qwen35moe`-compatible drafter is required → [details](docs/06-benchmarks/thor04-occamy-moe-200k-context-2026-09-23.md) |
| GPU hugepage pool | 20G → 42G (later expanded to 46G), persisted; ⚠️ **grow-only**: shrinking the pool breaks model loading (`unable to allocate CUDA0 buffer`, A/B measured) |
| Full-load temperature | 72–74 °C (passive cooling, stable) |

## Repo structure

### ⚡ Prebuilt binary quick download

Just want to run llama-server without compiling? **The prebuilt binary (74.7 MiB) is distributed via Baidu Netdisk**:

🔗 **<https://pan.baidu.com/s/16CNsjD3J2psvBo57oJZ2ZQ?pwd=d8bc>** (extraction code `d8bc`)

- Package `thor-prebuilt-2026-09-19`: llama-server + NVFP4 numerical self-check tools + `optional-drafter/` directory
- Verify sha256 after download (checksums in [docs/08-fp8-fastpath-server/README.md](docs/08-fp8-fastpath-server/README.md))
- ⚠️ **Host preparation (cold-start hugepage pool allocation) is mandatory before launching, otherwise throughput will be far below spec or the server won't start** — follow the "Quick start" two steps in that README

```
TROUBLESHOOTING.md        Pitfall reference (symptom index, verbatim errors ← read this first)
docs/
  01-hardware-recon/      Board environment recon: memory truth, carveout, tmpfs, storage layout
  02-cross-compile/       x86 host cross-compiling aarch64 + sm_101a full toolchain (11 pitfalls)
  03-model-conversion/    FP8 → GGUF conversion, three-layer obstacles + model file ledger
  04-nvfp4-optimization/  NVFP4 quantization + speculative-decoding tuning (incl. failed MTP K7, B3 kernel-level optimization decision chain, microbench breakdown, correctness-incident fix chain, final results, rejected follow-up paths, GPU deadlock discipline, multi-board parallel testing readiness, **drafting recipes: depth sweet spots / mixed-precision draft / content-type dependence**, **the DFlash2 verdict retraction**)
  05-system-tuning/       Hugepage pool expansion & persistence (**incl. proof that the pool is grow-only**), overlay, power-loss recovery, temperature, **KV budget & 256K measurements**, **runtime memory-growth investigation (solved: prompt-cache default exceeds usable RAM)**
  06-benchmarks/          Per-stage benchmarks + community comparison, **benchmark methodology (three measurement traps + paired design)**, **speculative-decoding matrix**, **multi-slot concurrency + KV/prefix reuse measurements**, **Q4_K_M MoE throughput + 200K KV measurements**
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

5. **A dirty power loss wipes the overlay** — `/etc`, `/home`, `/media` roll back to the vendor image
   after an unclean shutdown (fsck formats the overlay upper layer). Anything that must survive goes on
   the board's data partition, and auto-start has to be triggered from an always-on outside host over
   serial. See [docs/07-hermes-agent/](docs/07-hermes-agent/).

## Status

🚧 Content is being organized (source docs are being sanitized and restructured).
The private phase is the content-review window; a final sensitive-info pass is
done before going public.

## License

MIT (TBD)
