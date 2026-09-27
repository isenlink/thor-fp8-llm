# 01 — Cross-compiling torch 2.9.0 + vLLM 0.11.2 for sm_101 (aarch64)

Cross-compile on an x86_64 host with CUDA 12.8 aarch64 (sbsa) toolchain.
Everything below is battle-tested; the pitfalls are the real content.

## Toolchain setup

- gcc-14 cross (`aarch64-linux-gnu-gcc`) + CUDA 12.8.93 sbsa target
  (`<toolchain>/sbsa-linux`) + Ubuntu noble sysroot
- **nvcc must be a bash wrapper occupying the original name**
  `sbsa-linux/bin/nvcc`, with the real binary as `nvcc.real`. CMake's FindCUDA
  bypasses `CMAKE_CUDA_COMPILER` and executes the binary directly — the wrapper
  in the named path is what saves you.
- Pass `-DCMAKE_TOOLCHAIN_FILE=<toolchain>/toolchain-torch-xc128.cmake`

## Build isolation

Run the build in a systemd slice with a hard memory cap, NOT inside a desktop
session cgroup (memcg OOM kills mid-link):

```bash
sudo systemd-run --unit=torchbuild --slice=system.slice -p MemoryMax=80G bash build-torch.sh
export MAX_JOBS=16
```

**NEVER build under /tmp.** A reboot silently deletes a 19G build tree at
[217/417]. Use persistent storage and tee logs there.

## torch 2.9.0

- Explicit `USE_CUDA=1` — omit it and you get a silent CPU-only wheel
- Patch build.ninja guest-tool invocations (sleef, protoc) to go through the
  qemu wrapper
- Missing headers on demand: nvml.h, nvtx, cufftXt, cudalibxt, curand → copy
  into sbsa include dir
- torch 2.9 NVML compatibility block must go BEFORE the include guard `#endif`
- sbsa `crt/math_functions.h`: keep `double sincospi` ENABLED (older notes say
  to `#if 0` it — that was for an older CUDA, re-enabling is required now)
- **Wheel repack**: build machine tags .so suffix and WHEEL metadata as x86_64.
  Repack to aarch64 (fix .so filenames + WHEEL Tag + RECORD hashes). See
  `../patches/fix-wheel-aarch64.py` pattern.

## vLLM 0.11.2

- Link errors `-lcudadevrt -lcudart_static` → create `sbsa-linux/lib64`
  symlinks pointing back into `lib/`
- FA3 has no sm_101 support upstream (CC≥10 unsupported) — runtime falls back
  to FA2 via PTX JIT. Acceptance criterion: main `_C.so` and `_moe_C.so`
  contain sm_101 cubins. Check with `cuobjdump -lelfs`.
- Post-process wheel with the aarch64 fixer (same as torch)

## Board-side install (DRIVE OS quirks)

- No pip on board: create venv `--without-pip`, bootstrap pip from a zipfile
- Install vllm `--no-deps`, then resolve direct dependencies manually
  (following flashinfer's dependency chain drags in apache-tvm-ffi — avoid)
- Board lacks `libnvToolsExt.so.1` → copy the aarch64 build's
  `libnvToolsExt.so.1` into `venv/lib`, add `venv/lib` + cuda lib64 to
  LD_LIBRARY_PATH
- Offline wheel assembly: run pip install once with `--dry-run` to enumerate
  the true gap set. Do NOT do an unbounded dependency closure (one run pulled
  12G).

## Acceptance checks on the board

```bash
python3 -c "import torch; print(torch.cuda.get_device_capability(0))"
# expect: (10, 1)
python3 -c "import torch,time; a=torch.randn(2048,2048,device='cuda'); b=torch.randn(2048,2048,device='cuda'); torch.cuda.synchronize(); t=time.time(); [torch.mm(a,b) for _ in range(50)]; torch.cuda.synchronize(); print('50x matmul:', time.time()-t, 's')"
# expect: ~0.3s
python3 -c "import vllm; print(vllm.__version__)"
# expect: 0.11.2
```

## Dependency pins that matter (board-side)

| package | pin | why |
|---|---|---|
| compressed-tensors | 0.11.0 | 0.9 lacks transform; 0.19 needs torch≥2.10 |
| frozendict | any | missing from board image |
| transformers | per vllm 0.11.2 requirements | nested qwen3_vl config support |

## Known runtime patches (all in `../patches/`)

qwen3_next.py:1158 `set_moe_parameters` indentation bug breaks dense models
(`num_experts=0` misreports "No Qwen3Next layer") → `fix-moe-dense-indent.py`.
