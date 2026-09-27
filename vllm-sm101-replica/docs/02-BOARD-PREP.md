# 02 — Board preparation (DRIVE OS 7.0.3)

## Memory: hugepage pool = GPU memory

The board has 60G unified LPDDR and NO discrete VRAM. CUDA memory comes from a
hugepage pool you size at boot:

```
/etc/default/grub or boot param: default_hugepagesz=1G hugepagesz=1G hugepages=52
# or 2M pages: 27648 × 2M = 54G (what we validated)
```

Verify: `grep HugePages /proc/meminfo` → HugePages_Total × size = 52-54G.

**This pool IS the GPU memory.** torch.cuda reports it as device memory.
KV cache lives inside it → enlarging the pool directly enlarges KV capacity.
Physical ceiling: ~55G (system needs 6-8G).

## Offline gcc (required at runtime for torch.compile)

Board image has no gcc. Unpack a full aarch64 gcc root (e.g. Ubuntu noble
gcc-14 packages) to a fixed path, e.g. `/opt/gcc-root/root`, and export at
serve time:

```bash
export LD_LIBRARY_PATH=/opt/gcc-root/root/usr/lib/aarch64-linux-gnu:.../venv/lib:/usr/local/cuda-12.8/lib64
export CC=/opt/gcc-root/root/usr/bin/gcc
export CPATH=/opt/gcc-root/root/usr/include:.../aarch64-linux-gnu:.../python3.12
```

Templates in `../scripts/` do this — just set `GCC_ROOT`.

## NVML stub (board has no libnvidia-ml)

vLLM's platform probe and FP8 triton kernel lookup both call NVML. Two patches:

1. `patches/patch-pynvml-stub.py` — stub `nvmlInit` (falls back to torch.cuda),
   `nvmlDeviceGetCount/GetHandleByIndex/GetMemoryInfo/Shutdown`.
   **Stub defs must go at END of file or real defs override them.**
2. `patches/fix-pynvml-nvml-name.py` — `nvmlDeviceGetName` must return `str`,
   not `bytes` (FP8 kernel config table does `get_device_name().replace(" ","_")`
   and crashes on bytes).

## Python environment

```
python3.12 -m venv --without-pip /opt/vllm-p3/venv
# bootstrap pip from wheel:
python3 -m zipfile -e pip-*.whl /tmp/pip && PYTHONPATH=/tmp/pip python3 -m pip install ...
# then:
pip install torch-2.9.0-...aarch64.whl
pip install vllm-0.11.2-...aarch64.whl --no-deps
pip install compressed-tensors==0.11.0 frozendict <dry-run-derived-gap-set>
```

See `01-CROSS-COMPILE.md` for why `--no-deps` + manual gap fill.

## Process management rules (learned the hard way)

- NEVER `pkill -f <pattern>` for vllm — the pattern appears in your own ssh
  command line and kills your shell. Use pidfiles / `pgrep -x`.
- APIServer kill leaves `VLLM::EngineCore` zombie holding ~48G GPU memory.
  Kill it separately, then verify memory actually freed before restarting.
- Start serves with `setsid env ... nohup ... & echo $! > serve.pid` and WATCH
  the log to "startup complete" — don't fire-and-forget.

## Storage notes

- `/home` is read-only overlay on these images; put everything in the
  persistent data partition
- vLLM compile cache: `VLLM_CACHE_ROOT` must point at persistent storage
  (~1-2G, regenerated on demand)
