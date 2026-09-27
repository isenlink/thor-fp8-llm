#!/usr/bin/env python3
"""给板上 pynvml.py 打 NVML stub 补丁: 无 libnvidia-ml 库时用 torch 探测兜底"""
p = "${VENV:-/opt/vllm-p3/venv}/lib/python3.12/site-packages/pynvml.py"
src = open(p).read()

if "_nvml_stub_mode" in src:
    print("already patched")
    raise SystemExit

stub = '''
# --- P3 stub patch (DRIVE OS has no libnvidia-ml) ---
_nvml_stub_mode = False
_nvml_fallback_count = 0

def nvmlInit():
    global _nvml_stub_mode, _nvml_fallback_count
    try:
        _LoadNvmlLibrary()
        nvmlLibraryInit()
    except Exception:
        import torch
        if not torch.cuda.is_available():
            raise
        _nvml_fallback_count = torch.cuda.device_count()
        _nvml_stub_mode = True

def nvmlDeviceGetCount():
    if _nvml_stub_mode:
        return _nvml_fallback_count
    return _nvmlCheckReturn(nvml_lib().nvmlDeviceGetCount())
# --- end stub ---
'''

anchor = "def nvmlInit():"
assert anchor in src, "anchor not found"
src = src.replace(anchor, stub + "\ndef _disabled_nvmlInit():", 1)
src = src.replace("def nvmlDeviceGetCount():", "def _disabled_nvmlDeviceGetCount():", 1)
open(p, "w").write(src)
print("patched pynvml OK")
