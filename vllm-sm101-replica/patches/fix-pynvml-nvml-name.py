#!/usr/bin/env python3
"""修复 stub DeviceGetCount 顺序问题: 移到文件末尾"""
p = "${VENV:-/opt/vllm-p3/venv}/lib/python3.12/site-packages/pynvml.py"
src = open(p).read()
stub_def = "def nvmlDeviceGetCount():\n    if _nvml_stub_mode:\n        return _nvml_fallback_count\n    return _nvmlCheckReturn(nvml_lib().nvmlDeviceGetCount())\n"
assert stub_def in src, "stub_def not found"
src = src.replace(stub_def, "", 1)
src += "\n\n# P3 stub override (must be last)\n" + stub_def
open(p, "w").write(src)
print("fixed: stub DeviceGetCount moved to end")
