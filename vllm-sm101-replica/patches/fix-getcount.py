p = "${VENV:-/opt/vllm-p3/venv}/lib/python3.12/site-packages/vllm/third_party/pynvml.py"
lines = open(p).read().split("\n")
# 2407 stub 版(0-indexed 2406): _disabled_nvmlDeviceGetCount -> nvmlDeviceGetCount
assert lines[2406].strip() == "def _disabled_nvmlDeviceGetCount():", lines[2406]
lines[2406] = "def nvmlDeviceGetCount():"
# 2623 原版(0-indexed 2622): nvmlDeviceGetCount -> _orig_nvmlDeviceGetCount
assert lines[2622].strip() == "def nvmlDeviceGetCount():", lines[2622]
lines[2622] = "def _orig_nvmlDeviceGetCount():"
open(p, "w").write("\n".join(lines))
print("swapped: stub=active, orig=_orig_")
