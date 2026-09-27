for p in ["${VENV:-/opt/vllm-p3/venv}/lib/python3.12/site-packages/vllm/third_party/pynvml.py",
          "${VENV:-/opt/vllm-p3/venv}/lib/python3.12/site-packages/pynvml.py"]:
    src = open(p).read()
    if "def nvmlShutdown():" in src and "def _disabled_nvmlShutdown():" not in src:
        src = src.replace("def nvmlShutdown():",
"""def nvmlShutdown():
    if _nvml_stub_mode:
        return None
    return _orig_nvmlShutdown_impl()

def _orig_nvmlShutdown():""", 1)
        open(p, "w").write(src)
        print("patched shutdown in", p)
    else:
        print("skip", p)
