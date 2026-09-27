p = "${VENV:-/opt/vllm-p3/venv}/lib/python3.12/site-packages/vllm/model_executor/models/registry.py"
src = open(p).read()
old = '"Qwen3NextMTP": ("qwen3_next_mtp", "Qwen3NextMTP"),'
new = old + '\n    "EagleQwen3NextMTP": ("qwen3_next_mtp", "Qwen3NextMTP"),'
if "EagleQwen3NextMTP" in src:
    print("already patched")
else:
    assert old in src, "anchor not found"
    src = src.replace(old, new, 1)
    open(p, "w").write(src)
    print("registry patched")
