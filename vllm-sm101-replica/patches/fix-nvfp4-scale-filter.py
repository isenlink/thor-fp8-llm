p = "${VENV:-/opt/vllm-p3/venv}/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_next.py"
src = open(p).read()
if "P3 scale filter" in src:
    print("already patched")
    raise SystemExit
# 在 Qwen3NextModel.load_weights 的 mtp 过滤后加 scale 过滤
anchor = '''            if name.startswith("mtp."):
                continue
'''
patch = '''            if name.startswith("mtp."):
                continue

            # P3 scale filter: dynamic-activation quant scales not registered by vLLM
            if name.endswith((".k_scale", ".v_scale", ".q_scale", ".input_scale")):
                continue
'''
assert anchor in src, "anchor not found"
src = src.replace(anchor, patch, 1)
open(p, "w").write(src)
print("scale filter patched")
