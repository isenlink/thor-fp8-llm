p = "${VENV:-/opt/vllm-p3/venv}/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_next.py"
src = open(p).read()
if "P3 visual filter" in src:
    print("already")
    raise SystemExit
anchor = "            # P3 scale filter: dynamic-activation quant scales not registered by vLLM\n            if name.endswith((\".k_scale\", \".v_scale\", \".q_scale\", \".input_scale\")):\n                continue\n"
patch = anchor + "\n            # P3 visual filter: vision tower weights not used in text-only model\n            if \".visual.\" in name:\n                continue\n"
assert anchor in src, "anchor not found"
src = src.replace(anchor, patch, 1)
open(p, "w").write(src)
print("visual filter patched")
