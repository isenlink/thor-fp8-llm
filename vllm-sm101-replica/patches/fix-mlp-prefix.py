p = "${VENV:-/opt/vllm-p3/venv}/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_next.py"
src = open(p).read()
old = """        else:
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
            )"""
new = """        else:
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )"""
if old in src:
    src = src.replace(old, new, 1)
    open(p, "w").write(src)
    print("mlp prefix patched")
elif 'prefix=f"{prefix}.mlp"' in src:
    print("already")
else:
    raise SystemExit("pattern not found")
