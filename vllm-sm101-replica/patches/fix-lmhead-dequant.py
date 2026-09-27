p = "${VENV:-/opt/vllm-p3/venv}/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_next.py"
src = open(p).read()
old = """        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )"""
new = """        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )"""
if old in src:
    src = src.replace(old, new, 1)
    open(p, "w").write(src)
    print("lm_head quant_config patch applied")
elif "quant_config=self.quant_config," in src and "lm_head" in src:
    print("already patched")
else:
    raise SystemExit("pattern not found")
