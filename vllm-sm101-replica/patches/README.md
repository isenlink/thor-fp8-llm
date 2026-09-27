# patches/ — apply order and purpose

Run each ON THE BOARD against the installed tree
`$V/lib/python3.12/site-packages/vllm/` (V = venv root).
Order matters where noted. Each script self-verifies after applying.

| # | file | purpose | notes |
|---|------|---------|-------|
| 1 | fix-pynvml-nvml-name.py | NVML stub returns "Thor" string | board has no libnvidia-ml; platform probe + FP8 kernel lookup need it |
| 2 | patch-pynvml-stub.py | same stub inside vllm/third_party copy | |
| 3 | patch-graceful-shutdown.py | clean engine shutdown | |
| 4 | fix-lmhead-dequant.py | LM head loading fix | |
| 5 | fix-mlp-prefix.py | MLP weight name prefix mapping | |
| 6 | fix-moe-dense-indent.py | MoE path for qwen3_next | |
| 7 | fix-nvfp4-scale-filter.py | FP8/NVFP4 scale filtering | |
| 8 | fix-eagle-registry.py | MTP draft registration | draft route table |
| 9 | fix-getcount.py | kv cache get_count fix | |
| 10 | fix-visual-weight-filter.py | visual tower weight mapping | multimodal only |
| 11 | fix-cudagraph-concurrency-pad.py | CUDA graph concurrency guard | spec=2+CG+conc>2 crashes; production uses spec=1 |
| 12 | patch-nvml-full-stub.py | extended NVML stub | |
| 13 | extract-visual-tower.py | extracts visual tower weights | one-time, multimodal only |

Text-tower model class (replaces site-packages copy):
`../model-classes/qwen3_next_patched.py`
Multimodal graft class: `../model-classes/qwen38_vl.py` (+ its MTP patch).

Sanity check before publishing: run the maintainer's outbound-content
scan (the wordlist lives outside this repo); it must return nothing.
