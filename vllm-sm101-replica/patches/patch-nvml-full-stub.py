p = "${VENV:-/opt/vllm-p3/venv}/lib/python3.12/site-packages/vllm/third_party/pynvml.py"
src = open(p).read()
if "P3_FULL_STUB" in src:
    print("already"); raise SystemExit
full = 
