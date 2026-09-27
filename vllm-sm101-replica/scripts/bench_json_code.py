"""target-board vLLM 短 JSON/代码口径 (对齐 llama.cpp bench: temp 0, 全输出 token 计速)"""
import json
import time
import urllib.request

URL = "http://localhost:8996/v1/chat/completions"


def gen(prompt, mx):
    body = json.dumps({
        "model": "vl27",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": mx,
        "temperature": 0,
    }).encode()
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}), timeout=600))
    dt = time.time() - t0
    ct = r["usage"]["completion_tokens"]
    fr = r["choices"][0]["finish_reason"]
    print(f"{ct}tok/{dt:.1f}s = {ct/dt:.2f} tok/s fr={fr}", flush=True)


for i in range(3):
    gen("生成5个用户的JSON数组,每人含name/age/email字段,直接给JSON:", 400)
for i in range(3):
    gen("用Python写一个二分查找函数,含注释和测试用例:", 500)
