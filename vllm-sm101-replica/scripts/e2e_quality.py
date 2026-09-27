"""端到端质量复测: reasoning 分离后, 大 max_tokens 下答案完整性 + 速度"""
import json
import time
import urllib.request

URL = "http://localhost:8996/v1/chat/completions"


def ask(prompt, mx):
    body = json.dumps({
        "model": "vl27",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": mx,
        "temperature": 0.3,
    }).encode()
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}), timeout=600))
    dt = time.time() - t0
    m = r["choices"][0]["message"]
    ct = r["usage"]["completion_tokens"]
    fr = r["choices"][0]["finish_reason"]
    print(f"[{ct}tok/{dt:.1f}s={ct/dt:.1f}tok/s fr={fr}]", flush=True)
    print("  答案:", repr((m.get("content") or "").strip()[:120]), flush=True)


ask("9.9和9.11哪个大?只回答结论", 600)
ask("1+1等于几?只回答数字", 400)
ask("用一句话介绍大海", 500)
