"""VL27 200K+MTP 生成速度测试: 中短文本生成, 测 MTP 采纳率与 tok/s, 与文本版基线对比."""
import json
import time
import urllib.request

URL = "http://localhost:8996/v1/chat/completions"


def gen(prompt, max_tokens=200, temperature=0.3):
    body = json.dumps({
        "model": "vl27",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}), timeout=600))
    dt = time.time() - t0
    u = r["usage"]
    ct = u["completion_tokens"]
    msg = r["choices"][0]["message"]
    txt = (msg.get("content") or "") or "[思考未完] " + (msg.get("reasoning_content") or "")[-60:]
    return {
        "ct": ct, "sec": round(dt, 1),
        "tok_s": round(ct / dt, 2),
        "text": txt[-80:].replace("\n", " "),
        "fr": r["choices"][0]["finish_reason"],
    }


cases = [
    ("写一段100字左右的关于大海的短文", 200),
    ("请详细解释什么是光合作用，包括光反应和暗反应", 300),
    ("9.9和9.11哪个大？请一步步推理", 200),
]

for prompt, mx in cases:
    r = gen(prompt, mx)
    print(f"[{r['ct']}tok/{r['sec']}s = {r['tok_s']} tok/s fr={r['fr']}] {r['text']}")
