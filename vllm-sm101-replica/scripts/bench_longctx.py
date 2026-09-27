"""VL27 200K 长上下文测试: 100K+ prefill + 长文档问答, 与文本版基线(150K 245s)同口径对比."""
import json
import time
import urllib.request

URL = "http://localhost:8996/v1/chat/completions"

# 构造 ~100K token 的长 prompt（约 60K 汉字 ≈ 90-100K token）
para = ("森乐净水器采用五级过滤系统，包含PP棉、前置炭、RO反渗透膜、后置炭和UF超滤膜。"
        "反渗透膜的过滤精度为0.0001微米，能有效去除水中的重金属、细菌和病毒。"
        "滤芯寿命方面，PP棉建议3-6个月更换，前置炭6-12个月，RO膜24-36个月。")
long_doc = para * 1200  # ~ 96000 汉字

prompt = long_doc + "\n\n根据上文回答：RO反渗透膜的过滤精度是多少微米？滤芯中寿命最长的是哪一个？请只回答这两个问题的答案。"

body = json.dumps({
    "model": "vl27",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 150,
    "temperature": 0.3,
}).encode()

t0 = time.time()
req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=1800))
dt = time.time() - t0
u = r["usage"]
print(f"prompt_tokens={u['prompt_tokens']} completion_tokens={u['completion_tokens']}")
print(f"total={dt:.1f}s  (prefill+decode)")
ct = u["completion_tokens"]
print(f"decode speed ≈ {ct / max(dt - u['prompt_tokens'] / 2000, 1):.2f} tok/s (粗估)")
print("ANSWER:", r["choices"][0]["message"]["content"][-200:])
print("finish:", r["choices"][0]["finish_reason"])
