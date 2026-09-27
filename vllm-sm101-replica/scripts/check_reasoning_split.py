"""验证 reasoning-parser qwen3 是否把 think 内容分离到 reasoning_content."""
import json
import urllib.request

body = json.dumps({
    "model": "vl27",
    "messages": [{"role": "user", "content": "1+1等于几?只回答数字"}],
    "max_tokens": 100,
    "temperature": 0.3,
}).encode()
r = json.load(urllib.request.urlopen(urllib.request.Request(
    "http://localhost:8996/v1/chat/completions", data=body,
    headers={"Content-Type": "application/json"}), timeout=300))
msg = r["choices"][0]["message"]
print("content =", repr(msg.get("content", ""))[:120])
print("reasoning =", repr(msg.get("reasoning_content", ""))[:120])
print("has_think_in_content =", "<think>" in (msg.get("content") or ""))
