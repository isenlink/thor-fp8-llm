#!/usr/bin/env python3
"""复验：DNS + DeepSeek 可达性（供 restore-hermes.sh 调用）"""
import json, socket, sys, time, urllib.request
try:
    ip = socket.gethostbyname("api.deepseek.com")
except Exception as e:
    print(f"否（DNS 失败 {type(e).__name__}）"); sys.exit(0)
env = {}
try:
    for line in open("/brand_data/hermes/home/.env"):
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1); env[k.strip()] = v.strip()
except Exception:
    pass
key = env.get("DEEPSEEK_API_KEY", "")
if not key:
    print(f"否（.env 缺 DEEPSEEK_API_KEY；DNS 正常 {ip}）"); sys.exit(0)
body = json.dumps({"model": "deepseek-flash", "messages": [{"role": "user", "content": "ok"}],
                   "max_tokens": 3}).encode()
req = urllib.request.Request("https://api.deepseek.com/chat/completions", data=body,
                             headers={"Content-Type": "application/json",
                                      "Authorization": "Bearer " + key})
t = time.time()
try:
    json.load(urllib.request.urlopen(req, timeout=25))
    print(f"是（{time.time()-t:.2f}s，{ip}）")
except Exception as e:
    print(f"否（{type(e).__name__}，DNS {ip}）")
