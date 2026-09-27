import json
import urllib.request
import base64


def ask(content, mx=60, img=None):
    c = []
    if img:
        b64 = base64.b64encode(open(img, 'rb').read()).decode()
        c.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}})
    c.append({"type": "text", "text": content})
    body = json.dumps({
        "model": "vl27",
        "messages": [{"role": "user", "content": c}],
        "max_tokens": mx,
        "temperature": 0.3,
    }).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        "http://localhost:8996/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"}), timeout=300))
    u = r["usage"]
    txt = r["choices"][0]["message"]["content"]
    return txt, u, r["choices"][0].get("finish_reason")


txt, u, fr = ask("1+1等于几?只回答数字", 16)
print("TEXT:", txt, f"[pt={u['prompt_tokens']} ct={u['completion_tokens']} fr={fr}]")

# put your own test image next to this script (or edit the filename below)
txt, u, fr = ask("用一两句话描述这张图片的内容", 120, "test_image_00001_.png")
print("IMG:", txt, f"[pt={u['prompt_tokens']} ct={u['completion_tokens']} fr={fr}]")

txt, u, fr = ask("图片中有没有文字?有则读出", 100, "test_image_00001_.png")
print("OCR:", txt, f"[fr={fr}]")
