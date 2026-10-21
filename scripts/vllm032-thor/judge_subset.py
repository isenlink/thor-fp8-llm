import json,re,unicodedata
from pathlib import Path
QUESTIONS=json.loads(Path(__file__).with_name("questions16.json").read_text())

def norm_text(s):
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("（", "(").replace("）", ")").replace("：", ":").replace("，", ",").replace("。", ".")
    s = re.sub(r"\s+", "", s)
    return s

def strip_md(s):
    s = re.sub(r"```[a-zA-Z]*\s*", "", s or "")
    s = s.replace("```", "").strip()
    return s

def judge(q, raw):
    """返回 (correct: bool, note: str)"""
    cat = q["cat"]; ans = q["answer"]; text = strip_md(raw)
    if not text.strip():
        return False, "EMPTY"

    if cat == "算术":
        nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
        got = nums[-1] if nums else None
        return (got is not None and float(got) == float(ans)), f"got={got} want={ans}"

    if cat == "JSON":
        m = re.search(r"\{.*\}", text, re.S)
        if not m: return False, "NO_JSON_OBJECT"
        try:
            d = json.loads(m.group(0))
        except Exception as e:
            return False, f"JSON_PARSE_FAIL:{e}"
        if not isinstance(d, dict): return False, "NOT_DICT"
        probs = []
        for k, want in ans.items():
            if k not in d: probs.append(f"missing:{k}"); continue
            got = d[k]
            if k == "score":
                try:
                    if abs(float(got) - float(want)) > 0.05: probs.append(f"score:{got}!={want}")
                except Exception: probs.append(f"score_type:{got}")
            elif k == "tags":
                if got != want: probs.append(f"tags:{got}!={want}")
            else:
                if got != want: probs.append(f"{k}:{got}!={want}")
        return (not probs), ";".join(probs) or "OK"

    if cat == "列表计数":
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        items = [re.sub(r"^\s*\d+\s*[\.、\)）]\s*", "", l) for l in lines]
        items = [i for i in items if i and not re.match(r"^(好的|以下是|当然|注意)", i)]
        k = ans["count"]
        return len(items) == k, f"items={len(items)} want={k}"

    if cat == "标识符转录":
        want = ans
        cand = text.strip().strip("`'\" ")
        m = re.search(r"[A-Za-z0-9_\-]{" + str(max(4, len(want) - 3)) + r",}", cand)
        got = m.group(0) if m else cand
        return got == want, f"got={got[:40]} want={want}"

    if cat == "字母计数":
        nums = re.findall(r"\d+", text)
        got = nums[-1] if nums else None
        return (got is not None and int(got) == int(ans)), f"got={got} want={ans}"

    if cat == "数字排序":
        nums = [int(x) for x in re.findall(r"\d+", text)]
        want = ans
        if nums == want: return True, "OK"
        if len(nums) >= len(want) and nums[-len(want):] == want: return True, "OK(tail)"
        return False, f"got={nums[:14]} want={want[:14]}"

    if cat == "数字串转录":
        want = ans
        runs = re.findall(r"\d+", text)
        if not runs: return False, "NO_DIGITS"
        best = min(runs, key=lambda r: abs(len(r) - len(want)))
        return best == want, f"got={best[:26]} want={want}"

    if cat == "中英混排转录":
        want = norm_text(ans)
        got = norm_text(text)
        return got == want, ("OK" if got == want else f"DIFF len={len(got)}vs{len(want)}")

    return False, "UNKNOWN_CAT"
