# Author: AI assistant
# Deterministic paired prompts; report server timing separately from wall time.
import json
import sys
import time
import urllib.request

target = int(sys.argv[1])
trial = sys.argv[2]
urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
def api(path, payload):
    req = urllib.request.Request("http://127.0.0.1:8080" + path, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as response:
        return json.load(response)
def count(text):
    return len(api("/tokenize", {"content": text})["tokens"])

intro = "Below is a long document. Read it carefully. At the end, list 120 distinct observations, one per line, numbered.\n\nDOCUMENT:\n"
unit = "".join(t*2 for t in (
    "The quick brown fox jumps over the lazy dog near the riverbank while birds sing loudly",
    "In 1987 engineers tested a novel cooling system for a supercomputer in Osaka with mixed results",
    "She wrote letters every evening describing the old harbor, the fog, and the sound of bells"))
tail = "\n\nEND. Now list 120 observations:"
prefix = intro + "UNIQUE assistant-paired-" + trial + " "
density = (count(prefix+unit*100+tail)-count(prefix+tail))/100
units = max(1,int((target-count(prefix+tail))/density))
prompt = prefix+unit*units+tail
actual = count(prompt)
print(json.dumps(dict(author="AI assistant",stage="start", trial=trial, requested=target, prompt_tokens=actual)),flush=True)
start = time.monotonic()
result = api("/completion",dict(prompt=prompt,n_predict=384,temperature=0,seed=123,stream=False,cache_prompt=False,ignore_eos=True))
print(json.dumps(dict(stage="result",trial=trial,wall_seconds=time.monotonic()-start,
    **{k:result.get(k) for k in ("content","timings","tokens_predicted","tokens_evaluated","truncated","stop_type")}),ensure_ascii=False),flush=True)
