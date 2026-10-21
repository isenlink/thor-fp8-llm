"""100K end-to-end retrieval/rule/evidence-boundary acceptance; server defaults."""
import json, sys, urllib.request, os, uuid
from pathlib import Path

def answer_json(text):
    text = text.rsplit('</think>', 1)[-1].strip()
    if text.startswith('```json') and text.endswith('```'):
        text = text[7:-3].strip()
    return json.loads(text)

expected = {'任务号': 'THOR05-QUAL-20261021', '车辆号': 'TEST-CAR-001',
            '校验码': 'ZEBRA-6842', '处置': '复核', '精度结论': '证据不足'}
fixture = json.dumps(expected, ensure_ascii=False)
for text in [fixture, '先思考</think>\n' + fixture, '```json\n' + fixture + '\n```']:
    assert answer_json(text) == expected
try:
    answer_json('not-json')
    raise AssertionError('invalid JSON accepted')
except json.JSONDecodeError:
    pass
print('PARSER_FIXTURE pass=4 fail=0', flush=True)
if '--selftest' in sys.argv:
    sys.exit(0)

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def request(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(os.environ['THOR_BENCH_BASE'].rstrip('/') + path, data=data,
                                 headers={'Content-Type': 'application/json'})
    with opener.open(req, timeout=600) as response:
        return json.loads(response.read())
model = request('/v1/models')['data'][0]['id']
def tokenize(payload):
    return request('/tokenize', {'model': model, **payload})['tokens']
end = ('尾部权威记录：校验码为ZEBRA-6842。只依据权威记录，不把填充日志当成实际测量。'
       '请输出一个JSON对象，恰好包含任务号、车辆号、校验码、处置、精度结论这5个键。'
       '处置必须严格按首部规则计算；这些日志能否确认传感器测量精度，'
       '精度结论只能填已确认或证据不足。不要输出其他解释。')
template = tokenize({'messages': [{'role': 'user', 'content': end}]})
medium = tokenize({'messages': [{'role': 'user', 'content': end}],
                   'chat_template_kwargs': {'enable_thinking': True, 'reasoning_effort': 'medium'}})
assert template == medium
split = len(tokenize({'prompt': '<|im_start|>user\n', 'add_special_tokens': False}))
prefix = tokenize({'prompt': '首部权威记录：任务号为THOR05-QUAL-20261021。规则：只有温度大于等于45摄氏度并且电压小于360伏才输出报警，否则输出复核。\n', 'add_special_tokens': False})
mid = tokenize({'prompt': '\n中部权威记录：车辆号为TEST-CAR-001，温度47摄氏度，电压372伏。未提供校准基准、真值或精度实验结果。\n', 'add_special_tokens': False})
records = ''.join(f'填充日志{i}：车速{35+i%63}，电量{90-i%60}，温度{20+i%19}；本行是占位示例，不是权威记录。\n' for i in range(5000))
filler = tokenize({'prompt': records, 'add_special_tokens': False})
needed = 102400 - len(template) - len(prefix) - len(mid)
assert len(filler) >= needed > 0
half = needed // 2
prompt = template[:split] + prefix + filler[:half] + mid + filler[half:needed] + template[split:]
assert len(prompt) == 102400
Path('long-quality-expected.json').write_text(json.dumps(expected, ensure_ascii=False, indent=2))
print('LONG_QUALITY_START input=102400 fields=5 default_thinking=medium', flush=True)
result = request('/v1/completions', {'model': model, 'prompt': prompt, 'max_tokens': 2048, 'temperature':0.3, 'cache_salt':str(uuid.uuid4())})
Path('long-quality-response.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
assert result['usage']['prompt_tokens'] == 102400
choice = result['choices'][0]
assert choice['finish_reason'] == 'stop', choice['finish_reason']
actual = answer_json(choice['text'])
assert actual == expected, actual
Path('long-quality-verdict.json').write_text(json.dumps({'pass': True, 'fields': 5, 'input_tokens': 102400, 'usage': result['usage']}, indent=2))
print('LONG_QUALITY_PASS fields=5 input=102400', result['usage'], flush=True)
