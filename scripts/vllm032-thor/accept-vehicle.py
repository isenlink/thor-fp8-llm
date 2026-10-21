"""Check the live service's default sampling on three explicit vehicle rules."""
import json, os, sys
import urllib.request
from pathlib import Path

EXPECTED = [{'id': 'A', 'overheat': True, 'low_battery': True, 'moving': False},
            {'id': 'B', 'overheat': False, 'low_battery': False, 'moving': True},
            {'id': 'C', 'overheat': True, 'low_battery': False, 'moving': False}]

def check(content):
    content = content.strip()
    if content.startswith('```'):
        content = content.split('\n', 1)[1].rsplit('```', 1)[0]
    return json.loads(content) == EXPECTED

if __name__ == '__main__':
    assert check(json.dumps(EXPECTED))
    bad = json.loads(json.dumps(EXPECTED)); bad[0]['overheat'] = False
    assert not check(json.dumps(bad))
    if '--selftest' in sys.argv:
        print('VEHICLE_FIXTURE pass=2 fail=0'); sys.exit(0)
    base = os.environ.get('THOR_BENCH_BASE', 'http://127.0.0.1:8081')
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
    models = json.load(urllib.request.urlopen(base + '/v1/models'))
    prompt = ('按以下明确定义分类，不需额外建议：温度>=60摄氏度时overheat=true，'
              'SOC<=20时low_battery=true，车速>0时moving=true。'
              'A：温度65，SOC20，车速0；B：温度45，SOC80，车速35；'
              'C：温度60，SOC21，车速0。'
              '最终回答仅JSON数组，按A/B/C顺序，每项包含id/overheat/low_battery/moving。')
    payload = {'model': models['data'][0]['id'],
               'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 512}
    req = urllib.request.Request(base + '/v1/chat/completions',
                                 data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    result = json.load(urllib.request.urlopen(req, timeout=120))
    Path('accept-vehicle-defaults.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    assert check(result['choices'][0]['message'].get('content') or ''), result
    print('VEHICLE_DEFAULTS_PASS cases=3 fixture_pass=2 fixture_fail=0', result['usage'])
