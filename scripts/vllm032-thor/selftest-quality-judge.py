import json
from pathlib import Path
from judge_subset import QUESTIONS,judge
p=Path(__file__).parent;qs={q['id']:q for q in QUESTIONS}
fixtures=json.loads((p/'quality-judge-fixtures.json').read_text())
for x in fixtures:
 assert judge(qs[x['id']],x['content'])[0]==x['ok']
 assert judge(qs[x['id']],'')[0] is False
print('QUALITY_JUDGE_FIXTURE pass=32 fail=0')
