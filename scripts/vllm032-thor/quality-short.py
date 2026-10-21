#!/usr/bin/env python3
"""Balanced 16-question subset of existing 150-question suite; same judge."""
import argparse,concurrent.futures,importlib.util,json,pathlib,urllib.request,time
p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--out',required=True);p.add_argument('--concurrency',type=int,choices=[1,2,4],default=2);a=p.parse_args()
s=importlib.util.spec_from_file_location('original_judge',str(pathlib.Path(__file__).parent/'judge_subset.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
counts={};qs=[]
for q in m.QUESTIONS:
 if counts.get(q['cat'],0)<2:qs.append(q);counts[q['cat']]=counts.get(q['cat'],0)+1
assert len(qs)==16
out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
def run(q):
 op=urllib.request.build_opener(urllib.request.ProxyHandler({}));payload={'model':'qwen3.8-27b','messages':[{'role':'user','content':q['prompt']}],'temperature':0,'seed':123,'max_tokens':2048,'chat_template_kwargs':{'enable_thinking':True,'reasoning_effort':'medium'}}
 req=urllib.request.Request(a.base+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
 t=time.monotonic()
 with op.open(req,timeout=300) as r:d=json.load(r)
 choice=d['choices'][0];text=choice['message'].get('content') or '';ok,note=m.judge(q,text)
 result={'id':q['id'],'cat':q['cat'],'ok':ok,'note':note,'content':text,'usage':d.get('usage'),'finish_reason':choice.get('finish_reason'),'latency_s':time.monotonic()-t}
 (out/(str(q['id'])+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2));print('QUALITY',q['id'],ok,flush=True);return result
with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:results=list(pool.map(run,qs))
summary={'correct':sum(x['ok'] for x in results),'total':len(results),'empty':sum(not x['content'].strip() for x in results),'concurrency':a.concurrency,'ids':[q['id'] for q in qs],'note':'balanced 16/150 subset; not the full historical150 or long-context test'}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
assert summary['empty']==0 and all(x['finish_reason']=='stop' for x in results),'unfinished/empty quality answers'

assert summary['correct']==summary['total'], 'quality subset contains incorrect answers'
