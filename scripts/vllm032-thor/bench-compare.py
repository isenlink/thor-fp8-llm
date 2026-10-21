#!/usr/bin/env python3
"""Same workload, cold-prefix, streaming service comparison; no service mutations."""
import argparse,concurrent.futures,json,pathlib,time,urllib.request,uuid

def measure(chunks,start):
 first=last=None;first_tokens=None;usage=None;text=''
 for now,d in chunks:
  if d.get('usage'):usage=d['usage']
  for ch in d.get('choices',[]):
   piece=ch.get('text','')
   if piece:
    if first is None:
     first=now;first_tokens=usage['completion_tokens'] if usage else 1
     print('FIRST_TOKEN ttft_s='+str(first-start),flush=True)
    last=now;text+=piece
 if not usage or first is None or last<=first:raise ValueError('incomplete streamed measurement')
 n=usage['completion_tokens']
 return dict(usage=usage,input_tokens=usage['prompt_tokens'],output_tokens=n,first_chunk_tokens=first_tokens,ttft_s=first-start,decode_tokens_per_s=(n-first_tokens)/(last-first),prefill_effective_tokens_per_s=usage['prompt_tokens']/(first-start),output=text)

def main():
 p=argparse.ArgumentParser();p.add_argument('--base');p.add_argument('--model');p.add_argument('--input',type=int,default=8192);p.add_argument('--concurrency',type=int,choices=[1,2,4],default=1);p.add_argument('--out');p.add_argument('--selftest',action='store_true');a=p.parse_args()
 if a.selftest:
  d=measure([(12,{'usage':{'prompt_tokens':100,'completion_tokens':2},'choices':[{'text':'a'}]}),(14,{'usage':{'prompt_tokens':100,'completion_tokens':6},'choices':[{'text':'b'}]})],10)
  assert d['ttft_s']==2 and d['decode_tokens_per_s']==2 and d['prefill_effective_tokens_per_s']==50
  try:measure([],0)
  except ValueError:pass
  else:raise AssertionError('missing usage accepted')
  print('SELFTEST PASS fail=0');return
 if not a.base or not a.model or not a.out:p.error('--base --model --out required')
 out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
 op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 def request(path,payload=None):
  req=urllib.request.Request(a.base.rstrip('/')+path,data=None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
  return op.open(req,timeout=600)
 def tok(payload):
  with request('/tokenize',{'model':a.model,**payload}) as r:return json.load(r)['tokens']
 # Fixed medium template with exact input-token length for both versions.
 messages=[{'role':'user','content':'以上为车辆测试记录。请先简短分析这些记录能验证哪些功能，再列出100条需要检查的车辆项目，每行一项并说明检查目的。'}]
 template=tok({'messages':messages,'chat_template_kwargs':{'enable_thinking':True,'reasoning_effort':'medium'}})
 split=len(tok({'prompt':'<|im_start|>user\n','add_special_tokens':False}))
 records=''.join(f'记录{i}：车速{35+i%63}公里每小时，电量{90-i%60}%，电池温度{20+i%19}摄氏度，运行状态正常，需定期核对传感器与告警日志。\n' for i in range(4000))
 filler=tok({'prompt':records,'add_special_tokens':False});needed=a.input-len(template)
 assert 0<needed<=len(filler)
 tokens=template[:split]+filler[:needed]+template[split:];assert len(tokens)==a.input
 (out/'prompt-tokens.json').write_text(json.dumps(tokens))
 def run(i):
  payload={'model':a.model,'prompt':tokens,'max_tokens':512,'ignore_eos':True,'temperature':0.3,'seed':100+i,'cache_salt':str(uuid.uuid4()),'stream':True,'stream_options':{'include_usage':True,'continuous_usage_stats':True}}
  start=time.perf_counter()
  def chunks():
   with request('/v1/completions',payload) as r:
    for line in r:
     if not line.startswith(b'data: '):continue
     raw=line[6:].strip()
     if raw==b'[DONE]':break
     yield time.perf_counter(),json.loads(raw)
  result=measure(chunks(),start);result['total_s']=time.perf_counter()-start
  assert result['input_tokens']==a.input and result['output_tokens']==512,result
  (out/f'request-{i}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));return result
 with request('/metrics') as r:(out/'metrics-before.txt').write_bytes(r.read())
 start=time.perf_counter()
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:results=list(pool.map(run,range(a.concurrency)))
 wall=time.perf_counter()-start
 with request('/metrics') as r:(out/'metrics-after.txt').write_bytes(r.read())
 summary={'base':a.base,'model':a.model,'concurrency_submitted':a.concurrency,'input_each':a.input,'output_each':512,'wall_s':wall,'aggregate_output_tokens_per_wall_s':512*a.concurrency/wall,'results':[{k:v for k,v in r.items() if k!='output'} for r in results],'note':'cold unique cache_salt; per-request decode excludes prefill; aggregate includes prefill; verify running/queued separately'}
 (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
