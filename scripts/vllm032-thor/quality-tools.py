"""Check the HTTP tool-call contract; never execute a tool."""
import argparse,copy,json,pathlib,urllib.request

def check(d):
 try:
  c=d['choices'][0];calls=c['message'].get('tool_calls') or []
  return c['finish_reason']=='tool_calls' and len(calls)==1 and calls[0]['function']['name']=='get_vehicle_status' and json.loads(calls[0]['function']['arguments'])=={'car_id':'TEST-CAR-001'}
 except (KeyError,IndexError,TypeError,json.JSONDecodeError):return False

def main():
 p=argparse.ArgumentParser();p.add_argument('--base');p.add_argument('--out');p.add_argument('--selftest',action='store_true');a=p.parse_args()
 if a.selftest:
  good={'choices':[{'finish_reason':'tool_calls','message':{'tool_calls':[{'function':{'name':'get_vehicle_status','arguments':'{"car_id":"TEST-CAR-001"}'}}]}}]}
  assert check(good)
  bad=copy.deepcopy(good);bad['choices'][0]['message']['tool_calls']=[];assert not check(bad)
  bad=copy.deepcopy(good);bad['choices'][0]['message']['tool_calls'][0]['function']['arguments']='{"car_id":"OTHER"}';assert not check(bad)
  bad=copy.deepcopy(good);bad['choices'][0]['message']['tool_calls'][0]['function']['arguments']='not-json';assert not check(bad)
  print('TOOL_JUDGE_FIXTURE pass=4 fail=0');return
 if not a.base or not a.out:p.error('--base and --out required')
 op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 body={'model':'qwen3.8-27b','messages':[{'role':'user','content':'请调用 get_vehicle_status 查询车辆 TEST-CAR-001 的状态；car_id 必须原样填写 TEST-CAR-001。只调用工具，不要虚构状态。'}],'tools':[{'type':'function','function':{'name':'get_vehicle_status','description':'读取指定测试车辆状态','parameters':{'type':'object','properties':{'car_id':{'type':'string','enum':['TEST-CAR-001']}},'required':['car_id'],'additionalProperties':False}}}],'tool_choice':'auto','temperature':0,'seed':123,'max_tokens':1024}
 req=urllib.request.Request(a.base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with op.open(req,timeout=120) as r:d=json.load(r)
 out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/'response.json').write_text(json.dumps(d,ensure_ascii=False,indent=2))
 ok=check(d);(out/'verdict.json').write_text(json.dumps({'pass':ok,'expected_tool':'get_vehicle_status','expected_arguments':{'car_id':'TEST-CAR-001'}},indent=2));assert ok,'tool call contract failed'
 print('VEHICLE_TOOL_CALL_PASS no_tool_executed=1')
if __name__=='__main__':main()
