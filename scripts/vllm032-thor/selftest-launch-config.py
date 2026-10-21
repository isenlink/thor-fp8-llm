import pathlib,subprocess,json,os
src=(pathlib.Path(__file__).parent/'run-mixed-optimized.sh').read_text().split('ROOT=',1)[0]
count=0
for mtp in [4,5,6,7]:
 for n in [1,2,3,4]:
  env={**os.environ,'THOR_MTP':str(mtp),'THOR_MAX_NUM_SEQS':str(n)}
  d=subprocess.check_output(['bash','-c',src+'\necho "$THOR_GRAPH_CONFIG"; echo "$THOR_SPEC_CONFIG"'],env=env,text=True).splitlines()
  g,s=map(json.loads,d);v=mtp+1
  assert g['cudagraph_capture_sizes']==([1,v] if n==1 else [1,v,2*v] if n==2 else [1,v,2*v,n*v])
  assert s=={'method':'mtp','num_speculative_tokens':mtp}
  count+=1
for m,n in [('3','2'),('8','2'),('4','0'),('4','5')]:
 r=subprocess.run(['bash','-c',src],env={**os.environ,'THOR_MTP':m,'THOR_MAX_NUM_SEQS':n},capture_output=True)
 assert r.returncode==2
 count+=1
print(f'LAUNCH_CONFIG_FIXTURE pass={count} fail=0')
