"""Independent FP32 GQA oracle: 1..4 sequences, strided Q/KV, shuffled pages."""
import time,json,torch
from thor_hybrid_attention import hybrid_attention,decode_route

torch.set_grad_enabled(False);torch.manual_seed(8137)
torch.set_float32_matmul_precision('highest')
base=[17,513,4097,8193]
cases=[([q]*b,base[:b]) for b in [1,2,3,4] for q in [1,5]]
cases += [([1,5,3,2],base),([5,5,5,5],[129,577,4097,0]),([5,5],[102400,102400]),([1]*4,[65536]*4),([5]*4,[65536]*4)]
start=time.perf_counter()
for qlens,klens in cases:
 batch=len(qlens);nq=sum(qlens);counts=[(n+431)//432 for n in klens]
 blocks=sum(counts);width=max(counts)
 kvc=torch.randn(blocks,4,432,512,device='cuda',dtype=torch.bfloat16).transpose(1,2)
 kc,vc=kvc.split(256,dim=-1)
 order=torch.randperm(blocks,device='cuda',dtype=torch.int32)
 table=torch.zeros(batch,width,device='cuda',dtype=torch.int32);off=0
 for i,n in enumerate(counts):table[i,:n]=order[off:off+n];off+=n
 qbuf=torch.randn(nq,32,256,device='cuda',dtype=torch.bfloat16);q=qbuf[:,:24,:]
 cu=[0]
 for n in qlens:cu.append(cu[-1]+n)
 p=dict(q=q,k=kc,v=vc,cu_seqlens_q=torch.tensor(cu,device='cuda',dtype=torch.int32),max_seqlen_q=max(qlens),max_seqlen_k=max(klens),seqused_k=torch.tensor(klens,device='cuda',dtype=torch.int32),block_table=table,causal=True,fa_version=2)
 assert decode_route(p)
 out=hybrid_attention(**p)
 expected=[]
 for i,(n,k) in enumerate(zip(qlens,klens)):
  if k==0:
   expected.append(torch.zeros(n,24,256,device='cuda',dtype=q.dtype));continue
  keys=kc[table[i,:counts[i]].long()].reshape(-1,4,256)[:k].float()
  vals=vc[table[i,:counts[i]].long()].reshape(-1,4,256)[:k].float()
  queries=q[cu[i]:cu[i+1]].float().reshape(n,4,6,256)
  scores=torch.einsum('qhgd,khd->hgqk',queries,keys)/16
  mask=torch.arange(k,device='cuda')[None,:] <= torch.arange(n,device='cuda')[:,None]+k-n
  probs=scores.masked_fill(~mask,float('-inf')).softmax(dim=-1)
  ref=torch.einsum('hgqk,khd->qhgd',probs,vals).reshape(n,24,256).to(q.dtype)
  expected.append(ref)
 expected=torch.cat(expected)
 torch.testing.assert_close(out,expected,atol=1e-2,rtol=1e-2)
 torch.cuda.synchronize()
 print('BATCH_ATTN_CASE_PASS',qlens,klens,'maxerr',float((out.float()-expected.float()).abs().max()),flush=True)
print(json.dumps({'batch_cases_passed':len(cases),'seconds':time.perf_counter()-start,'peak_cuda_gib':torch.cuda.max_memory_allocated()/2**30}),flush=True)
