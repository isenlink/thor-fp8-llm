"""Thor native FA2 against explicit FP32 attention, GQA24/4 head256."""
import sys,json,time
import torch
from thor_hybrid_attention import hybrid_attention as flash_attn_varlen_func

torch.set_grad_enabled(False)
torch.manual_seed(7342)
t=torch.ones((16,16),dtype=torch.bfloat16,device='cuda')
assert (t@t).float().sum().item()==4096
print('CONTROL_PASS',flush=True)
cases=[(4,128,True)] if '--canary' in sys.argv else [(1,128,False),(4,128,False),(5,128,False),(6,128,False),(128,128,False),(1,8192,False),(4,1024,True),(1,8192,True),(6,102400,True),(7,102400,True),(8,102400,True),(5,102400,True),(1,102400,True)]
start=time.perf_counter()
for nq,nk,paged in cases:
    q=torch.randn(nq,24,256,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(nk,4,256,device='cuda',dtype=torch.bfloat16)
    v=torch.randn_like(k)
    cuq=torch.tensor([0,nq],device='cuda',dtype=torch.int32)
    cuk=torch.tensor([0,nk],device='cuda',dtype=torch.int32)
    if paged:
        block_size=448 if '--block448' in sys.argv else 432
        blocks=(nk+block_size-1)//block_size
        kvc=torch.randn(blocks,4,block_size,512,device='cuda',dtype=q.dtype).transpose(1,2)
        kc,vc=kvc.split(256,dim=-1)
        k=kc.flatten(0,1)[:nk];v=vc.flatten(0,1)[:nk]
        table=torch.arange(blocks,device='cuda',dtype=torch.int32)[None,:]
        used=torch.tensor([nk],device='cuda',dtype=torch.int32)
        out=flash_attn_varlen_func(q=q,k=kc,v=vc,cu_seqlens_q=cuq,max_seqlen_q=nq,max_seqlen_k=nk,seqused_k=used,block_table=table,causal=True,fa_version=2,q_descale=torch.ones((1,4),device='cuda'),k_descale=torch.ones((1,4),device='cuda'),v_descale=torch.ones((1,4),device='cuda'))
    else:
        out=flash_attn_varlen_func(q=q,k=k,v=v,cu_seqlens_q=cuq,cu_seqlens_k=cuk,max_seqlen_q=nq,max_seqlen_k=nk,causal=True,fa_version=2,q_descale=torch.ones((1,4),device='cuda'),k_descale=torch.ones((1,4),device='cuda'),v_descale=torch.ones((1,4),device='cuda'))
    torch.cuda.synchronize()
    qh=q.transpose(0,1).float();kh=k.repeat_interleave(6,dim=1).transpose(0,1).float();vh=v.repeat_interleave(6,dim=1).transpose(0,1).float()
    score=qh@kh.transpose(-1,-2)/16
    mask=torch.arange(nk,device='cuda')[None,:] <= torch.arange(nq,device='cuda')[:,None]+nk-nq
    expected=(score.masked_fill(~mask,float('-inf')).softmax(dim=-1)@vh).transpose(0,1).to(q.dtype)
    torch.testing.assert_close(out,expected,atol=1e-2,rtol=1e-2)
    print('FA2_CASE_PASS',nq,nk,'paged',paged,'maxerr',float((out.float()-expected.float()).abs().max()),flush=True)
print(json.dumps({'fa2_cases_passed':len(cases),'seconds':time.perf_counter()-start}),flush=True)
