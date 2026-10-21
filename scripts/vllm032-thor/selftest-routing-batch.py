import ast
from pathlib import Path
from types import SimpleNamespace as NS
s=(Path(__file__).parent/'opt-hybrid-batch/thor_hybrid_attention.py').read_text()
f=next(x for x in ast.parse(s).body if isinstance(x,ast.FunctionDef) and x.name=='decode_route')
ns={'torch':NS(bfloat16='bf16')};exec(compile(ast.Module(body=[f],type_ignores=[]),'routing','exec'),ns)
def case(b=1,m=5,rows=None,heads=24,block=432,dtype='bf16',paged=True,**kw):
 q=NS(shape=(b*m if rows is None else rows,heads,256),dtype=dtype)
 k=NS(shape=(10,block,4,256),dtype=dtype)
 return dict(q=q,k=k,v=k,cu_seqlens_q=NS(numel=lambda:b+1),max_seqlen_q=m,block_table=object() if paged else None,**kw)
cases=[(case(b=b,m=m),True) for b in [1,2,3,4] for m in [1,5]]
cases += [(case(b=2,rows=6),True),(case(rows=0),False),(case(b=4,rows=21),False),(case(b=0),False),(case(b=5),False),(case(m=0,rows=1),False),(case(m=6),False),(case(rows=6),False),(case(b=4,m=1,rows=20),False),(case(dtype='fp32'),False),(case(heads=12),False),(case(paged=False),False),(case(block=448),False),(case(block=816),False),(case(softcap=1),False),(case(window_size=(1,1)),False),(case(return_softmax_lse=True),False),(case(s_aux=object()),False),(case(output_scale=object()),False),(case(fa_version=3),False)]
for p,expected in cases:assert ns['decode_route'](p) is expected,(p,expected)
print(f'BATCH_ROUTE_FIXTURE pass={len(cases)} fail=0')
