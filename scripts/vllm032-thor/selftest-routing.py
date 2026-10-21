"""CPU-only boundary checks of the actual routing predicate."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
source = (Path(__file__).parent / 'opt-hybrid-qualification' / 'thor_hybrid_attention.py').read_text()
fn = next(x for x in ast.parse(source).body if isinstance(x, ast.FunctionDef) and x.name == 'decode_route')
ns = {'torch': NS(bfloat16='bf16')}
exec(compile(ast.Module(body=[fn], type_ignores=[]), 'routing', 'exec'), ns)
def case(rows=6, heads=24, paged=True, batch=1, dtype='bf16', **kw):
    q = NS(shape=(rows, heads, 256), dtype=dtype)
    k = NS(shape=(2, 816, 4, 256), dtype=dtype)
    return dict(q=q, k=k, v=k, block_table=object() if paged else None,
                cu_seqlens_q=NS(numel=lambda: batch + 1), **kw)
cases = [(case(rows=1), True), (case(), True), (case(rows=0), False),
         (case(rows=7), False), (case(heads=12), False), (case(paged=False), False),
         (case(batch=2), False), (case(dtype='fp32'), False), (case(s_aux=object()), False),
         (case(return_softmax_lse=True), False),
         (case(k_descale=object(), v_descale=object()), True), (case(q_descale=object()), True)]
count=0
for block in [432,816]:
 for args,expected in cases:
  args['k'].shape=(2,block,4,256)
  assert ns['decode_route'](args) is expected,(block,args)
  count+=1
bad=case();bad['k'].shape=(2,512,4,256)
assert ns['decode_route'](bad) is False
print(f'ROUTING_FIXTURE pass={count+1} fail=0')
