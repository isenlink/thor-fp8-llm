# SPDX-License-Identifier: Apache-2.0
# Adapted from upstream vLLM tests/kernels/quantization/test_nvfp4_scaled_mm.py.
import torch
from vllm import _custom_ops as ops
from nvfp4_reference import FLOAT4_E2M1_MAX,FLOAT8_E4M3_MAX,dequantize_nvfp4_to_dtype
with torch.inference_mode():
 for m in [1,5,6,10,20,128]:
  torch.manual_seed(42+m)
  a=torch.randn((m,256),dtype=torch.bfloat16,device="cuda")
  b=torch.randn((128,256),dtype=torch.bfloat16,device="cuda")
  ag=(FLOAT8_E4M3_MAX*FLOAT4_E2M1_MAX/a.abs().max()).float()
  bg=(FLOAT8_E4M3_MAX*FLOAT4_E2M1_MAX/b.abs().max()).float()
  aq,asc=ops.scaled_fp4_quant(a,ag);bq,bsc=ops.scaled_fp4_quant(b,bg)
  ad=dequantize_nvfp4_to_dtype(aq,asc,ag,torch.bfloat16,"cuda")
  bd=dequantize_nvfp4_to_dtype(bq,bsc,bg,torch.bfloat16,"cuda")
  ref=ad.float()@bd.float().T
  out=ops.cutlass_scaled_fp4_mm(aq,bq,asc,bsc,1/(ag*bg),torch.bfloat16)
  torch.cuda.synchronize()
  rel=((out.float()-ref).norm()/ref.norm()).item()
  assert out.shape==ref.shape and torch.isfinite(out).all() and rel<0.02,(m,rel)
  print("NVFP4_NATIVE_PASS",m,rel,flush=True)
print("NVFP4_NATIVE_SUITE PASS 6/6 fail=0",flush=True)
