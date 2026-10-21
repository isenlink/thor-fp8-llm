"""Compatibility for the existing dense Qwen3Next-exported mixed checkpoint."""
def register():
 from vllm.model_executor.models import qwen3_next as q
 from vllm.transformers_utils import config as cfg
 original_moe=q.QwenNextMixtureOfExperts.set_moe_parameters
 def set_moe(self):
  if getattr(self.config,"num_experts",None)==0:
   self.moe_layers=[]
   return
  return original_moe(self)
 q.QwenNextMixtureOfExperts.set_moe_parameters=set_moe
 original_mrope=cfg._uses_mrope
 def uses_mrope(config):
  if getattr(config,"model_type",None)=="qwen3_next":return False
  return original_mrope(config)
 cfg._uses_mrope=uses_mrope
 for cls in [q.Qwen3NextModel,q.Qwen3NextForCausalLM]:
  original=cls.load_weights
  def load(self,weights,_original=original):
   return _original(self,((name,w) for name,w in weights if ".visual." not in name and not name.startswith("visual")))
  cls.load_weights=load
 print("THOR04_MIXED_COMPAT_LOADED dense/mrope/visual",flush=True)
