# Qwen38VL: Qwen3.8-27B (Qwen3-Next hybrid 塔) + Qwen3VL 视觉塔
# 方案A: 0.11.2 的 Qwen3VLForConditionalGeneration 视觉壳(含 processor/deepstack)
#        + 已验证的 0.11.2 Qwen3NextForCausalLM 文本塔(含 CG 并发补丁)
# 手法与 0.30 qwen3_5.py 相同: 继承 Qwen3VLForConditionalGeneration, 换掉 language_model。
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_next import Qwen3NextForCausalLM
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.multimodal import MULTIMODAL_REGISTRY


class _NextModelAdapter(nn.Module):
    """把 Qwen3VL 壳传给 language_model.model 的 deepstack_input_embeds kwarg
    适配掉: Qwen3NextModel.forward 不收该参数。27B 的 deepstack_visual_indexes
    为空列表 → deepstack 恒为空, 直接丢弃即可。"""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, *args, deepstack_input_embeds=None, **kwargs):
        return self.inner(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner, name)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3VLProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen38VLForConditionalGeneration(Qwen3VLForConditionalGeneration, IsHybrid):
    hf_to_vllm_mapper = (
        Qwen3VLForConditionalGeneration.hf_to_vllm_mapper
    )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateShapeCalculator,
        )
        pc = vllm_config.parallel_config
        hf = vllm_config.model_config.hf_text_config
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            pc.tensor_parallel_size,
            hf.linear_num_key_heads,
            hf.linear_num_value_heads,
            hf.linear_key_head_dim,
            hf.linear_value_head_dim,
            hf.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateDtypeCalculator,
        )
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        import vllm.model_executor.models.qwen3_vl as qwen3_vl_mod

        orig_cls = qwen3_vl_mod.Qwen3LLMForCausalLM

        class _Qwen3NextAsLLM(Qwen3NextForCausalLM):
            def __init__(self, *, vllm_config=None, prefix=""):
                mc = vllm_config.model_config
                saved_hf = mc.hf_config
                mc.hf_config = mc.hf_text_config
                try:
                    super().__init__(vllm_config=vllm_config, prefix=prefix)
                finally:
                    mc.hf_config = saved_hf
                self.model = _NextModelAdapter(self.model)

        qwen3_vl_mod.Qwen3LLMForCausalLM = _Qwen3NextAsLLM
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            qwen3_vl_mod.Qwen3LLMForCausalLM = orig_cls

        config = vllm_config.model_config.hf_config
        text_config = vllm_config.model_config.hf_text_config
        assert text_config.hidden_size == config.vision_config.out_hidden_size, (
            f"hidden mismatch: text {text_config.hidden_size} vs visual "
            f"{config.vision_config.out_hidden_size}")
        # eagle投机壳兼容: 0.11.2 eagle.py 读 target_model.config.image_token_index,
        # 但 Qwen3VL 命名是 image_token_id — 补桥
        if hasattr(config, "image_token_id") and not hasattr(config, "image_token_index"):
            config.image_token_index = config.image_token_id

    def load_weights(self, weights):
        # 权重源 = 手术版 v4(键: model.layers.N.* / model.embed_tokens.* / lm_head.*)
        #        + visual-tower.safetensors(键: model.visual.*)
        # 模型树: self.language_model.model.layers.*(Qwen3NextForCausalLM), self.visual.*
        skip_prefixes = []
        if self.visual is None:
            skip_prefixes.extend(["visual."])
        mapper = WeightsMapper(
            orig_to_new_prefix={
                "model.visual.": "visual.",
                "model.layers.": "language_model.model.layers.",
                "model.embed_tokens.": "language_model.model.embed_tokens.",
                "model.norm.": "language_model.model.norm.",
                "lm_head.": "language_model.lm_head.",
            }
        )
        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(weights, mapper=mapper)
