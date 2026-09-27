# SPDX-License-Identifier: Apache-2.0
"""Qwen3Next MTP draft model — 移植适配 vLLM 0.11.2 (板上 aarch64 版).
参考 vllm 0.30 qwen3_next_mtp.py, 用 0.11.2 组件重写.
eagle proposer 期望: self.model.model.embed_tokens / self.model.lm_head / compute_logits
"""
from collections.abc import Iterable

import torch
from torch import nn

from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm as Qwen3NextRMSNorm
from vllm.model_executor.models.qwen3_next import Qwen3NextDecoderLayer

logger = init_logger(__name__)


class Qwen3NextMTPInner(nn.Module):
    """与 0.30 的 MultiTokenPredictor 对应, 顶层组件挂 self.model 下."""

    def __init__(self, *, vllm_config, config, prefix: str = ""):
        super().__init__()
        # MTP checkpoint(mtp.* 键)是 BF16 非量化 → 强制无量化构造,
        # 否则层参数被建成 NVFP4 weight_packed 而 BF16 权重装不进 → 随机权重
        vllm_config.quant_config = None
        quant_config = vllm_config.quant_config

        self.vocab_size = config.vocab_size
        self.num_mtp_layers = getattr(config, "num_nextn_predict_layers", 1)

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        self.fc = ColumnParallelLinear(
            config.hidden_size * 2,
            config.hidden_size,
            gather_output=True,
            bias=False,
            return_bias=False,
            quant_config=None,
            prefix=f"{prefix}.fc",
        )

        self.layers = torch.nn.ModuleList(
            Qwen3NextDecoderLayer(
                vllm_config,
                layer_type="full_attention",
                prefix=f"{prefix}.layers.{idx}",
            )
            for idx in range(self.num_mtp_layers)
        )

        self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_embedding = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        assert hidden_states.shape[-1] == inputs_embeds.shape[-1]
        inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
        hidden_states = self.pre_fc_norm_hidden(hidden_states)
        hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
        hidden_states = self.fc(hidden_states)
        residual = None

        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[current_step_idx]
        hidden_states, residual = mtp_layer(
            hidden_states=hidden_states,
            residual=residual,
            positions=positions,
        )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        from vllm.model_executor.models.utils import default_weight_loader

        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()

        for name, w in weights:
            if name.startswith("mtp."):
                name = name[len("mtp."):]
            # 注意: 保留 "layers.0." 前缀 — 类参数名就是 layers.0.xxx
            if "rotary_emb.inv_freq" in name:
                continue

            mapped = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                target = name.replace(weight_name, param_name, 1)
                if target not in params_dict:
                    continue
                param = params_dict[target]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, w, shard_id)
                loaded.add(target)
                mapped = True
                break
            if mapped:
                continue

            if name not in params_dict:
                logger.warning("MTP: unmatched weight %s", name)
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, w)
            loaded.add(name)
        return loaded


class Qwen3NextMTP(nn.Module):
    """eagle proposer 兼容外层: self.model.<inner>, self.lm_head, compute_logits."""

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__()
        model_config = vllm_config.model_config
        config = model_config.hf_config
        # VL 壳嫁接: hf_config 可能是 Qwen3VL 顶层(无 vocab_size/linear_*)。
        # 不能改共享 model_config(会污染 target 的 architectures → MM 注册表 KeyError),
        # 构造隔离的 vllm_config 副本(model_config 浅拷贝 + hf_config 指向 text_config)
        if not hasattr(config, "vocab_size") and hasattr(config, "text_config"):
            import dataclasses as _dc
            from copy import copy as _copy
            tc = config.text_config
            if getattr(tc, "model_type", "") not in ("qwen3_next", "qwen3_next_mtp"):
                tc.model_type = "qwen3_next"
            config = tc
            mc2 = _copy(model_config)
            mc2.hf_config = tc
            mc2.hf_text_config = tc
            try:
                vllm_config = _dc.replace(vllm_config, model_config=mc2)
            except Exception:
                vc = _copy(vllm_config)
                vc.model_config = mc2
                vllm_config = vc

        self.config = config
        self.vllm_config = vllm_config
        self.model = Qwen3NextMTPInner(
            vllm_config=vllm_config, config=config, prefix=f"{prefix}.model"
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=None,
            prefix=f"{prefix}.lm_head",
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

        # MTP checkpoint(15 键)无 embed_tokens 也无 lm_head → 全部共享 target 的,
        # 否则随机初始化 → draft logits 垃圾 → 0% 采纳
        self.has_own_embed_tokens = False
        self.has_own_lm_head = False
        del self.lm_head
        del self.model.embed_tokens

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def get_language_model(self) -> nn.Module:
        return self.model

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            hidden_states=hidden_states,
            inputs_embeds=inputs_embeds,
            spec_step_idx=spec_step_idx,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return self.model.load_weights(weights)
