from __future__ import annotations

import torch
import torch.nn.functional as F

from .attention import AttentionBatch, FIMetadata, FlashInferBackend
from .base import BaseOP, OPList
from .config import ModelConfig
from .layers import AttentionLayer, Embedding, MarlinLinear, RMSNormFused


class Glm4MoeMLP(BaseOP):
    def __init__(self, hidden_size: int, intermediate_size: int):
        self.gate_up_proj = MarlinLinear(hidden_size, 2 * intermediate_size)
        self.down_proj = MarlinLinear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj.forward(x).chunk(2, dim=-1)
        return self.down_proj.forward(F.silu(gate) * up)


class Glm4MoeRouter(BaseOP):
    def __init__(self, config: ModelConfig):
        self.weight = torch.empty(config.num_experts, config.hidden_size, dtype=torch.float32)
        self.e_score_correction_bias = torch.empty(
            config.num_experts, dtype=torch.float32
        )
        self._config = config

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        config = self._config
        scores = F.linear(x.float(), self.weight).sigmoid()
        choice = scores + self.e_score_correction_bias
        grouped = choice.view(
            -1, config.n_group, config.num_experts // config.n_group
        )
        groups = (
            grouped.topk(2, dim=-1)
            .values.sum(-1)
            .topk(config.topk_group, dim=-1)
            .indices
        )
        mask = torch.zeros_like(grouped[..., 0], dtype=torch.bool).scatter_(
            1, groups, True
        )
        choice = grouped.masked_fill(~mask.unsqueeze(-1), 0).flatten(1)
        ids = choice.topk(config.num_experts_per_tok, dim=-1, sorted=False).indices
        weights = scores.gather(1, ids)
        if config.norm_topk_prob:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        return ids, weights * config.routed_scaling_factor


class Glm4MoeSparseMLP(BaseOP):
    def __init__(self, config: ModelConfig):
        self.gate = Glm4MoeRouter(config)
        self.shared_experts = Glm4MoeMLP(
            config.hidden_size,
            config.n_shared_experts * config.moe_intermediate_size,
        )
        self._wrapper = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._wrapper is None:
            raise RuntimeError("KTransformers experts were not initialized")
        ids, weights = self.gate.forward(x)
        routed = self._wrapper.forward(
            x,
            ids,
            weights,
            torch.cuda.current_stream(x.device).cuda_stream,
        )
        return routed + self.shared_experts.forward(x)


class Glm4MoeAttention(BaseOP):
    def __init__(
        self, config: ModelConfig, layer_id: int, backend: FlashInferBackend
    ) -> None:
        qkv_size = (
            config.num_qo_heads + 2 * config.num_kv_heads
        ) * config.head_dim
        self.qkv_proj = MarlinLinear(config.hidden_size, qkv_size)
        self.qkv_bias = torch.empty(qkv_size)
        self.attn = AttentionLayer(config, layer_id, backend)
        self.o_proj = MarlinLinear(
            config.num_qo_heads * config.head_dim, config.hidden_size
        )

    def forward(
        self,
        x: torch.Tensor,
        batch: AttentionBatch,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj.forward(x) + self.qkv_bias
        return self.o_proj.forward(self.attn.forward(qkv, batch, k_cache, v_cache))


class Glm4MoeDecoderLayer(BaseOP):
    def __init__(
        self, config: ModelConfig, layer_id: int, backend: FlashInferBackend
    ) -> None:
        self.self_attn = Glm4MoeAttention(config, layer_id, backend)
        self.mlp = (
            Glm4MoeMLP(config.hidden_size, config.intermediate_size)
            if layer_id < config.first_k_dense_replace
            else Glm4MoeSparseMLP(config)
        )
        self.input_layernorm = RMSNormFused(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNormFused(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        batch: AttentionBatch,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x, batch, k_cache, v_cache)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        return self.mlp.forward(x), residual


class Glm4MoeModel(BaseOP):
    def __init__(self, config: ModelConfig, backend: FlashInferBackend):
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = OPList(
            [
                Glm4MoeDecoderLayer(config, layer_id, backend)
                for layer_id in range(config.num_layers)
            ]
        )
        self.norm = RMSNormFused(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        batch: AttentionBatch,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        x, residual = self.embed_tokens.forward(input_ids), None
        for layer_id, layer in enumerate(self.layers.op_list):
            x, residual = layer.forward(
                x,
                residual,
                batch,
                k_cache[layer_id],
                v_cache[layer_id],
            )
        return self.norm.forward(x, residual)[0]


class Glm4MoeForCausalLM(BaseOP):
    def __init__(self, config: ModelConfig, backend: FlashInferBackend):
        self.model = Glm4MoeModel(config, backend)
        self.lm_head = MarlinLinear(config.hidden_size, config.vocab_size)

    def forward(
        self,
        batch: AttentionBatch,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.model.forward(batch.input_ids, batch, k_cache, v_cache)
        metadata = batch.attn_metadata
        if not isinstance(metadata, FIMetadata):
            raise RuntimeError("FlashInfer metadata was not prepared")
        if not batch.is_decode:
            hidden = hidden[metadata.get_last_indices(batch.size)].contiguous()
        return self.lm_head.forward(hidden)


__all__ = [
    "Glm4MoeForCausalLM",
    "Glm4MoeMLP",
    "Glm4MoeRouter",
    "Glm4MoeSparseMLP",
]
