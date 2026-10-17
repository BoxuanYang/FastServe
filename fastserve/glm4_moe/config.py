from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from transformers import PretrainedConfig


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, Any] | None


@dataclass(frozen=True)
class ModelConfig:
    """The GLM-4 MoE fields used by the FastServe GGUF runtime."""

    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    model_type: str
    architectures: list[str]
    first_k_dense_replace: int
    n_shared_experts: int
    n_group: int
    topk_group: int
    routed_scaling_factor: float

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0

    @property
    def is_glm4_moe(self) -> bool:
        return self.model_type == "glm4_moe"

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
        if hasattr(config, "text_config") and config.text_config is not None:
            top = config
            config = config.text_config
            for attr in ("architectures", "rope_theta", "rope_scaling"):
                if not getattr(config, attr, None) and getattr(top, attr, None):
                    setattr(config, attr, getattr(top, attr))

        model_type = getattr(config, "model_type", None)
        architectures = list(getattr(config, "architectures", []) or [])
        if model_type != "glm4_moe" or architectures != ["Glm4MoeForCausalLM"]:
            raise ValueError(
                "FastServe's GGUF runtime only supports Glm4MoeForCausalLM "
                "with model_type=glm4_moe"
            )

        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None:
            rope_scaling = dict(rope_scaling)
            rope_scaling.setdefault("rope_type", rope_scaling.get("type", "default"))
        rope_theta = getattr(config, "rope_theta", None) or rope_scaling["rope_theta"]

        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=getattr(config, "tie_word_embeddings", False),
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=int(head_dim * getattr(config, "partial_rotary_factor", 1.0)),
                max_position=config.max_position_embeddings,
                base=rope_theta,
                scaling=rope_scaling,
            ),
            num_experts=getattr(config, "n_routed_experts"),
            num_experts_per_tok=config.num_experts_per_tok,
            moe_intermediate_size=config.moe_intermediate_size,
            norm_topk_prob=config.norm_topk_prob,
            model_type=model_type,
            architectures=architectures,
            first_k_dense_replace=config.first_k_dense_replace,
            n_shared_experts=config.n_shared_experts,
            n_group=config.n_group,
            topk_group=config.topk_group,
            routed_scaling_factor=config.routed_scaling_factor,
        )
