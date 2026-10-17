from __future__ import annotations

import functools
import math
from contextlib import contextmanager
from typing import Any, Callable

import torch
import torch.nn.functional as F

from .attention import AttentionBatch, FlashInferBackend
from .base import BaseOP
from .config import ModelConfig, RotaryConfig
from .marlin import MarlinLinear


@contextmanager
def torch_dtype(dtype: torch.dtype):
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


class RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import fused_add_rmsnorm, rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm
        self.fused_add_rmsnorm = fused_add_rmsnorm

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rmsnorm(x, self.weight, self.eps), x
        self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual


class RotaryEmbedding:
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        post_process: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.head_size = head_size
        inv_freq = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
        )
        if post_process is not None:
            inv_freq = post_process(inv_freq)
        positions = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", positions, inv_freq)
        self._cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
        if self.head_size not in (64, 128, 256, 512):
            raise ValueError(f"Unsupported FlashInfer RoPE head size {self.head_size}")
        from flashinfer import apply_rope_with_cos_sin_cache_inplace

        self.apply_rope = apply_rope_with_cos_sin_cache_inplace

    def forward(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.apply_rope(
            positions=positions,
            query=query,
            key=key,
            head_size=self.head_size,
            cos_sin_cache=self._cos_sin_cache,
        )
        return query, key


def _create_rope(config: RotaryConfig) -> RotaryEmbedding:
    rope_scaling = config.scaling
    if rope_scaling is None or rope_scaling.get("rope_type", "default") == "default":
        return RotaryEmbedding(
            config.head_dim, config.rotary_dim, config.max_position, config.base
        )
    if rope_scaling["rope_type"] == "llama3":
        factor = rope_scaling["factor"]
        low_factor = rope_scaling["low_freq_factor"]
        high_factor = rope_scaling["high_freq_factor"]
        original = rope_scaling["original_max_position_embeddings"]

        def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
            wave_len = 2 * math.pi / inv_freq
            if low_factor == high_factor:
                return torch.where(wave_len < original / high_factor, inv_freq, inv_freq / factor)
            smooth = (original / wave_len - low_factor) / (high_factor - low_factor)
            smooth = torch.clamp(smooth, 0, 1)
            return ((1 - smooth) / factor + smooth) * inv_freq

        return RotaryEmbedding(
            config.head_dim,
            config.rotary_dim,
            config.max_position,
            config.base,
            post_process,
        )
    if rope_scaling["rope_type"] == "yarn":
        factor = rope_scaling["factor"]
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        original = rope_scaling["original_max_position_embeddings"]

        def correction_dim(rotations: float) -> float:
            return config.rotary_dim * math.log(
                original / (rotations * 2 * math.pi)
            ) / (2 * math.log(config.base))

        low = max(math.floor(correction_dim(beta_fast)), 0)
        high = min(math.ceil(correction_dim(beta_slow)), config.rotary_dim // 2 - 1)

        def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
            ramp = torch.clamp(
                (torch.arange(config.rotary_dim // 2, dtype=torch.float32) - low)
                / max(high - low, 1),
                0,
                1,
            )
            return (inv_freq / factor) * ramp + inv_freq * (1 - ramp)

        return RotaryEmbedding(
            config.head_dim,
            config.rotary_dim,
            config.max_position,
            config.base,
            post_process,
        )
    raise ValueError(f"Unsupported rope_scaling={rope_scaling}")


_ROPE_DEVICE: torch.device | None = None


def set_rope_device(device: torch.device) -> None:
    global _ROPE_DEVICE
    _ROPE_DEVICE = device


@functools.cache
def get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    scaling: tuple[tuple[str, Any], ...] | None,
) -> RotaryEmbedding:
    config = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=max_position,
        base=base,
        scaling=dict(scaling) if scaling is not None else None,
    )
    probe = torch.tensor([])
    if probe.device == torch.device("meta"):
        if _ROPE_DEVICE is None:
            raise RuntimeError("Call set_rope_device before constructing the model")
        with torch.device(_ROPE_DEVICE):
            return _create_rope(config)
    return _create_rope(config)


class Embedding(BaseOP):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.weight = torch.empty(num_embeddings, embedding_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(token_ids, self.weight)


class AttentionLayer(BaseOP):
    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        backend: FlashInferBackend,
    ) -> None:
        self.layer_id = layer_id
        self.head_dim = config.head_dim
        self.num_qo_heads = config.num_qo_heads
        self.num_kv_heads = config.num_kv_heads
        self.qo_attn_dim = self.num_qo_heads * self.head_dim
        self.kv_attn_dim = self.num_kv_heads * self.head_dim
        rotary = config.rotary_config
        self.rotary = get_rope(
            rotary.head_dim,
            rotary.rotary_dim,
            rotary.max_position,
            rotary.base,
            tuple(rotary.scaling.items()) if rotary.scaling else None,
        )
        self._backend = backend

    def forward(
        self,
        qkv: torch.Tensor,
        batch: AttentionBatch,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        q, k, v = qkv.split(
            [self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1
        )
        q, k = self.rotary.forward(batch.positions, q, k)
        q = q.view(-1, self.num_qo_heads, self.head_dim)
        output = self._backend.forward(q, k, v, batch, k_cache, v_cache)
        return output.view(-1, self.qo_attn_dim)


__all__ = [
    "AttentionLayer",
    "Embedding",
    "MarlinLinear",
    "RMSNormFused",
    "set_rope_device",
    "torch_dtype",
]
