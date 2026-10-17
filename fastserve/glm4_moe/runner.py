from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from fastserve.config import (
    CacheConfig,
    ParallelConfig,
)
from fastserve.config import ModelConfig as FastServeModelConfig

from .attention import AttentionBatch, FlashInferBackend
from .config import ModelConfig
from .gguf import GGUFWeights
from .layers import set_rope_device, torch_dtype
from .marlin import marlin_gemm
from .model import Glm4MoeForCausalLM
from .moe import load_ktransformers_experts
from .sampling import Sampler


def build_attention_batch(
    input_tokens_batched: Sequence[Sequence[int]],
    first_token_indexes: Sequence[int],
    block_table: Sequence[Sequence[int]],
    device: torch.device,
    is_context_stage_batched: Sequence[bool] | None = None,
) -> AttentionBatch:
    if not (
        len(input_tokens_batched) == len(first_token_indexes) == len(block_table)
    ):
        raise ValueError("tokens, positions and block table must have the same batch size")
    if not input_tokens_batched:
        raise ValueError("Cannot execute an empty batch")
    if is_context_stage_batched is None:
        is_context_stage_batched = [first == 0 for first in first_token_indexes]
    if len(is_context_stage_batched) != len(input_tokens_batched):
        raise ValueError("Context-stage flags must have the same batch size as tokens")

    flat_tokens: list[int] = []
    positions: list[int] = []
    output_locations: list[int] = []
    kv_indices: list[int] = []
    seq_lens_q: list[int] = []
    seq_lens_k: list[int] = []
    cached_lens: list[int] = []

    for tokens, first, pages in zip(
        input_tokens_batched, first_token_indexes, block_table
    ):
        query_len = len(tokens)
        if query_len <= 0 or first < 0:
            raise ValueError("Every request needs non-empty input and a non-negative position")
        sequence_len = first + query_len
        if len(pages) < sequence_len:
            raise ValueError(
                f"Block table has {len(pages)} pages but sequence needs {sequence_len}"
            )
        flat_tokens.extend(tokens)
        positions.extend(range(first, sequence_len))
        output_locations.extend(pages[first:sequence_len])
        kv_indices.extend(pages[:sequence_len])
        seq_lens_q.append(query_len)
        seq_lens_k.append(sequence_len)
        cached_lens.append(first)

    return AttentionBatch(
        # FlashKVMoE's scheduler keeps token ids and RoPE positions in int32.
        input_ids=torch.tensor(flat_tokens, dtype=torch.int32, device=device),
        positions=torch.tensor(positions, dtype=torch.int32, device=device),
        out_loc=torch.tensor(output_locations, dtype=torch.int32, device=device),
        seq_lens_q=seq_lens_q,
        seq_lens_k=seq_lens_k,
        cached_lens=cached_lens,
        kv_indices=torch.tensor(kv_indices, dtype=torch.int32, device=device),
        is_decode=(
            not any(is_context_stage_batched)
            and all(length == 1 for length in seq_lens_q)
        ),
    )


class Glm4MoeOp:
    """Python model operator that preserves FastServe's worker interface."""

    def __init__(
        self,
        model_config: FastServeModelConfig,
        parallel_config: ParallelConfig,
        cache_config: CacheConfig,
    ) -> None:
        if parallel_config.tensor_parallel_size != 1 or parallel_config.pipeline_parallel_size != 1:
            raise ValueError("GLM-4.5-Air KTransformers support is single-GPU only")
        if cache_config.block_size != 1:
            raise ValueError(
                "GLM-4.5-Air requires --block-size 1 to match FlashKVMoE attention"
            )
        self.runtime_config = model_config
        self.cache_config = cache_config
        self.config = ModelConfig.from_hf(model_config.hf_config)
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        self.model: Glm4MoeForCausalLM | None = None
        self.attn_backend: FlashInferBackend | None = None
        self.sampler: Sampler | None = None

    def init_communicator(self, tensor_parallel_id, pipeline_parallel_id) -> None:
        # The aligned FlashKVMoE path is intentionally single GPU.
        return None

    def init_dummy_weights(self) -> None:
        raise ValueError("GLM-4.5-Air only supports complete GGUF weights")

    def load_weight(self, path: str | None = None) -> None:
        weight_path = self.runtime_config.kt_weight_path
        if path is not None and Path(path).resolve() != Path(weight_path).resolve():
            raise ValueError("GLM weight loading must use the configured GGUF path")
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("GLM-4.5-Air requires a CUDA GPU with BF16 support")
        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        set_rope_device(self.device)
        marlin_gemm()
        self.attn_backend = FlashInferBackend(self.config, self.device)
        self.sampler = Sampler(self.device, self.config.vocab_size)
        with torch.device("meta"), torch_dtype(self.dtype):
            self.model = Glm4MoeForCausalLM(self.config, self.attn_backend)

        weights = GGUFWeights(weight_path, self.config)
        load_ktransformers_experts(
            self.model, self.runtime_config, self.config, weights
        )
        state = dict(weights.weights(self.device, self.dtype))
        self.model.load_state_dict(state)
        del weights, state
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def forward(
        self,
        input_tokens_batched,
        first_token_indexes,
        is_context_stage_batched,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table,
        sampling_params,
    ) -> list[int]:
        if self.model is None or self.attn_backend is None or self.sampler is None:
            raise RuntimeError("GLM model weights have not been loaded")
        expected_prefix = (
            self.config.num_layers,
            k_cache.shape[1],
            1,
            self.config.num_kv_heads,
            self.config.head_dim,
        )
        if tuple(k_cache.shape) != expected_prefix or tuple(v_cache.shape) != expected_prefix:
            raise ValueError(
                "GLM KV cache must be [layers, pages, 1, kv_heads, head_dim]; "
                f"got K={tuple(k_cache.shape)}, V={tuple(v_cache.shape)}"
            )
        batch = build_attention_batch(
            input_tokens_batched,
            first_token_indexes,
            block_table,
            self.device,
            is_context_stage_batched,
        )
        self.attn_backend.prepare_metadata(batch, k_cache.dtype)
        logits = self.model.forward(batch, k_cache, v_cache)
        sampling_args = self.sampler.prepare(sampling_params)
        return self.sampler.sample(logits, sampling_args).to(torch.int32).tolist()


__all__ = ["Glm4MoeOp", "build_attention_batch"]
