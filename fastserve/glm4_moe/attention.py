from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Literal

import torch

from .store import store_cache

if TYPE_CHECKING:
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
    )

    from .config import ModelConfig


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << math.ceil(math.log2(n))


@dataclass
class AttentionBatch:
    """The FastServe request batch expressed in Mini-SGLang attention terms."""

    input_ids: torch.Tensor
    positions: torch.Tensor
    out_loc: torch.Tensor
    seq_lens_q: list[int]
    seq_lens_k: list[int]
    cached_lens: list[int]
    kv_indices: torch.Tensor
    is_decode: bool
    attn_metadata: FIMetadata | None = None

    @property
    def size(self) -> int:
        return len(self.seq_lens_q)


@dataclass
class FIMetadata:
    # Keep these fields and their device placement identical to
    # FlashKVMoE/python/minisgl/attention/fi.py.
    cu_seqlens_q_cpu: torch.Tensor
    cu_seqlens_k_cpu: torch.Tensor
    cu_seqlens_q_gpu: torch.Tensor
    indices: torch.Tensor
    last_page_len_cpu: torch.Tensor
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: Literal[1]
    pos_encoding_mode: str
    seq_lens_cpu: torch.Tensor
    dtype: torch.dtype
    wrapper: BatchPrefillWithPagedKVCacheWrapper | BatchDecodeWithPagedKVCacheWrapper
    initialized: bool = False

    def __post_init__(self) -> None:
        if self.page_size != 1:
            raise ValueError("FlashKVMoE-compatible attention requires page_size=1")
        assert (
            self.cu_seqlens_k_cpu.is_cpu
            and self.cu_seqlens_q_cpu.is_cpu
            and self.cu_seqlens_q_gpu.is_cuda
            and self.indices.is_cuda
            and self.last_page_len_cpu.is_cpu
            and self.seq_lens_cpu.is_cpu
        )

    def get_last_indices(self, batch_size: int) -> torch.Tensor:
        return self.cu_seqlens_q_gpu[1 : 1 + batch_size] - 1


class FlashInferBackend:
    """FlashInfer paged attention kept in lockstep with FlashKVMoE's FI backend."""

    def __init__(self, config: ModelConfig, device: torch.device) -> None:
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )

        self.config = config
        self.device = device
        self.float_workspace_buffer = torch.empty(
            256 * 1024 * 1024, dtype=torch.uint8, device=device
        )
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            backend="fa2",
        )
        self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            use_tensor_cores=self.use_tensor_cores,
            kv_layout="NHD",
            backend="fa2",
        )

        # FlashKVMoE deliberately shares this allocation between both wrappers.
        self.int_workspace_buffer = self.prefill_wrapper._int_workspace_buffer
        self.decode_wrapper._int_workspace_buffer = self.int_workspace_buffer

        self.cached_ones_cpu = torch.tensor([], dtype=torch.int32, pin_memory=True)
        self.last_event = torch.cuda.Event()
        self.last_event.record()

    @cached_property
    def use_tensor_cores(self) -> bool:
        override = os.getenv("MINISGL_FLASHINFER_USE_TENSOR_CORES")
        if override is not None:
            return override.lower() in ("1", "true", "yes")
        # GLM-4.5-Air uses GQA=8, matching FlashKVMoE's default selection.
        return self.config.num_qo_heads // self.config.num_kv_heads >= 4

    def _initialize_metadata_once(self, metadata: FIMetadata) -> None:
        if metadata.initialized:
            return

        from flashinfer import BatchDecodeWithPagedKVCacheWrapper

        metadata.initialized = True
        # FlashInfer plan() stages through pinned host memory. Reusing the same
        # wrapper before the asynchronous copy finishes can corrupt its plan.
        self.last_event.synchronize()
        if isinstance(metadata.wrapper, BatchDecodeWithPagedKVCacheWrapper):
            metadata.wrapper.plan(
                indptr=metadata.cu_seqlens_k_cpu,
                indices=metadata.indices,
                last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                data_type=metadata.dtype,
                q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype,
                non_blocking=True,
            )
        else:
            metadata.wrapper.plan(
                qo_indptr=metadata.cu_seqlens_q_cpu,
                paged_kv_indptr=metadata.cu_seqlens_k_cpu,
                paged_kv_indices=metadata.indices,
                paged_kv_last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim_qk=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype,
                non_blocking=True,
                causal=True,
            )
        self.last_event.record()

    def _get_ones_cpu(self, batch_size: int) -> torch.Tensor:
        if batch_size <= len(self.cached_ones_cpu):
            return self.cached_ones_cpu[:batch_size]
        next_len = _next_power_of_2(batch_size)
        self.cached_ones_cpu = torch.ones(next_len, dtype=torch.int32, pin_memory=True)
        return self.cached_ones_cpu[:batch_size]

    def prepare_metadata(self, batch: AttentionBatch, dtype: torch.dtype) -> None:
        if not batch.seq_lens_q:
            raise ValueError("Attention batch must contain at least one request")
        cpu_kwargs = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}
        seq_lens_cpu = torch.tensor(batch.seq_lens_k, **cpu_kwargs)
        cu_seqlens_k_cpu = torch.tensor(
            [0] + batch.seq_lens_k, **cpu_kwargs
        ).cumsum_(dim=0)
        max_seqlen_q = max(batch.seq_lens_q)
        if max_seqlen_q == 1:
            cu_seqlens_q_cpu = torch.arange(0, batch.size + 1, **cpu_kwargs)
        elif all(length == 0 for length in batch.cached_lens):
            cu_seqlens_q_cpu = cu_seqlens_k_cpu
        else:
            cu_seqlens_q_cpu = torch.tensor(
                [0] + batch.seq_lens_q, **cpu_kwargs
            ).cumsum_(dim=0)

        batch.attn_metadata = FIMetadata(
            cu_seqlens_q_cpu=cu_seqlens_q_cpu,
            cu_seqlens_k_cpu=cu_seqlens_k_cpu,
            cu_seqlens_q_gpu=cu_seqlens_q_cpu.to(self.device, non_blocking=True),
            indices=batch.kv_indices,
            last_page_len_cpu=self._get_ones_cpu(batch.size),
            num_qo_heads=self.config.num_qo_heads,
            num_kv_heads=self.config.num_kv_heads,
            head_dim=self.config.head_dim,
            page_size=1,
            pos_encoding_mode="NONE",
            seq_lens_cpu=seq_lens_cpu,
            dtype=dtype,
            wrapper=self.decode_wrapper if batch.is_decode else self.prefill_wrapper,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch: AttentionBatch,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        if not isinstance(metadata, FIMetadata):
            raise RuntimeError("FlashInfer metadata was not prepared")
        self._initialize_metadata_once(metadata)

        # The worker allocates [layer, page, 1, kv_head, head_dim]. The layer
        # slice below is exactly the NHD paged cache consumed by FlashKVMoE.
        flat_k = k_cache.view(-1, self.config.num_kv_heads, self.config.head_dim)
        flat_v = v_cache.view(-1, self.config.num_kv_heads, self.config.head_dim)
        store_cache(flat_k, flat_v, batch.out_loc, k, v)
        return metadata.wrapper.run(q=q, paged_kv_cache=(k_cache, v_cache))


__all__ = ["AttentionBatch", "FIMetadata", "FlashInferBackend"]
