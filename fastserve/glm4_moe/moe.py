from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from fastserve.config import ModelConfig as FastServeModelConfig

    from .config import ModelConfig
    from .gguf import GGUFWeights
    from .model import Glm4MoeForCausalLM


def _create_kt_wrapper(
    runtime: FastServeModelConfig, config: ModelConfig, layer_id: int
):
    from kt_kernel import KTMoEWrapper

    wrapper = KTMoEWrapper(
        layer_idx=layer_id,
        num_experts=config.num_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        hidden_size=config.hidden_size,
        moe_intermediate_size=config.moe_intermediate_size,
        gpu_experts_mask=None,
        cpuinfer_threads=runtime.kt_cpuinfer,
        threadpool_count=runtime.kt_threadpool_count,
        weight_path=runtime.kt_weight_path,
        chunked_prefill_size=runtime.kt_chunked_prefill_size,
        method=runtime.kt_method,
        max_deferred_experts_per_token=0,
    )
    wrapper.load_weights(
        torch.arange(config.num_experts, dtype=torch.int32, device="cpu")
    )
    return wrapper


def load_ktransformers_experts(
    model: Glm4MoeForCausalLM,
    runtime: FastServeModelConfig,
    config: ModelConfig,
    gguf_weights: GGUFWeights,
) -> None:
    from kt_kernel import KTMoEWrapper
    from kt_kernel.utils.llamafile import LlamafileMoEWrapper

    KTMoEWrapper.set_capture_batch_sizes([])
    LlamafileMoEWrapper._gguf_loaders_by_path[
        os.path.realpath(runtime.kt_weight_path)
    ] = gguf_weights
    for layer_id, layer in enumerate(model.model.layers.op_list):
        if layer_id >= config.first_k_dense_replace:
            layer.mlp._wrapper = _create_kt_wrapper(runtime, config, layer_id)


__all__ = ["load_ktransformers_experts"]
