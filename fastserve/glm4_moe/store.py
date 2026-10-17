from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module


@functools.cache
def _jit_store_module(element_size: int) -> Module:
    """Build the exact FlashKVMoE KV-store kernel specialization."""
    from tvm_ffi.cpp import load_inline

    csrc = Path(__file__).resolve().parent / "csrc"
    source = (csrc / "store.cu").resolve().as_posix()
    args = f"{element_size},128,1,false"
    return load_inline(
        f"fastserve__store_{element_size}_128_1_false",
        cuda_sources=[
            f'#include "{source}"',
            f'TVM_FFI_DLL_EXPORT_TYPED_FUNC(launch, (StoreKernel<{args}>::run));',
        ],
        extra_cuda_cflags=["-std=c++20", "-O3", "--expt-relaxed-constexpr"],
        extra_include_paths=[str((csrc / "include").resolve())],
    )


def store_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    """Store K and V together using FlashKVMoE's CUDA kernel."""
    num_tokens = k_cache.shape[0]
    k_cache = k_cache.view(num_tokens, -1)
    v_cache = v_cache.view(num_tokens, -1)
    element_size = k_cache.shape[1] * k_cache.element_size()
    module = _jit_store_module(element_size)
    module.launch(k_cache, v_cache, indices, k, v)


__all__ = ["store_cache"]
