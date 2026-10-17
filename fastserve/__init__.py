from __future__ import annotations

import os

import torch

BASE_DIR = os.path.dirname(__file__)
LIB_DIR = os.path.join(BASE_DIR, "../SwiftTransformer/build/lib")

_SWIFT_TRANSFORMER_LOADED = False


def load_swift_transformer() -> None:
    """Load the legacy FastServe operator only for models that use it."""
    global _SWIFT_TRANSFORMER_LOADED
    if _SWIFT_TRANSFORMER_LOADED:
        return
    if not os.path.exists(LIB_DIR):
        raise RuntimeError(
            f"Could not find the SwiftTransformer library libst_pybinding.so at {LIB_DIR}. "
            "Please build the SwiftTransformer library first or put it at the right place."
        )
    torch.ops.load_library(os.path.join(LIB_DIR, "libst_pybinding.so"))
    _SWIFT_TRANSFORMER_LOADED = True


from fastserve.llm import OfflineLLM as OfflineLLM  # noqa: E402
from fastserve.request import SamplingParams as SamplingParams  # noqa: E402
