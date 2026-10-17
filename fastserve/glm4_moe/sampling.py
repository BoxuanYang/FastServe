from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch

if TYPE_CHECKING:
    from fastserve.request import SamplingParams


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None


def _is_greedy(params: SamplingParams) -> bool:
    return (
        params.temperature <= 0.0 or params.top_k == 1
    ) and params.top_p == 1.0


def _make_device_tensor(
    data: Sequence[int | float], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(
        device, non_blocking=True
    )


def _is_sm90_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0)


def _sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
) -> torch.Tensor:
    import flashinfer.sampling as sampling

    probs = sampling.softmax(
        logits, temperatures, enable_pdl=_is_sm90_supported()
    )
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs)
    if top_p is None:
        return sampling.top_k_sampling_from_probs(probs, top_k)
    if top_k is None:
        return sampling.top_p_sampling_from_probs(probs, top_p)
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


@dataclass
class Sampler:
    """The same FlashInfer sampler used by FlashKVMoE Mini-SGLang."""

    device: torch.device
    vocab_size: int

    def prepare(self, params: Sequence[SamplingParams]) -> BatchSamplingArgs:
        if all(_is_greedy(item) for item in params):
            return BatchSamplingArgs(temperatures=None)

        min_probability = 1e-6
        min_temperature = 1e-6
        temperatures = [
            max(0.0 if _is_greedy(item) else item.temperature, min_temperature)
            for item in params
        ]
        top_ks = [
            item.top_k if item.top_k >= 1 else self.vocab_size for item in params
        ]
        top_ps = [
            min(max(item.top_p, min_probability), 1.0) for item in params
        ]
        temperature_tensor = _make_device_tensor(
            temperatures, torch.float32, self.device
        )
        top_k_tensor = None
        top_p_tensor = None
        if any(item != self.vocab_size for item in top_ks):
            top_k_tensor = _make_device_tensor(top_ks, torch.int32, self.device)
        if any(item < 1.0 for item in top_ps):
            top_p_tensor = _make_device_tensor(top_ps, torch.float32, self.device)
        return BatchSamplingArgs(
            temperature_tensor,
            top_k=top_k_tensor,
            top_p=top_p_tensor,
        )

    def sample(
        self, logits: torch.Tensor, args: BatchSamplingArgs
    ) -> torch.Tensor:
        if args.temperatures is None:
            return torch.argmax(logits, dim=-1)
        return _sample_impl(
            logits.float(), args.temperatures, args.top_k, args.top_p
        )


__all__ = ["BatchSamplingArgs", "Sampler"]
