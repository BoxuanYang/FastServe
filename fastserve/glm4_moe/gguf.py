"""GLM-4 MoE GGUF loading for CPU experts and GPU projection weights."""

from __future__ import annotations

from typing import Iterator

import gguf
import numpy as np
import torch
from tqdm import tqdm

from .config import ModelConfig
from .gguf_parts import open_gguf_readers
from .marlin import pack_marlin


def _gpu_tensor_specs(config: ModelConfig) -> Iterator[tuple[str, str, tuple[int, ...]]]:
    """Yield GGUF name, runtime name and PyTorch shape for every GPU tensor."""
    hidden = config.hidden_size
    yield "token_embd.weight", "model.embed_tokens.weight", (config.vocab_size, hidden)
    yield "output_norm.weight", "model.norm.weight", (hidden,)
    head = "token_embd.weight" if config.tie_word_embeddings else "output.weight"
    yield head, "lm_head.weight", (config.vocab_size, hidden)
    yield from _glm4_moe_tensor_specs(config)


def _glm4_moe_tensor_specs(c: ModelConfig) -> Iterator[tuple[str, str, tuple[int, ...]]]:
    for layer in range(c.num_layers):
        src, dst = f"blk.{layer}", f"model.layers.{layer}"
        for proj, heads in (
            ("q", c.num_qo_heads),
            ("k", c.num_kv_heads),
            ("v", c.num_kv_heads),
        ):
            size = heads * c.head_dim
            yield (
                f"{src}.attn_{proj}.weight",
                f"{dst}.self_attn.{proj}_proj.weight",
                (size, c.hidden_size),
            )
        for proj, heads in (
            ("q", c.num_qo_heads),
            ("k", c.num_kv_heads),
            ("v", c.num_kv_heads),
        ):
            size = heads * c.head_dim
            yield f"{src}.attn_{proj}.bias", f"{dst}.self_attn.{proj}_proj.bias", (size,)
        yield (
            f"{src}.attn_output.weight",
            f"{dst}.self_attn.o_proj.weight",
            (c.hidden_size, c.num_qo_heads * c.head_dim),
        )
        yield f"{src}.attn_norm.weight", f"{dst}.input_layernorm.weight", (c.hidden_size,)
        yield (
            f"{src}.post_attention_norm.weight",
            f"{dst}.post_attention_layernorm.weight",
            (c.hidden_size,),
        )
        dense = layer < c.first_k_dense_replace
        if not dense:
            yield (
                f"{src}.ffn_gate_inp.weight",
                f"{dst}.mlp.gate.weight",
                (c.num_experts, c.hidden_size),
            )
            yield (
                f"{src}.exp_probs_b.bias",
                f"{dst}.mlp.gate.e_score_correction_bias",
                (c.num_experts,),
            )
        intermediate = (
            c.intermediate_size if dense else c.n_shared_experts * c.moe_intermediate_size
        )
        suffix, mlp = ("", f"{dst}.mlp") if dense else ("_shexp", f"{dst}.mlp.shared_experts")
        for proj in ("gate", "up", "down"):
            shape = (
                (c.hidden_size, intermediate)
                if proj == "down"
                else (intermediate, c.hidden_size)
            )
            yield f"{src}.ffn_{proj}{suffix}.weight", f"{mlp}.{proj}_proj.weight", shape


class GGUFWeights:
    """Index and validate a complete GLM-4 MoE checkpoint.

    Expert tensors stay quantized and memory mapped for KTransformers. Every
    ordinary model tensor is loaded from the same GGUF checkpoint.
    """

    def __init__(self, path: str, config: ModelConfig):
        if config.model_type != "glm4_moe" or config.architectures != [
            "Glm4MoeForCausalLM"
        ]:
            raise ValueError("GGUF loading only supports Glm4MoeForCausalLM")

        architecture_name = "glm4moe"
        # Readers mmap packed data and therefore remain alive for this object's lifetime.
        self.readers = open_gguf_readers(path)
        self.tensors = {}
        self.config = config
        shard_ids = []
        for index, reader in enumerate(self.readers):
            if reader.endianess != gguf.GGUFEndian.LITTLE:
                raise ValueError("Only little-endian GGUF checkpoints are supported")
            fields = reader.fields
            architecture = fields.get("general.architecture")
            if (index == 0 and architecture is None) or (
                architecture is not None and architecture.contents() != architecture_name
            ):
                raise ValueError(
                    f"Expected general.architecture={architecture_name} in the first GGUF shard"
                )
            count = fields.get("split.count")
            if count is not None:
                if count.contents() != len(self.readers) or "split.no" not in fields:
                    raise ValueError("Incomplete GGUF shards: pass a directory with all shards")
                shard_ids.append(fields["split.no"].contents())
            for tensor in reader.tensors:
                if tensor.name in self.tensors:
                    raise ValueError(
                        f"Duplicate GGUF tensor {tensor.name}; use one quantization only"
                    )
                self.tensors[tensor.name] = tensor
        if shard_ids and sorted(shard_ids) != list(range(len(self.readers))):
            raise ValueError("Inconsistent GGUF split metadata or duplicate shard indices")
        if len(self.readers) > 1 and not shard_ids:
            raise ValueError("Multiple GGUF files without split metadata; select one checkpoint")

        expected = {name: shape for name, _, shape in _gpu_tensor_specs(config)}
        for layer in range(config.first_k_dense_replace, config.num_layers):
            for proj in ("gate", "up", "down"):
                dims = (config.moe_intermediate_size, config.hidden_size)
                if proj == "down":
                    dims = dims[::-1]
                expected[f"blk.{layer}.ffn_{proj}_exps.weight"] = (
                    config.num_experts,
                    *dims,
                )
        for name, shape in expected.items():
            tensor = self.tensors.get(name)
            if tensor is None:
                raise ValueError(f"Missing GGUF tensor {name}; pass a complete GGUF checkpoint")
            actual = tuple(int(dim) for dim in tensor.shape[::-1])
            if actual != shape:
                raise ValueError(f"GGUF tensor {name}: expected shape {shape}, got {actual}")

        # Detect a mismatched config even when tensor shapes alone cannot expose it.
        metadata = {
            "block_count": config.num_layers,
            "embedding_length": config.hidden_size,
            "attention.head_count": config.num_qo_heads,
            "attention.head_count_kv": config.num_kv_heads,
            "attention.key_length": config.head_dim,
            "attention.value_length": config.head_dim,
            "expert_count": config.num_experts,
            "expert_used_count": config.num_experts_per_tok,
            "expert_feed_forward_length": config.moe_intermediate_size,
            "rope.freq_base": config.rotary_config.base,
            "attention.layer_norm_rms_epsilon": config.rms_norm_eps,
        }
        nextn = self.readers[0].fields.get(f"{architecture_name}.nextn_predict_layers")
        metadata["block_count"] += nextn.contents() if nextn is not None else 0
        metadata.update(
            {
                "rope.dimension_count": config.rotary_config.rotary_dim,
                "leading_dense_block_count": config.first_k_dense_replace,
                "expert_shared_count": config.n_shared_experts,
            }
        )
        for reader in self.readers:
            for key, expected_value in metadata.items():
                field = reader.fields.get(f"{architecture_name}.{key}")
                if field is not None and not np.isclose(
                    field.contents(), expected_value, rtol=1e-6, atol=0
                ):
                    raise ValueError(
                        f"GGUF metadata {architecture_name}.{key}={field.contents()} "
                        f"does not match model config ({expected_value})"
                    )

    def get_undequanted_tensor_and_ggml_type(self, name: str):
        """KT loader protocol: return a packed byte view without copying experts."""
        tensor = self.tensors[name]
        return torch.from_numpy(tensor.data.view(np.uint8).reshape(-1)), tensor.tensor_type

    def _dequantize(
        self, name: str, device: torch.device, dtype: torch.dtype, chunk_bytes: int
    ) -> torch.Tensor:
        tensor = self.tensors[name]
        shape = tuple(int(dim) for dim in tensor.shape[::-1])
        result = torch.empty(shape, device=device, dtype=dtype)
        rows = result.reshape(-1, shape[-1])
        packed = tensor.data.reshape(rows.shape[0], -1)
        rows_per_chunk = max(1, chunk_bytes // (shape[-1] * 4))
        for start in range(0, rows.shape[0], rows_per_chunk):
            stop = min(start + rows_per_chunk, rows.shape[0])
            try:
                dense = gguf.dequantize(packed[start:stop], tensor.tensor_type)
            except (NotImplementedError, ValueError) as exc:
                raise ValueError(
                    f"Cannot dequantize GGUF GPU tensor {name} ({tensor.tensor_type.name}): {exc}"
                ) from exc
            # Copy detaches even F32/F16 arrays from the read-only file mapping.
            # Convert on CPU so no temporary FP32 copy is allocated on the GPU.
            chunk = torch.from_numpy(np.array(dense, copy=True)).to(dtype=dtype)
            rows[start:stop].copy_(chunk)
        return result

    def weights(
        self,
        device: torch.device,
        dtype: torch.dtype,
        *,
        chunk_bytes: int = 16 * 1024 * 1024,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Load all non-expert GGUF tensors and pack ordinary linear layers for Marlin."""
        if dtype != torch.bfloat16:
            raise ValueError("GLM-4 MoE GGUF loading requires BF16 runtime tensors")

        merged: list[torch.Tensor] = []
        specs = list(_gpu_tensor_specs(self.config))
        for source, target, _ in tqdm(specs, desc="Loading GGUF GPU weights"):
            is_qkv = target.endswith((".q_proj.weight", ".k_proj.weight", ".v_proj.weight"))
            is_qkv_bias = target.endswith((".q_proj.bias", ".k_proj.bias", ".v_proj.bias"))
            is_gate_up = target.endswith((".gate_proj.weight", ".up_proj.weight"))
            is_marlin = (
                is_qkv
                or is_gate_up
                or target.endswith((".o_proj.weight", "lm_head.weight", ".down_proj.weight"))
            )
            tensor_dtype = torch.float32 if ".mlp.gate." in target else dtype
            weight = self._dequantize(
                source,
                torch.device("cpu") if is_marlin else device,
                tensor_dtype,
                chunk_bytes,
            )
            if is_qkv or is_qkv_bias or is_gate_up:
                merged.append(weight)
                if len(merged) < (3 if is_qkv or is_qkv_bias else 2):
                    continue
                target = (
                    target.replace(".v_proj.weight", ".qkv_proj.weight")
                    if is_qkv
                    else (
                        target.replace(".v_proj.bias", ".qkv_bias")
                        if is_qkv_bias
                        else target.replace(".up_proj.weight", ".gate_up_proj.weight")
                    )
                )
                weight = torch.cat(merged)
                merged.clear()
            if is_marlin:
                packed, scales = pack_marlin(weight)
                yield target, packed.to(device)
                yield target.removesuffix(".weight") + ".scales", scales.to(device)
            else:
                yield target, weight

        if merged:
            raise RuntimeError("Incomplete GGUF projection group")
