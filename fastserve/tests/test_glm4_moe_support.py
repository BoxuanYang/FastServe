from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")


ROOT = Path(__file__).resolve().parents[2]
GLM4_MOE_DIR = ROOT / "fastserve" / "glm4_moe"


def _load_module(monkeypatch, name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def glm4_modules(monkeypatch):
    """Load the GLM implementation without importing FastServe's legacy stack."""

    package_name = "_fastserve_glm4_moe_under_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(GLM4_MOE_DIR)]
    monkeypatch.setitem(sys.modules, package_name, package)

    attention = _load_module(
        monkeypatch, f"{package_name}.attention", GLM4_MOE_DIR / "attention.py"
    )
    base = _load_module(monkeypatch, f"{package_name}.base", GLM4_MOE_DIR / "base.py")

    config = types.ModuleType(f"{package_name}.config")
    config.ModelConfig = object
    monkeypatch.setitem(sys.modules, config.__name__, config)

    class _UnusedLayer:
        def __init__(self, *args, **kwargs):
            raise AssertionError("This dependency is not used by the router test")

    layers = types.ModuleType(f"{package_name}.layers")
    layers.AttentionLayer = _UnusedLayer
    layers.Embedding = _UnusedLayer
    layers.MarlinLinear = _UnusedLayer
    layers.RMSNormFused = _UnusedLayer
    layers.set_rope_device = lambda device: None
    layers.torch_dtype = contextlib.nullcontext
    monkeypatch.setitem(sys.modules, layers.__name__, layers)

    model = _load_module(monkeypatch, f"{package_name}.model", GLM4_MOE_DIR / "model.py")

    # runner.py only needs these imports to define Glm4MoeOp. The batch builder
    # under test does not instantiate any of them.
    fastserve = types.ModuleType("fastserve")
    fastserve.__path__ = [str(ROOT / "fastserve")]
    fastserve_config = types.ModuleType("fastserve.config")
    fastserve_config.CacheConfig = object
    fastserve_config.ModelConfig = object
    fastserve_config.ParallelConfig = object
    monkeypatch.setitem(sys.modules, "fastserve", fastserve)
    monkeypatch.setitem(sys.modules, "fastserve.config", fastserve_config)

    gguf = types.ModuleType(f"{package_name}.gguf")
    gguf.GGUFWeights = object
    monkeypatch.setitem(sys.modules, gguf.__name__, gguf)

    marlin = types.ModuleType(f"{package_name}.marlin")
    marlin.marlin_gemm = lambda: None
    monkeypatch.setitem(sys.modules, marlin.__name__, marlin)

    moe = types.ModuleType(f"{package_name}.moe")
    moe.load_ktransformers_experts = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, moe.__name__, moe)

    runner = _load_module(monkeypatch, f"{package_name}.runner", GLM4_MOE_DIR / "runner.py")
    return SimpleNamespace(attention=attention, base=base, model=model, runner=runner)


@pytest.mark.parametrize(
    (
        "tokens",
        "first_token_indexes",
        "block_table",
        "context_flags",
        "expected",
    ),
    [
        pytest.param(
            [[10, 11, 12], [20, 21]],
            [0, 0],
            [[8, 2, 7], [5, 4]],
            [True, True],
            {
                "input_ids": [10, 11, 12, 20, 21],
                "positions": [0, 1, 2, 0, 1],
                "out_loc": [8, 2, 7, 5, 4],
                "kv_indices": [8, 2, 7, 5, 4],
                "seq_lens_q": [3, 2],
                "seq_lens_k": [3, 2],
                "cached_lens": [0, 0],
                "is_decode": False,
            },
            id="prefill",
        ),
        pytest.param(
            [[13], [22]],
            [3, 2],
            [[8, 2, 7, 9], [5, 4, 6]],
            [False, False],
            {
                "input_ids": [13, 22],
                "positions": [3, 2],
                "out_loc": [9, 6],
                "kv_indices": [8, 2, 7, 9, 5, 4, 6],
                "seq_lens_q": [1, 1],
                "seq_lens_k": [4, 3],
                "cached_lens": [3, 2],
                "is_decode": True,
            },
            id="decode",
        ),
        pytest.param(
            [[30], [40, 41]],
            [2, 1],
            [[10, 11, 12], [20, 21, 22]],
            [False, True],
            {
                "input_ids": [30, 40, 41],
                "positions": [2, 1, 2],
                "out_loc": [12, 21, 22],
                "kv_indices": [10, 11, 12, 20, 21, 22],
                "seq_lens_q": [1, 2],
                "seq_lens_k": [3, 3],
                "cached_lens": [2, 1],
                "is_decode": False,
            },
            id="mixed",
        ),
        pytest.param(
            [[42]],
            [0],
            [[3]],
            [True],
            {
                "input_ids": [42],
                "positions": [0],
                "out_loc": [3],
                "kv_indices": [3],
                "seq_lens_q": [1],
                "seq_lens_k": [1],
                "cached_lens": [0],
                "is_decode": False,
            },
            id="single-token-prefill",
        ),
    ],
)
def test_build_attention_batch_matches_flashkvmoe_page_table_convention(
    glm4_modules,
    tokens,
    first_token_indexes,
    block_table,
    context_flags,
    expected,
):
    batch = glm4_modules.runner.build_attention_batch(
        tokens,
        first_token_indexes,
        block_table,
        torch.device("cpu"),
        context_flags,
    )

    assert batch.input_ids.tolist() == expected["input_ids"]
    assert batch.positions.tolist() == expected["positions"]
    assert batch.out_loc.tolist() == expected["out_loc"]
    assert batch.kv_indices.tolist() == expected["kv_indices"]
    assert batch.seq_lens_q == expected["seq_lens_q"]
    assert batch.seq_lens_k == expected["seq_lens_k"]
    assert batch.cached_lens == expected["cached_lens"]
    assert batch.is_decode is expected["is_decode"]
    assert batch.input_ids.dtype == torch.int32
    assert batch.positions.dtype == torch.int32
    assert batch.out_loc.dtype == torch.int32
    assert batch.kv_indices.dtype == torch.int32


def test_glm_runner_keeps_mixed_batch_as_one_prefill_in_request_order(glm4_modules):
    calls = []

    class FakeAttentionBackend:
        def prepare_metadata(self, batch, dtype):
            calls.append(
                (
                    "attention",
                    batch.is_decode,
                    batch.input_ids.tolist(),
                    batch.kv_indices.tolist(),
                    dtype,
                )
            )

    class FakeModel:
        def forward(self, batch, k_cache, v_cache):
            del k_cache, v_cache
            calls.append(("model", batch.is_decode))
            last_tokens = []
            offset = 0
            for query_len in batch.seq_lens_q:
                last_tokens.append(batch.input_ids[offset + query_len - 1].item())
                offset += query_len
            return torch.tensor(last_tokens, dtype=torch.int64).unsqueeze(-1)

    class FakeSampler:
        def prepare(self, params):
            calls.append(("prepare", list(params)))
            return None

        def sample(self, logits, args):
            assert args is None
            return logits[:, 0] + 100

    op = object.__new__(glm4_modules.runner.Glm4MoeOp)
    op.config = SimpleNamespace(num_layers=1, num_kv_heads=1, head_dim=2)
    op.device = torch.device("cpu")
    op.attn_backend = FakeAttentionBackend()
    op.model = FakeModel()
    op.sampler = FakeSampler()
    k_cache = torch.empty((1, 16, 1, 1, 2), dtype=torch.bfloat16)
    v_cache = torch.empty_like(k_cache)

    sampled = op.forward(
        input_tokens_batched=[[10], [20, 21], [30], [40]],
        first_token_indexes=[2, 0, 3, 0],
        is_context_stage_batched=[False, True, False, True],
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=[
            [0, 1, 2],
            [3, 4],
            [5, 6, 7, 8],
            [9],
        ],
        sampling_params=["decode-0", "prefill-1", "decode-2", "prefill-3"],
    )

    assert sampled == [110, 121, 130, 140]
    assert calls == [
        (
            "attention",
            False,
            [10, 20, 21, 30, 40],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            torch.bfloat16,
        ),
        ("model", False),
        (
            "prepare",
            ["decode-0", "prefill-1", "decode-2", "prefill-3"],
        ),
    ]


def _reference_glm_router(x, weight, correction_bias, config):
    """Independent, explicit reference for GLM's grouped sigmoid router."""

    scores = torch.sigmoid(x.float() @ weight.float().T)
    adjusted = scores + correction_bias
    experts_per_group = config.num_experts // config.n_group

    group_scores = []
    for group_id in range(config.n_group):
        start = group_id * experts_per_group
        stop = start + experts_per_group
        top_two = torch.topk(adjusted[:, start:stop], 2, dim=1).values
        group_scores.append(top_two.sum(dim=1))
    group_scores = torch.stack(group_scores, dim=1)
    selected_groups = torch.topk(group_scores, config.topk_group, dim=1).indices

    allowed = torch.zeros_like(adjusted, dtype=torch.bool)
    for row, groups in enumerate(selected_groups.tolist()):
        for group_id in groups:
            start = group_id * experts_per_group
            allowed[row, start : start + experts_per_group] = True

    choices = adjusted.masked_fill(~allowed, 0)
    expert_ids = torch.topk(choices, config.num_experts_per_tok, dim=1, sorted=False).indices
    expert_weights = torch.gather(scores, 1, expert_ids)
    if config.norm_topk_prob:
        expert_weights = expert_weights / (expert_weights.sum(dim=1, keepdim=True) + 1e-20)
    return expert_ids, expert_weights * config.routed_scaling_factor


@pytest.mark.parametrize("norm_topk_prob", [False, True])
def test_glm_router_exactly_matches_independent_reference(glm4_modules, norm_topk_prob):
    config = SimpleNamespace(
        num_experts=12,
        hidden_size=6,
        n_group=3,
        topk_group=2,
        num_experts_per_tok=3,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=2.5,
    )
    router = glm4_modules.model.Glm4MoeRouter(config)
    generator = torch.Generator().manual_seed(20261017)
    router.weight = torch.randn(12, 6, generator=generator) * 0.3
    router.e_score_correction_bias = torch.tensor(
        [
            0.012,
            0.034,
            0.056,
            0.078,
            0.091,
            0.113,
            0.135,
            0.157,
            0.179,
            0.202,
            0.224,
            0.246,
        ],
        dtype=torch.float32,
    )
    x = torch.randn(5, 6, generator=generator)

    expected_ids, expected_weights = _reference_glm_router(
        x,
        router.weight,
        router.e_score_correction_bias,
        config,
    )
    actual_ids, actual_weights = router.forward(x)

    assert torch.equal(actual_ids, expected_ids)
    assert torch.equal(actual_weights, expected_weights)


def test_two_gguf_shards_load_model_and_kt(monkeypatch):
    """Exercise the complete GGUF/KT handoff with two in-memory shards."""

    package_name = "_fastserve_glm4_gguf_under_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(GLM4_MOE_DIR)]
    monkeypatch.setitem(sys.modules, package_name, package)

    _load_module(monkeypatch, f"{package_name}.base", GLM4_MOE_DIR / "base.py")
    config_module = types.ModuleType(f"{package_name}.config")
    config_module.ModelConfig = object
    monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
    marlin = _load_module(
        monkeypatch, f"{package_name}.marlin", GLM4_MOE_DIR / "marlin.py"
    )

    class FakeField:
        def __init__(self, value):
            self.value = value

        def contents(self):
            return self.value

    class FakeGGUFTensor:
        def __init__(self, name, value):
            self.name = name
            self.data = value
            self.shape = value.shape[::-1]
            self.tensor_type = SimpleNamespace(name="F32")

    class FakeReader:
        def __init__(self, shard_id, tensors):
            self.endianess = "little"
            self.fields = {
                "general.architecture": FakeField("glm4moe"),
                "split.no": FakeField(shard_id),
                "split.count": FakeField(2),
            }
            self.tensors = tensors

    fake_gguf = types.ModuleType("gguf")
    fake_gguf.GGUFEndian = SimpleNamespace(LITTLE="little")
    fake_gguf.dequantize = lambda value, _tensor_type: value
    monkeypatch.setitem(sys.modules, "gguf", fake_gguf)

    readers = []
    gguf_parts = types.ModuleType(f"{package_name}.gguf_parts")
    gguf_parts.open_gguf_readers = lambda _path: readers
    monkeypatch.setitem(sys.modules, gguf_parts.__name__, gguf_parts)
    gguf_weights_module = _load_module(
        monkeypatch, f"{package_name}.gguf", GLM4_MOE_DIR / "gguf.py"
    )
    moe = _load_module(monkeypatch, f"{package_name}.moe", GLM4_MOE_DIR / "moe.py")

    config = SimpleNamespace(
        num_layers=2,
        num_qo_heads=2,
        num_kv_heads=1,
        head_dim=4,
        hidden_size=8,
        vocab_size=8,
        intermediate_size=4,
        rms_norm_eps=1e-5,
        rotary_config=SimpleNamespace(base=10_000.0, rotary_dim=2),
        tie_word_embeddings=False,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=4,
        model_type="glm4_moe",
        architectures=["Glm4MoeForCausalLM"],
        first_k_dense_replace=1,
        n_shared_experts=1,
    )
    tensors = {}
    for index, (source, _target, shape) in enumerate(
        gguf_weights_module._gpu_tensor_specs(config)
    ):
        values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        tensors[source] = values / max(1, values.size) + index
    for projection in ("gate", "up", "down"):
        dims = (config.moe_intermediate_size, config.hidden_size)
        if projection == "down":
            dims = dims[::-1]
        shape = (config.num_experts, *dims)
        name = f"blk.1.ffn_{projection}_exps.weight"
        tensors[name] = np.zeros(shape, dtype=np.float32)

    entries = list(tensors.items())
    readers.extend(
        [
            FakeReader(
                shard_id,
                [FakeGGUFTensor(name, value) for name, value in entries[shard_id::2]],
            )
            for shard_id in range(2)
        ]
    )
    checkpoint_path = ROOT / "fake-two-shard-checkpoint"
    weights = gguf_weights_module.GGUFWeights(str(checkpoint_path), config)

    shard_names = [{tensor.name for tensor in reader.tensors} for reader in readers]
    assert shard_names[0].isdisjoint(shard_names[1])
    assert shard_names[0] | shard_names[1] == set(tensors)
    assert set(weights.tensors) == set(tensors)
    assert weights.readers == readers

    cache = {}

    class FakeKTMoEWrapper:
        capture_calls = []
        instances = []

        @classmethod
        def set_capture_batch_sizes(cls, sizes):
            cls.capture_calls.append(sizes)

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.loaded_experts = None
            self.__class__.instances.append(self)

        def load_weights(self, expert_map):
            self.loaded_experts = expert_map

    kt_kernel = types.ModuleType("kt_kernel")
    kt_kernel.KTMoEWrapper = FakeKTMoEWrapper
    llamafile = types.ModuleType("kt_kernel.utils.llamafile")
    llamafile.LlamafileMoEWrapper = SimpleNamespace(_gguf_loaders_by_path=cache)
    monkeypatch.setitem(sys.modules, "kt_kernel", kt_kernel)
    monkeypatch.setitem(sys.modules, "kt_kernel.utils", types.ModuleType("kt_kernel.utils"))
    monkeypatch.setitem(sys.modules, "kt_kernel.utils.llamafile", llamafile)

    layers = [
        SimpleNamespace(mlp=SimpleNamespace()),
        SimpleNamespace(mlp=SimpleNamespace()),
    ]
    model = SimpleNamespace(
        model=SimpleNamespace(layers=SimpleNamespace(op_list=layers))
    )
    runtime = SimpleNamespace(
        kt_weight_path=str(checkpoint_path),
        kt_cpuinfer=4,
        kt_threadpool_count=2,
        kt_chunked_prefill_size=32,
        kt_method="LLAMAFILE",
    )
    moe.load_ktransformers_experts(model, runtime, config, weights)

    assert cache == {os.path.realpath(checkpoint_path): weights}
    assert FakeKTMoEWrapper.capture_calls == [[]]
    assert not hasattr(layers[0].mlp, "_wrapper")
    assert layers[1].mlp._wrapper is FakeKTMoEWrapper.instances[0]
    wrapper = FakeKTMoEWrapper.instances[0]
    assert wrapper.kwargs["layer_idx"] == 1
    assert wrapper.kwargs["weight_path"] == str(checkpoint_path)
    assert wrapper.loaded_experts.dtype == torch.int32
    assert wrapper.loaded_experts.device.type == "cpu"
    assert wrapper.loaded_experts.tolist() == [0, 1]

    state = dict(weights.weights(torch.device("cpu"), torch.bfloat16))
    assert not any("_exps" in name for name in state)
    expected_marlin_keys = {
        "lm_head.weight",
        "lm_head.scales",
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.0.self_attn.qkv_proj.scales",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.o_proj.scales",
        "model.layers.0.mlp.gate_up_proj.weight",
        "model.layers.0.mlp.gate_up_proj.scales",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.mlp.down_proj.scales",
        "model.layers.1.mlp.shared_experts.gate_up_proj.weight",
        "model.layers.1.mlp.shared_experts.gate_up_proj.scales",
        "model.layers.1.mlp.shared_experts.down_proj.weight",
        "model.layers.1.mlp.shared_experts.down_proj.scales",
    }
    assert expected_marlin_keys <= state.keys()
    assert not any(
        name in state
        for name in (
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
        )
    )

    expected_qkv = torch.cat(
        [
            torch.from_numpy(tensors[f"blk.0.attn_{projection}.weight"])
            for projection in ("q", "k", "v")
        ]
    ).bfloat16()
    expected_packed, expected_scales = marlin.pack_marlin(expected_qkv)
    torch.testing.assert_close(
        state["model.layers.0.self_attn.qkv_proj.weight"], expected_packed
    )
    torch.testing.assert_close(
        state["model.layers.0.self_attn.qkv_proj.scales"], expected_scales
    )
    expected_bias = torch.cat(
        [
            torch.from_numpy(tensors[f"blk.0.attn_{projection}.bias"])
            for projection in ("q", "k", "v")
        ]
    ).bfloat16()
    torch.testing.assert_close(state["model.layers.0.self_attn.qkv_bias"], expected_bias)


class _FakeTensor:
    def __init__(self, values, *, device="cpu", dtype=None):
        self.values = list(values)
        self.device = str(device)
        self.dtype = dtype

    @property
    def is_cpu(self):
        return self.device == "cpu"

    @property
    def is_cuda(self):
        return self.device.startswith("cuda")

    def __len__(self):
        return len(self.values)

    def __getitem__(self, item):
        values = self.values[item]
        if isinstance(item, slice):
            return _FakeTensor(values, device=self.device, dtype=self.dtype)
        return values

    def __sub__(self, value):
        return _FakeTensor(
            [item - value for item in self.values],
            device=self.device,
            dtype=self.dtype,
        )

    def cumsum_(self, dim=0):
        assert dim == 0
        running = 0
        for index, value in enumerate(self.values):
            running += value
            self.values[index] = running
        return self

    def to(self, device, non_blocking=False):
        assert non_blocking is True
        return _FakeTensor(self.values, device=device, dtype=self.dtype)


class _FakeEvent:
    def __init__(self):
        self.record_count = 0
        self.synchronize_count = 0

    def record(self):
        self.record_count += 1

    def synchronize(self):
        self.synchronize_count += 1


class _FakePrefillWrapper:
    instances = []

    def __init__(self, workspace, **kwargs):
        self.workspace = workspace
        self.init_kwargs = kwargs
        self._int_workspace_buffer = object()
        self.plan_calls = []
        self.__class__.instances.append(self)

    def plan(self, **kwargs):
        self.plan_calls.append(kwargs)


class _FakeDecodeWrapper:
    instances = []

    def __init__(self, workspace, **kwargs):
        self.workspace = workspace
        self.init_kwargs = kwargs
        self._int_workspace_buffer = object()
        self.plan_calls = []
        self.__class__.instances.append(self)

    def plan(self, **kwargs):
        self.plan_calls.append(kwargs)


class _FakeTorch:
    uint8 = object()
    int32 = object()
    bfloat16 = object()

    def __init__(self):
        self.empty_calls = []
        self.workspace = object()
        self.cuda = SimpleNamespace(Event=_FakeEvent)

    def empty(self, size, *, dtype, device):
        self.empty_calls.append((size, dtype, device))
        return self.workspace

    def tensor(self, values, *, dtype, device="cpu", pin_memory=False):
        assert pin_memory is True
        return _FakeTensor(values, device=device, dtype=dtype)

    def ones(self, size, *, dtype, pin_memory=False):
        assert pin_memory is True
        return _FakeTensor([1] * size, dtype=dtype)

    def arange(self, start, stop, *, dtype, device="cpu", pin_memory=False):
        assert pin_memory is True
        return _FakeTensor(range(start, stop), device=device, dtype=dtype)


def _values(tensor):
    return tensor.values


def test_flashinfer_initialization_and_plans_match_flashkvmoe(glm4_modules, monkeypatch):
    attention = glm4_modules.attention
    _FakePrefillWrapper.instances.clear()
    _FakeDecodeWrapper.instances.clear()
    flashinfer = types.ModuleType("flashinfer")
    flashinfer.BatchPrefillWithPagedKVCacheWrapper = _FakePrefillWrapper
    flashinfer.BatchDecodeWithPagedKVCacheWrapper = _FakeDecodeWrapper
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)

    fake_torch = _FakeTorch()
    monkeypatch.setattr(attention, "torch", fake_torch)
    config = SimpleNamespace(num_qo_heads=32, num_kv_heads=4, head_dim=128)
    backend = attention.FlashInferBackend(config, "cuda:0")

    assert fake_torch.empty_calls == [(256 * 1024 * 1024, fake_torch.uint8, "cuda:0")]
    assert backend.prefill_wrapper.init_kwargs == {
        "kv_layout": "NHD",
        "backend": "fa2",
    }
    assert backend.decode_wrapper.init_kwargs == {
        "use_tensor_cores": True,
        "kv_layout": "NHD",
        "backend": "fa2",
    }
    assert backend.prefill_wrapper.workspace is fake_torch.workspace
    assert backend.decode_wrapper.workspace is fake_torch.workspace
    assert (
        backend.decode_wrapper._int_workspace_buffer
        is backend.prefill_wrapper._int_workspace_buffer
    )

    prefill_indices = _FakeTensor(
        [7, 8, 9, 10, 20, 21, 22], device="cuda:0", dtype=fake_torch.int32
    )
    prefill = SimpleNamespace(
        seq_lens_q=[2, 1],
        seq_lens_k=[4, 3],
        cached_lens=[2, 2],
        size=2,
        kv_indices=prefill_indices,
        is_decode=False,
        attn_metadata=None,
    )
    backend.prepare_metadata(prefill, fake_torch.bfloat16)
    prefill_metadata = prefill.attn_metadata
    assert _values(prefill_metadata.cu_seqlens_q_cpu) == [0, 2, 3]
    assert _values(prefill_metadata.cu_seqlens_k_cpu) == [0, 4, 7]
    assert _values(prefill_metadata.seq_lens_cpu) == [4, 3]
    assert _values(prefill_metadata.last_page_len_cpu) == [1, 1]
    assert prefill_metadata.page_size == 1
    assert prefill_metadata.pos_encoding_mode == "NONE"
    assert prefill_metadata.wrapper is backend.prefill_wrapper

    backend._initialize_metadata_once(prefill_metadata)
    assert backend.prefill_wrapper.plan_calls == [
        {
            "qo_indptr": prefill_metadata.cu_seqlens_q_cpu,
            "paged_kv_indptr": prefill_metadata.cu_seqlens_k_cpu,
            "paged_kv_indices": prefill_indices,
            "paged_kv_last_page_len": prefill_metadata.last_page_len_cpu,
            "num_qo_heads": 32,
            "num_kv_heads": 4,
            "head_dim_qk": 128,
            "page_size": 1,
            "pos_encoding_mode": "NONE",
            "seq_lens": prefill_metadata.seq_lens_cpu,
            "q_data_type": fake_torch.bfloat16,
            "kv_data_type": fake_torch.bfloat16,
            "non_blocking": True,
            "causal": True,
        }
    ]

    decode_indices = _FakeTensor(list(range(12)), device="cuda:0", dtype=fake_torch.int32)
    decode = SimpleNamespace(
        seq_lens_q=[1, 1],
        seq_lens_k=[5, 7],
        cached_lens=[4, 6],
        size=2,
        kv_indices=decode_indices,
        is_decode=True,
        attn_metadata=None,
    )
    backend.prepare_metadata(decode, fake_torch.bfloat16)
    decode_metadata = decode.attn_metadata
    assert _values(decode_metadata.cu_seqlens_q_cpu) == [0, 1, 2]
    assert _values(decode_metadata.cu_seqlens_k_cpu) == [0, 5, 12]
    assert decode_metadata.page_size == 1
    assert decode_metadata.pos_encoding_mode == "NONE"
    assert decode_metadata.wrapper is backend.decode_wrapper

    backend._initialize_metadata_once(decode_metadata)
    assert backend.decode_wrapper.plan_calls == [
        {
            "indptr": decode_metadata.cu_seqlens_k_cpu,
            "indices": decode_indices,
            "last_page_len": decode_metadata.last_page_len_cpu,
            "num_qo_heads": 32,
            "num_kv_heads": 4,
            "head_dim": 128,
            "page_size": 1,
            "pos_encoding_mode": "NONE",
            "seq_lens": decode_metadata.seq_lens_cpu,
            "data_type": fake_torch.bfloat16,
            "q_data_type": fake_torch.bfloat16,
            "kv_data_type": fake_torch.bfloat16,
            "non_blocking": True,
        }
    ]

    # Planning is once per metadata object and is fenced exactly as in fi.py.
    backend._initialize_metadata_once(decode_metadata)
    assert len(backend.decode_wrapper.plan_calls) == 1
    assert backend.last_event.synchronize_count == 2
    assert backend.last_event.record_count == 3
