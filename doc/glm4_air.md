# GLM-4.5-Air GGUF runtime

FastServe's GLM-4.5-Air path is intended for controlled comparisons with
FlashKVMoE. It uses FlashInfer for attention and KTransformers `kt-kernel` for
the routed MoE experts. The attention metadata, cache layout, and wrapper
options match `FlashKVMoE/python/minisgl/attention/fi.py`.

## Supported configuration

- **Weights:** GGUF only. Safetensors, converted FastServe weights, and dummy
  weights are not accepted by this runtime.
- **Activation and KV-cache dtype:** BF16 (`--dtype bf16`). The
  routed experts remain in their GGUF quantization, such as IQ4_XS.
- **Device topology:** one CUDA GPU. Tensor parallel and pipeline parallel
  sizes must both be 1.
- **KV cache:** `--block-size 1`. Each FlashInfer page represents one token.
- **KTransformers backend:** `LLAMAFILE` with the source pinned to commit
  `d0fe8ee2e7aa8594e4d7551c4dd2049cdcc93b1b` (`d0fe8ee`).

The lab checkpoint has two distinct paths:

```text
/data2/models/GLM-4.5-Air-GGUF/
  chat_template.jinja
  config.json
  generation_config.json
  model.safetensors.index.json
  README.md
  tokenizer_config.json
  tokenizer.json
  IQ4_XS/
    GLM-4.5-Air-IQ4_XS-00001-of-00002.gguf
    GLM-4.5-Air-IQ4_XS-00002-of-00002.gguf
```

Pass the top-level directory to `--model` and `--tokenizer`; FastServe reads
the model configuration, chat template, and tokenizer there. Pass the IQ4_XS
directory to `--kt-weight-path`; FastServe discovers and validates both GGUF
shards in that directory. The safetensors index is not used for weight loading.

## Build and install

Run these commands on the Linux lab machine from the FastServe repository root.
Keep the checkout at `third_party/ktransformers`: the FastServe Marlin build and
the FlashKVMoE reference tree both resolve KTransformers sources from that path.

```bash
conda env create -f environment-glm.yml
conda activate fastserve-glm

git submodule update --init third_party/ktransformers
git -C third_party/ktransformers checkout \
  d0fe8ee2e7aa8594e4d7551c4dd2049cdcc93b1b
git -C third_party/ktransformers submodule update --init --recursive \
  third_party/llama.cpp third_party/pybind11

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export CUDAHOSTCXX="$CXX"
export CMAKE_PREFIX_PATH="$CONDA_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig:$CONDA_PREFIX/share/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"

python -m pip install -e .
CMAKE_ARGS="-DUSE_CONDA_TOOLCHAIN=ON" \
  CPUINFER_USE_CUDA=1 CPUINFER_CPU_INSTRUCT=NATIVE CPUINFER_PARALLEL=8 \
  python -m pip install --no-build-isolation -v -e \
  third_party/ktransformers/kt-kernel
python -c "from kt_kernel import KTMoEWrapper; print('kt-kernel import OK')"
python -m pip check
```

`kt-kernel` builds CPU-specific code. Build it on the machine that will run the
experiment, with the same Python/PyTorch environment used by FastServe. The
LLAMAFILE backend requires an AVX2-capable CPU; `CPUINFER_CPU_INSTRUCT=NATIVE`
selects the available instruction set on that machine.

`environment-glm.yml` pins PyTorch 2.9.1 with CUDA 12.8 because the selected
KTransformers commit pins that PyTorch ABI. Use this same environment for the
FastServe and FlashKVMoE comparison, and record the installed FlashInfer version.

## Start the server

The following command is complete for the checkpoint layout above. Tune the
CPU thread counts only when the comparison protocol calls for it; record any
change alongside the benchmark result.

```bash
CUDA_VISIBLE_DEVICES=0 python -m fastserve.api_server.fastserve_api_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model /data2/models/GLM-4.5-Air-GGUF \
  --tokenizer /data2/models/GLM-4.5-Air-GGUF \
  --dtype bf16 \
  --seed 42 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --block-size 1 \
  --max-num-blocks-per-req 131072 \
  --swap-space 16 \
  --kt-weight-path /data2/models/GLM-4.5-Air-GGUF/IQ4_XS \
  --kt-method LLAMAFILE \
  --kt-cpuinfer 128 \
  --kt-threadpool-count 2 \
  --kt-chunked-prefill-size 131072
```

`--block-size 1` is required because the matching FlashInfer implementation
uses one token per page. Set `--max-num-blocks-per-req` and
`--kt-chunked-prefill-size` to the maximum sequence length used by the
experiment; 131072 matches the full checkpoint context in this example. The
runtime rejects non-BF16 execution, multiple GPUs, non-unit block sizes, dummy
weights, and missing GGUF paths.

The GLM backend does not replace or constrain FastServe's scheduler,
BlockManager, block tables, proactive offloading, or CPU KV swap policy. The
usual `--sched-policy`, `--proactive-offloading`, and `--swap-space` settings
retain their original meaning. The worker only changes the physical tensor
layout for GLM so each layer is a contiguous FlashInfer NHD cache; the same
FastServe block IDs and swap-in/swap-out decisions address its page dimension.
The command above keeps the server's original 16 GiB API swap setting enabled.

FastServe may place prefill and decode requests in one batch. Such a mixed
batch remains one model forward in the original request order and uses the
FlashInfer prefill wrapper, which supports per-request query and cached
lengths. A batch containing only one-token decode requests uses the decode
wrapper. No phase splitting is added to the scheduler or model execution.

## FlashKVMoE attention parity

FastServe deliberately keeps these settings aligned with
`python/minisgl/attention/fi.py`:

| Item | FastServe and FlashKVMoE value |
| --- | --- |
| Floating workspace | one 256 MiB `torch.uint8` buffer |
| Wrapper cache layout | `kv_layout="NHD"` |
| FlashInfer backend | `backend="fa2"` for prefill and decode |
| Integer workspace | shared by the prefill and decode wrappers |
| Page size | 1 token |
| Position encoding in FlashInfer | `pos_encoding_mode="NONE"`; RoPE is applied before attention |
| Prefill mask | `causal=True` |
| Decode tensor cores | enabled when `num_qo_heads / num_kv_heads >= 4` |
| Planning copies | `non_blocking=True`, fenced by the previous CUDA event |
| Execution mode | eager FlashInfer `plan()` / `run()` |

For every request, `positions` is the absolute range from the cached length to
the new sequence length. `out_loc` contains the physical page for each new K/V
token. `kv_indices` concatenates every request's physical pages from token 0
through its current sequence length. Query and KV indptr arrays, per-request
sequence lengths, and last-page lengths use the same pinned-host conventions as
FlashKVMoE. K and V are written to the NHD cache before the FlashInfer wrapper
runs. FlashInfer receives no RoPE mode because the model applies RoPE to Q and K
first; prefill remains causal.

For a fair comparison, disable CUDA graph capture in the FlashKVMoE reference
run by adding `--cuda-graph-max-bs 0` to its launch command. FastServe's current
GLM path uses the eager FlashInfer `plan()` / `run()` path, so enabling graphs
only on FlashKVMoE would compare different attention execution modes.

Run the CPU-safe parity tests with:

```bash
python -m pytest -q fastserve/tests/test_glm4_moe_support.py
```

The FlashInfer wrappers and CUDA tensors are mocked in this test, so these unit
checks do not require a GPU. End-to-end generation still requires a CUDA GPU
with BF16 support and the complete two-shard GGUF checkpoint.
