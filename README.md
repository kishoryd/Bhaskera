# Branch: `inference`

## Summary

Rebuilds the inference engine around vLLM running on a Ray actor. The existing Hugging Face `generate()` path is kept as a fallback. The branch also adds multi-GPU tensor parallelism, a concurrent-request benchmark, and SLURM launch scripts.

## Changes

### Inference backends (`inference/backends/`)

| Backend | Used when | Notes |
|---|---|---|
| `RayVLLMBackend` | vLLM is installed and CUDA is available (default) | vLLM runs inside a Ray actor on a GPU worker, with PagedAttention and continuous batching. `tensor_parallel_size` defaults to the number of visible GPUs. |
| `HFBackend` | vLLM is unavailable, the device is CPU or MPS, or the backend is forced to `hf` | Hugging Face `generate()`. Supports the TurboQuant KV cache and speculative decoding. |

- **Choosing a backend:** set `--backend vllm|hf` or `BHASKERA_BACKEND`.
- **Ray cluster:** the engine joins the cluster given by `RAY_ADDRESS` or `--ray-address`; otherwise it starts a local Ray session.
- **TurboQuant:** in `InferenceEngine`, the TurboQuant KV cache (`kv_cache: turboquant`) is available on the HF backend only. On the vLLM backend it logs a warning and uses vLLM's own KV cache.
  - The separate vLLM-TurboQuant module (`vllm_turboquant.py`) can run through Ray. `BHASKERA_VLLM_ORCHESTRATION=auto|ray|direct` controls this.

### Command line (`bhaskera-infer`)

- **New options:** `--backend`, `--ray-address`, `--tensor-parallel-size`, `--output-file`, and Param2 options (`--param2`, `--show-thinking`, `--system-prompt`).
- **Throughput report:** tokens per second is printed after each run.

### Benchmarking and scripts

| File | Purpose |
|---|---|
| `scripts/benchmark_system.py` | Sends benchmark prompts concurrently (`--concurrency`, default 8). Reports latency and throughput, and logs the number of active requests to the experiment tracker. |
| `scripts/inference.slurm`, `scripts/inference.sh` | Single-node, 2-GPU inference job. |

### Configuration and tests

- `configs/param2.yaml` adds an `inference` section and switches the tracker to W&B.
- `configs/inference_param2.yaml` enables flash-attention 2.
- `inference/tests/test_inference.py` adds tests for Ray initialization and backend selection.

## Merge notes

- **Build artifacts:** compiled files under `src/bhaskera/inference/__pycache__/` are committed. Remove them before merging.
- **Scratch and duplicate files:**
  - `check_tq.py` and `test_vllm_stream.py` at the repository root are scratch scripts.
  - `src/bhaskera/inference/test_inference.py` duplicates the test suite in `inference/tests/`.
  - `inference/tests/__init__.py` is deleted, which may affect test discovery.
- **Network access:** `benchmark_system.py` downloads its prompts from a GitHub URL by default, but compute nodes have no internet access. Use a local prompt file there.
- **Conflict:** merging conflicts with `main` in `configs/param2.yaml`. The branch is 27 commits behind `main`.
