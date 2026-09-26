# Branch: `serve`

## Summary

Adds `bhaskera-serve`, an HTTP server with an OpenAI-compatible API for any model Bhaskera supports. It is built on Ray Serve and FastAPI, with a vLLM or Hugging Face backend. An optional gateway adds API-key authentication, Langfuse tracing and a public Cloudflare tunnel. The branch also adds QLoRA model loading.

## Changes

### Serving (`serve/`, `launcher/serve.py`)

| Endpoint | Description |
|---|---|
| `GET /health` | Returns 200 once the model is loaded |
| `GET /v1/models` | Lists the served model |
| `POST /v1/chat/completions` | OpenAI Chat Completions, with streaming |

- **Backends:**
  - `vllm`: tensor parallelism, `gpu_memory_utilization`, `max_model_len`
  - `hf`: batching and concurrency limits
- **Scaling:** `num_replicas`, with optional autoscaling between minimum and maximum replica counts.
- **Configuration:** a new `serve` section in `config.py`, with `vllm`, `hf` and `gateway` sub-sections.
- **Documentation:** full CLI and API reference in `src/bhaskera/serve/README.md`.

```bash
pip install -e ".[serve,vllm,gateway]"
bhaskera-serve --config configs/serve_test.yaml
```

### Gateway (`gateway.py`)

A FastAPI proxy in front of the server. It checks bearer API keys, traces requests to Langfuse, and can publish a public URL through a Cloudflare quick tunnel.

### QLoRA loading (`models/quantization.py`, `models/loader.py`)

- `model.quantization: qlora` loads the base model in 4-bit NF4 with BitsAndBytes, using double quantization and bf16 compute.
- It is supported with DDP only. Combining it with FSDP raises an error at load time.

### Configurations

- **Added:** `gemma.yaml`, `openbiollm.yaml`, `param_vllm.yaml`, `serve_test.yaml`.

## Merge notes

Merging as-is would break training on `main`:

- **`pyproject.toml` is replaced.**
  - The training dependencies (`ray[train,data]`, `peft`, `datasets`, `liger-kernel` and others) are removed.
  - Every console script except `bhaskera-serve` is removed: `bhaskera-train`, `-tokenize`, `-infer`, `-diag` and `-dashboard`.
  - The package-discovery setting (`packages.find where = ["src"]`) and the `ruff` configuration are removed.
  - The serving extras should be added to the existing file instead.
- **16 configuration files are deleted**, including `2node.yaml`, `config.yaml`, `qwen_ddp.yaml`, `qwen_hybrid_shard.yaml`, `tokenize.yaml` and the inference configs.
- **API keys are hard-coded in `gateway.py`**, and the gateway can expose the server publicly. Move the keys to environment variables or a secrets file before merging.
- **`config.py` conflicts with `main`.**
- **QLoRA is also on the `qlora` branch**, from the same commit. Merge it from one branch only.
