# Branch: `qlora`

## Summary

Adds:
- QLoRA fine-tuning
- continual pre-training (CPT) with sequence packing
- an evaluation subsystem with validation metrics and LM benchmarks
- a pluggable optimizer system (any `torch.optim` class, Lion, GaLore)
- a revised FLOPs/MFU calculation

## Changes

### QLoRA (`models/quantization.py`)

- `model.quantization: qlora` loads the base model in 4-bit NF4 with BitsAndBytes (double quantization, bf16 compute) and trains LoRA adapters in full precision.
- It is supported with DDP only. BitsAndBytes 4-bit weights cannot be sharded by FSDP2, so FSDP raises an error at load time.
- Each GPU holds the whole 4-bit model, about 0.5 bytes per parameter (roughly 35 GB for a 70B model).
- Install `bitsandbytes` separately; it is not in `pyproject.toml`.

### Continual pre-training (`data/tokenize.py`, `data/datasets/local_cpt.py`)

- `data.is_cpt: true`, or its alias `data.pack_sequences`, reads raw JSONL corpora and packs tokens into fixed-length chunks with no padding.
- The flag is part of the tokenization cache key, so packed and unpacked caches never collide.

### Evaluation (`evaluation/`, `plugins/`, `trainer/eval_lifecycle.py`)

| Type | Included |
|---|---|
| Validation metrics | Loss, perplexity, token accuracy |
| Benchmarks | HellaSwag, ARC-Easy, ARC-Challenge |

- Evaluation runs during training under `evaluation.enabled: true` and works under both FSDP2 and DDP.
- Before evaluation, optimizer state is moved to CPU to free GPU memory. Afterwards it is restored and the data position resumes exactly.
- New metrics and benchmarks register with `@register_metric` and `@register_benchmark`.

### Optimizers (`trainer/optim.py`, `trainer/optimizer_registry.py`, `plugins/optimizers/`)

```yaml
training:
  optimizer:
    backend: torch          # default (fused AdamW) | torch | plugin
    class_name: Adadelta    # any torch.optim class when backend: torch
    # name: lion            # registered plugin when backend: plugin
```

- **Built-in plugins:** Lion and GaLore. GaLore has been tested with DDP only.
- **Custom optimizers:** plugin modules listed under `plugins.optimizers` are loaded on both the driver and the workers.

### Throughput and MFU (`utils/throughput.py`, `trainer/loop.py`)

The FLOPs-per-token estimate now:
- includes the attention term
- adds the extra forward pass when activation checkpointing is on

### Documentation and configuration

- **Documentation:**
  - Adds `CODE_FLOW.md`, `CLAUDE.md` and `AGENTS.md`.
  - Extends `ARCHITECTURE.md`.
- **Example configs:** `param_qlora.yaml`, `fsdp_cpt.yaml`, `pre.yaml`, `cpt_token.yaml`, `eval.yaml`, `galore.yaml`, `lion.yaml`, `adadelta.yaml`.

## Merge notes

- **`trainer/checkpointing.py` conflicts with `main`.**
- **MFU definition:** counting the recomputed forward pass measures hardware utilization (HFU), not model FLOPs utilization (MFU) as usually defined. Consider reporting both.
- **Overlapping fixes:** `fsdp-moe-fixes` separately fixes a double division by `world_size` in `throughput.py`. Reconcile the two changes when both are merged.
- **Machine-specific files:**
  - `bhaskera-activate.sh` points to a machine-specific virtual environment (`/mnt/disk1/...`) and CUDA 12.0.
  - `scripts/submit.sh` changes the default node count to 5.
  - Revert both before merging.
- **Agent guidance files:** `main` deliberately removed `CLAUDE.md`. Decide whether `CLAUDE.md` and `AGENTS.md` should be added back.
- **QLoRA is shared with the `serve` branch.** Both branches start from the QLoRA commit (`ef99b8e`). Git applies it once, whichever branch is merged first.
