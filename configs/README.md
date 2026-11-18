# configs/

Configs kept after the 18 Nov 2026 clean-up. Older configs (per-node copies, profiling, Param2, Llama-2 and
experiments) were removed; they are in git history at commit `581e183` (`git show 581e183:configs/<file>`).

Model and data paths are absolute paths on this cluster
(`/scratch/nsmapplication/dlapp/wk-kishor/bhaskera/Bhaskera/...`). Change `model.name`,
`data.tokenized_path` and `data.val_tokenized_path` for another machine.

## `release/` — configs tested in this release

Each is a short test run (6–10 steps). For a real run raise `training.max_steps` and set
`data.num_workers` / `--num-workers` to 2 × nodes. Files ending `-a` / `-b` are the two halves of the
checkpoint-resume test: `-a` trains and saves, `-b` uses the same save directory and continues.

| File | Model | Strategy | Settings |
|---|---|---|---|
| `rel-ddp-qwen14b.yaml` | Qwen3-14B | DDP | batch 1 |
| `rel-fsdp-llama70b.yaml` | Llama-3.3-70B-Instruct | FSDP | batch 1 |
| `rel-fsdp-moe-qwen30b.yaml` | Qwen3-30B-A3B-Instruct-2507 | FSDP | batch 1 |
| `rel-fsdp-qwen14b.yaml` | Qwen3-14B | FSDP | batch 1 |
| `rel-hsdp-qwen14b.yaml` | Qwen3-14B | HSDP | shard 4 GPUs, batch 1 |
| `rel-pp-llama70b-a.yaml` | Llama-3.3-70B-Instruct | Pipeline | 4 stages, 16 micro-batches, 1f1b, packed, batch 1 |
| `rel-pp-llama70b-b.yaml` | Llama-3.3-70B-Instruct | Pipeline | 4 stages, 16 micro-batches, 1f1b, packed, batch 1 |
| `rel-pp-mistral123b.yaml` | Mistral-Large-Instruct-2411 | Pipeline | 8 stages, 32 micro-batches, 1f1b, packed, batch 1 |
| `rel-pp-moe-qwen235b.yaml` | Qwen3-235B-A22B-Instruct-2507 | Pipeline | 16 stages, 32 micro-batches, 1f1b, packed, batch 2 |
| `rel-pp-moe-qwen30b-a.yaml` | Qwen3-30B-A3B-Instruct-2507 | Pipeline | 4 stages, 16 micro-batches, 1f1b, packed, batch 4 |
| `rel-pp-moe-qwen30b-b.yaml` | Qwen3-30B-A3B-Instruct-2507 | Pipeline | 4 stages, 16 micro-batches, 1f1b, packed, batch 4 |
| `rel-pp-moe-qwen30b.yaml` | Qwen3-30B-A3B-Instruct-2507 | Pipeline | 4 stages, 16 micro-batches, 1f1b, packed, batch 4 |
| `rel-pp-qwen14b-1f1b-packed-b2.yaml` | Qwen3-14B | Pipeline | 4 stages, 8 micro-batches, 1f1b, packed, batch 2 |
| `rel-pp-qwen14b-gpipe-ackpt-b2.yaml` | Qwen3-14B | Pipeline | 2 stages, 8 micro-batches, gpipe, batch 2 |
| `rel-qlora-llama70b.yaml` | Llama-3.3-70B-Instruct | DDP | 4-bit QLoRA, packed, batch 2 |

Run one, from the repo root:

```bash
bhaskera-preflight --config configs/release/rel-ddp-qwen14b.yaml --num-gpus 16
sbatch --nodes=8 scripts/train.sbatch configs/release/rel-ddp-qwen14b.yaml
```

## Tokenizing

| File | Use |
|---|---|
| `tokenize_qwen3_14b_nemotron.yaml`, `tokenize_qwen3_30b_a3b_nemotron.yaml`, `tokenize_qwen3_235b_a22b_nemotron.yaml`, `tokenize_llama33_70b_nemotron.yaml`, `tokenize_mistral_large_nemotron.yaml` | Build the data caches used by the release configs |
| `local_data_tokenize_chatml.yaml` | Example for your own data: local JSONL in chat (`messages`) format. `alpaca` and `sharegpt` formats are also supported (`data.format`). |

```bash
bhaskera-tokenize --config configs/tokenize_qwen3_14b_nemotron.yaml --split both
```

## Inference

`inference_turboquant.yaml`, `inference_speculative.yaml`, `inference_param2.yaml`: examples for `bhaskera-infer`
(TurboQuant KV cache, speculative decoding, Param2). Not re-tested in this release.

## `tests/`

`llama33_70b_pipeline_4node.yaml`: the base config used by `scripts/pp_validate.py` and
`src/bhaskera/trainer/tests/test_pipeline.py`. Keep it.
