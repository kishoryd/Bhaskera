# Branch: `feature/cpt-eval-bench`

## Summary

Adds continual pre-training (CPT), benchmark evaluation and training benchmarks for dense, Mixture-of-Experts and agentic (tool-calling) models.

- **CPT:** documents are packed into full-length rows with no padding. Works for raw text, chat data and tool-use transcripts.
- **`bhaskera-eval`:** scores standard benchmarks and an agentic tool-calling benchmark on the same FSDP/DDP path as training, so any model that can be trained can be evaluated, including models too large for one node.
- **Training runs** can evaluate benchmarks periodically and write a throughput/memory benchmark file at the end.

Depends on `fsdp-moe-fixes`; merge that branch first.

## Changes

### Continual pre-training data — `data.cpt: true`

| | Existing (SFT) layout | CPT layout |
|---|---|---|
| Rows | One example per row, padded to `seq_len` | Documents concatenated (EOS after each) and cut into rows of exactly `seq_len` |
| Training tokens | Real tokens only; padding is masked | Every token |
| Long documents | Truncated | Continue into the next row |

- **One tokenizer stage for all data:** CPT is a mode of the existing stage, so raw text, `chatml`/`alpaca` data and the new `agentic` format all produce CPT rows.
- **Separate caches:** the layout is part of the cache key, so CPT and non-CPT caches of the same corpus never collide. Existing cache keys are unchanged.
- **Specials handled correctly:** text rendered by a chat template keeps its own special tokens, and none are added a second time (for example, no double BOS for Llama).

```yaml
data:
  name: local
  cpt: true
  seq_len: 4096
  train_path: /path/to/corpus.jsonl      # {"text": ...} per line
```

### Agentic format — `data.format: agentic`

Tool-use conversations (`messages` with tool calls and tool results, plus `tools`) are rendered through the model's own chat template, so each model trains and is evaluated on its native tool-call syntax:

| Model family | Tool-call syntax produced |
|---|---|
| Qwen3 | `<tool_call>{"name": …, "arguments": …}</tool_call>` |
| Llama 3.3 | `{"name": …, "parameters": …}` |
| Mixtral | `[TOOL_CALLS] [{"name": …, "arguments": …, "id": …}]` |
| gpt-oss | `to=functions.<name><|channel|>commentary json<|message|>{…}` |

- **Input handling:** tools and messages may be stored as JSON strings, which avoids Arrow schema merging. BFCL's `"type": "dict"` schemas are normalised to `"object"`.
- **Template options:** `format_options.template_kwargs` passes variables to the template, for example `{enable_thinking: false}` for Qwen3.
- **Templates without tool support:** Mistral-Large-Instruct-2411's Hugging Face template drops tool definitions. This is detected and reported instead of silently producing wrong data.

### Evaluation — `bhaskera-eval`

```bash
python scripts/fetch_eval_data.py --out data1/eval            # once, on a node with internet
bhaskera-eval --config configs/eval_qwen3_30b_a3b.yaml --num-workers 2
bhaskera-eval --config <training config> --checkpoint <save_dir>/step_0001000 --tasks hellaswag,bfcl_multiple
```

| Task | Metric(s) | Prompt format |
|---|---|---|
| `hellaswag`, `arc_easy`, `arc_challenge` | `acc`, `acc_norm` | lm-evaluation-harness defaults |
| `mmlu` | `acc` | harness default (subject header, A–D) |
| `winogrande` | `acc` | harness partial evaluation |
| `wikitext2` | `word_perplexity`, `bits_per_byte` | harness detokenizer, disjoint windows |
| `bfcl_simple` | `exact_match`, `gold_call_nll` | model's own tool-calling template |
| `bfcl_multiple` | `exact_match`, `tool_selection_acc`, `gold_call_nll` | model's own tool-calling template |

How it works:
- **Log-likelihood scoring, no text generation.** A forward pass gives the log-probability of each candidate answer. A tool call counts as exact when every one of its tokens is the model's top choice, which is exactly when greedy decoding would produce it.
- **FSDP-safe.** Under FSDP every forward pass is a collective, so each rank runs the same number of batches; the padding requests are discarded.
- **Memory-efficient batching.** Rows are left-padded with explicit position IDs, and only the positions being scored are projected onto the vocabulary (`logits_to_keep`).
- **Offline data.** Benchmark data is read from the local snapshot written by `scripts/fetch_eval_data.py`, never from the Hub.
- **Training checkpoints** are loaded with `distributed.load_model_weights`. DCP reshards on load, so the evaluation GPU count may differ from training's.
- **Output:** `<evaluation.output_dir>/<run_name>.json`, plus a markdown table. Each task records its scoring throughput (`tokens_per_sec`).

`exact_match` requires one canonical serialisation of the gold call, so it is a lower bound on BFCL's official AST accuracy.

### Benchmarks during training

| Setting | Effect |
|---|---|
| `evaluation.tasks` | Evaluated at the end of training; results logged as `bench/<task>/<metric>` |
| `evaluation.every_n_steps` | Also evaluated every N steps; one JSON per evaluation under `<output_dir>/<run_name>_steps/` |
| `monitoring.metrics.benchmark_file` | JSON at the end of the run: median tokens/s, MFU and step time after warm-up; peak GPU memory (maximum over ranks) |

The step clock is reset after every evaluation, so tokens/s and MFU measure training only. This also applies to the existing validation-loss evaluation.

### Files

| File | Change |
|---|---|
| `data/cpt.py` | New. Packing function. |
| `data/formats/agentic.py` | New. Tool-use rendering helpers. |
| `data/formats/builtins.py` | `agentic` format registration |
| `data/tokenize.py` | CPT mode in the tokenizer stage; layout in cache key and metadata |
| `evaluation/` | New. `scoring.py`, `tasks.py`, `runner.py` |
| `launcher/evaluate.py` | New. `bhaskera-eval` |
| `trainer/benchmark.py` | New. Training benchmark JSON |
| `trainer/loop.py` | Periodic and final evaluation, benchmark recording, step-clock reset after evaluation |
| `distributed/checkpoint.py` | `load_model_weights` (model-only checkpoint load) |
| `config.py` | `data.cpt`, `evaluation.*`, `monitoring.metrics.benchmark_file` |
| `scripts/fetch_eval_data.py` | New. Offline benchmark snapshot |
| `configs/` | `eval_qwen3_14b`, `eval_qwen3_30b_a3b`, `cpt_tokenize_wikitext103_*`, `cpt_train_wikitext103_*` |

## Validation

### Unit tests (CPU)

```bash
python -m bhaskera.evaluation.tests.test_scoring
BHASKERA_TEST_TOKENIZER=<model dir> python -m bhaskera.data.tests.test_cpt
BHASKERA_TEST_TOKENIZER=<model dir> BHASKERA_EVAL_DATA=data1/eval python -m bhaskera.evaluation.tests.test_tasks
```

| Test | Result |
|---|---|
| Batched scores vs. each request scored alone, unpadded (Qwen3, Llama, Qwen3-MoE, Mixtral) | Identical (max difference 0.0); a deliberate misalignment bug is detected |
| Two ranks, uneven request count | Equal forwards per rank; results identical to one process |
| CPT packing | Exact token stream, one EOS per document, full rows, cache key isolation |
| Prompt formats | HellaSwag example 0 matches the harness query exactly |

### Benchmark evaluation (A100, 2 GPUs, first 200 examples per task, 0-shot)

| Task | Qwen3-14B (dense) | Qwen3-30B-A3B (MoE) |
|---|---|---|
| HellaSwag `acc_norm` | 0.665 | 0.690 |
| ARC-Easy / ARC-Challenge `acc_norm` | 0.825 / 0.590 | 0.840 / 0.620 |
| MMLU `acc` | 0.710 | 0.775 |
| WinoGrande `acc` | 0.750 | 0.720 |
| WikiText-2 word perplexity | 18.01 | 14.09 |
| BFCL simple `exact_match` | 0.760 | 0.750 |
| BFCL multiple `tool_selection_acc` / `exact_match` | 0.990 / 0.665 | 0.990 / 0.645 |

### Continual pre-training (WikiText-103, 124M tokens packed at 4096; 20 steps on 4 GPUs)

| | Qwen3-14B, full-parameter | Qwen3-30B-A3B, LoRA |
|---|---|---|
| Held-out WikiText-2 perplexity: before → step 10 → step 20 | 18.01 → 15.62 → 15.44 | 14.09 → 13.38 → 13.12 |
| Validation loss, step 10 → 20 | 1.881 → 1.872 | 1.802 → 1.784 |
| Median throughput / MFU | 1,622 tok/s / 11.5% | 641 tok/s / 1.0% |
| Peak GPU memory | 38.6 GiB | 23.6 GiB |

MoE MFU is limited by Hugging Face's per-expert loop, which is addressed separately on `perf/grouped-experts`.

### Checkpoint evaluation and benchmark accounting (Qwen3-14B)

| Check | Result |
|---|---|
| Checkpoint round trip: 6 CPT steps on 4 GPUs, saved, then evaluated with `--checkpoint` on 2 GPUs | WikiText-2 perplexity 16.4774 in training vs. 16.4775 from the checkpoint (base model: 18.01) |
| Evaluation every 4 steps | Steps immediately after an evaluation run at 1,625–1,629 tok/s, the same as other steps; the benchmark file excludes 3 warm-up steps |

## Merge notes

- **Overlap with `qlora`.** The `qlora` branch also adds CPT and HellaSwag/ARC evaluation. This branch's evaluation reads a local data snapshot (compute nodes are offline) and keeps FSDP ranks in lockstep. The two should be reconciled rather than merged side by side.
- **New pipeline dependency.** `trainer/loop.py` now imports `transformers.AutoTokenizer` when `evaluation.tasks` is set.
