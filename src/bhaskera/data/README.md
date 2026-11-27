# data

Turns your files into training batches: a chat format, a tokenised cache on disk, and a Ray Data pipeline that reads it.

## What you control

In the `data` section of a recipe:

| Key | Meaning |
|---|---|
| `name` | `local` for your own files, or a built-in: `ultrachat`, `openassistant`, `redpajama` |
| `train_path`, `val_path` | A file, a folder, or a glob of `.jsonl`, `.json`, `.csv`, `.tsv`, `.parquet`, `.arrow` or plain text `.txt` (one kind per path). Plain text: `format_options: {text_split: file|paragraphs|lines, chunk: true}` |
| `format` | How a row becomes text: `chatml`, `alpaca`, `sharegpt`, or your own. `format_options` passes settings to it |
| `seq_len` | Tokens per row |
| `cache_dir`, `overwrite_cache` | Where the tokenised cache goes, and whether to redo it |
| `tokenized_path`, `val_tokenized_path` | The cache to train from (the two paths `bhaskera-tokenize` prints) |
| `runtime_packing`, `packing_window_rows` | Pack several examples into each row so no compute goes on padding. Needs `attn_impl: flash_attention_2` |
| `num_workers`, `tokenize_batch_size`, `tokenize_compression`, `prefetch_batches` | Speed settings |

Row shapes: `chatml` expects `messages: [{role, content}]`, `alpaca` expects `instruction`, optional `input`, and `output`, `sharegpt` expects `conversations: [{from, value}]`.

To add a format of your own:

```python
from bhaskera.data.formats import register_format

@register_format("my_custom")
def render(row, tokenizer, options):
    return f"### Q: {row['question']}\n### A: {row['answer']}"
```

then set `data.format: my_custom`. A new dataset is a module that applies `@register_raw` and `@register` from `registry.py`; nothing else needs editing.

## What ExaAI does itself

- **The cache.** Tokenising runs once and writes Parquet files plus a `metadata.json` (model, `seq_len`, dataset, format, row count). The folder name is a hash of those settings, so a changed setting never reuses a stale cache. At load time a different `seq_len` is an error, because rows are never re-padded; a different model is only a warning, since models that share a tokenizer can share a cache.
- **Rendering.** The `chatml` format uses the model's own chat template when it has one and falls back to plain ChatML markup. Labels are the input ids with padding masked to `-100`; empty rows are dropped.
- **Packing.** First-fit-decreasing within a window of rows. Each example's positions restart at 0 and its first label is masked, so nothing is predicted across examples. In one test it cut padding from 42.7% to 0.1%. Only the training split is packed.
- **Partitions.** The dataset is split into a multiple of the worker count, at least four per worker, so no shard is empty or uneven.
- **Images, video and audio.** See `docs/multimodal/README.md`; the cache keeps token ids and file paths, never pixels.
