# Branch: `custom-data`

## Summary

Adds training on datasets outside the built-in registry: any Hugging Face Hub dataset, or local files in JSONL, JSON, CSV, Parquet or plain text. Supports single-column text data and prompt/completion pairs for instruction tuning.

## Changes

| File | Change |
|---|---|
| `data/__init__.py` | `build_ray_dataset()` selects the data source: registry, `huggingface` or `local`. Adds loaders for Hugging Face and local files, prompt/completion rendering, and column validation. |
| `config.py` | Adds `data.source`, `hf_dataset`, `hf_split`, `local_path`, `local_format`, `text_column`, `prompt_column`, `completion_column` and `prompt_completion_template`. |
| `configs/config.yaml` | Documents the new keys. |
| `configs/config_custom_data.yaml` | New example config for a local dataset. |

## How it works

- **Source selection.** `data.source` chooses the loader. If it is not set, the source is inferred: `local_path` means `local`, `hf_dataset` means `huggingface`, otherwise the built-in registry.
- **Hugging Face datasets** are loaded with `datasets.load_dataset(hf_dataset, split=hf_split)` and converted to a Ray dataset.
- **Local files** are read with the matching Ray Data reader. `local_path` can be a file, a directory or a glob.
- **Text source:**
  - By default, training uses the column named by `text_column`.
  - If `prompt_column` and `completion_column` are both set, each row is rendered with `prompt_completion_template`.
- **Column check.** Before tokenization, required columns are checked against the first row. A missing column raises an error that lists the columns available.

## Usage

```yaml
data:
  source: local
  local_path: ./data/train.jsonl
  local_format: jsonl          # jsonl | json | csv | parquet | text

  # Single text column
  text_column: text

  # Or prompt/completion pairs
  # prompt_column: instruction
  # completion_column: response
  # prompt_completion_template: "### Instruction:\n{prompt}\n\n### Response:\n{completion}"
```

## Merge notes

- This branch is 34 commits behind `main`. `main` has since added its own local-file and chat-format pipeline (`data.format`, `local_chat`, persistent tokenization cache), which overlaps with this feature.
- Merging conflicts in `config.py` and `data/__init__.py`. The new sources should be integrated with the current `main` pipeline, not merged as-is.
