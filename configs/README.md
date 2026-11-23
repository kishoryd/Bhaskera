# configs

Starters to copy. Each file is a complete config; replace the `<project>/<username>` paths and run it with `./submit.sh <file> <nodes> <hours>`. Everything in here is yours to edit.

## Text models

| Parallelism | Config |
|---|---|
| DDP | `text/ddp.yaml` |
| FSDP2 | `text/fsdp.yaml` |
| HSDP | `text/hsdp.yaml` |
| Pipeline | `text/pipeline.yaml` |
| Tokenise first | `text/tokenize.yaml` |

## Multimodal models

| Input | Config | Tokenise first |
|---|---|---|
| Image | `multimodal/fsdp.yaml`, `ddp.yaml`, `hsdp.yaml` | `multimodal/tokenize_vision.yaml` |
| Video | `multimodal/video.yaml` | `multimodal/tokenize_vision.yaml` |
| Audio | `multimodal/audio.yaml` | `multimodal/tokenize_audio.yaml` |

Pipeline is not available for multimodal models. A row can mix inputs: list them in `data.modalities`.

## Everything else

- `config.yaml` lists every option with its default.
- `inference_turboquant.yaml` generates text with a quantised KV cache.
- Tested recipes for the supported models are in `slurm/recipes/`.

A key that does nothing, like a typo, is reported with the closest real name when the config loads. The meaning of each key is in `src/bhaskera/config.py`.
