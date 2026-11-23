# configs/

Starter configs. Each file is complete and loads with `bhaskera.config.load_config(path)`: copy one, replace the paths,
and run it.

| | Text | Multimodal (image, video, audio) |
|---|---|---|
| DDP | `text/ddp.yaml` | `multimodal/ddp.yaml` |
| FSDP2 | `text/fsdp.yaml` | `multimodal/fsdp.yaml` |
| HSDP | `text/hsdp.yaml` | `multimodal/hsdp.yaml` |
| Pipeline | `text/pipeline.yaml` | not supported for multimodal models |
| Tokenise first | `text/tokenize.yaml` | `multimodal/tokenize_vision.yaml` (image, video), `multimodal/tokenize_audio.yaml` |

`config.yaml` lists every option with its default. `inference_turboquant.yaml` generates text with a quantised KV cache.

Ready-to-run recipes for the supported models are in `slurm/recipes/`.

A key that does nothing (a typo, a misplaced option) is reported with the closest real name when the config is loaded.
The meaning of each key is in `src/bhaskera/config.py`.
