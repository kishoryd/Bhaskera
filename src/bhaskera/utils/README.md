# utils

What a run reports: GPU and host readings, throughput and MFU, and the trackers that receive them.

## What you control

Under `logging` in a recipe:

| Key | Meaning |
|---|---|
| `tracker` | `mlflow`, `wandb`, `ray`, a list of them, or `none`. The Ray Dashboard is added by default |
| `project`, `run_name` | Names in the tracker |
| `log_gpu_every_n_steps` | How often GPU and host readings join the metrics |
| `mlflow_metrics` | `essential` (about 20 curves, default), `detailed` or `full` (also per GPU and per node) |
| `mlflow_log_adapter`, `mlflow_register_model` (or later, `python -m bhaskera.launcher.register add`) | Store the trained LoRA adapter (PEFT format) with the run, and optionally register it |
| `mlflow_eval_report`, `mlflow_eval_samples`, `mlflow_tracing` | The evaluation table, example predictions, and traces |
| `mlflow_tracking_uri` | Where MLflow writes (default `mlflow-runs` in your run folder, see the main README) |

Under `monitoring.metrics`: `throughput`, `peak_tflops_per_gpu` (default 312, an A100 in bf16, so set it for other GPUs), `node_stats_every_n_steps` and `node_stats_per_node_max`. Weights & Biases is optional and imported only when you ask for it.

## What ExaAI does itself

- **GPU and host readings** (`system_stats.py`). One cheap call per step gives per-GPU utilisation, memory, temperature, power and clocks, PCIe and NVLink traffic, ECC errors and throttle reasons (through pynvml), plus CPU, memory, disk and network (through psutil). A missing library just drops that part. Rates such as disk and network bytes per second are computed against the previous call.
- **Throughput** (`throughput.py`). Step time, tokens per second and MFU. MFU is `6 × parameters × tokens / step time` over the GPU's peak, using all the model's parameters (LoRA's FLOPs still flow through the frozen base). The attention term is left out, which changes it by under 5%. The first few steps are skipped in the moving average. For multimodal models the encoder parameters are not counted, so MFU covers the language model only.
- **Trackers** (`loggers/`). `build_logger` makes one logger per backend and wraps them so that a tracker that fails or loses its connection never stops training. `ray` runs on every rank; `wandb` and `mlflow` on rank 0 only. MLflow is push-based into a file store, so a login node that shares your home folder can show a live run without any network setup.
- **Metric names** are slash-separated and stable: `gpu/<i>/util_pct`, `cpu/system_pct`, `throughput/tokens_per_sec`, `throughput/mfu_pct`, `model/trainable_params`, `expert/imbalance_ratio`, and so on.
