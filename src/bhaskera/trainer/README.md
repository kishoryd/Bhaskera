# trainer

The training loop: optimizer, learning-rate schedule, MoE losses, evaluation and checkpointing. It gets a model that is already wrapped for its parallelism and a Ray Data shard, and runs.

## What you control

Under `training` in a recipe:

| Key | Meaning |
|---|---|
| `batch_size`, `grad_accum` | Micro-batch per GPU, and micro-batches per optimizer step |
| `lr`, `weight_decay`, `warmup_steps` | AdamW and the schedule |
| `max_steps`, `num_epochs` | Whichever is reached first stops the run |
| `grad_clip` | Gradient clipping threshold |
| `eval_steps`, `eval_batch_size` | How often to evaluate on the validation split, and its batch size |
| `seed`, `deterministic` | Seeding; the rank is added to the seed |
| `startup_timeout_s` | Stop a worker that has not finished a step this long after start (default 3600, `0` turns it off) |

Under `moe`: the weights of the load-balancing and router z-losses, and `log_expert_utilization`. Under `monitoring.metrics`: `step_timing` (default on) and `straggler_every_n_steps` (default 10).

## What ExaAI does itself

- **The loop.** Builds the optimizer and scheduler, resumes from the latest checkpoint if there is one, then runs epochs until `max_steps`. Each micro-step does forward and backward; the last one of each optimizer step does the gradient sync, clipping, the optimizer and scheduler step, logging, and, when due, a checkpoint.
- **Gradient accumulation.** Gradient sync is switched off for every micro-step but the last, for FSDP2 and for DDP (not with DDP's `static_graph`), so the all-reduce happens once per optimizer step.
- **Optimizer.** AdamW with betas (0.9, 0.95) and a fused kernel on CUDA. Matrices get weight decay; biases and norm gains do not. The schedule is a linear warmup into cosine decay.
- **Precision.** Under DDP the loop wraps the forward in `torch.autocast` with the model's dtype. Under FSDP the sharding policy handles precision, so the loop adds nothing.
- **MoE.** The auxiliary loss is taken from the model if it computes one, and otherwise built from the router logits (a Switch-Transformer-style balance loss). It works from the model's profile, never its name, and handles the different router-logit shapes of Mixtral, Qwen-MoE and Param2. Expert load and imbalance are logged.
- **Step timing.** Each step is split into data wait, compute, optimizer and other, measured with CUDA events at the existing sync point, so it adds no overhead. Every few steps rank 0 logs the slowest rank and its host, and warns when it is both 25% and a second slower than the median.
- **Watchdog.** Workers log their start-up phase. If no step completes before the timeout, the worker prints every thread's stack and exits with code 86.
- **Throughput and MFU.** Tokens per second, step time and MFU are logged every step (see `utils`).
- **Checkpoints.** Wrappers around `distributed/checkpoint.py`: save with the newest-N rule, and resume with the schedule replayed to the saved step.
