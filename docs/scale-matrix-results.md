# Scale-matrix results

125 test runs: 5 models × 5 parallelism strategies × 5 node counts (8, 16, 32, 64, 128 nodes).
Cluster: NVIDIA A100 80 GB, 2 GPUs per node, InfiniBand, no GPUDirect RDMA. Run on 17–18 November 2026.
Every run trains for 10 steps, then evaluates once on about 256 held-out conversations. Totals: **105 passed, 20 failed**.

| Column | Meaning |
|---|---|
| tok/s | Real (non-padding) tokens per second, all GPUs together: total work ÷ total time over steps 3–10 |
| MFU | Share of the GPUs' peak maths speed used for model work (MoE models count active parameters only) |
| Peak GPU mem | Highest memory used on any GPU, out of 80 GB |
| Actual run time | Wall-clock time of the whole test job: startup, model loading, 10 steps and eval |
| Estimated time to train | Projected time for one pass over a 1.24-billion-token training set, from measured speed; startup excluded |
| Train loss | Mean of steps 8–10 (noisy over so few steps) |
| Eval loss / acc | After step 10; accuracy is next-token accuracy. These show that strategies agree, not model quality |
| `*` | At least one step took over 1.5× the median step (slow node or network) |
| attempt N | The run was retried after a hardware or network fault |

**The 20 failures are limits, not bugs.** DDP on Llama-70B, Mistral-123B and Qwen3-235B (15 runs): each GPU holds a
full bf16 copy of the model, which is 140–470 GB against 80 GB. Qwen3-235B with QLoRA (5 runs): even in 4-bit the
model is about 118 GB per GPU.

**MoE models are only fast with the Pipeline strategy.** With FSDP, HSDP, DDP and QLoRA they run at 0.2–1.7% MFU
(see *Known limitations* in the README).

| Model | Parallelism | Nodes | Status | tok/s | MFU | Peak GPU mem | Actual run time (10 steps + eval) | Estimated time to train (1 epoch) | Train loss | Eval loss | Eval acc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-14B | FSDP | 8 | ✅ pass | 9,139 | 11.8% | 12 / 80 GB | 7m 25s | 37.7 h | 1.711 | 1.5538 | 66.00% |
| Qwen3-14B | FSDP | 16 | ✅ pass | 16,991 | 11.0% | 11 / 80 GB | 7m 03s | 20.3 h | 1.276 | 1.5893 | 65.78% |
| Qwen3-14B | FSDP | 32 | ✅ pass | 24,048 | 7.8% | 11 / 80 GB | 7m 54s | 14.3 h | 1.448 | 1.5866 | 65.47% |
| Qwen3-14B | FSDP | 64 | ✅ pass | 39,151 | 6.8% | 10 / 80 GB | 8m 12s | 8.8 h | 1.439 | 1.5802 | 65.24% |
| Qwen3-14B | FSDP | 128 | ✅ pass | 57,997 | 4.7% | 10 / 80 GB | 9m 54s | 5.9 h | 1.419 | 1.5680 | 66.07% |
| Qwen3-14B | HSDP | 8 | ✅ pass | 9,893 | 12.8% | 24 / 80 GB | 7m 37s | 34.8 h | 0.827 | 1.5542 | 65.99% |
| Qwen3-14B | HSDP | 16 | ✅ pass | 19,015 | 12.3% | 24 / 80 GB | 7m 01s | 18.1 h | 1.526 | 1.5894 | 65.77% |
| Qwen3-14B | HSDP | 32 | ✅ pass | 18,518* | 6.0% | 24 / 80 GB | 8m 55s | 18.6 h | 1.336 | 1.5783 | 65.72% |
| Qwen3-14B | HSDP | 64 | ✅ pass | 75,659 | 12.2% | 24 / 80 GB | 8m 15s | 4.5 h | 1.349 | 1.5748 | 65.24% |
| Qwen3-14B | HSDP | 128 | ✅ pass | 93,861 | 7.6% | 24 / 80 GB | 10m 30s | 3.7 h | 1.573 | 1.5650 | 66.10% |
| Qwen3-14B | DDP | 8 | ✅ pass | 24,427 | 31.6% | 32 / 80 GB | 6m 02s | 14.1 h | 1.558 | 1.5531 | 66.00% |
| Qwen3-14B | DDP | 16 | ✅ pass | 48,106 | 31.1% | 32 / 80 GB | 6m 35s | 7.2 h | 1.135 | 1.5890 | 65.78% |
| Qwen3-14B | DDP | 32 | ✅ pass | 95,702 | 30.9% | 32 / 80 GB | 6m 07s | 3.6 h | 1.751 | 1.5773 | 65.71% |
| Qwen3-14B | DDP | 64 | ✅ pass | 187,941 | 30.4% | 32 / 80 GB | 6m 47s | 1.8 h | 1.505 | 1.5891 | 64.75% |
| Qwen3-14B | DDP | 128 | ✅ pass | 385,299 | 31.1% | 32 / 80 GB | 8m 10s | 54 min | 1.321 | 1.5439 | 66.29% |
| Qwen3-14B | QLoRA+DDP | 8 | ✅ pass | 27,463 | 35.5% | 16 / 80 GB | 6m 52s | 12.5 h | 1.377 | 1.5705 | 65.57% |
| Qwen3-14B | QLoRA+DDP | 16 | ✅ pass | 54,677 | 35.4% | 16 / 80 GB | 6m 57s | 6.3 h | 1.236 | 1.6061 | 65.37% |
| Qwen3-14B | QLoRA+DDP | 32 | ✅ pass | 109,032 | 35.2% | 16 / 80 GB | 7m 09s | 3.2 h | 1.224 | 1.5941 | 65.32% |
| Qwen3-14B | QLoRA+DDP | 64 | ✅ pass | 191,301* | 30.9% | 16 / 80 GB | 8m 01s | 1.8 h | 1.603 | 1.5966 | 64.83% |
| Qwen3-14B | QLoRA+DDP | 128 | ✅ pass | 424,899 | 34.4% | 16 / 80 GB | 9m 14s | 49 min | 1.463 | 1.5842 | 65.52% |
| Qwen3-14B | Pipeline | 8 | ✅ pass | 18,017 | 23.2% | 16 / 80 GB | 7m 32s | 19.1 h | 1.453 | 1.4744 | 66.00% |
| Qwen3-14B | Pipeline | 16 | ✅ pass | 36,033 | 23.2% | 16 / 80 GB | 7m 15s | 9.6 h | 1.411 | 1.5202 | 65.78% |
| Qwen3-14B | Pipeline | 32 | ✅ pass | 72,285 | 23.3% | 16 / 80 GB | 7m 34s | 4.8 h | 1.459 | 1.5060 | 65.64% |
| Qwen3-14B | Pipeline | 64 | ✅ pass | 143,441 | 23.1% | 16 / 80 GB | 7m 43s | 2.4 h | 1.436 | 1.5287 | 65.24% |
| Qwen3-14B | Pipeline | 128 | ✅ pass (attempt 3) | 284,960 | 22.9% | 16 / 80 GB | 8m 50s | 1.2 h | 1.444 | 1.5129 | 65.31% |
| Qwen3-30B-A3B (MoE) | FSDP | 8 | ✅ pass | 1,873 | 0.6% | 13 / 80 GB | 16m 41s | 7.7 days | 1.398 | 1.7065 | 65.85% |
| Qwen3-30B-A3B (MoE) | FSDP | 16 | ✅ pass | 2,289* | 0.4% | 11 / 80 GB | 19m 50s | 6.3 days | 1.349 | 1.6829 | 66.11% |
| Qwen3-30B-A3B (MoE) | FSDP | 32 | ✅ pass | 4,398 | 0.4% | 10 / 80 GB | 19m 59s | 3.3 days | 1.438 | 1.7155 | 66.03% |
| Qwen3-30B-A3B (MoE) | FSDP | 64 | ✅ pass | 7,072 | 0.3% | 10 / 80 GB | 21m 51s | 2.0 days | 1.690 | 1.7200 | 65.85% |
| Qwen3-30B-A3B (MoE) | FSDP | 128 | ✅ pass | 10,316 | 0.2% | 10 / 80 GB | 26m 45s | 33.4 h | 1.352 | 1.6519 | 66.43% |
| Qwen3-30B-A3B (MoE) | HSDP | 8 | ✅ pass | 2,146 | 0.7% | 38 / 80 GB | 16m 56s | 6.7 days | 1.208 | 1.7067 | 65.85% |
| Qwen3-30B-A3B (MoE) | HSDP | 16 | ✅ pass | 3,916 | 0.6% | 38 / 80 GB | 16m 04s | 3.7 days | 1.451 | 1.6825 | 66.09% |
| Qwen3-30B-A3B (MoE) | HSDP | 32 | ✅ pass | 6,873 | 0.5% | 38 / 80 GB | 17m 21s | 2.1 days | 1.492 | 1.7329 | 65.93% |
| Qwen3-30B-A3B (MoE) | HSDP | 64 | ✅ pass | 11,728 | 0.4% | 38 / 80 GB | 18m 24s | 29.4 h | 1.865 | 1.7180 | 65.86% |
| Qwen3-30B-A3B (MoE) | HSDP | 128 | ✅ pass | 18,550 | 0.4% | 38 / 80 GB | 20m 57s | 18.6 h | 1.434 | 1.6368 | 66.96% |
| Qwen3-30B-A3B (MoE) | DDP | 8 | ✅ pass | 4,692 | 1.5% | 61 / 80 GB | 10m 58s | 3.1 days | 1.332 | 1.7065 | 65.83% |
| Qwen3-30B-A3B (MoE) | DDP | 16 | ✅ pass | 9,117 | 1.4% | 61 / 80 GB | 11m 19s | 37.8 h | 1.269 | 1.6826 | 66.11% |
| Qwen3-30B-A3B (MoE) | DDP | 32 | ✅ pass | 16,847 | 1.4% | 61 / 80 GB | 11m 10s | 20.4 h | 0.957 | 1.7152 | 66.06% |
| Qwen3-30B-A3B (MoE) | DDP | 64 | ✅ pass | 33,914 | 1.3% | 61 / 80 GB | 11m 29s | 10.2 h | 1.747 | 1.7168 | 65.85% |
| Qwen3-30B-A3B (MoE) | DDP | 128 | ✅ pass | 66,776 | 1.3% | 61 / 80 GB | 14m 14s | 5.2 h | 1.513 | 1.6178 | 67.18% |
| Qwen3-30B-A3B (MoE) | QLoRA+DDP | 8 | ✅ pass | 5,314 | 1.7% | 23 / 80 GB | 14m 07s | 2.7 days | 1.535 | 1.7119 | 65.48% |
| Qwen3-30B-A3B (MoE) | QLoRA+DDP | 16 | ✅ pass | 10,426 | 1.7% | 23 / 80 GB | 13m 10s | 33.0 h | 1.517 | 1.6858 | 65.71% |
| Qwen3-30B-A3B (MoE) | QLoRA+DDP | 32 | ✅ pass | 20,250 | 1.6% | 23 / 80 GB | 13m 39s | 17.0 h | 1.388 | 1.7233 | 65.57% |
| Qwen3-30B-A3B (MoE) | QLoRA+DDP | 64 | ✅ pass | 31,706* | 1.3% | 23 / 80 GB | 15m 35s | 10.9 h | 1.520 | 1.7488 | 65.69% |
| Qwen3-30B-A3B (MoE) | QLoRA+DDP | 128 | ✅ pass | 79,210 | 1.6% | 23 / 80 GB | 15m 42s | 4.3 h | 1.389 | 1.6498 | 66.39% |
| Qwen3-30B-A3B (MoE) | Pipeline | 8 | ✅ pass | 28,108 | 8.9% | 25 / 80 GB | 11m 57s | 12.3 h | 1.495 | 1.5312 | 65.86% |
| Qwen3-30B-A3B (MoE) | Pipeline | 16 | ✅ pass | 56,190 | 9.0% | 25 / 80 GB | 11m 45s | 6.1 h | 1.486 | 1.5459 | 66.11% |
| Qwen3-30B-A3B (MoE) | Pipeline | 32 | ✅ pass | 111,384 | 8.8% | 25 / 80 GB | 12m 03s | 3.1 h | 1.481 | 1.5777 | 65.90% |
| Qwen3-30B-A3B (MoE) | Pipeline | 64 | ✅ pass | 219,829 | 8.7% | 25 / 80 GB | 12m 25s | 1.6 h | 1.491 | 1.5334 | 66.76% |
| Qwen3-30B-A3B (MoE) | Pipeline | 128 | ✅ pass | 386,473 | 7.7% | 25 / 80 GB | 14m 18s | 53 min | 1.475 | 1.5063 | 67.07% |
| Llama-3.3-70B | FSDP | 8 | ✅ pass | 1,863 | 11.5% | 27 / 80 GB | 19m 01s | 7.7 days | 1.484 | 1.7359 | 63.55% |
| Llama-3.3-70B | FSDP | 16 | ✅ pass | 3,748 | 11.2% | 22 / 80 GB | 17m 50s | 3.8 days | 1.544 | 1.7242 | 63.42% |
| Llama-3.3-70B | FSDP | 32 | ✅ pass | 7,365 | 11.0% | 20 / 80 GB | 16m 23s | 46.6 h | 1.688 | 1.6946 | 64.03% |
| Llama-3.3-70B | FSDP | 64 | ✅ pass | 13,346 | 10.7% | 19 / 80 GB | 17m 00s | 25.7 h | 1.628 | 1.6840 | 63.62% |
| Llama-3.3-70B | FSDP | 128 | ✅ pass | 16,485 | 6.2% | 19 / 80 GB | 21m 41s | 20.8 h | 1.367 | 1.7088 | 64.30% |
| Llama-3.3-70B | HSDP | 8 | ✅ pass | 2,013 | 12.1% | 51 / 80 GB | 19m 29s | 7.1 days | 1.771 | 1.7362 | 63.54% |
| Llama-3.3-70B | HSDP | 16 | ✅ pass | 3,965 | 11.9% | 51 / 80 GB | 18m 23s | 3.6 days | 1.677 | 1.7225 | 63.42% |
| Llama-3.3-70B | HSDP | 32 | ✅ pass | 7,885 | 11.9% | 51 / 80 GB | 17m 43s | 43.6 h | 1.432 | 1.6945 | 64.04% |
| Llama-3.3-70B | HSDP | 64 | ✅ pass | 15,639 | 11.8% | 51 / 80 GB | 17m 53s | 22.0 h | 2.146 | 1.7031 | 63.46% |
| Llama-3.3-70B | HSDP | 128 | ✅ pass | 18,750 | 7.0% | 51 / 80 GB | 40m 10s | 18.3 h | 1.570 | 1.6894 | 64.37% |
| Llama-3.3-70B | DDP | 8 | ❌ GPU OOM |  |  |  | 14m 53s |  |  |  |  |
| Llama-3.3-70B | DDP | 16 | ❌ GPU OOM |  |  |  | 14m 45s |  |  |  |  |
| Llama-3.3-70B | DDP | 32 | ❌ GPU OOM |  |  |  | 16m 14s |  |  |  |  |
| Llama-3.3-70B | DDP | 64 | ❌ GPU OOM |  |  |  | 16m 19s |  |  |  |  |
| Llama-3.3-70B | DDP | 128 | ❌ GPU OOM |  |  |  | 17m 44s |  |  |  |  |
| Llama-3.3-70B | QLoRA+DDP | 8 | ✅ pass | 6,508 | 39.1% | 55 / 80 GB | 16m 43s | 2.2 days | 1.300 | 1.7539 | 62.99% |
| Llama-3.3-70B | QLoRA+DDP | 16 | ✅ pass | 12,977 | 39.0% | 55 / 80 GB | 16m 27s | 26.5 h | 1.346 | 1.7436 | 62.79% |
| Llama-3.3-70B | QLoRA+DDP | 32 | ✅ pass | 25,877 | 38.9% | 55 / 80 GB | 16m 27s | 13.3 h | 1.571 | 1.7204 | 63.40% |
| Llama-3.3-70B | QLoRA+DDP | 64 | ✅ pass | 51,809 | 38.9% | 55 / 80 GB | 18m 20s | 6.6 h | 1.504 | 1.7251 | 62.89% |
| Llama-3.3-70B | QLoRA+DDP | 128 | ✅ pass | 103,481 | 38.8% | 55 / 80 GB | 19m 56s | 3.3 h | 1.471 | 1.7256 | 62.76% |
| Llama-3.3-70B | Pipeline | 8 | ✅ pass | 5,166 | 31.0% | 44 / 80 GB | 16m 56s | 2.8 days | 1.601 | 1.7306 | 63.55% |
| Llama-3.3-70B | Pipeline | 16 | ✅ pass | 10,194 | 30.6% | 44 / 80 GB | 15m 27s | 33.7 h | 1.556 | 1.7212 | 63.42% |
| Llama-3.3-70B | Pipeline | 32 | ✅ pass (attempt 3) | 20,356 | 30.5% | 44 / 80 GB | 15m 02s | 16.9 h | 1.569 | 1.6924 | 64.06% |
| Llama-3.3-70B | Pipeline | 64 | ✅ pass | 40,706 | 30.5% | 44 / 80 GB | 15m 09s | 8.4 h | 1.560 | 1.7118 | 63.11% |
| Llama-3.3-70B | Pipeline | 128 | ✅ pass | 74,723* | 28.1% | 44 / 80 GB | 17m 58s | 4.6 h | 1.553 | 1.7075 | 63.66% |
| Mistral-Large-123B | FSDP | 8 | ✅ pass | 1,082 | 11.3% | 36 / 80 GB | 30m 30s | 14.2 days | 1.404 | 1.2785 | 71.97% |
| Mistral-Large-123B | FSDP | 16 | ✅ pass | 2,207 | 11.5% | 29 / 80 GB | 26m 48s | 7.0 days | 1.137 | 1.2735 | 71.61% |
| Mistral-Large-123B | FSDP | 32 | ✅ pass | 4,326 | 11.2% | 26 / 80 GB | 26m 25s | 3.5 days | 1.260 | 1.2465 | 71.44% |
| Mistral-Large-123B | FSDP | 64 | ✅ pass | 8,299 | 10.8% | 24 / 80 GB | 25m 29s | 44.4 h | 1.091 | 1.2342 | 71.90% |
| Mistral-Large-123B | FSDP | 128 | ✅ pass | 9,712 | 6.3% | 23 / 80 GB | 33m 10s | 37.9 h | 1.197 | 1.2395 | 71.17% |
| Mistral-Large-123B | HSDP | 8 | ✅ pass | 1,141 | 11.8% | 50 / 80 GB | 34m 42s | 13.4 days | 0.943 | 1.2787 | 71.98% |
| Mistral-Large-123B | HSDP | 16 | ✅ pass | 2,267 | 11.8% | 50 / 80 GB | 27m 03s | 6.8 days | 1.335 | 1.2743 | 71.63% |
| Mistral-Large-123B | HSDP | 32 | ✅ pass (attempt 2) | 4,533 | 11.8% | 50 / 80 GB | 26m 01s | 3.4 days | 0.990 | 1.2460 | 71.46% |
| Mistral-Large-123B | HSDP | 64 | ✅ pass | 8,971 | 11.6% | 50 / 80 GB | 27m 02s | 41.0 h | 1.038 | 1.2542 | 71.60% |
| Mistral-Large-123B | HSDP | 128 | ✅ pass (attempt 2) | 10,472 | 6.8% | 50 / 80 GB | 34m 03s | 35.2 h | 1.130 | 1.2407 | 71.75% |
| Mistral-Large-123B | DDP | 8 | ❌ GPU OOM |  |  |  | 16m 04s |  |  |  |  |
| Mistral-Large-123B | DDP | 16 | ❌ GPU OOM |  |  |  | 15m 24s |  |  |  |  |
| Mistral-Large-123B | DDP | 32 | ❌ GPU OOM |  |  |  | 16m 03s |  |  |  |  |
| Mistral-Large-123B | DDP | 64 | ❌ GPU OOM |  |  |  | 17m 00s |  |  |  |  |
| Mistral-Large-123B | DDP | 128 | ❌ GPU OOM |  |  |  | 19m 22s |  |  |  |  |
| Mistral-Large-123B | QLoRA+DDP | 8 | ✅ pass (attempt 3) | 3,740 | 38.8% | 78 / 80 GB | 30m 42s | 4.1 days | 1.064 | 1.2721 | 71.98% |
| Mistral-Large-123B | QLoRA+DDP | 16 | ✅ pass (attempt 3) | 7,412 | 38.6% | 78 / 80 GB | 25m 04s | 2.1 days | 1.023 | 1.2636 | 71.65% |
| Mistral-Large-123B | QLoRA+DDP | 32 | ✅ pass (attempt 3) | 14,784 | 38.6% | 78 / 80 GB | 24m 55s | 24.9 h | 1.208 | 1.2350 | 71.50% |
| Mistral-Large-123B | QLoRA+DDP | 64 | ✅ pass (attempt 3) | 29,638 | 38.5% | 78 / 80 GB | 25m 58s | 12.4 h | 1.102 | 1.2292 | 72.13% |
| Mistral-Large-123B | QLoRA+DDP | 128 | ✅ pass (attempt 3) | 57,457 | 37.3% | 78 / 80 GB | 28m 25s | 6.4 h | 1.312 | 1.2401 | 71.91% |
| Mistral-Large-123B | Pipeline | 8 | ✅ pass (attempt 2) | 2,973 | 30.8% | 50 / 80 GB | 30m 05s | 5.2 days | 1.144 | 1.2729 | 72.14% |
| Mistral-Large-123B | Pipeline | 16 | ✅ pass | 5,957 | 30.9% | 50 / 80 GB | 25m 13s | 2.6 days | 1.164 | 1.2748 | 71.70% |
| Mistral-Large-123B | Pipeline | 32 | ✅ pass (attempt 2) | 11,745 | 30.5% | 50 / 80 GB | 25m 54s | 31.3 h | 1.164 | 1.2632 | 71.38% |
| Mistral-Large-123B | Pipeline | 64 | ✅ pass | 23,760 | 30.8% | 50 / 80 GB | 22m 01s | 15.5 h | 1.167 | 1.2426 | 72.14% |
| Mistral-Large-123B | Pipeline | 128 | ✅ pass (attempt 2) | 47,350 | 30.7% | 50 / 80 GB | 24m 40s | 7.8 h | 1.154 | 1.2554 | 71.48% |
| Qwen3-235B-A22B (MoE) | FSDP | 8 | ✅ pass | 467 | 0.9% | 54 / 80 GB | 1h 01m | 30.7 days | 0.825 | 1.4793 | 68.19% |
| Qwen3-235B-A22B (MoE) | FSDP | 16 | ✅ pass | 839 | 0.8% | 41 / 80 GB | 56m 41s | 17.1 days | 1.589 | 1.4589 | 68.22% |
| Qwen3-235B-A22B (MoE) | FSDP | 32 | ✅ pass (attempt 2) | 1,157 | 0.6% | 34 / 80 GB | 1h 04m | 12.4 days | 1.326 | 1.4565 | 68.76% |
| Qwen3-235B-A22B (MoE) | FSDP | 64 | ✅ pass | 1,430 | 0.4% | 30 / 80 GB | 1h 26m | 10.0 days | 1.484 | 1.4836 | 68.05% |
| Qwen3-235B-A22B (MoE) | FSDP | 128 | ✅ pass (attempt 4) | 2,710 | 0.3% | 29 / 80 GB | 1h 29m | 5.3 days | 1.448 | 1.4233 | 68.50% |
| Qwen3-235B-A22B (MoE) | HSDP | 8 | ✅ pass | 472 | 0.9% | 54 / 80 GB | 1h 00m | 30.4 days | 1.680 | 1.4786 | 68.20% |
| Qwen3-235B-A22B (MoE) | HSDP | 16 | ✅ pass | 896 | 0.9% | 54 / 80 GB | 54m 42s | 16.0 days | 1.318 | 1.4596 | 68.24% |
| Qwen3-235B-A22B (MoE) | HSDP | 32 | ✅ pass | 1,484* | 0.7% | 54 / 80 GB | 58m 21s | 9.7 days | 1.318 | 1.4583 | 68.75% |
| Qwen3-235B-A22B (MoE) | HSDP | 64 | ✅ pass | 2,171 | 0.6% | 54 / 80 GB | 1h 07m | 6.6 days | 1.354 | 1.4897 | 68.16% |
| Qwen3-235B-A22B (MoE) | HSDP | 128 | ✅ pass (attempt 2) | 4,034 | 0.5% | 54 / 80 GB | 1h 09m | 3.6 days | 1.674 | 1.3943 | 69.17% |
| Qwen3-235B-A22B (MoE) | DDP | 8 | ❌ GPU OOM |  |  |  | 17m 27s |  |  |  |  |
| Qwen3-235B-A22B (MoE) | DDP | 16 | ❌ GPU OOM |  |  |  | 17m 30s |  |  |  |  |
| Qwen3-235B-A22B (MoE) | DDP | 32 | ❌ GPU OOM |  |  |  | 17m 43s |  |  |  |  |
| Qwen3-235B-A22B (MoE) | DDP | 64 | ❌ GPU OOM |  |  |  | 18m 59s |  |  |  |  |
| Qwen3-235B-A22B (MoE) | DDP | 128 | ❌ GPU OOM |  |  |  | 20m 39s |  |  |  |  |
| Qwen3-235B-A22B (MoE) | QLoRA+DDP | 8 | ❌ host RAM OOM |  |  |  | 10m 30s |  |  |  |  |
| Qwen3-235B-A22B (MoE) | QLoRA+DDP | 16 | ❌ host RAM OOM |  |  |  | 8m 27s |  |  |  |  |
| Qwen3-235B-A22B (MoE) | QLoRA+DDP | 32 | ❌ host RAM OOM |  |  |  | 7m 38s |  |  |  |  |
| Qwen3-235B-A22B (MoE) | QLoRA+DDP | 64 | ❌ host RAM OOM |  |  |  | 11m 09s |  |  |  |  |
| Qwen3-235B-A22B (MoE) | QLoRA+DDP | 128 | ❌ host RAM OOM |  |  |  | 17m 25s |  |  |  |  |
| Qwen3-235B-A22B (MoE) | Pipeline | 8 | ✅ pass | 5,538 | 10.9% | 45 / 80 GB | 22m 30s | 2.6 days | 1.300 | 1.3887 | 68.18% |
| Qwen3-235B-A22B (MoE) | Pipeline | 16 | ✅ pass | 11,119 | 11.0% | 45 / 80 GB | 18m 16s | 31.0 h | 1.307 | 1.3966 | 68.23% |
| Qwen3-235B-A22B (MoE) | Pipeline | 32 | ✅ pass | 19,361 | 9.5% | 45 / 80 GB | 17m 57s | 17.8 h | 1.306 | 1.3853 | 68.66% |
| Qwen3-235B-A22B (MoE) | Pipeline | 64 | ✅ pass | 43,951 | 10.8% | 45 / 80 GB | 15m 50s | 7.8 h | 1.308 | 1.4098 | 68.06% |
| Qwen3-235B-A22B (MoE) | Pipeline | 128 | ✅ pass (attempt 4) | 80,110* | 9.9% | 45 / 80 GB | 18m 07s | 4.3 h | 1.297 | 1.4065 | 68.00% |

FAIL: 20, PASS: 105 (of 125)
