# feature/megablocks-moe

Built on `fsdp-moe-fixes` + `feature/expert-parallel` + `feature/robustness` (runtime
packing, startup watchdog, step timing, preflight, node health), with the goal of
replacing the manual per-expert token-dispatch/gather/loop/scatter computation with
Megablocks' block-sparse grouped-GEMM kernels for MoE expert FFNs.

Motivation: measured combined-fixes MFU for Qwen3-30B-A3B on this cluster plateaus at
~4-6.4% (8-64 nodes, A100 80GB PCIe, 2 GPUs/node). Profiling showed the remaining
bottleneck at small node counts (8n/16n) is `fwd+bwd` compute time, not communication —
Megablocks' block-sparse kernels are the standard fix for this in production MoE stacks
(used by Mixtral/DeepSpeed-MoE-style training).

Each of the three merged-in branches is its own independent deliverable, reviewed and
merged on its own elsewhere. This branch adds Megablocks on top for throughput
measurement; whether it merges anywhere depends on what the comparison shows.

See each branch's own prior README content (in git history) for what each one changes.
