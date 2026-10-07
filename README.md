# EXPERIMENTAL — do not push, do not merge into any real branch

This is a disposable local combination of three independently-developed branches, built solely to measure MFU when their fixes are stacked together on Qwen3-30B-A3B:

- `perf/grouped-experts` (batched expert computation)
- `feature/expert-parallel` (expert parallelism for MoE)
- `feature/robustness` (runtime sequence packing, startup watchdog, step timing, preflight, node health)

Each of those branches is its own independent deliverable, reviewed and merged on its own. This combination exists only in this local worktree (`Bhaskera-mfu-experiment`, branch `experiment/mfu-combined-DO-NOT-PUSH`) for a one-off throughput measurement and is not pushed to origin.

See each branch's own README for what it actually changes and why.
