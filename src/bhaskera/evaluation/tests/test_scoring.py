"""
Log-likelihood scoring must equal scoring each request alone (CPU, fp32):
    python -m bhaskera.evaluation.tests.test_scoring

For dense (Qwen3, Llama) and MoE (Qwen3-MoE, Mixtral) tiny random models:
  1. batched, left-padded, logits_to_keep scores == each request scored
     alone with a plain unpadded forward and a manual log-softmax; greedy
     flags equal; context longer than max_length is left-truncated exactly
  2. two processes (gloo) with an uneven request count: every rank runs the
     same number of forwards (the FSDP requirement) and the results equal
     the single-process results
"""
from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from bhaskera.evaluation.scoring import Request, score

VOCAB = 97


def _models():
    from transformers import (
        LlamaConfig,
        LlamaForCausalLM,
        MixtralConfig,
        MixtralForCausalLM,
        Qwen3Config,
        Qwen3ForCausalLM,
        Qwen3MoeConfig,
        Qwen3MoeForCausalLM,
    )

    small = {"vocab_size": VOCAB, "hidden_size": 32, "num_hidden_layers": 2,
             "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
             "attn_implementation": "eager", "max_position_embeddings": 256}
    return {
        "qwen3": lambda: Qwen3ForCausalLM(Qwen3Config(**small, intermediate_size=64)),
        "llama": lambda: LlamaForCausalLM(LlamaConfig(**small, intermediate_size=64)),
        "qwen3moe": lambda: Qwen3MoeForCausalLM(Qwen3MoeConfig(
            **small, intermediate_size=64, moe_intermediate_size=16, num_experts=8,
            num_experts_per_tok=2)),
        "mixtral": lambda: MixtralForCausalLM(MixtralConfig(
            **small, intermediate_size=16, num_local_experts=8, num_experts_per_tok=2)),
    }


def _requests(seed: int = 0) -> list[Request]:
    g = torch.Generator().manual_seed(seed)
    reqs = []
    for i in range(11):
        n_ctx = int(torch.randint(0, 40, (1,), generator=g))
        n_cont = int(torch.randint(1, 9, (1,), generator=g))
        ids = torch.randint(3, VOCAB, (n_ctx + n_cont,), generator=g).tolist()
        reqs.append(Request(ids[:n_ctx], ids[n_ctx:]))
    reqs.append(Request(list(range(3, 60)) + list(range(3, 60)), [5, 6, 7]))   # > max_length
    return reqs


@torch.no_grad()
def _alone(model, req: Request, max_length: int, prefix: int) -> tuple[float, bool]:
    ctx = req.context or [prefix]
    full = (ctx + req.continuation)[-(max_length + 1):]
    ids = torch.tensor([full[:-1]])
    logits = model(input_ids=ids).logits[0, -len(req.continuation):].float()
    target = torch.tensor(req.continuation)
    lp = F.log_softmax(logits, -1).gather(-1, target[:, None]).sum().item()
    return lp, bool((logits.argmax(-1) == target).all())


def check_exact(name, make) -> bool:
    torch.manual_seed(0)
    model = make().float().eval()
    reqs = _requests()
    got = score(model, reqs, batch_size=4, max_length=64, pad_id=0, prefix_token=1,
                device=torch.device("cpu"))
    err, greedy_ok = 0.0, True
    for r, s in zip(reqs, got):
        lp, gr = _alone(model, r, 64, 1)
        err = max(err, abs(lp - s.logprob))
        greedy_ok &= gr == s.greedy
    ok = err < 1e-4 and greedy_ok
    print(f"[{name}] 1. batched left-padded == alone: max |dlogprob| = {err:.2e}, "
          f"greedy flags equal: {greedy_ok}  {'OK' if ok else 'FAIL'}")
    return ok


def _dist_worker(rank: int, world: int, port: int, q) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.manual_seed(0)
    model = _models()["qwen3moe"]().float().eval()
    calls = [0]
    model.register_forward_pre_hook(lambda *_: calls.__setitem__(0, calls[0] + 1))
    reqs = _requests()[:7]          # 7 requests over 2 ranks: uneven on purpose
    got = score(model, reqs, batch_size=2, max_length=64, pad_id=0, prefix_token=1,
                device=torch.device("cpu"))
    counts = [None] * world
    dist.all_gather_object(counts, calls[0])
    if rank == 0:
        q.put((counts, [(s.logprob, s.greedy) for s in got]))
    dist.destroy_process_group()


def check_distributed() -> bool:
    torch.manual_seed(0)
    model = _models()["qwen3moe"]().float().eval()
    reqs = _requests()[:7]
    ref = score(model, reqs, batch_size=2, max_length=64, pad_id=0, prefix_token=1,
                device=torch.device("cpu"))
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 29500 + os.getpid() % 1000
    procs = [ctx.Process(target=_dist_worker, args=(r, 2, port, q)) for r in range(2)]
    for p in procs:
        p.start()
    counts, got = q.get(timeout=300)
    for p in procs:
        p.join()
    same = all(abs(a - s.logprob) < 1e-5 and b == s.greedy for (a, b), s in zip(got, ref))
    ok = len(set(counts)) == 1 and same
    print(f"2. 2 ranks, 7 requests: forwards per rank {counts} (must be equal), "
          f"results == single process: {same}  {'OK' if ok else 'FAIL'}")
    return ok


def main() -> int:
    results = [check_exact(n, m) for n, m in _models().items()]
    results.append(check_distributed())
    ok = all(results)
    print("=== SCORING:", "PASS" if ok else "FAIL", "===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
