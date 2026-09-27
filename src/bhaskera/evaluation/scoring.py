"""
bhaskera.evaluation.scoring
===========================
Log-likelihood scoring that is safe to run on an FSDP-sharded model.

Every benchmark here reduces to one primitive: log P(continuation | context)
summed over the continuation's tokens, plus whether each continuation token
was the model's argmax ("greedy"). Multiple-choice picks the highest-scoring
choice; perplexity sums over rolling windows; an agentic tool call counts as
exactly right when every one of its tokens is greedy, which is precisely
when greedy decoding would reproduce it. No generation is needed, so a
model too large for one node (sharded across many) is scored with forward
passes only.

FSDP constraint: every forward is a collective (each layer all-gathers its
weights across ranks), so all ranks must run the same number of forwards
or the job hangs. Requests are dealt round-robin and every rank's list is
padded with a repeat of a real request to the same length; padding results
are discarded.

Batching: rows are left-padded so every continuation ends at the last
position; the model then only projects the last K positions onto the
vocabulary (logits_to_keep), instead of materialising [batch, length,
vocab] logits. Position ids are passed explicitly (counting real tokens
only), so a left-padded row is scored exactly as it would be alone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F


@dataclass
class Request:
    context: list[int]
    continuation: list[int]


@dataclass
class Score:
    logprob: float          # sum over continuation tokens
    greedy: bool            # every continuation token was the argmax
    n_tokens: int


def _dist() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _inputs(req: Request, max_length: int, prefix_token: int) -> tuple[list[int], int]:
    """Model input ids and continuation length, harness-style: predict every
    continuation token; left-truncate the context if too long."""
    if not req.continuation:
        raise ValueError("empty continuation: nothing to score")
    if len(req.continuation) > max_length:
        raise ValueError(f"continuation of {len(req.continuation)} tokens exceeds "
                         f"max_length={max_length}; split it (see tasks.rolling_windows)")
    context = req.context or [prefix_token]
    full = (context + req.continuation)[-(max_length + 1):]
    return full[:-1], len(req.continuation)


@torch.no_grad()
def _score_batch(model, rows: list[tuple[list[int], int, list[int]]], pad_id: int,
                 device: torch.device) -> list[Score]:
    width = max(len(ids) for ids, _, _ in rows)
    keep = max(n for _, n, _ in rows)
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, (ids, _, _) in enumerate(rows):
        input_ids[i, width - len(ids):] = torch.tensor(ids)
        mask[i, width - len(ids):] = 1
    position_ids = (mask.cumsum(-1) - 1).clamp(min=0)
    input_ids, mask, position_ids = input_ids.to(device), mask.to(device), position_ids.to(device)
    try:
        logits = model(input_ids=input_ids, attention_mask=mask, position_ids=position_ids,
                       logits_to_keep=keep, use_cache=False).logits
    except TypeError:  # forward without logits_to_keep (custom architectures)
        logits = model(input_ids=input_ids, attention_mask=mask, position_ids=position_ids,
                       use_cache=False).logits[:, -keep:]
    out = []
    for i, (_, n, cont) in enumerate(rows):
        row = logits[i, keep - n:].float()                      # [n, vocab]
        target = torch.tensor(cont, device=device)
        logp = F.log_softmax(row, dim=-1).gather(-1, target[:, None]).squeeze(-1)
        out.append(Score(float(logp.sum()), bool((row.argmax(-1) == target).all()), n))
    return out


def score(model, requests: list[Request], *, batch_size: int, max_length: int,
          pad_id: int, prefix_token: int, device: torch.device) -> list[Score]:
    """Score every request; returns one Score per request, in order, on
    every rank. Must be called on all ranks with the same requests."""
    rank, world = _dist()
    prepared = [_inputs(r, max_length, prefix_token) for r in requests]
    # Longest first: similar lengths share a batch (less padding), and an
    # out-of-memory shows up on the first batch rather than hours in.
    order = sorted(range(len(requests)), key=lambda i: -len(prepared[i][0]))
    mine = order[rank::world]
    per_rank = math.ceil(len(order) / world) if order else 0
    n_pad = per_rank - len(mine)
    if order:
        mine = mine + [order[0]] * n_pad
    was_training = model.training
    model.eval()
    results: dict[int, Score] = {}
    try:
        for start in range(0, len(mine), batch_size):
            idx = mine[start:start + batch_size]
            rows = [(prepared[i][0], prepared[i][1], requests[i].continuation) for i in idx]
            for j, (i, s) in enumerate(zip(idx, _score_batch(model, rows, pad_id, device))):
                if start + j < len(mine) - n_pad:   # drop the padding repeats
                    results[i] = s
    finally:
        model.train(was_training)
    if world > 1:
        gathered: list = [None] * world
        dist.all_gather_object(gathered, results)
        results = {k: v for part in gathered for k, v in part.items()}
    return [results[i] for i in range(len(requests))]
