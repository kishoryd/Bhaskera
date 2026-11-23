"""
bhaskera.trainer.eval_metrics
=============================
Next-token accuracy for evaluation: the share of target tokens (labels != -100) where the
model's highest-scoring token is the label. Position t predicts label t+1 within each row
(HF causal-LM convention), so the same number comes out of every strategy.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def token_accuracy_counts(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    """logits [B, S, V], labels [B, S] -> (correct, counted)."""
    pred = logits[:, :-1].argmax(dim=-1)
    tgt = labels[:, 1:].to(pred.device)
    keep = tgt != -100
    return int((pred.eq(tgt) & keep).sum()), int(keep.sum())


@torch.no_grad()
def token_accuracy_from_hidden(hidden: torch.Tensor, lm_head, labels: torch.Tensor,
                               rows: int, chunk_rows: int = 1) -> tuple[int, int]:
    """Pipeline last stage: final hidden states [1, rows*S, H] (after the norm) -> (correct,
    counted), computing logits a few rows at a time so the [S, vocab] tensor stays small."""
    h = hidden.reshape(rows, -1, hidden.shape[-1])
    y = labels.reshape(rows, -1)
    correct = counted = 0
    for i in range(0, rows, chunk_rows):
        c, n = token_accuracy_counts(lm_head(h[i:i + chunk_rows]), y[i:i + chunk_rows])
        correct, counted = correct + c, counted + n
    return correct, counted
