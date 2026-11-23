"""Next-token accuracy helper (eval)."""

import torch
from bhaskera.trainer.eval_metrics import token_accuracy_counts, token_accuracy_from_hidden


def test_token_accuracy():
    torch.manual_seed(0)
    V, H, R, S = 11, 6, 3, 7
    head = torch.nn.Linear(H, V, bias=False)
    h = torch.randn(1, R * S, H); y = torch.randint(0, V, (R, S)); y[1, 3] = -100
    logits = head(h.reshape(R, S, H))
    c, k = token_accuracy_counts(logits, y)
    pred = logits[:, :-1].argmax(-1); t = y[:, 1:]; m = t != -100
    assert (c, k) == (int((pred == t)[m].sum()), int(m.sum())), (c, k)
    assert token_accuracy_from_hidden(h, head, y, R) == (c, k)
    y2 = logits.argmax(-1); y2 = torch.cat([y2[:, :1], y2[:, :-1]], 1)   # labels = shifted predictions -> 100%
    assert token_accuracy_counts(logits, y2) == (R * (S - 1), R * (S - 1))
