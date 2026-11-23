"""Base class for experiment trackers."""
from __future__ import annotations

import contextlib
from typing import Any


class NullSpan:
    """What a phase span yields when tracing is off: accepts and ignores everything."""

    def set_outputs(self, *_a, **_k) -> None:
        pass

    def set_attribute(self, *_a, **_k) -> None:
        pass

    def set_attributes(self, *_a, **_k) -> None:
        pass


def phase_span(tracker, name: str, **attributes):
    """A context manager around one phase of training (an evaluation, a checkpoint, an epoch). With MLflow
    tracing on, the phase becomes a trace in the run; otherwise it costs nothing."""
    span = getattr(tracker, "span", None)          # a custom tracker may not have it
    if span is None:
        return contextlib.nullcontext(NullSpan())
    return span(name, **attributes)


class BaseLogger:
    """
    Minimal logger interface.

    Subclasses implement ``log()`` (publish a flat metrics dict at the
    given training step) and ``finish()`` (release resources, flush
    buffers).  Both are best-effort: a logger that loses connectivity
    must never raise into the training loop.
    """

    def log(self, metrics: dict[str, Any], step: int) -> None:  # pragma: no cover
        ...

    def finish(self, failed: bool = False) -> None:  # pragma: no cover
        ...

    def log_adapter(self, path, register_as=None, tags=None):  # pragma: no cover
        """Store the exported LoRA adapter with the run; return the registered model version, or None."""
        return None

    def span(self, name, **attributes):  # pragma: no cover
        """Context manager for one phase of training; the default does nothing."""
        return contextlib.nullcontext(NullSpan())

    def log_eval_samples(self, rows, step):  # pragma: no cover
        """Store a few evaluation examples (prompt, reference, prediction) with the run."""
        return None

