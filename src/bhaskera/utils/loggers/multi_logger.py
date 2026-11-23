"""
bhaskera.utils.loggers.multi_logger
===================================
Fan-out logger.  Forwards each call to a list of child backends and
swallows individual failures so one flaky tracker never breaks training.

This is what ``build_logger`` returns when multiple backends are
configured (which is the common case now that Ray Dashboard runs
alongside W&B / MLflow).
"""
from __future__ import annotations

import logging
from typing import Any, Iterable

from .base import BaseLogger

logger = logging.getLogger(__name__)


class MultiLogger(BaseLogger):
    def __init__(self, children: Iterable[BaseLogger]) -> None:
        self._children: list[BaseLogger] = [c for c in children if c is not None]

    def __len__(self) -> int:
        return len(self._children)

    def __iter__(self):
        return iter(self._children)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        for child in self._children:
            try:
                child.log(metrics, step=step)
            except Exception as e:
                logger.warning(
                    f"{type(child).__name__}.log raised: {e} — "
                    "continuing with remaining loggers"
                )

    def finish(self, failed: bool = False) -> None:
        for child in self._children:
            try:
                child.finish(failed=failed)
            except Exception as e:
                logger.debug(f"{type(child).__name__}.finish raised: {e}")

    def log_adapter(self, path, register_as=None, tags=None):
        version = None
        for child in self._children:
            try:
                found = child.log_adapter(path, register_as=register_as, tags=tags)
                version = version or found
            except Exception as e:
                logger.warning(f"{type(child).__name__}.log_adapter raised: {e}")
        return version

    def span(self, name, **attributes):
        import contextlib
        from .base import NullSpan
        stack = contextlib.ExitStack()
        spans = []
        for child in self._children:
            try:
                spans.append(stack.enter_context(child.span(name, **attributes)))
            except Exception as e:
                logger.debug(f"{type(child).__name__}.span raised: {e}")

        class _Both:
            def __getattr__(self_inner, attr):
                def call(*a, **k):
                    for sp in spans:
                        getattr(sp, attr)(*a, **k)
                return call

        @contextlib.contextmanager
        def manager():
            with stack:
                yield _Both() if spans else NullSpan()
        return manager()

    def log_eval_samples(self, rows, step):
        for child in self._children:
            try:
                child.log_eval_samples(rows, step)
            except Exception as e:
                logger.warning(f"{type(child).__name__}.log_eval_samples raised: {e}")

