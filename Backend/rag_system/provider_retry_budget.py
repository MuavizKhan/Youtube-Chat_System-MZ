"""Request-scoped retry budget shared by Groq overview generation and evidence audits.

The budget is stored in a ContextVar so concurrent API requests never share
retry time. It counts only deliberate Retry-After/backoff sleeps, not provider
request latency.
"""
from __future__ import annotations

from contextvars import ContextVar, Token
import math
from typing import Any


_GROQ_RETRY_BUDGET: ContextVar[dict[str, float] | None] = ContextVar(
    "youtube_rag_groq_retry_budget",
    default=None,
)


def begin_groq_retry_budget(*, max_wait_seconds: float) -> Token:
    """Start a request-local cumulative Groq retry-wait budget."""
    if (
        isinstance(max_wait_seconds, bool)
        or not isinstance(max_wait_seconds, (int, float))
        or not math.isfinite(float(max_wait_seconds))
        or max_wait_seconds < 0
    ):
        raise ValueError("max_wait_seconds must be a finite non-negative number.")
    return _GROQ_RETRY_BUDGET.set({
        "max_wait_seconds": float(max_wait_seconds),
        "waited_seconds": 0.0,
    })


def end_groq_retry_budget(token: Token) -> None:
    """Restore the previous request context, including nested contexts."""
    _GROQ_RETRY_BUDGET.reset(token)


def reserve_groq_retry_wait(seconds: float) -> bool:
    """Reserve retry sleep time; return False without sleeping if the budget is spent."""
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(float(seconds))
        or seconds < 0
    ):
        raise ValueError("retry wait must be a finite non-negative number.")

    budget = _GROQ_RETRY_BUDGET.get()
    if budget is None:
        return True

    next_total = budget["waited_seconds"] + float(seconds)
    if next_total > budget["max_wait_seconds"]:
        return False

    budget["waited_seconds"] = next_total
    return True


def current_groq_retry_budget() -> dict[str, float] | None:
    """Return a copy of the current request budget for safe diagnostics/tests."""
    budget: dict[str, float] | None = _GROQ_RETRY_BUDGET.get()
    return None if budget is None else dict(budget)
