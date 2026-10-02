"""
Deterministic quality checks for generated RAG responses.

These checks are intentionally lightweight and do not claim to replace
human or model-based evaluation. They enforce hard application contracts:
fallback behavior, non-empty answers, and absence of internal source
metadata in the user-facing answer.
"""

import re
from dataclasses import dataclass


FALLBACK_ANSWER = (
    "I couldn't find enough information about that "
    "in the video. It may not be covered by this video."
)


@dataclass(frozen=True)
class DeterministicQualityResult:
    passed: bool
    checks: dict[str, bool]
    failures: tuple[str, ...]


_INTERNAL_SOURCE_PATTERNS = (
    re.compile(r"[SOURCEs+d+]", re.IGNORECASE),
    re.compile(r"sources+d+", re.IGNORECASE),
    re.compile(r"videos+ids*:", re.IGNORECASE),
    re.compile(r"timestamps*:", re.IGNORECASE),
)


def evaluate_response_contract(
    *,
    answer: str,
    answerable: bool,
) -> DeterministicQualityResult:
    """Apply deterministic end-to-end response contracts."""

    checks: dict[str, bool] = {}
    failures: list[str] = []

    normalized = answer.strip() if isinstance(answer, str) else ""

    checks["non_empty"] = bool(normalized)
    if not checks["non_empty"]:
        failures.append("answer_empty")

    if answerable:
        checks["not_fallback"] = normalized != FALLBACK_ANSWER
        if not checks["not_fallback"]:
            failures.append("answerable_case_returned_fallback")
    else:
        checks["exact_fallback"] = normalized == FALLBACK_ANSWER
        if not checks["exact_fallback"]:
            failures.append("unanswerable_case_did_not_return_exact_fallback")

    contains_internal_metadata = any(
        pattern.search(normalized)
        for pattern in _INTERNAL_SOURCE_PATTERNS
    )

    checks["no_internal_source_metadata"] = not contains_internal_metadata
    if not checks["no_internal_source_metadata"]:
        failures.append("internal_source_metadata_in_answer")

    return DeterministicQualityResult(
        passed=not failures,
        checks=checks,
        failures=tuple(failures),
    )
