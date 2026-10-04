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




def _validate_sources(
    *,
    sources: list[dict] | None,
    retrieved_chunks: int | None,
    source_segments: int | None,
) -> tuple[dict[str, bool], list[str]]:
    """Validate provenance metadata returned by the RAG chain."""

    checks: dict[str, bool] = {}
    failures: list[str] = []

    normalized_sources = sources if isinstance(sources, list) else []

    checks["source_count_matches"] = (
        source_segments is None
        or int(source_segments) == len(normalized_sources)
    )
    if not checks["source_count_matches"]:
        failures.append("source_count_mismatch")

    checks["retrieval_count_valid"] = (
        retrieved_chunks is None
        or int(retrieved_chunks) >= 0
    )
    if not checks["retrieval_count_valid"]:
        failures.append("retrieved_chunk_count_invalid")

    valid_source_bounds = True
    valid_source_identity = True

    for source in normalized_sources:
        try:
            source_id = int(source.get("source_id", 0))
            start = float(source.get("start", 0.0))
            end = float(source.get("end", 0.0))
            duration = float(source.get("duration", -1.0))
        except (AttributeError, TypeError, ValueError):
            valid_source_bounds = False
            valid_source_identity = False
            break

        if source_id <= 0 or not source.get("video_id"):
            valid_source_identity = False

        if start < 0 or end < start:
            valid_source_bounds = False

        if abs(duration - max(0.0, end - start)) > 0.01:
            valid_source_bounds = False

    checks["source_bounds_valid"] = valid_source_bounds
    if not valid_source_bounds:
        failures.append("source_bounds_invalid")

    checks["source_identity_valid"] = valid_source_identity
    if not valid_source_identity:
        failures.append("source_identity_invalid")

    return checks, failures


_INTERNAL_SOURCE_PATTERNS = (
    re.compile(r"\[SOURCE\s+\d+\]", re.IGNORECASE),
    re.compile(r"\bSource\s+\d+\b", re.IGNORECASE),
    re.compile(r"\bVideo\s+ID\s*:", re.IGNORECASE),
    re.compile(r"\bTimestamp\s*:", re.IGNORECASE),
)


def evaluate_response_contract(
    *,
    answer: str,
    answerable: bool | None,
    retrieved_chunks: int | None = None,
    source_segments: int | None = None,
    sources: list[dict] | None = None,
) -> DeterministicQualityResult:
    """Apply deterministic end-to-end response contracts."""

    checks: dict[str, bool] = {}
    failures: list[str] = []

    normalized = answer.strip() if isinstance(answer, str) else ""

    checks["non_empty"] = bool(normalized)
    if not checks["non_empty"]:
        failures.append("answer_empty")

    if answerable is True:
        checks["not_fallback"] = normalized != FALLBACK_ANSWER
        if not checks["not_fallback"]:
            failures.append("answerable_case_returned_fallback")
    elif answerable is False:
        checks["exact_fallback"] = normalized == FALLBACK_ANSWER
        if not checks["exact_fallback"]:
            failures.append("unanswerable_case_did_not_return_exact_fallback")
    else:
        checks["answerability_deferred"] = True

    contains_internal_metadata = any(
        pattern.search(normalized)
        for pattern in _INTERNAL_SOURCE_PATTERNS
    )

    checks["no_internal_source_metadata"] = not contains_internal_metadata
    if not checks["no_internal_source_metadata"]:
        failures.append("internal_source_metadata_in_answer")

    provenance_checks, provenance_failures = _validate_sources(
        sources=sources,
        retrieved_chunks=retrieved_chunks,
        source_segments=source_segments,
    )
    checks.update(provenance_checks)
    failures.extend(provenance_failures)

    if answerable is True and retrieved_chunks is not None:
        checks["answerable_has_retrieval"] = retrieved_chunks > 0
        if not checks["answerable_has_retrieval"]:
            failures.append("answerable_case_missing_retrieval")

        checks["answerable_has_sources"] = (
            source_segments is not None
            and source_segments > 0
            and bool(sources)
        )
        if not checks["answerable_has_sources"]:
            failures.append("answerable_case_missing_sources")

    elif answerable is False and retrieved_chunks is not None:
        checks["unanswerable_has_no_retrieval"] = retrieved_chunks == 0
        if not checks["unanswerable_has_no_retrieval"]:
            failures.append("unanswerable_case_retrieved_evidence")

        checks["unanswerable_has_no_sources"] = (
            (source_segments or 0) == 0
            and not sources
        )
        if not checks["unanswerable_has_no_sources"]:
            failures.append("unanswerable_case_has_sources")

    return DeterministicQualityResult(
        passed=not failures,
        checks=checks,
        failures=tuple(failures),
    )
