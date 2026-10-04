"""
evaluation.py

Retrieval regression evaluation for the YouTube RAG system.

This module evaluates retrieval behavior only. It never calls the
generation model.

The regression contract intentionally separates strict retrieval
requirements from cases where an answerability decision belongs to
the generation/grounding layer.
"""

import argparse
import json
import math
from typing import Any

from .config import (
    MAX_DISTANCE,
    MMR_FETCH_K,
    MMR_LAMBDA,
    TOP_K,
)

from .gold_evidence import (
    GOLD_EVIDENCE_CASES,
    GOLD_EVIDENCE_VIDEO_ID,
    get_gold_evidence_case,
    matched_gold_groups,
    normalize_evidence_text,
)
from .temporal_evidence import (
    GOLD_TEMPORAL_VIDEO_ID,
    TEMPORAL_BENCHMARK_THRESHOLDS,
    TEMPORAL_EVIDENCE_CASES,
    get_temporal_evidence_case,
)

from .retrieval import (
    extract_video_id,
    load_vector_store,
    retrieve_question_context,
)


DEFAULT_VIDEO = "https://www.youtube.com/watch?v=MdeQMVBuGgY"

TEST_QUESTIONS = [
    {
        "id": "Q01",
        "type": "direct_fact",
        "question": "What is Kingfisher Airlines?",
    },
    {
        "id": "Q02",
        "type": "direct_fact",
        "question": (
            "What connection does Vijay Mallya discuss "
            "with Formula One?"
        ),
    },
    {
        "id": "Q03",
        "type": "direct_fact",
        "question": (
            "What businesses or companies does Vijay Mallya "
            "discuss in the interview?"
        ),
    },
    {
        "id": "Q04",
        "type": "explanation",
        "question": (
            "Why does Vijay Mallya say Kingfisher Airlines "
            "struggled or failed?"
        ),
    },
    {
        "id": "Q05",
        "type": "explanation",
        "question": (
            "How does Vijay Mallya describe the challenges "
            "he faced while running Kingfisher Airlines?"
        ),
    },
    {
        "id": "Q06",
        "type": "explanation",
        "question": (
            "What does Vijay Mallya say about building "
            "the Kingfisher brand?"
        ),
    },
    {
        "id": "Q07",
        "type": "specific_detail",
        "question": (
            "What does Vijay Mallya say about the amount "
            "of debt or money involved in the Kingfisher "
            "situation?"
        ),
    },
    {
        "id": "Q08",
        "type": "specific_detail",
        "question": (
            "What does Vijay Mallya say about banks and "
            "the recovery of money?"
        ),
    },
    {
        "id": "Q09",
        "type": "specific_detail",
        "question": (
            "What does Vijay Mallya say about the role of "
            "Indian government policy in the problems "
            "faced by Kingfisher Airlines?"
        ),
    },
    {
        "id": "Q10",
        "type": "multi_part",
        "question": (
            "What does Vijay Mallya say about the failure "
            "of Kingfisher Airlines, and what reasons does "
            "he give for it?"
        ),
    },
    {
        "id": "Q11",
        "type": "temporal",
        "question": (
            "Where in the video does Vijay Mallya discuss "
            "Kingfisher Airlines, and what is discussed "
            "there?"
        ),
    },
    {
        "id": "Q12",
        "type": "unanswerable",
        "question": (
            "What programming language does Vijay Mallya "
            "say is his favorite?"
        ),
    },
    {
        "id": "Q13",
        "type": "unrelated",
        "question": "What is the capital of Australia?",
    },
]

EXPECTED_RETRIEVAL = {
    **{f"Q{i:02d}": "must_retrieve" for i in range(1, 12)},
    "Q12": "may_retrieve",
    "Q13": "must_not_retrieve",
}

# The benchmark deliberately keeps hard thresholds tied to the existing
# retrieval contract instead of inventing arbitrary semantic-quality scores.
# A future benchmark can add labeled relevant chunks or human judgments
# without changing these deterministic regression metrics.
BENCHMARK_THRESHOLDS = {
    "strict_pass_rate": 1.0,
    "must_retrieve_recall": 1.0,
    "must_not_retrieve_rejection_rate": 1.0,
    "strict_max_best_distance": MAX_DISTANCE,
}

GOLD_BENCHMARK_THRESHOLDS = {
    "hit_rate_at_k": 1.0,
    "mean_group_coverage": 0.66,
    "mrr": 0.50,
    "mean_precision_at_k": 0.15,
}

# The production retriever intentionally expands strong anchors with nearby
# transcript chunks. This benchmark evaluates the final production context
# separately from the raw-anchor diagnostic so chunk-boundary expansion is
# measured rather than hidden.
GOLD_CONTEXT_BENCHMARK_THRESHOLDS = {
    "evidence_presence_rate": 1.0,
    "mean_group_coverage": 0.66,
    "case_pass_rate": 1.0,
}


def _expected_for(question_id: str) -> str:
    try:
        return EXPECTED_RETRIEVAL[question_id]
    except KeyError as error:
        raise ValueError(
            f"Unknown evaluation question ID: {question_id}"
        ) from error


def evaluate_retrieval_result(
    question_id: str,
    retrieved_count: int,
) -> tuple[bool, str]:
    """Evaluate one retrieval result against the regression contract."""

    if retrieved_count < 0:
        raise ValueError("retrieved_count cannot be negative.")

    expected = _expected_for(question_id)

    if expected == "must_retrieve":
        passed = retrieved_count > 0
        message = (
            "Relevant retrieval found."
            if passed
            else "Expected retrieval, but nothing was retrieved."
        )
        return passed, message

    if expected == "may_retrieve":
        return True, (
            "Retrieval may return context; answerability is "
            "validated by generation."
        )

    if expected == "must_not_retrieve":
        passed = retrieved_count == 0
        message = (
            "Correctly rejected as unrelated."
            if passed
            else "Unexpected context retrieved."
        )
        return passed, message

    raise ValueError(f"Unsupported retrieval expectation: {expected}")


def _build_result_record(
    test_case: dict[str, str],
    results: list[tuple[Any, float]],
) -> dict[str, Any]:
    retrieved_count = len(results)
    passed, evaluation_message = evaluate_retrieval_result(
        question_id=test_case["id"],
        retrieved_count=retrieved_count,
    )

    distances = [float(distance) for _, distance in results]

    chunk_ids = []
    for document, _ in results:
        chunk_id = document.metadata.get("chunk_id")
        if chunk_id is not None and chunk_id not in chunk_ids:
            chunk_ids.append(chunk_id)

    return {
        "id": test_case["id"],
        "type": test_case["type"],
        "question": test_case["question"],
        "expected": EXPECTED_RETRIEVAL[test_case["id"]],
        "retrieved": retrieved_count,
        "passed": passed,
        "message": evaluation_message,
        "best_distance": min(distances) if distances else None,
        "retrieved_chunk_ids": chunk_ids,
    }


def _rate(
    numerator: int,
    denominator: int,
) -> float:
    return numerator / denominator if denominator else 1.0


def build_benchmark_metrics(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build deterministic retrieval-quality metrics for the suite."""

    must_retrieve = [
        result
        for result in results
        if result["expected"] == "must_retrieve"
    ]
    must_not_retrieve = [
        result
        for result in results
        if result["expected"] == "must_not_retrieve"
    ]

    must_retrieve_passed = sum(
        1 for result in must_retrieve if result["passed"]
    )
    must_not_retrieve_passed = sum(
        1 for result in must_not_retrieve if result["passed"]
    )

    strict_results = [
        result
        for result in results
        if result["expected"] != "may_retrieve"
    ]
    strict_passed = sum(
        1 for result in strict_results if result["passed"]
    )

    best_distances = [
        float(result["best_distance"])
        for result in must_retrieve
        if result["best_distance"] is not None
    ]

    return {
        "strict_pass_rate": _rate(
            strict_passed,
            len(strict_results),
        ),
        "must_retrieve_recall": _rate(
            must_retrieve_passed,
            len(must_retrieve),
        ),
        "must_not_retrieve_rejection_rate": _rate(
            must_not_retrieve_passed,
            len(must_not_retrieve),
        ),
        "strict_max_best_distance": (
            max(best_distances) if best_distances else None
        ),
        "strict_mean_best_distance": (
            sum(best_distances) / len(best_distances)
            if best_distances
            else None
        ),
        "must_retrieve_cases": len(must_retrieve),
        "must_retrieve_passed": must_retrieve_passed,
        "must_not_retrieve_cases": len(must_not_retrieve),
        "must_not_retrieve_passed": must_not_retrieve_passed,
    }


def evaluate_benchmark_thresholds(
    benchmark_metrics: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Apply the deterministic retrieval benchmark thresholds."""

    failures: list[str] = []

    strict_pass_rate = benchmark_metrics["strict_pass_rate"]
    if strict_pass_rate < BENCHMARK_THRESHOLDS["strict_pass_rate"]:
        failures.append("strict_pass_rate_below_threshold")

    recall = benchmark_metrics["must_retrieve_recall"]
    if recall < BENCHMARK_THRESHOLDS["must_retrieve_recall"]:
        failures.append("must_retrieve_recall_below_threshold")

    rejection_rate = benchmark_metrics[
        "must_not_retrieve_rejection_rate"
    ]
    if (
        rejection_rate
        < BENCHMARK_THRESHOLDS[
            "must_not_retrieve_rejection_rate"
        ]
    ):
        failures.append(
            "must_not_retrieve_rejection_rate_below_threshold"
        )

    max_distance = benchmark_metrics["strict_max_best_distance"]
    if (
        max_distance is not None
        and max_distance
        > BENCHMARK_THRESHOLDS["strict_max_best_distance"]
    ):
        failures.append("strict_best_distance_above_threshold")

    return not failures, failures


def _build_gold_result_record(
    test_case: dict[str, str],
    results: list[tuple[Any, float]],
) -> dict[str, Any]:
    """Score raw top-k retrieval against curated evidence groups."""

    gold_case = get_gold_evidence_case(test_case["id"])
    top_results = results[:TOP_K]

    matched_group_indexes: set[int] = set()
    relevant_ranks: list[int] = []
    relevant_documents = 0

    for rank, (document, _distance) in enumerate(
        top_results,
        start=1,
    ):
        matched_groups = matched_gold_groups(
            document.page_content,
            gold_case,
        )

        if matched_groups:
            relevant_documents += 1
            relevant_ranks.append(rank)
            matched_group_indexes.update(matched_groups)

    group_coverage = (
        len(matched_group_indexes)
        / len(gold_case.groups)
    )

    first_relevant_rank = (
        min(relevant_ranks)
        if relevant_ranks
        else None
    )

    mrr = (
        1.0 / first_relevant_rank
        if first_relevant_rank is not None
        else 0.0
    )

    precision_at_k = (
        relevant_documents / TOP_K
        if TOP_K
        else 0.0
    )

    passed = (
        group_coverage
        >= gold_case.min_group_coverage
    )

    return {
        "id": test_case["id"],
        "question": test_case["question"],
        "gold_groups": len(gold_case.groups),
        "matched_gold_groups": len(matched_group_indexes),
        "group_coverage": group_coverage,
        "required_group_coverage": gold_case.min_group_coverage,
        "retrieved": len(top_results),
        "relevant_retrieved": relevant_documents,
        "hit_at_k": bool(relevant_ranks),
        "first_relevant_rank": first_relevant_rank,
        "mrr": mrr,
        "precision_at_k": precision_at_k,
        "passed": passed,
    }


def _build_gold_context_result_record(
    test_case: dict[str, str],
    results: list[tuple[Any, float]],
) -> dict[str, Any]:
    """Score the full production-expanded context against gold evidence."""

    gold_case = get_gold_evidence_case(test_case["id"])
    matched_group_indexes: set[int] = set()
    relevant_documents = 0

    for document, _distance in results:
        matched_groups = matched_gold_groups(
            document.page_content,
            gold_case,
        )

        if matched_groups:
            relevant_documents += 1
            matched_group_indexes.update(matched_groups)

    group_coverage = (
        len(matched_group_indexes)
        / len(gold_case.groups)
    )

    context_size = len(results)
    relevance_ratio = (
        relevant_documents / context_size
        if context_size
        else 0.0
    )

    hit = bool(matched_group_indexes)
    passed = (
        group_coverage
        >= gold_case.min_group_coverage
    )

    return {
        "id": test_case["id"],
        "question": test_case["question"],
        "gold_groups": len(gold_case.groups),
        "matched_gold_groups": len(matched_group_indexes),
        "group_coverage": group_coverage,
        "required_group_coverage": gold_case.min_group_coverage,
        "context_chunks": context_size,
        "relevant_context_chunks": relevant_documents,
        "evidence_presence": hit,
        "relevance_ratio": relevance_ratio,
        "passed": passed,
    }


def build_gold_context_benchmark_metrics(
    context_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate evidence coverage over final production context."""

    total = len(context_results)
    if not total:
        return {
            "cases": 0,
            "evidence_presence_rate": 1.0,
            "mean_group_coverage": 1.0,
            "mean_relevance_ratio": 1.0,
            "passed_cases": 0,
            "case_pass_rate": 1.0,
        }

    return {
        "cases": total,
        "evidence_presence_rate": sum(
            1
            for result in context_results
            if result["evidence_presence"]
        ) / total,
        "mean_group_coverage": sum(
            result["group_coverage"]
            for result in context_results
        ) / total,
        "mean_relevance_ratio": sum(
            result["relevance_ratio"]
            for result in context_results
        ) / total,
        "passed_cases": sum(
            1 for result in context_results if result["passed"]
        ),
        "case_pass_rate": sum(
            1 for result in context_results if result["passed"]
        ) / total,
    }


def evaluate_gold_context_benchmark_thresholds(
    context_metrics: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Apply thresholds to the production-expanded gold-evidence benchmark."""

    if not context_metrics["cases"]:
        return True, []

    failures: list[str] = []

    if (
        context_metrics["evidence_presence_rate"]
        < GOLD_CONTEXT_BENCHMARK_THRESHOLDS[
            "evidence_presence_rate"
        ]
    ):
        failures.append("gold_context_evidence_presence_below_threshold")

    if (
        context_metrics["mean_group_coverage"]
        < GOLD_CONTEXT_BENCHMARK_THRESHOLDS[
            "mean_group_coverage"
        ]
    ):
        failures.append("gold_context_group_coverage_below_threshold")

    if (
        context_metrics["case_pass_rate"]
        < GOLD_CONTEXT_BENCHMARK_THRESHOLDS["case_pass_rate"]
    ):
        failures.append("gold_context_case_pass_rate_below_threshold")

    return not failures, failures


def build_gold_benchmark_metrics(
    gold_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate gold-evidence retrieval metrics."""

    total = len(gold_results)
    if not total:
        return {
            "cases": 0,
            "hit_rate_at_k": 1.0,
            "mean_group_coverage": 1.0,
            "mrr": 1.0,
            "mean_precision_at_k": 1.0,
            "passed_cases": 0,
            "pass_rate": 1.0,
        }

    return {
        "cases": total,
        "hit_rate_at_k": sum(
            1 for result in gold_results if result["hit_at_k"]
        ) / total,
        "mean_group_coverage": sum(
            result["group_coverage"]
            for result in gold_results
        ) / total,
        "mrr": sum(
            result["mrr"]
            for result in gold_results
        ) / total,
        "mean_precision_at_k": sum(
            result["precision_at_k"]
            for result in gold_results
        ) / total,
        "passed_cases": sum(
            1 for result in gold_results if result["passed"]
        ),
        "pass_rate": sum(
            1 for result in gold_results if result["passed"]
        ) / total,
    }


def evaluate_gold_benchmark_thresholds(
    gold_metrics: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Apply hard thresholds to the curated gold-evidence benchmark."""

    if not gold_metrics["cases"]:
        return True, []

    failures: list[str] = []

    if (
        gold_metrics["hit_rate_at_k"]
        < GOLD_BENCHMARK_THRESHOLDS["hit_rate_at_k"]
    ):
        failures.append("gold_hit_rate_below_threshold")

    if (
        gold_metrics["mean_group_coverage"]
        < GOLD_BENCHMARK_THRESHOLDS["mean_group_coverage"]
    ):
        failures.append("gold_group_coverage_below_threshold")

    if gold_metrics["mrr"] < GOLD_BENCHMARK_THRESHOLDS["mrr"]:
        failures.append("gold_mrr_below_threshold")

    if (
        gold_metrics["mean_precision_at_k"]
        < GOLD_BENCHMARK_THRESHOLDS["mean_precision_at_k"]
    ):
        failures.append("gold_precision_below_threshold")

    return not failures, failures



def _validated_timestamp_bounds(
    document: Any,
) -> tuple[float, float] | None:
    """Return finite, ordered timestamp bounds or None for invalid metadata."""

    metadata = getattr(document, "metadata", None) or {}
    start = metadata.get("start")
    end = metadata.get("end")

    # bool is an int subclass, but True/False are not meaningful timestamps.
    if isinstance(start, bool) or isinstance(end, bool):
        return None

    try:
        start_value = float(start)
        end_value = float(end)
    except (TypeError, ValueError, OverflowError):
        return None

    if (
        not math.isfinite(start_value)
        or not math.isfinite(end_value)
        or start_value < 0
        or end_value <= start_value
    ):
        return None

    return start_value, end_value


def _build_temporal_result_record(
    test_case: dict[str, str],
    results: list[tuple[Any, float]],
    *,
    expected_video_id: str = GOLD_TEMPORAL_VIDEO_ID,
) -> dict[str, Any]:
    """Score top-k retrieval against chapter-level timestamp gold windows."""

    temporal_case = get_temporal_evidence_case(test_case["id"])
    top_results = results[:TOP_K]
    valid_timestamp_count = 0
    valid_identity_count = 0
    matched_window_indexes: set[int] = set()
    relevant_ranks: list[int] = []

    for rank, (document, _distance) in enumerate(top_results, start=1):
        bounds = _validated_timestamp_bounds(document)
        metadata = getattr(document, "metadata", None) or {}
        identity_valid = metadata.get("video_id") == expected_video_id

        if identity_valid:
            valid_identity_count += 1

        if bounds is None:
            continue

        valid_timestamp_count += 1
        if not identity_valid:
            # A valid timestamp from another video is not valid evidence for
            # this benchmark's video, and must not count as a temporal hit.
            continue

        start, end = bounds
        normalized_content = normalize_evidence_text(
            getattr(document, "page_content", "") or ""
        )
        matched_this_result = False

        for window_index, window in enumerate(temporal_case.windows):
            overlap = min(end, window.end_seconds) - max(
                start,
                window.start_seconds,
            )
            has_topic_evidence = any(
                normalize_evidence_text(phrase) in normalized_content
                for phrase in window.topic_phrases
            )
            if overlap > 0 and has_topic_evidence:
                matched_window_indexes.add(window_index)
                matched_this_result = True

        if matched_this_result:
            relevant_ranks.append(rank)

    denominator = len(top_results)
    timestamp_validity_rate = (
        valid_timestamp_count / denominator if denominator else 0.0
    )
    source_identity_validity_rate = (
        valid_identity_count / denominator if denominator else 0.0
    )
    window_coverage = (
        len(matched_window_indexes) / len(temporal_case.windows)
        if temporal_case.windows
        else 0.0
    )
    first_relevant_rank = min(relevant_ranks) if relevant_ranks else None
    mrr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
    hit_at_k = bool(relevant_ranks)
    passed = (
        hit_at_k
        and window_coverage >= temporal_case.min_window_coverage
        and timestamp_validity_rate == 1.0
        and source_identity_validity_rate == 1.0
    )

    return {
        "id": test_case["id"],
        "question": test_case["question"],
        "gold_windows": len(temporal_case.windows),
        "matched_windows": len(matched_window_indexes),
        "matched_section_ids": [
            window.section_id
            for index, window in enumerate(temporal_case.windows)
            if index in matched_window_indexes
        ],
        "matched_section_labels": [
            window.label
            for index, window in enumerate(temporal_case.windows)
            if index in matched_window_indexes
        ],
        "window_coverage": window_coverage,
        "required_window_coverage": temporal_case.min_window_coverage,
        "retrieved": denominator,
        "hit_at_k": hit_at_k,
        "first_relevant_rank": first_relevant_rank,
        "mrr": mrr,
        "valid_timestamp_count": valid_timestamp_count,
        "timestamp_validity_rate": timestamp_validity_rate,
        "valid_source_identity_count": valid_identity_count,
        "source_identity_validity_rate": source_identity_validity_rate,
        "passed": passed,
    }


def build_temporal_benchmark_metrics(
    temporal_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate temporal retrieval, timestamp, and identity metrics."""

    total = len(temporal_results)
    if not total:
        return {
            "cases": 0,
            "hit_rate_at_k": 0.0,
            "mean_window_coverage": 0.0,
            "timestamp_validity_rate": 0.0,
            "source_identity_validity_rate": 0.0,
            "mrr": 0.0,
            "passed_cases": 0,
            "pass_rate": 0.0,
        }

    return {
        "cases": total,
        "hit_rate_at_k": sum(
            1 for result in temporal_results if result["hit_at_k"]
        ) / total,
        "mean_window_coverage": sum(
            result["window_coverage"] for result in temporal_results
        ) / total,
        "timestamp_validity_rate": sum(
            result["timestamp_validity_rate"] for result in temporal_results
        ) / total,
        "source_identity_validity_rate": sum(
            result["source_identity_validity_rate"]
            for result in temporal_results
        ) / total,
        "mrr": sum(result["mrr"] for result in temporal_results) / total,
        "passed_cases": sum(
            1 for result in temporal_results if result["passed"]
        ),
        "pass_rate": sum(
            1 for result in temporal_results if result["passed"]
        ) / total,
    }


def evaluate_temporal_benchmark_thresholds(
    temporal_metrics: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Apply strict thresholds to temporal evidence and timestamp metadata."""

    if not temporal_metrics["cases"]:
        return False, ["temporal_benchmark_has_no_cases"]

    failures: list[str] = []
    if (
        temporal_metrics["hit_rate_at_k"]
        < TEMPORAL_BENCHMARK_THRESHOLDS["hit_rate_at_k"]
    ):
        failures.append("temporal_hit_rate_below_threshold")
    if (
        temporal_metrics["mean_window_coverage"]
        < TEMPORAL_BENCHMARK_THRESHOLDS["mean_window_coverage"]
    ):
        failures.append("temporal_window_coverage_below_threshold")
    if (
        temporal_metrics["timestamp_validity_rate"]
        < TEMPORAL_BENCHMARK_THRESHOLDS["timestamp_validity_rate"]
    ):
        failures.append("temporal_timestamp_validity_below_threshold")
    if (
        temporal_metrics["source_identity_validity_rate"]
        < TEMPORAL_BENCHMARK_THRESHOLDS["source_identity_validity_rate"]
    ):
        failures.append("temporal_source_identity_below_threshold")
    if temporal_metrics["mrr"] < TEMPORAL_BENCHMARK_THRESHOLDS["mrr"]:
        failures.append("temporal_mrr_below_threshold")

    return not failures, failures


def build_evaluation_summary(
    video_id: str,
    results: list[dict[str, Any]],
    gold_results: list[dict[str, Any]] | None = None,
    gold_context_results: list[dict[str, Any]] | None = None,
    temporal_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a stable machine-readable evaluation summary."""

    total = len(results)
    passed = sum(1 for result in results if result["passed"])
    failed = total - passed

    strict_results = [
        result
        for result in results
        if result["expected"] != "may_retrieve"
    ]
    strict_total = len(strict_results)
    strict_passed = sum(
        1 for result in strict_results if result["passed"]
    )
    strict_failed = strict_total - strict_passed

    benchmark_metrics = build_benchmark_metrics(results)
    benchmark_passed, benchmark_failures = evaluate_benchmark_thresholds(
        benchmark_metrics
    )

    gold_benchmark = None
    if gold_results is not None:
        gold_metrics = build_gold_benchmark_metrics(
            gold_results
        )
        gold_passed, gold_failures = evaluate_gold_benchmark_thresholds(
            gold_metrics
        )
        gold_benchmark = {
            "thresholds": GOLD_BENCHMARK_THRESHOLDS,
            "metrics": gold_metrics,
            "passed": gold_passed,
            "failures": gold_failures,
            "results": gold_results,
        }

    gold_context_benchmark = None
    if gold_context_results is not None:
        context_metrics = build_gold_context_benchmark_metrics(
            gold_context_results
        )
        context_passed, context_failures = (
            evaluate_gold_context_benchmark_thresholds(
                context_metrics
            )
        )
        gold_context_benchmark = {
            "thresholds": GOLD_CONTEXT_BENCHMARK_THRESHOLDS,
            "metrics": context_metrics,
            "passed": context_passed,
            "failures": context_failures,
            "results": gold_context_results,
        }

    temporal_benchmark = None
    if temporal_results is not None:
        temporal_metrics = build_temporal_benchmark_metrics(temporal_results)
        temporal_passed, temporal_failures = (
            evaluate_temporal_benchmark_thresholds(temporal_metrics)
        )
        temporal_benchmark = {
            "thresholds": TEMPORAL_BENCHMARK_THRESHOLDS,
            "metrics": temporal_metrics,
            "passed": temporal_passed,
            "failures": temporal_failures,
            "results": temporal_results,
        }

    return {
        "video_id": video_id,
        "retrieval_strategy": "question_aware_mmr_lexical",
        "retrieval_config": {
            "top_k": TOP_K,
            "fetch_k": MMR_FETCH_K,
            "lambda_mult": MMR_LAMBDA,
            "max_distance": MAX_DISTANCE,
        },
        "benchmark_thresholds": BENCHMARK_THRESHOLDS,
        "benchmark_metrics": benchmark_metrics,
        "benchmark_passed": benchmark_passed,
        "benchmark_failures": benchmark_failures,
        "gold_benchmark": gold_benchmark,
        "gold_context_benchmark": gold_context_benchmark,
        "temporal_benchmark": temporal_benchmark,
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / total if total else 1.0,
        "strict_total": strict_total,
        "strict_passed": strict_passed,
        "strict_failed": strict_failed,
        "strict_pass_rate": (
            strict_passed / strict_total if strict_total else 1.0
        ),
        "results": results,
    }


def display_results(results: list[tuple[Any, float]]) -> None:
    """Print retrieved evidence for human inspection."""

    if not results:
        print("\nNO RELEVANT DOCUMENTS RETRIEVED.")
        return

    for rank, (document, distance) in enumerate(results, start=1):
        metadata = document.metadata
        print(f"\n--- Result {rank} ---")
        print(f"Distance: {float(distance):.6f}")
        print(f"Chunk ID: {metadata.get('chunk_id')}")
        print(
            f"Timestamp: {metadata.get('start', 0):.2f}s"
            f" -> {metadata.get('end', 0):.2f}s"
        )
        print("\nTEXT:")
        print(document.page_content)


def evaluate_video(
    video_reference: str,
    *,
    display: bool = True,
) -> dict[str, Any]:
    """Run the retrieval regression suite and return a summary."""

    video_id = extract_video_id(video_reference)
    vector_store = load_vector_store(video_id)

    results = []
    gold_results = []
    gold_context_results = []
    temporal_results = []
    run_video_specific_gold = video_id == GOLD_EVIDENCE_VIDEO_ID
    run_temporal_gold = video_id == GOLD_TEMPORAL_VIDEO_ID

    if display:
        print("\n" + "=" * 80)
        print("RAG RETRIEVAL REGRESSION EVALUATION")
        print("=" * 80)
        print(f"\nVideo ID: {video_id}")
        print("\nRetrieval Configuration:")
        print("  strategy     = question-aware retrieval")
        print(f"  top_k        = {TOP_K}")
        print(f"  fetch_k      = {MMR_FETCH_K}")
        print(f"  lambda       = {MMR_LAMBDA}")
        print(f"  max_distance = {MAX_DISTANCE}")
        print("\nBenchmark Thresholds:")
        print(
            "  strict_pass_rate               = "
            f"{BENCHMARK_THRESHOLDS['strict_pass_rate']:.1%}"
        )
        print(
            "  must_retrieve_recall           = "
            f"{BENCHMARK_THRESHOLDS['must_retrieve_recall']:.1%}"
        )
        print(
            "  must_not_retrieve_rejection    = "
            f"{BENCHMARK_THRESHOLDS['must_not_retrieve_rejection_rate']:.1%}"
        )
        print(
            "  strict_max_best_distance       = "
            f"{BENCHMARK_THRESHOLDS['strict_max_best_distance']:.3f}"
        )
        print("\nLoading vector store...")

    for test_case in TEST_QUESTIONS:
        question_id = test_case["id"]
        question_type = test_case["type"]
        question = test_case["question"]

        if display:
            print("\n" + "#" * 80)
            print(f"{question_id} | {question_type}")
            print(f"QUESTION: {question}")
            print("#" * 80)

        retrieved = retrieve_question_context(
            vector_store=vector_store,
            query=question,
        )

        record = _build_result_record(
            test_case=test_case,
            results=retrieved,
        )
        results.append(record)

        if (
            run_video_specific_gold
            and question_id in {
                case.question_id
                for case in GOLD_EVIDENCE_CASES
            }
        ):
            raw_retrieved = retrieve_question_context(
                vector_store=vector_store,
                query=question,
                expand_context=False,
            )
            gold_results.append(
                _build_gold_result_record(
                    test_case=test_case,
                    results=raw_retrieved,
                )
            )
            gold_context_results.append(
                _build_gold_context_result_record(
                    test_case=test_case,
                    results=retrieved,
                )
            )

        if run_temporal_gold and question_id in {
            case.question_id
            for case in TEMPORAL_EVIDENCE_CASES
        }:
            raw_retrieved = retrieve_question_context(
                vector_store=vector_store,
                query=question,
                expand_context=False,
            )
            temporal_results.append(
                _build_temporal_result_record(
                    test_case=test_case,
                    results=raw_retrieved,
                    expected_video_id=GOLD_TEMPORAL_VIDEO_ID,
                )
            )

        if display:
            display_results(retrieved)
            print(f"\nExpected: {record['expected']}")
            print(f"Result: {'PASS' if record['passed'] else 'FAIL'}")
            print(f"Evaluation: {record['message']}")

    summary = build_evaluation_summary(
        video_id=video_id,
        results=results,
        gold_results=gold_results if run_video_specific_gold else None,
        gold_context_results=(
            gold_context_results
            if run_video_specific_gold
            else None
        ),
        temporal_results=temporal_results if run_temporal_gold else None,
    )

    if display:
        print("\n" + "=" * 80)
        print("RETRIEVAL SUMMARY")
        print("=" * 80)

        for result in results:
            print(
                f"\n{result['id']} | {result['type']} | "
                f"Retrieved: {result['retrieved']} | "
                f"{'PASS' if result['passed'] else 'FAIL'}"
            )

        metrics = summary["benchmark_metrics"]
        print("\n" + "-" * 80)
        print(
            f"Overall: {summary['passed']}/{summary['total']} passed "
            f"({summary['pass_rate']:.1%})"
        )
        print(
            f"Strict:  {summary['strict_passed']}/"
            f"{summary['strict_total']} passed "
            f"({summary['strict_pass_rate']:.1%})"
        )
        print("\nBenchmark Metrics:")
        print(
            "  must-retrieve recall        = "
            f"{metrics['must_retrieve_recall']:.1%}"
        )
        print(
            "  unrelated rejection         = "
            f"{metrics['must_not_retrieve_rejection_rate']:.1%}"
        )
        print(
            "  strict max best distance    = "
            f"{metrics['strict_max_best_distance']}"
        )
        print(
            "  benchmark status             = "
            f"{'PASS' if summary['benchmark_passed'] else 'FAIL'}"
        )
        if summary["benchmark_failures"]:
            print(
                "  benchmark failures          = "
                + ", ".join(summary["benchmark_failures"])
            )

        gold_benchmark = summary["gold_benchmark"]
        if gold_benchmark is not None:
            gold_metrics = gold_benchmark["metrics"]
            print("\nGold Evidence Benchmark:")
            print(
                "  hit@k                       = "
                f"{gold_metrics['hit_rate_at_k']:.1%}"
            )
            print(
                "  mean group coverage        = "
                f"{gold_metrics['mean_group_coverage']:.1%}"
            )
            print(
                "  MRR                         = "
                f"{gold_metrics['mrr']:.3f}"
            )
            print(
                "  mean precision@k            = "
                f"{gold_metrics['mean_precision_at_k']:.1%}"
            )
            print(
                "  gold benchmark status       = "
                f"{'PASS' if gold_benchmark['passed'] else 'FAIL'}"
            )
            if gold_benchmark["failures"]:
                print(
                    "  gold benchmark failures     = "
                    + ", ".join(gold_benchmark["failures"])
                )
        gold_context_benchmark = summary["gold_context_benchmark"]
        if gold_context_benchmark is not None:
            context_metrics = gold_context_benchmark["metrics"]
            print("\nGold Context Benchmark (production):")
            print(
                "  evidence presence           = "
                f"{context_metrics['evidence_presence_rate']:.1%}"
            )
            print(
                "  mean group coverage         = "
                f"{context_metrics['mean_group_coverage']:.1%}"
            )
            print(
                "  mean relevance ratio        = "
                f"{context_metrics['mean_relevance_ratio']:.1%}"
            )
            print(
                "  case pass rate              = "
                f"{context_metrics['case_pass_rate']:.1%}"
            )
            print(
                "  production gold status      = "
                f"{'PASS' if gold_context_benchmark['passed'] else 'FAIL'}"
            )
            if gold_context_benchmark["failures"]:
                print(
                    "  production gold failures    = "
                    + ", ".join(gold_context_benchmark["failures"])
                )

        temporal_benchmark = summary["temporal_benchmark"]
        if temporal_benchmark is not None:
            temporal_metrics = temporal_benchmark["metrics"]
            print("\nTemporal Evidence Benchmark:")
            print(
                "  temporal hit@k              = "
                f"{temporal_metrics['hit_rate_at_k']:.1%}"
            )
            print(
                "  section coverage            = "
                f"{temporal_metrics['mean_window_coverage']:.1%}"
            )
            print(
                "  timestamp validity          = "
                f"{temporal_metrics['timestamp_validity_rate']:.1%}"
            )
            print(
                "  source identity validity    = "
                f"{temporal_metrics['source_identity_validity_rate']:.1%}"
            )
            print(
                "  temporal MRR                = "
                f"{temporal_metrics['mrr']:.3f}"
            )
            print(
                "  temporal benchmark status   = "
                f"{'PASS' if temporal_benchmark['passed'] else 'FAIL'}"
            )
            if temporal_benchmark["failures"]:
                print(
                    "  temporal benchmark failures = "
                    + ", ".join(temporal_benchmark["failures"])
                )
        print("-" * 80)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the YouTube RAG retrieval regression suite."
    )
    parser.add_argument(
        "video",
        nargs="?",
        default=DEFAULT_VIDEO,
        help="YouTube video ID or URL.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print only machine-readable JSON.",
    )

    args = parser.parse_args()

    summary = evaluate_video(
        args.video,
        display=not args.json,
    )

    if args.json:
        print(json.dumps(summary, indent=2))

    # The raw-anchor gold benchmark remains diagnostic: production uses
    # context expansion, so the production-expanded gold benchmark is the
    # quality gate for the live evaluator.
    gold_context_passed = (
        summary["gold_context_benchmark"] is None
        or summary["gold_context_benchmark"]["passed"]
    )
    temporal_passed = (
        summary["temporal_benchmark"] is None
        or summary["temporal_benchmark"]["passed"]
    )

    return 0 if (
        summary["failed"] == 0
        and summary["benchmark_passed"]
        and gold_context_passed
        and temporal_passed
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
