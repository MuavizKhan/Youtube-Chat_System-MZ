"""
Multi-video, multilingual retrieval benchmark runner and report aggregation.

This module deliberately separates deterministic report aggregation from
running retrieval against local vector stores. Unit tests can exercise the
aggregation without network calls, model-provider calls, or local indexes.
"""

import argparse
import json
import math
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .benchmark_registry import (
    BENCHMARK_CURATION_BLOCKERS,
    VIDEO_BENCHMARKS,
    VideoBenchmark,
)
from .evaluation import evaluate_video


REPORT_SCHEMA_VERSION = "1.1.0"
REPORT_ID = "phase10_5_query_language_coverage_v1"

_STAGE_SPECS = {
    "raw_anchor": {
        "summary_key": "gold_benchmark",
        "metric_keys": (
            "hit_rate_at_k",
            "mean_group_coverage",
            "mrr",
            "mean_precision_at_k",
        ),
    },
    "production_context": {
        "summary_key": "gold_context_benchmark",
        "metric_keys": (
            "evidence_presence_rate",
            "mean_group_coverage",
            "mean_relevance_ratio",
            "case_pass_rate",
        ),
    },
    "temporal": {
        "summary_key": "temporal_benchmark",
        "metric_keys": (
            "hit_rate_at_k",
            "mean_window_coverage",
            "timestamp_validity_rate",
            "source_identity_validity_rate",
            "mrr",
            "pass_rate",
        ),
    },
}


def _finite_number(value: Any) -> bool:
    """Whether a value can safely participate in numeric aggregation."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _stage_metrics(summary: dict[str, Any], stage: str) -> dict[str, Any] | None:
    """Read metrics and pass/fail for one stage of a single-video summary."""
    spec = _STAGE_SPECS[stage]
    benchmark = summary.get(spec["summary_key"])
    if not isinstance(benchmark, dict):
        return None

    metrics = benchmark.get("metrics")
    if not isinstance(metrics, dict):
        return None

    cases = metrics.get("cases", len(benchmark.get("results") or []))
    if not isinstance(cases, int) or isinstance(cases, bool) or cases <= 0:
        return None

    selected = {
        key: float(metrics[key])
        for key in spec["metric_keys"]
        if key in metrics and _finite_number(metrics[key])
    }
    return {
        "cases": cases,
        "passed": benchmark.get("passed"),
        "failures": list(benchmark.get("failures") or []),
        "metrics": selected,
    }


def _group_details(
    *,
    gold_case: Any,
    matched_indexes: Sequence[int],
    missing_indexes: Sequence[int],
    index_diagnostics: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Join annotation groups with their stage-level match and index trace."""
    groups_by_index = {
        item.get("group"): item
        for item in (index_diagnostics or {}).get("groups", [])
        if isinstance(item, dict)
    }
    matched = set(matched_indexes)
    missing = set(missing_indexes)
    details = []

    for group_index, phrases in enumerate(gold_case.groups):
        index_record = groups_by_index.get(group_index, {})
        details.append({
            "group_index": group_index,
            "phrases": list(phrases),
            "matched_at_stage": group_index in matched,
            "missing_at_stage": group_index in missing,
            "index_candidate_count": index_record.get("candidate_count"),
            "index_candidate_chunk_ids": index_record.get(
                "candidate_chunk_ids", []
            ),
            "production_context_candidate_chunk_ids": index_record.get(
                "retrieved_candidate_chunk_ids", []
            ),
            "matched_index_phrase": index_record.get("matched_index_phrase"),
        })

    return details


def _build_case_diagnostics(
    *,
    benchmark: VideoBenchmark,
    summary: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return complete case diagnostics and the subset that failed a stage."""
    video_id = benchmark.video_id
    language = benchmark.transcript_language
    gold_cases = benchmark.gold_evidence_by_question
    temporal_cases = benchmark.temporal_evidence_by_question
    index_diagnostics = {
        item.get("id"): item
        for item in (summary.get("gold_retrieval_diagnostics") or [])
        if isinstance(item, dict)
    }
    query_language_by_id = {
        item["id"]: item.get("query_language", "en")
        for item in benchmark.question_cases
    }

    diagnostics: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    # Keep per-query retrieval-contract failures visible even if no gold phrase
    # annotation exists for that control question.
    for result in summary.get("results") or []:
        if result.get("passed") is True:
            continue
        diagnostic = {
            "video_id": video_id,
            "language": language,
            "question_id": result.get("id"),
            "question": result.get("question"),
            "query_language": result.get(
                "query_language",
                query_language_by_id.get(result.get("id"), "en"),
            ),
            "stage": "retrieval_contract",
            "passed": False,
            "failure_reasons": [result.get("message", "retrieval_contract_failed")],
            "expected": result.get("expected"),
            "retrieved": result.get("retrieved"),
            "best_distance": result.get("best_distance"),
            "retrieved_chunk_ids": result.get("retrieved_chunk_ids", []),
        }
        diagnostics.append(diagnostic)
        failures.append(diagnostic)

    stage_result_keys = (
        ("raw_anchor", "gold_benchmark"),
        ("production_context", "gold_context_benchmark"),
        ("temporal", "temporal_benchmark"),
    )
    for stage, summary_key in stage_result_keys:
        stage_benchmark = summary.get(summary_key)
        if not isinstance(stage_benchmark, dict):
            continue
        for result in stage_benchmark.get("results") or []:
            question_id = result.get("id")
            diagnostic: dict[str, Any] = {
                "video_id": video_id,
                "language": language,
                "question_id": question_id,
                "question": result.get("question"),
                "query_language": query_language_by_id.get(question_id, "en"),
                "stage": stage,
                "passed": bool(result.get("passed")),
                "metrics": {},
            }

            if stage in {"raw_anchor", "production_context"}:
                gold_case = gold_cases.get(question_id)
                matched_indexes = result.get("matched_gold_group_indexes", [])
                missing_indexes = result.get(
                    "missing_gold_group_indexes",
                    [
                        i for i in range(int(result.get("gold_groups", 0)))
                        if i not in set(matched_indexes)
                    ],
                )
                if gold_case is not None:
                    diagnostic["groups"] = _group_details(
                        gold_case=gold_case,
                        matched_indexes=matched_indexes,
                        missing_indexes=missing_indexes,
                        index_diagnostics=index_diagnostics.get(question_id),
                    )
                diagnostic["metrics"] = {
                    key: result[key]
                    for key in (
                        "hit_at_k",
                        "group_coverage",
                        "mrr",
                        "precision_at_k",
                        "evidence_presence",
                        "relevance_ratio",
                        "matched_gold_groups",
                        "gold_groups",
                    )
                    if key in result
                }
                diagnostic["missing_gold_group_indexes"] = list(missing_indexes)
                diagnostic["failure_reasons"] = (
                    [f"{stage}_evidence_coverage_below_required"]
                    if not diagnostic["passed"]
                    else []
                )
            else:
                temporal_case = temporal_cases.get(question_id)
                diagnostic["metrics"] = {
                    key: result[key]
                    for key in (
                        "hit_at_k",
                        "window_coverage",
                        "mrr",
                        "timestamp_validity_rate",
                        "source_identity_validity_rate",
                        "matched_windows",
                        "gold_windows",
                    )
                    if key in result
                }
                diagnostic["matched_section_ids"] = result.get(
                    "matched_section_ids", []
                )
                diagnostic["matched_section_labels"] = result.get(
                    "matched_section_labels", []
                )
                diagnostic["failure_reasons"] = (
                    ["temporal_evidence_or_provenance_below_required"]
                    if not diagnostic["passed"]
                    else []
                )
                if temporal_case is not None:
                    diagnostic["expected_windows"] = [
                        {
                            "section_id": window.section_id,
                            "label": window.label,
                            "start_seconds": window.start_seconds,
                            "end_seconds": window.end_seconds,
                            "topic_phrases": list(window.topic_phrases),
                        }
                        for window in temporal_case.windows
                    ]

            diagnostics.append(diagnostic)
            if not diagnostic["passed"]:
                failures.append(diagnostic)

    return diagnostics, failures


def _video_quality_passed(summary: dict[str, Any]) -> bool:
    """Match the existing single-video CLI hard gates; raw gold stays diagnostic."""
    context = summary.get("gold_context_benchmark")
    temporal = summary.get("temporal_benchmark")
    return (
        summary.get("failed", 0) == 0
        and summary.get("benchmark_passed") is True
        and (context is None or context.get("passed") is True)
        and (temporal is None or temporal.get("passed") is True)
    )


def evaluate_registered_videos(
    *,
    video_ids: Sequence[str] | None = None,
    evaluator: Callable[..., dict[str, Any]] | None = None,
    registry: dict[str, VideoBenchmark] | None = None,
) -> list[dict[str, Any]]:
    """Run each selected registered video, retaining errors instead of hiding them."""
    if evaluator is None:
        evaluator = evaluate_video
    if registry is None:
        registry = VIDEO_BENCHMARKS

    selected_ids = list(video_ids) if video_ids is not None else list(registry)
    unknown_ids = [video_id for video_id in selected_ids if video_id not in registry]
    if unknown_ids:
        raise ValueError(f"Unregistered benchmark video IDs: {unknown_ids}")
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("video_ids must not contain duplicates.")

    video_reports: list[dict[str, Any]] = []
    for video_id in selected_ids:
        benchmark = registry[video_id]
        base = {
            "video_id": benchmark.video_id,
            "title": benchmark.title,
            "language": benchmark.transcript_language,
            "video_url": benchmark.video_url,
            "caption_track_kind": benchmark.caption_track_kind,
            "caption_validation_note": benchmark.caption_validation_note,
            "runtime_seconds": benchmark.runtime_seconds,
            "transcript_word_count": benchmark.transcript_word_count,
            "transcript_source_url": benchmark.transcript_source_url,
            "annotation_source_urls": list(benchmark.annotation_source_urls),
        }

        try:
            summary = evaluator(benchmark.video_url, display=False)
        except Exception as error:  # retain the failed video and continue the pilot
            video_reports.append({
                **base,
                "status": "evaluation_error",
                "overall_passed": False,
                "failure_reasons": [
                    f"evaluation_error:{type(error).__name__}: {error}"
                ],
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
                "stage_metrics": {
                    "raw_anchor": None,
                    "production_context": None,
                    "temporal": None,
                },
                "case_diagnostics": [],
                "failure_diagnostics": [],
            })
            continue

        benchmark_status = (
            summary.get("benchmark_configuration") or {}
        ).get("status", "configured")
        if benchmark_status != "configured":
            video_reports.append({
                **base,
                "status": "not_configured",
                "overall_passed": False,
                "failure_reasons": ["no_benchmark_configured"],
                "stage_metrics": {
                    "raw_anchor": None,
                    "production_context": None,
                    "temporal": None,
                },
                "case_diagnostics": [],
                "failure_diagnostics": [],
            })
            continue

        case_diagnostics, failure_diagnostics = _build_case_diagnostics(
            benchmark=benchmark,
            summary=summary,
        )
        stage_metrics = {
            stage: _stage_metrics(summary, stage)
            for stage in _STAGE_SPECS
        }
        retrieval_contract = {
            "total": summary.get("total", 0),
            "passed": summary.get("passed", 0),
            "failed": summary.get("failed", 0),
            "strict_total": summary.get("strict_total", 0),
            "strict_passed": summary.get("strict_passed", 0),
            "strict_failed": summary.get("strict_failed", 0),
            "benchmark_passed": summary.get("benchmark_passed"),
            "benchmark_failures": list(summary.get("benchmark_failures") or []),
        }
        quality_passed = _video_quality_passed(summary)
        reasons = []
        if summary.get("failed", 0):
            reasons.append("retrieval_contract_failed")
        if summary.get("benchmark_passed") is not True:
            reasons.extend(
                summary.get("benchmark_failures") or ["retrieval_benchmark_failed"]
            )
        context = summary.get("gold_context_benchmark")
        if context is not None and context.get("passed") is not True:
            reasons.extend(context.get("failures") or ["production_context_failed"])
        temporal = summary.get("temporal_benchmark")
        if temporal is not None and temporal.get("passed") is not True:
            reasons.extend(temporal.get("failures") or ["temporal_benchmark_failed"])

        query_language_by_id = {
            case["id"]: case.get("query_language", "en")
            for case in benchmark.question_cases
        }
        question_results = [
            {
                **item,
                "query_language": item.get(
                    "query_language",
                    query_language_by_id.get(item.get("id"), "en"),
                ),
            }
            for item in (summary.get("results") or [])
            if isinstance(item, dict)
        ]

        video_reports.append({
            **base,
            "status": "evaluated",
            "overall_passed": quality_passed,
            "failure_reasons": sorted(set(reasons)),
            "retrieval_contract": retrieval_contract,
            "stage_metrics": stage_metrics,
            "question_results": question_results,
            "case_diagnostics": case_diagnostics,
            "failure_diagnostics": failure_diagnostics,
            "retrieval_config": summary.get("retrieval_config", {}),
        })

    return video_reports


def _aggregate_stage(
    video_reports: Sequence[dict[str, Any]],
    stage: str,
) -> dict[str, Any]:
    """Produce video-macro and case-weighted means for a benchmark stage."""
    eligible = []
    for report in video_reports:
        if report.get("status") != "evaluated":
            continue
        stage_data = (report.get("stage_metrics") or {}).get(stage)
        if not isinstance(stage_data, dict) or stage_data.get("cases", 0) <= 0:
            continue
        eligible.append(stage_data)

    metric_keys = _STAGE_SPECS[stage]["metric_keys"]
    macro: dict[str, float] = {}
    weighted: dict[str, float] = {}

    for metric in metric_keys:
        values = [
            (float(item["metrics"][metric]), int(item["cases"]))
            for item in eligible
            if metric in item.get("metrics", {})
            and _finite_number(item["metrics"][metric])
        ]
        if not values:
            continue
        macro[metric] = sum(value for value, _ in values) / len(values)
        total_cases = sum(cases for _, cases in values)
        weighted[metric] = (
            sum(value * cases for value, cases in values) / total_cases
            if total_cases
            else 0.0
        )

    return {
        "videos_contributing": len(eligible),
        "case_count": sum(int(item["cases"]) for item in eligible),
        "macro_average_by_video": macro,
        "case_weighted_average": weighted,
    }


def _aggregate_by_query_language(
    video_reports: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize annotated retrieval quality by the language of each query.

    Transcript language and query language are intentionally separate axes:
    English questions against Hindi/Arabic transcripts measure cross-lingual
    retrieval, while native-language questions measure in-language retrieval.
    """
    metric_sources = {
        "raw_anchor": {
            "hit_rate_at_k": "hit_at_k",
            "mean_group_coverage": "group_coverage",
            "mrr": "mrr",
            "mean_precision_at_k": "precision_at_k",
        },
        "production_context": {
            "evidence_presence_rate": "evidence_presence",
            "mean_group_coverage": "group_coverage",
            "mean_relevance_ratio": "relevance_ratio",
        },
        "temporal": {
            "hit_rate_at_k": "hit_at_k",
            "mean_window_coverage": "window_coverage",
            "timestamp_validity_rate": "timestamp_validity_rate",
            "source_identity_validity_rate": "source_identity_validity_rate",
            "mrr": "mrr",
        },
    }
    questions_by_language: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    cases_by_language_stage: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for report in video_reports:
        if report.get("status") != "evaluated":
            continue
        video_id = report.get("video_id", "unknown")
        for question in report.get("question_results") or []:
            if not isinstance(question, dict):
                continue
            language = question.get("query_language", "en")
            questions_by_language.setdefault(language, []).append((video_id, question))

        for diagnostic in report.get("case_diagnostics") or []:
            if not isinstance(diagnostic, dict):
                continue
            language = diagnostic.get("query_language", "en")
            stage = diagnostic.get("stage")
            if stage not in metric_sources:
                continue
            cases_by_language_stage.setdefault(language, {}).setdefault(
                stage, []
            ).append(diagnostic)

    languages = sorted(set(questions_by_language) | set(cases_by_language_stage))
    output: dict[str, Any] = {}
    for language in languages:
        question_rows = questions_by_language.get(language, [])
        stage_metrics: dict[str, Any] = {}
        language_stages = cases_by_language_stage.get(language, {})
        for stage, source_map in metric_sources.items():
            rows = language_stages.get(stage, [])
            if not rows:
                continue

            values_by_metric: dict[str, list[float]] = {}
            for output_key, source_key in source_map.items():
                metric_values: list[float] = []
                for row in rows:
                    value = (row.get("metrics") or {}).get(source_key)
                    if isinstance(value, bool):
                        metric_values.append(float(value))
                    elif _finite_number(value):
                        metric_values.append(float(value))
                if metric_values:
                    values_by_metric[output_key] = metric_values

            means = {
                key: sum(values) / len(values)
                for key, values in values_by_metric.items()
            }
            if stage in {"raw_anchor", "production_context"}:
                means["case_pass_rate"] = sum(
                    1 for row in rows if row.get("passed") is True
                ) / len(rows)
            elif stage == "temporal":
                means["pass_rate"] = sum(
                    1 for row in rows if row.get("passed") is True
                ) / len(rows)

            stage_metrics[stage] = {
                "case_count": len(rows),
                "passed_cases": sum(
                    1 for row in rows if row.get("passed") is True
                ),
                "metrics": means,
            }

        output[language] = {
            "video_count": len({video_id for video_id, _ in question_rows}),
            "question_count": len(question_rows),
            "stage_metrics": stage_metrics,
        }

    return output


def build_multi_video_report(
    video_reports: Sequence[dict[str, Any]],
    *,
    generated_at: str | None = None,
    curation_blockers: dict[str, Any] | None = None,
    registered_video_count: int | None = None,
) -> dict[str, Any]:
    """Aggregate already-run single-video summaries without doing retrieval."""
    reports = [dict(item) for item in video_reports]
    if generated_at is None:
        generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if curation_blockers is None:
        curation_blockers = BENCHMARK_CURATION_BLOCKERS

    selected_video_count = len(reports)
    if registered_video_count is None:
        registered_video_count = selected_video_count
    evaluated = [item for item in reports if item.get("status") == "evaluated"]
    eval_errors = [
        item for item in reports
        if item.get("status") in {"evaluation_error", "not_configured"}
    ]
    quality_failures = [
        item for item in evaluated if item.get("overall_passed") is not True
    ]

    stage_metrics = {
        stage: _aggregate_stage(reports, stage)
        for stage in _STAGE_SPECS
    }
    languages = sorted({
        item.get("language", "unknown")
        for item in reports
    })
    by_language: dict[str, Any] = {}
    for language in languages:
        all_language_reports = [
            item for item in reports
            if item.get("language", "unknown") == language
        ]
        language_reports = [
            item for item in all_language_reports
            if item.get("status") == "evaluated"
        ]
        by_language[language] = {
            "video_count": len(all_language_reports),
            "evaluated_video_count": len(language_reports),
            "evaluation_error_video_count": sum(
                1 for item in all_language_reports
                if item.get("status") in {"evaluation_error", "not_configured"}
            ),
            "quality_passed_video_count": sum(
                1 for item in language_reports
                if item.get("overall_passed") is True
            ),
            "quality_failure_video_count": sum(
                1 for item in language_reports
                if item.get("overall_passed") is not True
            ),
            "stage_metrics": {
                stage: _aggregate_stage(language_reports, stage)
                for stage in _STAGE_SPECS
            },
        }

    overall_passed = (
        selected_video_count > 0
        and len(evaluated) == selected_video_count
        and not quality_failures
    )
    by_query_language = _aggregate_by_query_language(reports)

    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "report_id": REPORT_ID,
        "generated_at": generated_at,
        "summary": {
            "registered_video_count": registered_video_count,
            "selected_video_count": selected_video_count,
            "evaluated_video_count": len(evaluated),
            "evaluation_error_video_count": len(eval_errors),
            "quality_failure_video_count": len(quality_failures),
            "overall_passed": overall_passed,
            "stage_metrics": stage_metrics,
            "by_language": by_language,
            "by_query_language": by_query_language,
            "urdu_curation_status": {
                "status": curation_blockers.get("status"),
                "reason": curation_blockers.get("reason"),
                "verification_reference": curation_blockers.get(
                    "verification_reference"
                ),
            },
        },
        "videos": reports,
    }


def run_multi_video_evaluation(
    *,
    video_ids: Sequence[str] | None = None,
    evaluator: Callable[..., dict[str, Any]] | None = None,
    registry: dict[str, VideoBenchmark] | None = None,
) -> dict[str, Any]:
    """Run registered video benchmarks and build a versioned aggregate report."""
    if registry is None:
        registry = VIDEO_BENCHMARKS
    reports = evaluate_registered_videos(
        video_ids=video_ids,
        evaluator=evaluator,
        registry=registry,
    )
    return build_multi_video_report(
        reports,
        registered_video_count=len(registry),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run all (or selected) registered YouTube RAG benchmarks and "
            "write a versioned multi-video aggregate report."
        )
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        metavar="VIDEO_ID",
        help="Optional exact list of registered video IDs to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Optional JSON report path, e.g. "
            "evaluation_reports/phase10_5_query_language_coverage_v1.json"
        ),
    )
    args = parser.parse_args(argv)

    try:
        report = run_multi_video_evaluation(video_ids=args.videos)
    except ValueError as error:
        parser.error(str(error))

    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")

    print(serialized)
    return 0 if report["summary"]["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
