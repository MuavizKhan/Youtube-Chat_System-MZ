import pytest

from Backend.rag_system.benchmark_registry import (
    KINGFISHER_BENCHMARK,
    VIDEO_BENCHMARKS,
)
from Backend.rag_system.multi_video_evaluation import (
    REPORT_ID,
    REPORT_SCHEMA_VERSION,
    build_multi_video_report,
    evaluate_registered_videos,
)


def _stage(cases, values, passed=True, failures=None):
    return {
        "cases": cases,
        "passed": passed,
        "failures": list(failures or []),
        "metrics": values,
    }


def _video_report(
    video_id,
    language,
    raw_cases,
    raw_values,
    *,
    status="evaluated",
    context_cases=None,
    context_values=None,
    overall_passed=True,
):
    return {
        "video_id": video_id,
        "title": f"Benchmark {video_id}",
        "language": language,
        "status": status,
        "overall_passed": overall_passed,
        "stage_metrics": {
            "raw_anchor": (
                _stage(raw_cases, raw_values)
                if raw_cases is not None
                else None
            ),
            "production_context": (
                _stage(context_cases, context_values)
                if context_cases is not None
                else None
            ),
            "temporal": None,
        },
        "case_diagnostics": [],
        "failure_diagnostics": [],
    }


def test_multi_video_report_distinguishes_video_macro_and_case_weighted_metrics():
    reports = [
        _video_report(
            "english-short",
            "en",
            2,
            {
                "hit_rate_at_k": 1.0,
                "mean_group_coverage": 0.5,
                "mrr": 0.75,
                "mean_precision_at_k": 0.25,
            },
        ),
        _video_report(
            "english-long",
            "en",
            6,
            {
                "hit_rate_at_k": 0.5,
                "mean_group_coverage": 0.8,
                "mrr": 0.5,
                "mean_precision_at_k": 0.2,
            },
        ),
        _video_report(
            "unavailable-video",
            "hi",
            None,
            {},
            status="evaluation_error",
            overall_passed=False,
        ),
    ]

    report = build_multi_video_report(
        reports,
        generated_at="2026-10-16T12:00:00Z",
        curation_blockers={"status": "blocked_pending_source_verification"},
    )

    assert report["report_schema_version"] == REPORT_SCHEMA_VERSION
    assert report["report_id"] == REPORT_ID
    raw = report["summary"]["stage_metrics"]["raw_anchor"]
    assert raw["videos_contributing"] == 2
    assert raw["case_count"] == 8
    assert raw["macro_average_by_video"]["mean_group_coverage"] == pytest.approx(0.65)
    assert raw["case_weighted_average"]["mean_group_coverage"] == pytest.approx(0.725)
    assert raw["macro_average_by_video"]["mrr"] == pytest.approx(0.625)
    assert raw["case_weighted_average"]["mrr"] == pytest.approx(0.5625)

    english = report["summary"]["by_language"]["en"]["stage_metrics"]["raw_anchor"]
    assert english["videos_contributing"] == 2
    assert english["case_weighted_average"]["hit_rate_at_k"] == pytest.approx(0.625)
    assert report["summary"]["evaluation_error_video_count"] == 1
    assert report["summary"]["registered_video_count"] == 3
    assert report["summary"]["selected_video_count"] == 3
    hindi = report["summary"]["by_language"]["hi"]
    assert hindi["video_count"] == 1
    assert hindi["evaluation_error_video_count"] == 1
    assert hindi["evaluated_video_count"] == 0
    assert report["summary"]["overall_passed"] is False


def test_runner_attaches_video_query_stage_and_missing_group_diagnostics():
    benchmark = KINGFISHER_BENCHMARK
    gold_case = benchmark.gold_evidence_by_question["Q01"]
    question = next(
        case for case in benchmark.question_cases
        if case["id"] == "Q01"
    )
    summary = {
        "video_id": benchmark.video_id,
        "benchmark_configuration": {"status": "configured"},
        "total": 1,
        "passed": 1,
        "failed": 0,
        "strict_total": 1,
        "strict_passed": 1,
        "strict_failed": 0,
        "benchmark_passed": True,
        "benchmark_failures": [],
        "retrieval_config": {"top_k": 4},
        "results": [
            {
                "id": "Q01",
                "type": question["type"],
                "question": question["question"],
                "expected": "must_retrieve",
                "retrieved": 1,
                "passed": True,
                "message": "Relevant retrieval found.",
                "best_distance": 0.2,
                "retrieved_chunk_ids": [12],
            }
        ],
        "gold_benchmark": {
            "passed": False,
            "failures": ["gold_group_coverage_below_threshold"],
            "metrics": {
                "cases": 1,
                "hit_rate_at_k": 1.0,
                "mean_group_coverage": 1 / 3,
                "mrr": 1.0,
                "mean_precision_at_k": 0.25,
            },
            "results": [
                {
                    "id": "Q01",
                    "question": question["question"],
                    "gold_groups": len(gold_case.groups),
                    "matched_gold_groups": 1,
                    "matched_gold_group_indexes": [0],
                    "missing_gold_group_indexes": [1, 2],
                    "group_coverage": 1 / 3,
                    "hit_at_k": True,
                    "mrr": 1.0,
                    "precision_at_k": 0.25,
                    "passed": False,
                }
            ],
        },
        "gold_context_benchmark": {
            "passed": True,
            "failures": [],
            "metrics": {
                "cases": 1,
                "evidence_presence_rate": 1.0,
                "mean_group_coverage": 2 / 3,
                "mean_relevance_ratio": 0.25,
                "case_pass_rate": 1.0,
            },
            "results": [
                {
                    "id": "Q01",
                    "question": question["question"],
                    "gold_groups": len(gold_case.groups),
                    "matched_gold_groups": 2,
                    "matched_gold_group_indexes": [0, 1],
                    "missing_gold_group_indexes": [2],
                    "group_coverage": 2 / 3,
                    "evidence_presence": True,
                    "relevance_ratio": 0.25,
                    "passed": True,
                }
            ],
        },
        "temporal_benchmark": None,
        "gold_retrieval_diagnostics": [
            {
                "id": "Q01",
                "groups": [
                    {
                        "group": 0,
                        "candidate_count": 2,
                        "candidate_chunk_ids": [12, 13],
                        "retrieved_candidate_chunk_ids": [12],
                        "matched_index_phrase": gold_case.groups[0][0],
                    },
                    {
                        "group": 1,
                        "candidate_count": 1,
                        "candidate_chunk_ids": [14],
                        "retrieved_candidate_chunk_ids": [],
                        "matched_index_phrase": gold_case.groups[1][0],
                    },
                    {
                        "group": 2,
                        "candidate_count": 0,
                        "candidate_chunk_ids": [],
                        "retrieved_candidate_chunk_ids": [],
                        "matched_index_phrase": None,
                    },
                ],
            }
        ],
    }

    reports = evaluate_registered_videos(
        video_ids=[benchmark.video_id],
        registry={benchmark.video_id: benchmark},
        evaluator=lambda _url, display=False: summary,
    )
    video = reports[0]

    assert video["status"] == "evaluated"
    assert video["overall_passed"] is True  # raw-anchor benchmark remains diagnostic
    raw_failure = next(
        item for item in video["failure_diagnostics"]
        if item["stage"] == "raw_anchor" and item["question_id"] == "Q01"
    )
    assert raw_failure["video_id"] == benchmark.video_id
    assert raw_failure["question"] == question["question"]
    assert raw_failure["missing_gold_group_indexes"] == [1, 2]
    assert raw_failure["groups"][1]["phrases"] == list(gold_case.groups[1])
    assert raw_failure["groups"][1]["index_candidate_chunk_ids"] == [14]
    assert raw_failure["groups"][1]["production_context_candidate_chunk_ids"] == []


def test_runner_retains_per_video_errors_instead_of_skipping_them():
    benchmarks = list(VIDEO_BENCHMARKS.values())[:2]
    registry = {benchmark.video_id: benchmark for benchmark in benchmarks}
    broken_id = benchmarks[0].video_id
    good_id = benchmarks[1].video_id

    def fake_evaluator(video_url, display=False):
        if broken_id in video_url:
            raise FileNotFoundError("FAISS index is not present")
        return {
            "video_id": good_id,
            "benchmark_configuration": {"status": "configured"},
            "failed": 0,
            "passed": 0,
            "total": 0,
            "strict_total": 0,
            "strict_passed": 0,
            "strict_failed": 0,
            "benchmark_passed": True,
            "benchmark_failures": [],
            "results": [],
            "gold_benchmark": None,
            "gold_context_benchmark": None,
            "temporal_benchmark": None,
        }

    reports = evaluate_registered_videos(
        evaluator=fake_evaluator,
        registry=registry,
    )

    assert [report["video_id"] for report in reports] == [
        benchmark.video_id for benchmark in benchmarks
    ]
    assert reports[0]["status"] == "evaluation_error"
    assert reports[0]["error"]["type"] == "FileNotFoundError"
    assert reports[1]["status"] == "evaluated"
