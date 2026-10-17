from pathlib import Path

import pytest

from Backend.rag_system import multi_video_evaluation
from Backend.rag_system.benchmark_registry import (
    ARABIC_SILENCE_BENCHMARK,
    HINDI_HISTORY_BENCHMARK,
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

    assert report["report_schema_version"] == REPORT_SCHEMA_VERSION == "1.4.0"
    assert report["report_id"] == REPORT_ID == "phase10_6_experiment_provenance_v1"
    assert report["experiment_metadata"] is None
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



def test_experiment_metadata_records_model_index_runtime_and_effective_config(monkeypatch):
    config = multi_video_evaluation.rag_config
    monkeypatch.setattr(config, "PROJECT_ROOT", Path("/repo"))
    monkeypatch.setattr(
        config,
        "VECTOR_STORE_ROOT",
        Path("/repo/Backend/rag_system/vector_stores/experiments/experiment-bge-m3"),
    )
    monkeypatch.setattr(
        config,
        "INDEX_JOB_DB_PATH",
        Path("/repo/Backend/rag_system/index_jobs_experiment-bge-m3.sqlite3"),
    )
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "BAAI/bge-m3")
    monkeypatch.setattr(config, "RAG_INDEX_NAMESPACE", "experiment-bge-m3")
    monkeypatch.setattr(config, "MAX_DISTANCE", 1.12)
    monkeypatch.setattr(config, "TOP_K", 4)
    monkeypatch.setattr(config, "CHUNK_SIZE", 1000)
    monkeypatch.setattr(config, "CHUNK_OVERLAP", 200)
    monkeypatch.setattr(
        multi_video_evaluation,
        "_read_git_provenance",
        lambda: {
            "git_commit": "abc123",
            "git_worktree_tracked_changes": False,
        },
    )

    metadata = multi_video_evaluation._build_experiment_metadata()

    assert metadata["embedding_model"] == "BAAI/bge-m3"
    assert metadata["index_namespace"] == "experiment-bge-m3"
    assert metadata["vector_store_root"] == (
        "Backend/rag_system/vector_stores/experiments/experiment-bge-m3"
    )
    assert metadata["index_job_db_path"] == (
        "Backend/rag_system/index_jobs_experiment-bge-m3.sqlite3"
    )
    assert metadata["git_commit"] == "abc123"
    assert metadata["git_worktree_tracked_changes"] is False
    assert metadata["index_config"]["chunk_size"] == 1000
    assert metadata["index_config"]["chunk_overlap"] == 200
    assert metadata["effective_retrieval_config"]["max_distance"] == pytest.approx(1.12)
    assert metadata["effective_retrieval_config"]["top_k"] == 4


def test_multi_video_report_preserves_supplied_experiment_metadata():
    metadata = {
        "embedding_model": "BAAI/bge-m3",
        "index_namespace": "experiment-bge-m3",
        "git_commit": "abc123",
        "effective_retrieval_config": {"max_distance": 1.12},
    }

    report = build_multi_video_report(
        [],
        generated_at="2026-10-17T16:00:00Z",
        experiment_metadata=metadata,
    )

    assert report["experiment_metadata"] == metadata


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
    assert video["question_results"][0]["query_language"] == "en"
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


def test_temporal_validity_metrics_are_aggregated_when_annotations_exist():
    report_row = _video_report(
        "english-temporal",
        "en",
        1,
        {
            "hit_rate_at_k": 1.0,
            "mean_group_coverage": 0.8,
            "mrr": 1.0,
            "mean_precision_at_k": 0.25,
        },
    )
    report_row["stage_metrics"]["temporal"] = _stage(
        2,
        {
            "hit_rate_at_k": 1.0,
            "mean_window_coverage": 0.75,
            "timestamp_validity_rate": 1.0,
            "source_identity_validity_rate": 0.5,
            "mrr": 0.75,
            "pass_rate": 0.5,
        },
    )

    report = build_multi_video_report([report_row])
    temporal = report["summary"]["stage_metrics"]["temporal"]

    assert temporal["videos_contributing"] == 1
    assert temporal["case_count"] == 2
    assert temporal["macro_average_by_video"]["timestamp_validity_rate"] == pytest.approx(1.0)
    assert temporal["case_weighted_average"]["source_identity_validity_rate"] == pytest.approx(0.5)
    assert temporal["case_weighted_average"]["mean_window_coverage"] == pytest.approx(0.75)


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


def test_non_english_benchmarks_keep_crosslingual_cases_and_add_native_queries():
    for benchmark, language, native_ids, english_ids in (
        (
            HINDI_HISTORY_BENCHMARK,
            "hi",
            ("HI06", "HI07", "HI08", "HI09", "HI10"),
            ("HI01", "HI02", "HI03", "HI04", "HI05"),
        ),
        (
            ARABIC_SILENCE_BENCHMARK,
            "ar",
            ("AR06", "AR07", "AR08", "AR09", "AR10"),
            ("AR01", "AR02", "AR03", "AR04", "AR05"),
        ),
    ):
        cases = {
            case["id"]: case
            for case in benchmark.question_cases_for_evaluation()
        }

        assert set(native_ids).issubset(cases)
        assert set(english_ids).issubset(cases)
        assert all(cases[item]["query_language"] == language for item in native_ids)
        assert all(cases[item]["query_language"] == "en" for item in english_ids)
        assert set(native_ids[:-1]).issubset(benchmark.gold_evidence_by_question)
        assert all(
            benchmark.expected_retrieval[item] == "must_retrieve"
            for item in native_ids[:-1]
        )
        assert benchmark.expected_retrieval[native_ids[-1]] == "must_not_retrieve"


def test_multi_video_report_breaks_stage_metrics_out_by_query_language():
    report_row = _video_report(
        "arabic-demo",
        "ar",
        1,
        {
            "hit_rate_at_k": 0.0,
            "mean_group_coverage": 0.0,
            "mrr": 0.0,
            "mean_precision_at_k": 0.0,
        },
        overall_passed=False,
    )
    report_row["question_results"] = [
        {"id": "AR02", "query_language": "en"},
        {"id": "AR07", "query_language": "ar"},
    ]
    report_row["case_diagnostics"] = [
        {
            "query_language": "en",
            "stage": "raw_anchor",
            "passed": False,
            "metrics": {
                "hit_at_k": False,
                "group_coverage": 0.0,
                "mrr": 0.0,
                "precision_at_k": 0.0,
            },
        },
        {
            "query_language": "ar",
            "stage": "raw_anchor",
            "passed": True,
            "metrics": {
                "hit_at_k": True,
                "group_coverage": 1.0,
                "mrr": 1.0,
                "precision_at_k": 0.25,
            },
        },
        {
            "query_language": "ar",
            "stage": "production_context",
            "passed": True,
            "metrics": {
                "evidence_presence": True,
                "group_coverage": 1.0,
                "relevance_ratio": 0.25,
            },
        },
    ]

    report = build_multi_video_report([report_row])
    query_languages = report["summary"]["by_query_language"]

    assert query_languages["en"]["question_count"] == 1
    assert query_languages["en"]["stage_metrics"]["raw_anchor"]["metrics"][
        "hit_rate_at_k"
    ] == pytest.approx(0.0)
    assert query_languages["ar"]["question_count"] == 1
    assert query_languages["ar"]["stage_metrics"]["raw_anchor"]["metrics"][
        "hit_rate_at_k"
    ] == pytest.approx(1.0)
    assert query_languages["ar"]["stage_metrics"]["production_context"][
        "metrics"
    ]["evidence_presence_rate"] == pytest.approx(1.0)


def test_report_separates_transcript_query_language_pairs_and_negative_controls():
    arabic = _video_report(
        "arabic-demo",
        "ar",
        1,
        {
            "hit_rate_at_k": 0.0,
            "mean_group_coverage": 0.0,
            "mrr": 0.0,
            "mean_precision_at_k": 0.0,
        },
        overall_passed=False,
    )
    arabic["question_results"] = [
        {
            "id": "AR02", "query_language": "en",
            "expected": "must_retrieve", "retrieved": 3, "passed": True,
        },
        {
            "id": "AR05", "query_language": "en",
            "expected": "must_not_retrieve", "retrieved": 0, "passed": True,
        },
        {
            "id": "AR07", "query_language": "ar",
            "expected": "must_retrieve", "retrieved": 3, "passed": True,
        },
        {
            "id": "AR10", "query_language": "ar",
            "expected": "must_not_retrieve", "retrieved": 0, "passed": True,
        },
    ]
    arabic["case_diagnostics"] = [
        {
            "query_language": "en", "stage": "raw_anchor", "passed": False,
            "metrics": {"hit_at_k": False, "group_coverage": 0.0},
        },
        {
            "query_language": "ar", "stage": "raw_anchor", "passed": True,
            "metrics": {"hit_at_k": True, "group_coverage": 1.0},
        },
        {
            "query_language": "ar", "stage": "production_context",
            "passed": True,
            "metrics": {"evidence_presence": True, "group_coverage": 1.0},
        },
    ]

    hindi = _video_report(
        "hindi-demo",
        "hi",
        1,
        {
            "hit_rate_at_k": 0.0,
            "mean_group_coverage": 0.0,
            "mrr": 0.0,
            "mean_precision_at_k": 0.0,
        },
        overall_passed=False,
    )
    hindi["question_results"] = [
        {
            "id": "HI04", "query_language": "en",
            "expected": "must_retrieve", "retrieved": 5, "passed": True,
        },
        {
            "id": "HI10", "query_language": "hi",
            "expected": "must_not_retrieve", "retrieved": 11, "passed": False,
        },
    ]

    report = build_multi_video_report([arabic, hindi])
    by_pair = report["summary"]["by_transcript_and_query_language"]

    arabic_english = by_pair["ar"]["en"]
    assert arabic_english["question_count"] == 2
    assert arabic_english["retrieval_contract"]["must_retrieve"]["retrieval_presence_rate"] == pytest.approx(1.0)
    assert "recall" not in arabic_english["retrieval_contract"]["must_retrieve"]
    assert arabic_english["retrieval_contract"]["must_not_retrieve"]["rejection_rate"] == pytest.approx(1.0)
    assert arabic_english["stage_metrics"]["raw_anchor"]["metrics"]["hit_rate_at_k"] == pytest.approx(0.0)

    arabic_native = by_pair["ar"]["ar"]
    assert arabic_native["question_count"] == 2
    assert arabic_native["retrieval_contract"]["must_not_retrieve"]["rejection_rate"] == pytest.approx(1.0)
    assert arabic_native["stage_metrics"]["production_context"]["metrics"]["case_pass_rate"] == pytest.approx(1.0)

    hindi_native = by_pair["hi"]["hi"]
    assert hindi_native["retrieval_contract"]["must_not_retrieve"]["case_count"] == 1
    assert hindi_native["retrieval_contract"]["must_not_retrieve"]["rejection_rate"] == pytest.approx(0.0)
    assert hindi_native["retrieval_contract"]["strict"]["pass_rate"] == pytest.approx(0.0)

    # Same English query language is deliberately separated by transcript language.
    assert by_pair["hi"]["en"]["question_count"] == 1
    assert by_pair["ar"]["en"]["question_count"] == 2
