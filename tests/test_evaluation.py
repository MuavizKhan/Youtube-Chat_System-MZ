import pytest
from langchain_core.documents import Document

from Backend.rag_system.evaluation import (
    EXPECTED_RETRIEVAL,
    TEST_QUESTIONS,
    BENCHMARK_THRESHOLDS,
    MAX_DISTANCE,
    build_benchmark_metrics,
    build_evaluation_summary,
    build_gold_benchmark_metrics,
    evaluate_benchmark_thresholds,
    evaluate_gold_benchmark_thresholds,
    evaluate_temporal_benchmark_thresholds,
    build_temporal_benchmark_metrics,
    _build_temporal_result_record,
    evaluate_retrieval_result,
    _build_gold_result_record,
)
from Backend.rag_system.gold_evidence import (
    GOLD_EVIDENCE_CASES,
    get_gold_evidence_case,
    matched_gold_groups,
)
from Backend.rag_system.temporal_evidence import (
    GOLD_TEMPORAL_VIDEO_ID,
    TEMPORAL_BENCHMARK_THRESHOLDS,
    TEMPORAL_EVIDENCE_CASES,
    get_temporal_evidence_case,
)


@pytest.mark.regression
def test_ids_and_expectations_are_complete():
    ids = [item["id"] for item in TEST_QUESTIONS]

    assert len(ids) == len(set(ids)) == 13
    assert set(EXPECTED_RETRIEVAL) == set(ids)
    assert EXPECTED_RETRIEVAL["Q01"] == "must_retrieve"
    assert EXPECTED_RETRIEVAL["Q11"] == "must_retrieve"
    assert EXPECTED_RETRIEVAL["Q12"] == "may_retrieve"
    assert EXPECTED_RETRIEVAL["Q13"] == "must_not_retrieve"


@pytest.mark.regression
def test_gold_evidence_annotations_cover_content_cases():
    assert [case.question_id for case in GOLD_EVIDENCE_CASES] == [
        f"Q{i:02d}" for i in range(1, 11)
    ]
    assert all(
        0.0 < case.min_group_coverage <= 1.0
        for case in GOLD_EVIDENCE_CASES
    )


@pytest.mark.regression
def test_gold_phrase_matching_is_case_insensitive():
    case = get_gold_evidence_case("Q06")

    matched = matched_gold_groups(
        "A BRAND NEEDS A PERSONALITY and surrogate advertising were central.",
        case,
    )

    assert matched == {0, 1}


@pytest.mark.regression
@pytest.mark.parametrize(
    "question_id,count,expected",
    [
        ("Q01", 1, True),
        ("Q01", 0, False),
        ("Q12", 0, True),
        ("Q12", 4, True),
        ("Q13", 0, True),
        ("Q13", 1, False),
    ],
)
def test_expectation_contract(question_id, count, expected):
    passed, _ = evaluate_retrieval_result(question_id, count)
    assert passed is expected


@pytest.mark.regression
def test_negative_retrieval_count_is_rejected():
    with pytest.raises(ValueError):
        evaluate_retrieval_result("Q01", -1)


@pytest.mark.regression
def test_benchmark_metrics_measure_recall_and_rejection():
    results = [
        {
            "id": "Q01",
            "expected": "must_retrieve",
            "passed": True,
            "retrieved": 2,
            "best_distance": 0.42,
        },
        {
            "id": "Q02",
            "expected": "must_retrieve",
            "passed": True,
            "retrieved": 3,
            "best_distance": 0.58,
        },
        {
            "id": "Q12",
            "expected": "may_retrieve",
            "passed": True,
            "retrieved": 0,
            "best_distance": None,
        },
        {
            "id": "Q13",
            "expected": "must_not_retrieve",
            "passed": True,
            "retrieved": 0,
            "best_distance": None,
        },
    ]

    metrics = build_benchmark_metrics(results)

    assert metrics["must_retrieve_cases"] == 2
    assert metrics["must_retrieve_passed"] == 2
    assert metrics["must_retrieve_recall"] == pytest.approx(1.0)
    assert metrics["must_not_retrieve_cases"] == 1
    assert metrics["must_not_retrieve_passed"] == 1
    assert metrics["must_not_retrieve_rejection_rate"] == pytest.approx(1.0)
    assert metrics["strict_pass_rate"] == pytest.approx(1.0)
    assert metrics["strict_max_best_distance"] == pytest.approx(0.58)
    assert metrics["strict_mean_best_distance"] == pytest.approx(0.50)


@pytest.mark.regression
def test_benchmark_thresholds_pass_for_clean_suite():
    metrics = {
        "strict_pass_rate": 1.0,
        "must_retrieve_recall": 1.0,
        "must_not_retrieve_rejection_rate": 1.0,
        "strict_max_best_distance": MAX_DISTANCE,
    }

    passed, failures = evaluate_benchmark_thresholds(metrics)

    assert passed
    assert failures == []


@pytest.mark.regression
def test_benchmark_thresholds_fail_when_positive_case_is_missed():
    metrics = {
        "strict_pass_rate": 11 / 12,
        "must_retrieve_recall": 10 / 11,
        "must_not_retrieve_rejection_rate": 1.0,
        "strict_max_best_distance": 1.0,
    }

    passed, failures = evaluate_benchmark_thresholds(metrics)

    assert not passed
    assert "strict_pass_rate_below_threshold" in failures
    assert "must_retrieve_recall_below_threshold" in failures


@pytest.mark.regression
def test_benchmark_thresholds_fail_when_unrelated_query_retrieves():
    metrics = {
        "strict_pass_rate": 11 / 12,
        "must_retrieve_recall": 1.0,
        "must_not_retrieve_rejection_rate": 0.0,
        "strict_max_best_distance": 1.0,
    }

    passed, failures = evaluate_benchmark_thresholds(metrics)

    assert not passed
    assert (
        "must_not_retrieve_rejection_rate_below_threshold"
        in failures
    )


@pytest.mark.regression
def test_benchmark_thresholds_fail_when_distance_contract_is_violated():
    metrics = {
        "strict_pass_rate": 1.0,
        "must_retrieve_recall": 1.0,
        "must_not_retrieve_rejection_rate": 1.0,
        "strict_max_best_distance": BENCHMARK_THRESHOLDS[
            "strict_max_best_distance"
        ] + 0.01,
    }

    passed, failures = evaluate_benchmark_thresholds(metrics)

    assert not passed
    assert "strict_best_distance_above_threshold" in failures


@pytest.mark.regression
def test_gold_result_record_scores_top_k_against_curated_groups():
    test_case = TEST_QUESTIONS[0]
    results = [
        (
            Document(
                page_content=(
                    "Kingfisher Airlines started off as a single class "
                    "airline, low cost, but with a difference. The best "
                    "flying experience we had ever seen. It had in flight, "
                    "entertainment and it offered meals."
                ),
                metadata={},
            ),
            0.10,
        ),
        (
            Document(
                page_content="Unrelated transcript text.",
                metadata={},
            ),
            0.20,
        ),
        (
            Document(
                page_content="More unrelated transcript text.",
                metadata={},
            ),
            0.30,
        ),
        (
            Document(
                page_content="Another unrelated transcript text.",
                metadata={},
            ),
            0.40,
        ),
    ]

    result = _build_gold_result_record(
        test_case=test_case,
        results=results,
    )

    assert result["hit_at_k"]
    assert result["first_relevant_rank"] == 1
    assert result["mrr"] == pytest.approx(1.0)
    assert result["matched_gold_groups"] == 3
    assert result["group_coverage"] == pytest.approx(1.0)
    assert result["precision_at_k"] == pytest.approx(0.25)
    assert result["passed"]


@pytest.mark.regression
def test_gold_benchmark_metrics_and_thresholds_pass():
    gold_results = [
        {
            "id": "Q01",
            "hit_at_k": True,
            "group_coverage": 1.0,
            "mrr": 1.0,
            "precision_at_k": 0.50,
            "passed": True,
        },
        {
            "id": "Q02",
            "hit_at_k": True,
            "group_coverage": 1.0,
            "mrr": 0.50,
            "precision_at_k": 0.25,
            "passed": True,
        },
    ]

    metrics = build_gold_benchmark_metrics(gold_results)

    assert metrics["cases"] == 2
    assert metrics["hit_rate_at_k"] == pytest.approx(1.0)
    assert metrics["mean_group_coverage"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(0.75)
    assert metrics["mean_precision_at_k"] == pytest.approx(0.375)
    assert metrics["pass_rate"] == pytest.approx(1.0)

    passed, failures = evaluate_gold_benchmark_thresholds(metrics)

    assert passed
    assert failures == []


@pytest.mark.regression
def test_gold_benchmark_thresholds_fail_on_missing_evidence():
    metrics = {
        "cases": 10,
        "hit_rate_at_k": 0.9,
        "mean_group_coverage": 0.55,
        "mrr": 0.4,
        "mean_precision_at_k": 0.10,
    }

    passed, failures = evaluate_gold_benchmark_thresholds(metrics)

    assert not passed
    assert "gold_hit_rate_below_threshold" in failures
    assert "gold_group_coverage_below_threshold" in failures
    assert "gold_mrr_below_threshold" in failures
    assert "gold_precision_below_threshold" in failures


@pytest.mark.regression
def test_summary_contains_benchmark_and_gold_contract():
    results = [
        {
            "id": "Q01",
            "expected": "must_retrieve",
            "passed": True,
            "retrieved": 2,
            "best_distance": 0.42,
        },
        {
            "id": "Q12",
            "expected": "may_retrieve",
            "passed": True,
            "retrieved": 0,
            "best_distance": None,
        },
        {
            "id": "Q13",
            "expected": "must_not_retrieve",
            "passed": True,
            "retrieved": 0,
            "best_distance": None,
        },
    ]

    gold_results = [
        {
            "id": "Q01",
            "hit_at_k": True,
            "group_coverage": 1.0,
            "mrr": 1.0,
            "precision_at_k": 0.25,
            "passed": True,
        }
    ]

    summary = build_evaluation_summary(
        "MdeQMVBuGgY",
        results,
        gold_results=gold_results,
    )

    assert summary["total"] == 3
    assert summary["passed"] == 3
    assert summary["failed"] == 0
    assert summary["pass_rate"] == pytest.approx(1.0)
    assert summary["strict_total"] == 2
    assert summary["strict_passed"] == 2
    assert summary["strict_pass_rate"] == pytest.approx(1.0)
    assert summary["benchmark_passed"]
    assert summary["benchmark_failures"] == []
    assert summary["gold_benchmark"]["passed"]
    assert summary["gold_benchmark"]["metrics"]["hit_rate_at_k"] == pytest.approx(
        1.0
    )



def _temporal_document(
    start,
    end,
    *,
    video_id=GOLD_TEMPORAL_VIDEO_ID,
    text="Kingfisher Airlines and its financial turmoil.",
):
    metadata = {"video_id": video_id}
    if start is not None:
        metadata["start"] = start
    if end is not None:
        metadata["end"] = end
    return Document(page_content=text, metadata=metadata)


@pytest.mark.regression
def test_temporal_gold_windows_are_ordered_and_question_specific():
    cases = TEMPORAL_EVIDENCE_CASES
    assert [case.question_id for case in cases] == ["Q11"]

    case = get_temporal_evidence_case("Q11")
    assert len(case.windows) == 2
    assert case.windows[0].start_seconds == pytest.approx(4040.0)
    assert case.windows[0].end_seconds == pytest.approx(5700.0)
    assert case.windows[1].start_seconds == pytest.approx(7171.0)
    assert case.windows[1].end_seconds == pytest.approx(8172.0)
    assert all(
        window.start_seconds < window.end_seconds
        for window in case.windows
    )
    assert 0.0 < case.min_window_coverage <= 1.0


@pytest.mark.regression
def test_temporal_result_scores_relevant_sections_and_valid_provenance():
    test_case = next(item for item in TEST_QUESTIONS if item["id"] == "Q11")
    results = [
        (
            _temporal_document(4100, 4200),
            0.1,
        ),
        (
            _temporal_document(7200, 7300, text="Banks and Kingfisher debt."),
            0.2,
        ),
        (
            _temporal_document(9000, 9100, text="Unrelated later section."),
            0.3,
        ),
    ]

    result = _build_temporal_result_record(test_case, results)

    assert result["hit_at_k"]
    assert result["first_relevant_rank"] == 1
    assert result["mrr"] == pytest.approx(1.0)
    assert result["matched_windows"] == 2
    assert result["matched_section_ids"] == [
        "kingfisher_rise_and_fall",
        "kingfisher_financial_turmoil",
    ]
    assert result["window_coverage"] == pytest.approx(1.0)
    assert result["timestamp_validity_rate"] == pytest.approx(1.0)
    assert result["source_identity_validity_rate"] == pytest.approx(1.0)
    assert result["passed"]


@pytest.mark.regression
def test_temporal_result_fails_when_retrieved_timestamps_miss_gold_windows():
    test_case = next(item for item in TEST_QUESTIONS if item["id"] == "Q11")
    results = [(_temporal_document(9000, 9100), 0.1)]

    result = _build_temporal_result_record(test_case, results)

    assert not result["hit_at_k"]
    assert result["matched_windows"] == 0
    assert result["window_coverage"] == pytest.approx(0.0)
    assert not result["passed"]


@pytest.mark.regression
@pytest.mark.parametrize(
    "start,end",
    [
        (None, 4100),
        (-1, 4100),
        (4200, 4100),
        (float("nan"), 4200),
        (4100, float("inf")),
        (True, 4200),
    ],
)
def test_temporal_result_rejects_invalid_timestamp_bounds(start, end):
    test_case = next(item for item in TEST_QUESTIONS if item["id"] == "Q11")
    result = _build_temporal_result_record(
        test_case,
        [(_temporal_document(start, end), 0.1)],
    )

    assert result["timestamp_validity_rate"] == pytest.approx(0.0)
    assert not result["passed"]


@pytest.mark.regression
def test_temporal_result_rejects_sources_from_another_video():
    test_case = next(item for item in TEST_QUESTIONS if item["id"] == "Q11")
    result = _build_temporal_result_record(
        test_case,
        [(_temporal_document(4100, 4200, video_id="Gfr50f6ZBvo"), 0.1)],
    )

    assert result["timestamp_validity_rate"] == pytest.approx(1.0)
    assert result["source_identity_validity_rate"] == pytest.approx(0.0)
    assert not result["hit_at_k"]
    assert not result["passed"]


@pytest.mark.regression
def test_temporal_benchmark_metrics_and_thresholds_pass():
    metrics = {
        "cases": 1,
        "hit_rate_at_k": 1.0,
        "mean_window_coverage": 0.5,
        "timestamp_validity_rate": 1.0,
        "source_identity_validity_rate": 1.0,
        "mrr": 0.5,
    }

    passed, failures = evaluate_temporal_benchmark_thresholds(metrics)

    assert passed
    assert failures == []


@pytest.mark.regression
def test_temporal_benchmark_thresholds_fail_on_bad_timestamps_or_misses():
    metrics = {
        "cases": 1,
        "hit_rate_at_k": 0.0,
        "mean_window_coverage": 0.0,
        "timestamp_validity_rate": 0.5,
        "source_identity_validity_rate": 0.0,
        "mrr": 0.0,
    }

    passed, failures = evaluate_temporal_benchmark_thresholds(metrics)

    assert not passed
    assert "temporal_hit_rate_below_threshold" in failures
    assert "temporal_window_coverage_below_threshold" in failures
    assert "temporal_timestamp_validity_below_threshold" in failures
    assert "temporal_source_identity_below_threshold" in failures
    assert "temporal_mrr_below_threshold" in failures


@pytest.mark.regression
def test_summary_includes_temporal_benchmark():
    results = [
        {
            "id": "Q11",
            "expected": "must_retrieve",
            "passed": True,
            "retrieved": 1,
            "best_distance": 0.1,
        }
    ]
    temporal_results = [
        {
            "id": "Q11",
            "hit_at_k": True,
            "window_coverage": 0.5,
            "timestamp_validity_rate": 1.0,
            "source_identity_validity_rate": 1.0,
            "mrr": 1.0,
            "passed": True,
        }
    ]

    summary = build_evaluation_summary(
        GOLD_TEMPORAL_VIDEO_ID,
        results,
        temporal_results=temporal_results,
    )

    assert summary["temporal_benchmark"]["passed"]
    assert summary["temporal_benchmark"]["failures"] == []
    assert summary["temporal_benchmark"]["metrics"]["hit_rate_at_k"] == 1.0


@pytest.mark.regression
def test_temporal_benchmark_does_not_treat_empty_results_as_pass():
    metrics = build_temporal_benchmark_metrics([])
    passed, failures = evaluate_temporal_benchmark_thresholds(metrics)

    assert not passed
    assert failures == ["temporal_benchmark_has_no_cases"]
