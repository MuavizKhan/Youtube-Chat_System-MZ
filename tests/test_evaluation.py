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
    evaluate_retrieval_result,
    _build_gold_result_record,
)
from Backend.rag_system.gold_evidence import (
    GOLD_EVIDENCE_CASES,
    get_gold_evidence_case,
    matched_gold_groups,
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
