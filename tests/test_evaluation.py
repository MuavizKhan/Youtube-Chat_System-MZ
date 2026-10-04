import pytest

from Backend.rag_system.evaluation import (
    EXPECTED_RETRIEVAL,
    TEST_QUESTIONS,
    BENCHMARK_THRESHOLDS,
    MAX_DISTANCE,
    build_benchmark_metrics,
    build_evaluation_summary,
    evaluate_benchmark_thresholds,
    evaluate_retrieval_result,
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
def test_summary_contains_benchmark_contract():
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

    summary = build_evaluation_summary("MdeQMVBuGgY", results)

    assert summary["total"] == 3
    assert summary["passed"] == 3
    assert summary["failed"] == 0
    assert summary["pass_rate"] == pytest.approx(1.0)
    assert summary["strict_total"] == 2
    assert summary["strict_passed"] == 2
    assert summary["strict_pass_rate"] == pytest.approx(1.0)
    assert summary["benchmark_passed"]
    assert summary["benchmark_failures"] == []
    assert summary["benchmark_metrics"]["must_retrieve_recall"] == pytest.approx(
        1.0
    )
