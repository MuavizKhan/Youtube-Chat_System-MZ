import pytest

from Backend.rag_system.chain import FALLBACK_ANSWER
from Backend.rag_system.rag_evaluation import (
    EVALUATION_CASES,
    evaluate_response_contract,
)


@pytest.mark.regression
def test_end_to_end_evaluation_dataset_is_stable():
    assert [case.id for case in EVALUATION_CASES] == [
        "E01", "E02", "E03", "E04", "E05", "E06"
    ]
    assert all(case.retrieval_case_id for case in EVALUATION_CASES)


@pytest.mark.regression
@pytest.mark.parametrize(
    "answer,answerable,expected",
    [
        ("Kingfisher Airlines was discussed in the interview.", True, True),
        (FALLBACK_ANSWER, False, True),
        ("I do not know.", False, False),
        ("", True, False),
        ("[SOURCE 1] The speaker discusses the airline.", True, False),
        ("Video ID: abc123. The speaker discusses the airline.", True, False),
        ("Timestamp: 01:20 - 02:00", True, False),
    ],
)
def test_response_contract(answer, answerable, expected):
    result = evaluate_response_contract(
        answer=answer,
        answerable=answerable,
    )
    assert result.passed is expected


@pytest.mark.regression
def test_answerable_cases_are_not_marked_unanswerable():
    for case in EVALUATION_CASES:
        result = evaluate_response_contract(
            answer="Grounded transcript answer.",
            answerable=case.answerable,
        )

        if case.answerable:
            assert result.passed


@pytest.mark.regression
def test_unanswerable_cases_require_exact_fallback():
    for case in EVALUATION_CASES:
        if case.answerable:
            continue

        result = evaluate_response_contract(
            answer=FALLBACK_ANSWER,
            answerable=False,
        )
        assert result.passed
