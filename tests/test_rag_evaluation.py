import pytest

from Backend.rag_system.chain import FALLBACK_ANSWER
from Backend.rag_system.rag_evaluation import EVALUATION_CASES
from Backend.rag_system.metrics import evaluate_response_contract


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



def source_record(
    *,
    source_id=1,
    video_id="Gfr50f6ZBvo",
    start=10.0,
    end=25.0,
):
    return {
        "source_id": source_id,
        "chunk_ids": [source_id],
        "video_id": video_id,
        "start": start,
        "end": end,
        "duration": end - start,
        "distance": 0.4,
    }


@pytest.mark.regression
def test_answerable_response_requires_retrieval_and_sources():
    result = evaluate_response_contract(
        answer="Grounded transcript answer.",
        answerable=True,
        retrieved_chunks=2,
        source_segments=1,
        sources=[source_record()],
    )

    assert result.passed
    assert result.checks["answerable_has_retrieval"]
    assert result.checks["answerable_has_sources"]
    assert result.checks["source_bounds_valid"]


@pytest.mark.regression
def test_answerable_response_fails_without_retrieval():
    result = evaluate_response_contract(
        answer="Grounded transcript answer.",
        answerable=True,
        retrieved_chunks=0,
        source_segments=0,
        sources=[],
    )

    assert not result.passed
    assert "answerable_case_missing_retrieval" in result.failures
    assert "answerable_case_missing_sources" in result.failures


@pytest.mark.regression
def test_unanswerable_response_rejects_retrieved_evidence():
    result = evaluate_response_contract(
        answer=FALLBACK_ANSWER,
        answerable=False,
        retrieved_chunks=1,
        source_segments=1,
        sources=[source_record()],
    )

    assert not result.passed
    assert "unanswerable_case_retrieved_evidence" in result.failures
    assert "unanswerable_case_has_sources" in result.failures


@pytest.mark.regression
@pytest.mark.parametrize(
    "sources,source_segments,expected_failure",
    [
        (
            [source_record(start=30.0, end=20.0)],
            1,
            "source_bounds_invalid",
        ),
        (
            [source_record(video_id="")],
            1,
            "source_identity_invalid",
        ),
        (
            [source_record()],
            2,
            "source_count_mismatch",
        ),
    ],
)
def test_source_provenance_contract_detects_invalid_metadata(
    sources,
    source_segments,
    expected_failure,
):
    result = evaluate_response_contract(
        answer="Grounded transcript answer.",
        answerable=None,
        retrieved_chunks=1,
        source_segments=source_segments,
        sources=sources,
    )

    assert not result.passed
    assert expected_failure in result.failures
