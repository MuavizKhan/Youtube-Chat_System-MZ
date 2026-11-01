import pytest

from scripts.phase11_2_live_validation import evaluate_case_response, select_cases


def _case(
    *,
    expected_answerable=True,
    expected_route="targeted",
):
    return {
        "id": "TEST01",
        "video_id": "INpnuL7qnzY",
        "question": "Explain the major historical developments and their consequences.",
        "kind": "test",
        "expected_answerable": expected_answerable,
        "expected_route": expected_route,
        "purpose": "Unit-test case.",
    }


def _response(
    *,
    final_stage="initial_strict",
    answer="The transcript describes the historical events.",
    sources=None,
    attempts=None,
):
    return {
        "answer": answer,
        "retrieved_chunks": 2,
        "source_segments": len(sources or []),
        "sources": (
            sources
            if sources is not None
            else [
                {
                    "source_id": "source-1",
                    "start": 60,
                    "end": 90,
                    "chunk_ids": [1],
                    "url": "https://www.youtube.com/watch?v=INpnuL7qnzY&t=60s",
                }
            ]
        ),
        "observability": {
            "retrieval": {
                "final_stage": final_stage,
                "planner_used": True,
                "planner_intent": "factual",
                "attempts": attempts or [
                    {"stage": "initial_strict", "retrieved_chunks": 1}
                ],
            },
            "generation": {
                "generated": True,
                "fallback_used": False,
                "latency_ms": 120.0,
                "context_budget": {
                    "budget_chars": 8000,
                    "final_chars": 3200,
                },
            },
        },
    }


@pytest.mark.unit
def test_targeted_case_rejects_best_effort_overview_final_stage():
    audit = evaluate_case_response(
        _case(),
        _response(final_stage="best_effort_overview"),
    )

    assert audit["passed"] is False
    assert "specific_question_avoids_overview_sampling" in audit["failed_checks"]


@pytest.mark.regression
def test_targeted_case_rejects_overview_attempt_even_when_final_stage_is_rewritten():
    audit = evaluate_case_response(
        _case(),
        _response(
            final_stage="rewritten_strict",
            attempts=[
                {"stage": "initial_strict", "retrieved_chunks": 0},
                {"stage": "overview", "retrieved_chunks": 6},
                {"stage": "rewritten_strict", "retrieved_chunks": 2},
            ],
        ),
    )

    assert audit["passed"] is False
    assert "specific_question_avoids_overview_sampling" in audit["failed_checks"]


@pytest.mark.unit
def test_targeted_case_accepts_planner_failure_original_query_recovery():
    audit = evaluate_case_response(
        _case(),
        _response(final_stage="planner_failed_original_query_relaxed"),
    )

    assert audit["passed"] is True
    assert "specific_question_avoids_overview_sampling" not in audit["failed_checks"]


@pytest.mark.unit
def test_unanswerable_case_requires_abstention_and_suppresses_sources_and_timestamps():
    case = _case(expected_answerable=False)
    response = _response(
        answer=(
            "I couldn't find enough information about that in the video. "
            "It may not be covered by this video."
        ),
        sources=[],
    )

    audit = evaluate_case_response(case, response)

    assert audit["passed"] is True
    assert audit["failed_checks"] == []


@pytest.mark.unit
def test_unanswerable_case_fails_when_sources_or_timestamp_summary_are_returned():
    case = _case(expected_answerable=False)
    response = _response(
        answer=(
            "I couldn't find enough information about that in the video. "
            "Relevant timestamps: 00:01–00:15"
        ),
    )

    audit = evaluate_case_response(case, response)

    assert audit["passed"] is False
    assert "unanswerable_case_suppresses_sources" in audit["failed_checks"]
    assert "unanswerable_case_suppresses_timestamps" in audit["failed_checks"]


@pytest.mark.unit
def test_answerable_case_rejects_malformed_source_intervals():
    response = _response(
        sources=[
            {
                "source_id": "source-1",
                "start": 90,
                "end": 60,
                "chunk_ids": [1],
            }
        ],
    )

    audit = evaluate_case_response(_case(), response)

    assert audit["passed"] is False
    assert "source_intervals_are_valid" in audit["failed_checks"]


@pytest.mark.unit
def test_overview_allowed_case_does_not_apply_targeted_route_restriction():
    audit = evaluate_case_response(
        _case(expected_route="overview_allowed"),
        _response(final_stage="overview"),
    )

    assert audit["passed"] is True
    assert "specific_question_avoids_overview_sampling" not in audit["failed_checks"]


@pytest.mark.unit
def test_select_cases_rejects_duplicate_and_unknown_ids():
    with pytest.raises(ValueError, match="duplicate"):
        select_cases("HI_ANALYTICAL_01,HI_ANALYTICAL_01")

    with pytest.raises(ValueError, match="Unknown"):
        select_cases("UNKNOWN_CASE")


@pytest.mark.unit
def test_select_cases_preserves_requested_order():
    selected = select_cases("AR_MULTIPART_01,HI_ANALYTICAL_01")

    assert [case["id"] for case in selected] == [
        "AR_MULTIPART_01",
        "HI_ANALYTICAL_01",
    ]
