import pytest

from Backend.rag_system import evaluate_e2e
from Backend.rag_system.rag_evaluation import EVALUATION_CASES


@pytest.mark.unit
def test_evaluate_video_includes_observability_in_case_output(monkeypatch):
    case = EVALUATION_CASES[0]
    observability = {
        "retrieval": {
            "final_stage": "initial_strict",
            "candidate_retrieved_chunks": 1,
            "source_segments": 1,
        },
        "generation": {
            "generated": True,
            "fallback_used": False,
        },
    }
    source = {
        "source_id": 1,
        "video_id": "MdeQMVBuGgY",
        "start": 10.0,
        "end": 20.0,
        "duration": 10.0,
    }

    monkeypatch.setattr(
        evaluate_e2e,
        "EVALUATION_CASES",
        (case,),
    )
    monkeypatch.setattr(
        evaluate_e2e,
        "answer_question",
        lambda **_kwargs: {
            "answer": "A transcript-grounded answer.",
            "retrieved_chunks": 1,
            "source_segments": 1,
            "sources": [source],
            "observability": observability,
        },
    )

    report = evaluate_e2e.evaluate_video("MdeQMVBuGgY")

    assert report["passed"] == 1
    assert report["failed"] == 0
    assert report["results"][0]["observability"] == observability
