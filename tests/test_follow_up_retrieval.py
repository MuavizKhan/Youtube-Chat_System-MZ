from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from Backend.rag_system import chain, query_understanding


@pytest.mark.parametrize(
    "question",
    [
        "Which studies?",
        "What sources is the speaker referring to?",
        "What about their methodology?",
        "How about the second example?",
        "What does that result mean?",
        "Which company was involved?",
    ],
)
def test_generic_follow_up_language_is_detected(question):
    assert query_understanding.is_likely_follow_up(question)


def test_follow_up_query_plan_retains_alternative_searches(monkeypatch):
    calls = []

    def fake_understand(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            intent="follow_up",
            standalone_question="Which research studies did the speaker mention?",
            search_queries=[
                "Which research studies did the speaker mention?",
                "study participant count research findings",
                "study publication source and methods",
            ],
        )

    monkeypatch.setattr(chain, "understand_query", fake_understand)
    history = [
        {"role": "user", "content": "What did the presenter say about this method?"},
        {"role": "assistant", "content": "The presenter described a study."},
    ]

    resolved, variants = chain.resolve_conversational_query_plan(
        question="Which studies?",
        conversation_history=history,
        llm_client=object(),
    )

    assert resolved == "Which research studies did the speaker mention?"
    assert variants == [
        "study participant count research findings",
        "study publication source and methods",
    ]
    assert calls[0]["conversation_history"] == history


def test_non_follow_up_does_not_call_query_planner(monkeypatch):
    monkeypatch.setattr(
        chain,
        "understand_query",
        lambda **kwargs: pytest.fail("Planner must not run for standalone questions."),
    )
    resolved, variants = chain.resolve_conversational_query_plan(
        question="Explain how an engine works.",
        conversation_history=[
            {"role": "user", "content": "What is a wheel?"},
            {"role": "assistant", "content": "A wheel is a component."},
        ],
        llm_client=object(),
    )
    assert resolved == "Explain how an engine works."
    assert variants == []


def test_query_variant_retrieval_keeps_only_strong_transcript_evidence(monkeypatch):
    weak = Document(
        page_content="Generic surrounding transcript words.",
        metadata={"chunk_id": 1, "video_id": "video-1", "start": 0, "end": 10},
    )
    strong = Document(
        page_content="The presenter identifies the research methodology and sample.",
        metadata={"chunk_id": 2, "video_id": "video-1", "start": 10, "end": 20},
    )
    calls = []

    def fake_retrieve(**kwargs):
        calls.append(kwargs["query"])
        if kwargs["query"] == "which studies":
            return [], [], {"weak_native_query_rejected": False}
        if kwargs["query"] == "research methodology sample size":
            return [(weak, 1.2)], [(weak, 1.2)], {}
        if kwargs["query"] == "study source publication":
            return [(strong, 0.1)], [(strong, 0.1)], {}
        raise AssertionError(kwargs["query"])

    monkeypatch.setattr(chain, "retrieve_temporal_context", lambda **kwargs: None)
    monkeypatch.setattr(chain, "retrieve_question_context", fake_retrieve)
    monkeypatch.setattr(
        chain,
        "passes_strong_semantic_gate",
        lambda results, *_args, **_kwargs: bool(results and results[0][1] <= 0.5),
    )
    monkeypatch.setattr(
        chain,
        "expand_question_context",
        lambda _store, _query, anchors: anchors,
    )
    monkeypatch.setattr(
        chain,
        "_probe_q03_acquisition_evidence",
        lambda **kwargs: (
            kwargs["retrieved_results"],
            kwargs["citation_results"],
            None,
            False,
        ),
    )
    monkeypatch.setattr(
        chain,
        "_probe_q08_bank_payment_attribution_evidence",
        lambda **kwargs: (
            kwargs["retrieved_results"],
            kwargs["citation_results"],
            None,
            False,
        ),
    )

    results, trace = chain._retrieve_with_query_recovery_trace(
        vector_store=object(),
        question="which studies",
        video_id="video-1",
        query_variants=[
            "research methodology sample size",
            "study source publication",
        ],
    )

    assert calls == [
        "which studies",
        "research methodology sample size",
        "study source publication",
    ]
    assert [doc.metadata["chunk_id"] for doc, _distance in results] == [2]
    assert trace["conversation_variant_attempted"] is True
    assert trace["conversation_variant_accepted"] is True
    assert trace["conversation_variant_count"] == 2
