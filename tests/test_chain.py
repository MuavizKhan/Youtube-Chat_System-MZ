from types import SimpleNamespace
import pytest
from langchain_core.documents import Document
from Backend.rag_system import chain


def doc(cid,start,end,text):
    return Document(page_content=text,metadata={
        "chunk_id":cid,"start":start,"end":end,"video_id":"Gfr50f6ZBvo"
    })


@pytest.mark.unit
@pytest.mark.parametrize("seconds,expected",[(0,"00:00"),(59.9,"00:59"),(60,"01:00"),(3661,"01:01:01")])
def test_format_timestamp(seconds,expected):
    assert chain.format_timestamp(seconds)==expected


@pytest.mark.unit
def test_append_requested_temporal_timestamp_from_retrieval_window():
    trace = {
        "route": "temporal_window",
        "temporal_window": {
            "start_seconds": 1535.0,
            "end_seconds": 1584.52,
        },
    }

    answer = chain.append_requested_temporal_timestamp(
        answer="No winner is declared, and no score is announced.",
        question=(
            "What is said during the final 50 seconds? "
            "Cite the relevant timestamp."
        ),
        retrieval_trace=trace,
        has_evidence=True,
    )

    assert answer.endswith("Relevant timestamp: 25:35–26:24")


@pytest.mark.unit
@pytest.mark.parametrize(
    "question,trace,has_evidence,answer",
    [
        (
            "What happened during 25:35–26:25?",
            {"route": "temporal_window", "temporal_window": {
                "start_seconds": 1535.0, "end_seconds": 1584.52,
            }},
            True,
            "A factual answer.",
        ),
        (
            "What happened? Cite the relevant timestamp.",
            {"route": "question_aware", "temporal_window": {
                "start_seconds": 1535.0, "end_seconds": 1584.52,
            }},
            True,
            "A factual answer.",
        ),
        (
            "What happened during 25:35–26:25? Cite the timestamp.",
            {"route": "temporal_window", "temporal_window": {
                "start_seconds": 1535.0, "end_seconds": 1584.52,
            }},
            False,
            "A factual answer.",
        ),
    ],
)
def test_append_requested_temporal_timestamp_is_guarded(
    question, trace, has_evidence, answer
):
    assert chain.append_requested_temporal_timestamp(
        answer=answer,
        question=question,
        retrieval_trace=trace,
        has_evidence=has_evidence,
    ) == answer


@pytest.mark.unit
def test_append_requested_temporal_timestamp_replaces_model_summary_with_window():
    trace = {
        "route": "temporal_window",
        "temporal_window": {"start_seconds": 1535.0, "end_seconds": 1584.52},
    }
    answer = (
        "A factual answer.\n\n"
        "Relevant timestamps: 00:10–00:20"
    )

    assert chain.append_requested_temporal_timestamp(
        answer=answer,
        question="Cite the relevant timestamp for 25:35–26:25.",
        retrieval_trace=trace,
        has_evidence=True,
    ) == "A factual answer.\n\nRelevant timestamp: 25:35–26:24"


@pytest.mark.unit
def test_append_verified_response_timestamps_preserves_requested_temporal_window():
    result = chain.append_verified_response_timestamps(
        answer=(
            "A factual answer.\n\n"
            "Relevant timestamps: 00:10–00:20"
        ),
        question="Cite the relevant timestamp for 25:35–26:25.",
        sources=[{"start": 120, "end": 160, "distance": 0.1}],
        retrieval_trace={
            "route": "temporal_window",
            "temporal_window": {
                "start_seconds": 1535.0,
                "end_seconds": 1584.52,
            },
        },
        has_evidence=True,
    )

    assert result == "A factual answer.\n\nRelevant timestamp: 25:35–26:24"
    assert "00:10–00:20" not in result


@pytest.mark.unit
def test_append_relevant_source_timestamps_uses_three_strongest_in_chronological_order():
    answer = chain.append_relevant_source_timestamps(
        answer="A grounded answer.",
        sources=[
            {"start": 400, "end": 430, "distance": 0.2},
            {"start": 120, "end": 160, "distance": 0.1},
            {"start": 10, "end": 20, "distance": 0.3},
            {"start": 60, "end": 70, "distance": 0.4},
        ],
        has_evidence=True,
    )

    assert answer.endswith(
        "Relevant timestamps: 00:10–00:20, 02:00–02:40, 06:40–07:10"
    )


@pytest.mark.unit
def test_append_relevant_source_timestamps_aligns_multi_amount_claims_to_transcript_text():
    answer = (
        "Vijay says he put 3,000 crore of UB Group money into Kingfisher to keep it afloat, "
        "and debt was approximately 6,200 crore plus interest. "
        "The Ministry of Finance report showed 14,100 crore recovered. "
        "The interviewer mentions a CBI-related 423 crore figure and an ED "
        "money-laundering allegation involving 3,000 crore."
    )
    result = chain.append_relevant_source_timestamps(
        answer=answer,
        sources=[
            {
                "start": 10,
                "end": 20,
                "distance": 0.01,
                "transcript_text": "Kingfisher was known for guest service and on-time departures.",
            },
            {
                "start": 100,
                "end": 140,
                "distance": 0.8,
                "transcript_text": (
                    "I put 3,000 crores of UB Group money into Kingfisher Airlines "
                    "and infused cash to keep it afloat."
                ),
            },
            {
                "start": 200,
                "end": 240,
                "distance": 0.95,
                "transcript_text": (
                    "The DRT debt certificate records 6,203 crores plus interest. "
                    "The Ministry of Finance report showed 14,100 crores recovered."
                ),
            },
            {
                "start": 300,
                "end": 340,
                "distance": 1.01,
                "transcript_text": (
                    "The interviewer says CBI identified about 423 crores and ED "
                    "reported a 3,000 crore money-laundering issue."
                ),
            },
        ],
        has_evidence=True,
    )

    assert result.endswith(
        "Relevant timestamps: 01:40–02:20, 03:20–04:00, 05:00–05:40"
    )
    assert "00:10–00:20" not in result


@pytest.mark.unit
def test_append_relevant_source_timestamps_skips_fallback_missing_evidence_and_invalid_ranges():
    sources = [{"start": 10, "end": 20, "distance": 0.1}]

    assert chain.append_relevant_source_timestamps(
        answer=chain.FALLBACK_ANSWER,
        sources=sources,
        has_evidence=True,
    ) == chain.FALLBACK_ANSWER
    assert chain.append_relevant_source_timestamps(
        answer="A grounded answer.",
        sources=sources,
        has_evidence=False,
    ) == "A grounded answer."
    assert chain.append_relevant_source_timestamps(
        answer="A grounded answer.",
        sources=[
            {"start": float("nan"), "end": 20, "distance": 0.1},
            {"start": 20, "end": 20, "distance": 0.1},
            {"start": 30},
        ],
        has_evidence=True,
    ) == "A grounded answer."


@pytest.mark.unit
def test_append_relevant_source_timestamps_replaces_model_summary_with_source_ranges():
    answer = "A grounded answer.\n\nRelevant timestamp: 00:10–00:20"
    result = chain.append_relevant_source_timestamps(
        answer=answer,
        sources=[{"start": 120, "end": 160, "distance": 0.1}],
        has_evidence=True,
    )

    assert result == "A grounded answer.\n\nRelevant timestamps: 02:00–02:40"
    assert "00:10–00:20" not in result


@pytest.mark.unit
def test_append_relevant_source_timestamps_removes_unverified_summary_without_sources():
    answer = "A grounded answer.\n\nRelevant timestamps: 99:00–99:01"

    assert chain.append_relevant_source_timestamps(
        answer=answer,
        sources=[],
        has_evidence=False,
    ) == "A grounded answer."


@pytest.mark.unit
def test_merge_source_groups():
    groups=chain.merge_overlapping_results([
        (doc(2,40,80,"second"),0.8),
        (doc(3,120,150,"third"),0.9),
        (doc(1,10,50,"first"),0.7),
    ])
    assert len(groups)==2
    assert (groups[0]["start"],groups[0]["end"])==(10,80)
    assert groups[0]["distance"]==pytest.approx(0.7)
    assert [d.metadata["chunk_id"] for d in groups[0]["documents"]]==[1,2]


@pytest.mark.unit
def test_merge_source_groups_respects_gap():
    groups=chain.merge_overlapping_results([
        (doc(1,10,50,"first"),0.5),(doc(2,55,80,"second"),0.6)
    ],merge_gap_seconds=5)
    assert len(groups)==1


@pytest.mark.unit
def test_merge_overlapping_text():
    assert chain.merge_overlapping_text(["Alpha Beta Gamma","Gamma Delta Epsilon"],5,20)=="Alpha Beta Gamma Delta Epsilon"
    assert chain.merge_overlapping_text(["Alpha","Beta"])=="Alpha\n\nBeta"


@pytest.mark.unit
def test_format_source_group_deduplicates_chunks():
    formatted=chain.format_source_group({
        "documents":[doc(1,10,50,"Alpha"),doc(1,40,80,"Beta")],
        "start":10,"end":80,"distance":0.4
    },1)
    assert "[SOURCE 1]" in formatted
    assert "00:10 - 01:20" in formatted
    assert "Alpha" in formatted
    assert "Beta" not in formatted


@pytest.mark.unit
def test_build_context_empty():
    assert chain.build_context([])=="No relevant transcript context was retrieved."


@pytest.mark.unit
def test_sanitize_generated_answer_removes_model_authored_timestamp_summary():
    answer = (
        "A grounded answer.\n\n"
        "Relevant timestamps: 99:00–99:01"
    )

    assert chain.sanitize_generated_answer(answer) == "A grounded answer."


@pytest.mark.unit
def test_prompt_role_mapping():
    prompt=chain.RAG_PROMPT.invoke({"context":"Transcript evidence","question":"What happened?"})
    messages=chain.prompt_to_messages(prompt)
    assert [m["role"] for m in messages]==["system","user"]


@pytest.mark.unit
def test_prompt_handles_temporal_absence_and_requested_timestamp():
    prompt = chain.RAG_PROMPT.invoke({
        "context": (
            "[SOURCE 1]\nTimestamp: 25:25 - 26:24\n"
            "Transcript: The passage ends with a question about interpretation."
        ),
        "question": (
            "What is said between 25:35 and 26:25? "
            "Does either AI declare a winner or announce a score? "
            "Cite the relevant timestamp."
        ),
    })
    messages = chain.prompt_to_messages(prompt)
    system_prompt = messages[0]["content"]

    assert "Do not use this fallback solely because a named event" in system_prompt
    assert "Absence of a statement is answerable" in system_prompt
    assert "application appends a separate" in system_prompt
    assert "timestamp ranges remain part of the answer" in system_prompt
    assert "distinct figures as stated in the transcript" in system_prompt
    assert "Do not round" in system_prompt
    normalized_system_prompt = " ".join(system_prompt.split())
    assert (
        "Never turn an allegation described by the interviewer"
        in normalized_system_prompt
    )


class FakeCompletions:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.kwargs = None
        self.call_count = 0

    def create(self, **kwargs):
        self.kwargs = kwargs
        self.call_count += 1

        if self.error is not None:
            raise self.error

        return self.response


def client(response=None,error=None):
    completions=FakeCompletions(response,error)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)),completions




@pytest.mark.unit
def test_groq_generation_returns_clean_answer(monkeypatch):
    monkeypatch.setattr(chain, "GROQ_MAX_RETRIES", 0)

    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="  grounded answer  "
                )
            )
        ]
    )

    fake, completions = client(response=response)
    prompt = chain.RAG_PROMPT.invoke(
        {
            "context": "X",
            "question": "Q",
        }
    )

    assert chain.generate_with_groq(fake, prompt) == "grounded answer"
    assert completions.kwargs["model"] == chain.GROQ_MODEL_ID
    assert completions.kwargs["max_completion_tokens"] == chain.GROQ_MAX_TOKENS
    assert completions.kwargs["temperature"] == chain.GROQ_TEMPERATURE
    assert completions.kwargs["reasoning_effort"] == chain.GROQ_REASONING_EFFORT
    assert completions.kwargs["include_reasoning"] is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "error_text,expected",
    [
        ("401 unauthorized", "authentication failed"),
        ("403 forbidden", "rejected the inference request"),
        ("429 rate limit", "rate limit was reached"),
        ("network timeout", "generation failed"),
    ],
)
def test_groq_generation_translates_errors(error_text, expected):
    fake, _ = client(error=RuntimeError(error_text))
    prompt = chain.RAG_PROMPT.invoke(
        {
            "context": "X",
            "question": "Q",
        }
    )

    with pytest.raises(RuntimeError, match=expected):
        chain.generate_with_groq(fake, prompt)


@pytest.mark.unit
def test_generate_with_llm_dispatches_to_groq(monkeypatch):
    sentinel = "groq answer"
    monkeypatch.setattr(chain, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(
        chain,
        "generate_with_groq",
        lambda **kwargs: sentinel,
    )

    assert chain.generate_with_llm(object(), object()) == sentinel


@pytest.mark.unit
def test_build_rag_chain_uses_separate_generation_client(monkeypatch):
    generation_client = object()
    observed = {}

    monkeypatch.setattr(
        chain,
        "_retrieve_with_query_recovery_trace",
        lambda **kwargs: (
            [(doc(1, 0, 10, "evidence"), 0.2)],
            {
                "recovery_used": False,
                "final_stage": "initial_strict",
                "retrieval_latency_ms": 1.0,
                "final_chunk_ids": [1],
            },
        ),
    )
    monkeypatch.setattr(
        chain,
        "generate_with_llm",
        lambda **kwargs: (
            observed.update(kwargs) or "grounded answer"
        ),
    )

    result = chain.build_rag_chain(
        object(),
        object(),
        generation_client=generation_client,
    ).invoke(
        {
            "question": "What happened?",
            "video_id": "Gfr50f6ZBvo",
        }
    )

    assert result["answer"] == "grounded answer"
    assert result["retrieval_trace"]["final_stage"] == "initial_strict"
    assert result["generation_trace"]["generated"] is True
    assert observed["client"] is generation_client

@pytest.mark.unit
def test_generation_returns_clean_answer():
    response=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="  grounded answer  "))])
    fake,completions=client(response=response)
    prompt=chain.RAG_PROMPT.invoke({"context":"X","question":"Q"})
    assert chain.generate_with_huggingface(fake,prompt)=="grounded answer"
    assert completions.kwargs["model"]==chain.HF_MODEL_ID


@pytest.mark.unit
@pytest.mark.parametrize("error_text,expected",[
    ("model_not_supported","could not find an enabled Inference Provider"),
    ("401 unauthorized","authentication failed"),
    ("403 forbidden","rejected the inference request"),
    ("network timeout","generation failed"),
])
def test_generation_translates_errors(error_text,expected):
    fake,_=client(error=RuntimeError(error_text))
    prompt=chain.RAG_PROMPT.invoke({"context":"X","question":"Q"})
    with pytest.raises(RuntimeError,match=expected):
        chain.generate_with_huggingface(fake,prompt)


@pytest.mark.unit
def test_generation_rejects_empty_answer(monkeypatch):
    monkeypatch.setattr(chain, "HF_MAX_RETRIES", 0)

    fake, completions = client(
        response=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="")
                )
            ]
        )
    )

    prompt = chain.RAG_PROMPT.invoke(
        {
            "context": "X",
            "question": "Q",
        }
    )

    with pytest.raises(
        RuntimeError,
        match="empty answer",
    ):
        chain.generate_with_huggingface(
            fake,
            prompt,
        )

    assert completions.call_count == 1

@pytest.mark.unit
def test_generation_retries_empty_answer_then_succeeds(monkeypatch):
    monkeypatch.setattr(chain, "HF_MAX_RETRIES", 2)
    monkeypatch.setattr(chain, "HF_RETRY_DELAY_SECONDS", 0)

    responses = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="")
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="grounded answer"
                    )
                )
            ]
        ),
    ]

    class SequenceCompletions:
        def __init__(self):
            self.call_count = 0
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            response = responses[
                min(
                    self.call_count,
                    len(responses) - 1,
                )
            ]
            self.call_count += 1
            return response

    completions = SequenceCompletions()

    fake = SimpleNamespace(
        chat=SimpleNamespace(
            completions=completions
        )
    )

    prompt = chain.RAG_PROMPT.invoke(
        {
            "context": "X",
            "question": "Q",
        }
    )

    assert (
        chain.generate_with_huggingface(
            fake,
            prompt,
        )
        == "grounded answer"
    )

    assert completions.call_count == 2


@pytest.mark.unit
def test_generation_retries_multiple_empty_answers_then_succeeds(
    monkeypatch,
):
    monkeypatch.setattr(chain, "HF_MAX_RETRIES", 2)
    monkeypatch.setattr(chain, "HF_RETRY_DELAY_SECONDS", 0)

    responses = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="")
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="   ")
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="grounded answer"
                    )
                )
            ]
        ),
    ]

    class SequenceCompletions:
        def __init__(self):
            self.call_count = 0

        def create(self, **kwargs):
            response = responses[self.call_count]
            self.call_count += 1
            return response

    completions = SequenceCompletions()

    fake = SimpleNamespace(
        chat=SimpleNamespace(
            completions=completions
        )
    )

    prompt = chain.RAG_PROMPT.invoke(
        {
            "context": "X",
            "question": "Q",
        }
    )

    assert (
        chain.generate_with_huggingface(
            fake,
            prompt,
        )
        == "grounded answer"
    )

    assert completions.call_count == 3


@pytest.mark.unit
def test_generation_exhausts_empty_answer_retries(monkeypatch):
    monkeypatch.setattr(chain, "HF_MAX_RETRIES", 2)
    monkeypatch.setattr(chain, "HF_RETRY_DELAY_SECONDS", 0)

    empty_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="")
            )
        ]
    )

    class EmptyCompletions:
        def __init__(self):
            self.call_count = 0

        def create(self, **kwargs):
            self.call_count += 1
            return empty_response

    completions = EmptyCompletions()

    fake = SimpleNamespace(
        chat=SimpleNamespace(
            completions=completions
        )
    )

    prompt = chain.RAG_PROMPT.invoke(
        {
            "context": "X",
            "question": "Q",
        }
    )

    with pytest.raises(
        RuntimeError,
        match="empty answer after 3 attempts",
    ):
        chain.generate_with_huggingface(
            fake,
            prompt,
        )

    assert completions.call_count == 3


@pytest.mark.unit
def test_rag_chain_falls_back_without_generation(monkeypatch):
    monkeypatch.setattr(
        chain,
        "_retrieve_with_query_recovery_trace",
        lambda **k: (
            [],
            {
                "recovery_used": True,
                "final_stage": "no_evidence",
                "retrieval_latency_ms": 1.0,
                "final_chunk_ids": [],
            },
        ),
    )
    monkeypatch.setattr(
        chain,
        "generate_with_huggingface",
        lambda **k: pytest.fail("generation must not run"),
    )
    result = chain.build_rag_chain(
        object(),
        object(),
    ).invoke(
        {
            "question": "Missing?",
            "video_id": "Gfr50f6ZBvo",
        }
    )
    assert result["answer"] == chain.FALLBACK_ANSWER
    assert (
        result["retrieved_results"] == []
        and result["source_groups"] == []
    )


@pytest.mark.unit
def test_retrieval_trace_separates_selected_anchors_from_expanded_context(monkeypatch):
    anchors = [
        (doc(1, 10, 20, "answer-bearing anchor"), 0.2),
        (doc(3, 30, 40, "second selected anchor"), 0.5),
    ]
    expanded = [
        (doc(1, 10, 20, "answer-bearing anchor"), 0.2),
        (doc(2, 20, 30, "context-only neighbor"), chain.MAX_DISTANCE),
        (doc(3, 30, 40, "second selected anchor"), 0.5),
    ]

    monkeypatch.setattr(chain, "retrieve_temporal_context", lambda **kwargs: None)
    monkeypatch.setattr(
        chain,
        "retrieve_question_context",
        lambda **kwargs: (expanded, anchors),
    )

    results, trace = chain._retrieve_with_query_recovery_trace(
        object(),
        "How much does it cost?",
        object(),
    )

    assert results == expanded
    assert trace["final_chunk_ids"] == [1, 2, 3]
    assert trace["citation_chunk_ids"] == [1, 3]


@pytest.mark.unit
def test_select_citation_results_excludes_context_expansion_neighbors():
    expanded = [
        (doc(1, 10, 20, "answer-bearing anchor"), 0.2),
        (doc(2, 20, 30, "context-only neighbor"), chain.MAX_DISTANCE),
        (doc(3, 30, 40, "second selected anchor"), 0.5),
    ]

    selected = chain._select_citation_results(
        expanded,
        {"citation_chunk_ids": [1, 3]},
    )

    assert [document.metadata["chunk_id"] for document, _ in selected] == [1, 3]


@pytest.mark.regression
def test_select_citation_results_can_include_context_neighbors_in_experiment(monkeypatch):
    expanded = [
        (doc(1, 10, 20, "answer-bearing anchor"), 0.2),
        (doc(2, 20, 30, "context-only neighbor"), chain.MAX_DISTANCE),
        (doc(3, 30, 40, "second selected anchor"), 0.5),
    ]
    monkeypatch.setattr(
        chain,
        "RAG_CITATION_INCLUDE_CONTEXT_NEIGHBORS",
        True,
    )

    selected = chain._select_citation_results(
        expanded,
        {
            "final_chunk_ids": [1, 2, 3],
            "citation_chunk_ids": [1, 3],
        },
    )

    assert [document.metadata["chunk_id"] for document, _ in selected] == [1, 2, 3]


@pytest.mark.unit
def test_retrieval_trace_records_initial_strict_success(monkeypatch):
    expected = [(doc(4, 10, 20, "evidence"), 0.35)]

    monkeypatch.setattr(
        chain,
        "retrieve_question_context",
        lambda **kwargs: expected,
    )

    results, trace = chain._retrieve_with_query_recovery_trace(
        object(),
        "What happened?",
        object(),
    )

    assert results == expected
    assert trace["recovery_used"] is False
    assert trace["final_stage"] == "initial_strict"
    assert trace["final_chunk_ids"] == [4]
    assert trace["attempts"][0]["stage"] == "initial_strict"
    assert trace["attempts"][0]["retrieved_chunks"] == 1
    assert trace["attempts"][0]["best_distance"] == pytest.approx(0.35)
    assert trace["attempts"][0]["query"] == "What happened?"


@pytest.mark.unit
def test_retrieval_trace_records_recovery_stage(monkeypatch):
    recovered = [(doc(8, 80, 90, "recovered"), 0.75)]
    calls = []

    monkeypatch.setattr(
        chain,
        "retrieve_question_context",
        lambda **kwargs: (
            calls.append(kwargs["query"])
            or (
                []
                if len(calls) == 1
                else recovered
            )
        ),
    )
    monkeypatch.setattr(
        chain,
        "understand_query",
        lambda **kwargs: SimpleNamespace(
            intent="causal",
            standalone_question="reasons for the event",
            search_queries=["event reasons"],
        ),
    )

    results, trace = chain._retrieve_with_query_recovery_trace(
        object(),
        "Why did it happen?",
        object(),
    )

    assert results == recovered
    assert trace["recovery_used"] is True
    assert trace["final_stage"] == "rewritten_strict"
    assert trace["planner_used"] is True
    assert trace["planner_intent"] == "causal"
    assert [attempt["stage"] for attempt in trace["attempts"]] == [
        "initial_strict",
        "rewritten_strict",
    ]
    assert trace["attempts"][0]["query"] == "Why did it happen?"
    assert trace["attempts"][1]["query"] == "reasons for the event"


@pytest.mark.unit
def test_query_recovery_keeps_fast_path_when_initial_retrieval_succeeds(monkeypatch):
    expected = [(doc(1, 0, 10, "direct evidence"), 0.2)]
    calls = {"understand": 0}

    monkeypatch.setattr(
        chain,
        "retrieve_question_context",
        lambda **k: expected,
    )

    def forbidden_understanding(**kwargs):
        calls["understand"] += 1
        raise AssertionError(
            "query understanding should not run after successful retrieval"
        )

    monkeypatch.setattr(
        chain,
        "understand_query",
        forbidden_understanding,
    )

    result = chain.retrieve_with_query_recovery(
        object(),
        "What happened?",
        object(),
    )

    assert result == expected
    assert calls["understand"] == 0


@pytest.mark.unit
def test_query_recovery_retries_with_reformulated_queries(monkeypatch):
    empty = []
    recovered = [(doc(7, 70, 80, "recovered evidence"), 0.4)]
    attempts = []

    monkeypatch.setattr(
        chain,
        "retrieve_question_context",
        lambda **kwargs: (
            attempts.append(kwargs["query"])
            or (empty if len(attempts) == 1 else recovered)
        ),
    )
    monkeypatch.setattr(
        chain,
        "understand_query",
        lambda **kwargs: SimpleNamespace(
            intent="causal",
            standalone_question="reasons people are protesting",
            search_queries=[
                "reasons for protest",
                "protest demands",
            ],
        ),
    )

    result = chain.retrieve_with_query_recovery(
        object(),
        "Why are they protesting?",
        object(),
    )

    assert result == recovered
    assert attempts == [
        "Why are they protesting?",
        "reasons people are protesting",
    ]


@pytest.mark.unit
def test_retrieval_trace_preserves_recovery_after_empty_overview(monkeypatch):
    recovered = [(doc(9, 90, 100, "recovered"), 0.6)]
    calls = []

    monkeypatch.setattr(
        chain,
        "retrieve_question_context",
        lambda **kwargs: (
            calls.append(kwargs["query"])
            or (
                []
                if len(calls) == 1
                else recovered
            )
        ),
    )
    monkeypatch.setattr(
        chain,
        "retrieve_overview",
        lambda _vector_store: [],
    )
    monkeypatch.setattr(
        chain,
        "understand_query",
        lambda **kwargs: SimpleNamespace(
            intent="overview",
            standalone_question="main topic",
            search_queries=["main topic"],
        ),
    )

    results, trace = chain._retrieve_with_query_recovery_trace(
        object(),
        "What is the main subject?",
        object(),
    )

    assert results == recovered
    assert trace["final_stage"] == "rewritten_strict"
    assert [attempt["stage"] for attempt in trace["attempts"]] == [
        "initial_strict",
        "overview",
        "rewritten_strict",
    ]


@pytest.mark.unit
def test_query_recovery_routes_unrecognized_overview_question(monkeypatch):
    overview = [(doc(1, 0, 10, "overview evidence"), 1.3)]

    monkeypatch.setattr(
        chain,
        "retrieve_question_context",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        chain,
        "understand_query",
        lambda **kwargs: SimpleNamespace(
            intent="overview",
            standalone_question="What is the main subject?",
            search_queries=["main topic"],
        ),
    )
    monkeypatch.setattr(
        chain,
        "retrieve_overview",
        lambda vector_store: overview,
    )

    result = chain.retrieve_with_query_recovery(
        object(),
        "What is the main subject discussed here?",
        object(),
    )

    assert result == overview


@pytest.mark.unit
def test_query_recovery_fails_open_when_understanding_fails(monkeypatch):
    monkeypatch.setattr(
        chain,
        "retrieve_question_context",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        chain,
        "understand_query",
        lambda **kwargs: (_ for _ in ()).throw(
            chain.QueryUnderstandingError("planner unavailable")
        ),
    )

    assert chain.retrieve_with_query_recovery(
        object(),
        "What happened?",
        object(),
    ) == []


@pytest.mark.regression
def test_query_recovery_rejects_lexical_only_rewrites_without_overview_fallback(
    monkeypatch,
):
    candidate = doc(173, 11595, 11675, "Australia capital city")
    expanded_neighbor = doc(174, 11675, 11755, "unrelated neighboring transcript text")
    calls = []

    def fake_retrieve(**kwargs):
        calls.append(kwargs.copy())
        if kwargs["query"] == "Australia capital city":
            anchor = (candidate, kwargs["max_distance"])
            if (
                kwargs.get("expand_context") is False
                and kwargs.get("allow_overview") is False
            ):
                # Recovery must gate the scored anchor before context expansion.
                return [anchor]
            # This simulates the old false positive: an unscored neighbor carries
            # MAX_DISTANCE (1.3), which falls inside the relaxed gate (1.472).
            return [anchor, (expanded_neighbor, chain.MAX_DISTANCE)]
        return []

    monkeypatch.setattr(chain, "retrieve_question_context", fake_retrieve)
    monkeypatch.setattr(
        chain,
        "expand_question_context",
        lambda *_args, **_kwargs: pytest.fail(
            "weak recovery anchors must be rejected before context expansion"
        ),
    )
    monkeypatch.setattr(
        chain,
        "understand_query",
        lambda **kwargs: SimpleNamespace(
            intent="factual",
            standalone_question="capital of Australia",
            search_queries=["Australia capital city"],
        ),
    )
    monkeypatch.setattr(
        chain,
        "retrieve_overview",
        lambda _vector_store: pytest.fail(
            "weak rejected recovery candidates must not be replaced by overview context"
        ),
    )

    results, trace = chain._retrieve_with_query_recovery_trace(
        object(),
        "What is the capital of Australia?",
        object(),
    )

    assert results == []
    assert trace["final_stage"] == "no_evidence_weak_recovery_rejected"
    assert trace["final_chunk_ids"] == []
    assert len(calls) == 5
    recovery_calls = [
        call for call in calls
        if call["query"] != "What is the capital of Australia?"
    ]
    assert len(recovery_calls) == 4
    assert all(
        call.get("expand_context") is False
        and call.get("allow_overview") is False
        for call in recovery_calls
    )

    rejected_attempts = [
        attempt
        for attempt in trace["attempts"]
        if attempt["retrieved_chunks"] > 0
    ]
    assert [attempt["stage"] for attempt in rejected_attempts] == [
        "rewritten_strict",
        "rewritten_relaxed",
    ]
    assert [attempt["query"] for attempt in rejected_attempts] == [
        "Australia capital city",
        "Australia capital city",
    ]
    assert all(
        attempt["semantic_gate_passed"] is False
        and attempt["accepted_for_generation"] is False
        and attempt["rejection_reason"] == "weak_semantic_support"
        and attempt["context_expansion_performed"] is False
        and attempt["expanded_context_chunks"] == 0
        for attempt in rejected_attempts
    )


@pytest.mark.regression
def test_query_recovery_rejects_relaxed_anchors_above_normal_distance_ceiling(
    monkeypatch,
):
    question = "What is the capital of Australia?"
    rewritten_strict_candidate = (doc(174, 11675, 11755, "Australia capital city"), chain.MAX_DISTANCE)
    relaxed_candidates = [
        (doc(cid, cid * 10, cid * 10 + 5, "unrelated airline discussion"), distance)
        for cid, distance in [
            (25, 1.407618),
            (83, 1.42),
            (152, 1.43),
            (124, 1.44),
            (9, 1.45),
            (87, 1.46),
        ]
    ]
    calls = []

    def fake_retrieve(**kwargs):
        calls.append(kwargs.copy())
        query = kwargs["query"]
        cutoff = kwargs["max_distance"]

        if query == question:
            return []
        if query == "capital of Australia":
            if cutoff == chain.RECOVERY_MAX_DISTANCE:
                return relaxed_candidates
            return []
        if query == "Australia capital city":
            if cutoff == chain.RECOVERY_MAX_DISTANCE:
                # At the relaxed cutoff this represents lexical-only fusion:
                # the sentinel is the configured relaxed distance, not a score.
                return [(rewritten_strict_candidate[0], cutoff)]
            return [rewritten_strict_candidate]
        return []

    monkeypatch.setattr(chain, "retrieve_question_context", fake_retrieve)
    monkeypatch.setattr(
        chain,
        "expand_question_context",
        lambda *_args, **_kwargs: pytest.fail(
            "anchors above the normal distance ceiling must be rejected before expansion"
        ),
    )
    monkeypatch.setattr(
        chain,
        "understand_query",
        lambda **kwargs: SimpleNamespace(
            intent="factual",
            standalone_question="capital of Australia",
            search_queries=["Australia capital city"],
        ),
    )
    monkeypatch.setattr(
        chain,
        "retrieve_overview",
        lambda _vector_store: pytest.fail(
            "weak rejected recovery candidates must not trigger overview fallback"
        ),
    )

    results, trace = chain._retrieve_with_query_recovery_trace(
        object(),
        question,
        object(),
    )

    assert results == []
    assert trace["final_stage"] == "no_evidence_weak_recovery_rejected"
    assert trace["final_chunk_ids"] == []
    relaxed_attempts = [
        attempt for attempt in trace["attempts"]
        if attempt["stage"] == "rewritten_relaxed"
        and attempt["retrieved_chunks"] > 0
    ]
    assert relaxed_attempts
    australia_attempt = next(
        attempt for attempt in relaxed_attempts
        if attempt["query"] == "capital of Australia"
    )
    assert australia_attempt["best_distance"] == pytest.approx(1.407618)
    assert australia_attempt["semantic_gate_distance_ceiling"] == chain.MAX_DISTANCE
    assert australia_attempt["semantic_gate_passed"] is False
    assert australia_attempt["accepted_for_generation"] is False
    assert australia_attempt["context_expansion_performed"] is False
    assert australia_attempt["expanded_context_chunks"] == 0
    assert australia_attempt["rejection_reason"] == "weak_semantic_support"


@pytest.mark.unit
def test_query_recovery_uses_relaxed_retrieval_after_strict_recovery_fails(
    monkeypatch,
):
    # Relaxed recovery may admit evidence above the 0.92 * normal cutoff,
    # but not beyond the normal cutoff itself.
    recovered = [
        (
            doc(7, 70, 80, "relaxed evidence"),
            chain.MAX_DISTANCE * 0.95,
        )
    ]
    neighbor = (doc(8, 80, 90, "neighbor context"), chain.MAX_DISTANCE)
    expanded_context = [*recovered, neighbor]
    distances = []
    expansion_calls = []

    monkeypatch.setattr(
        chain,
        "retrieve_question_context",
        lambda **kwargs: (
            distances.append(kwargs["max_distance"])
            or (
                recovered
                if kwargs["max_distance"] == chain.RECOVERY_MAX_DISTANCE
                else []
            )
        ),
    )
    monkeypatch.setattr(
        chain,
        "expand_question_context",
        lambda vector_store, query, anchors: (
            expansion_calls.append((query, anchors)) or expanded_context
        ),
    )
    monkeypatch.setattr(
        chain,
        "understand_query",
        lambda **kwargs: SimpleNamespace(
            intent="general",
            standalone_question="important issue",
            search_queries=["main issue", "central concern"],
        ),
    )

    result = chain.retrieve_with_query_recovery(
        object(),
        "What seems to be the biggest issue?",
        object(),
    )

    assert result == expanded_context
    assert expansion_calls == [("important issue", recovered)]
    assert distances[-1] == chain.RECOVERY_MAX_DISTANCE
    assert all(
        distance == chain.MAX_DISTANCE
        for distance in distances[:-1]
    )


@pytest.mark.unit
def test_query_recovery_uses_representative_context_as_final_fallback(
    monkeypatch,
):
    overview = [
        (doc(1, 0, 10, "opening context"), chain.MAX_DISTANCE),
        (doc(2, 20, 30, "representative context"), chain.MAX_DISTANCE),
    ]

    monkeypatch.setattr(
        chain,
        "retrieve_question_context",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        chain,
        "understand_query",
        lambda **kwargs: SimpleNamespace(
            intent="opinion",
            standalone_question="which side is good and which is bad",
            search_queries=[
                "arguments for each side",
                "positions of opposing sides",
            ],
        ),
    )
    monkeypatch.setattr(
        chain,
        "retrieve_overview",
        lambda vector_store: overview,
    )

    assert chain.retrieve_with_query_recovery(
        object(),
        "Which side is good and which is bad?",
        object(),
    ) == overview


@pytest.mark.unit
def test_generation_rejects_malformed_response():
    fake, _ = client(
        response=SimpleNamespace(
            choices=[]
        )
    )

    prompt = chain.RAG_PROMPT.invoke(
        {
            "context": "X",
            "question": "Q",
        }
    )

    with pytest.raises(
        RuntimeError,
        match="invalid chat completion response",
    ):
        chain.generate_with_huggingface(
            fake,
            prompt,
        )


@pytest.mark.unit
def test_create_llm_client_requires_huggingface_token(monkeypatch):
    monkeypatch.setattr(chain, "LLM_PROVIDER", "huggingface")
    monkeypatch.setattr(chain, "HF_TOKEN", None)
    with pytest.raises(RuntimeError, match="HF_TOKEN was not found"):
        chain.create_llm_client()


@pytest.mark.unit
def test_create_llm_client_selects_groq(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(chain, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(
        chain,
        "create_groq_client",
        lambda: sentinel,
    )

    assert chain.create_llm_client() is sentinel


@pytest.mark.unit
def test_create_groq_client_requires_api_key(monkeypatch):
    monkeypatch.setattr(chain, "GROQ_API_KEY", None)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY was not found"):
        chain.create_groq_client()


@pytest.mark.unit
def test_create_groq_client_uses_configured_key(monkeypatch):
    captured = {}

    class FakeGroq:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(chain, "GROQ_API_KEY", "gsk_test_key")
    monkeypatch.setattr(chain, "Groq", FakeGroq)

    client = chain.create_groq_client()

    assert isinstance(client, FakeGroq)
    assert captured == {"api_key": "gsk_test_key"}

@pytest.mark.unit
def test_answer_question_uses_index_lifecycle(monkeypatch):
    vector_store = object()

    calls = {
        "load_ready_index": 0,
        "load_vector_store": 0,
    }

    monkeypatch.setattr(
        chain,
        "extract_video_id",
        lambda reference: "Gfr50f6ZBvo",
    )

    def fake_load_ready_index(video_id):
        calls["load_ready_index"] += 1
        assert video_id == "Gfr50f6ZBvo"
        return vector_store

    monkeypatch.setattr(
        chain,
        "load_ready_index",
        fake_load_ready_index,
    )

    def forbidden_load(video_id):
        calls["load_vector_store"] += 1
        raise AssertionError(
            "answer_question must not call load_vector_store directly"
        )

    monkeypatch.setattr(
        chain,
        "load_vector_store",
        forbidden_load,
        raising=False,
    )

    configured_client = object()
    captured_clients = {}

    monkeypatch.setattr(
        chain,
        "create_llm_client",
        lambda: configured_client,
    )

    source_doc = doc(1, 10, 20, "evidence")

    class FakeRagChain:
        def invoke(self, inputs):
            assert inputs["video_id"] == "Gfr50f6ZBvo"
            assert inputs["question"] == "What happened?"

            return {
                "answer": "grounded answer",
                "retrieved_results": [(source_doc, 0.25)],
                "source_groups": [
                    {
                        "documents": [source_doc],
                        "start": 10.0,
                        "end": 20.0,
                        "distance": 0.25,
                    }
                ],
            }

    def fake_build_rag_chain(
        vector_store,
        llm_client,
        generation_client,
    ):
        captured_clients["query"] = llm_client
        captured_clients["generation"] = generation_client
        return FakeRagChain()

    monkeypatch.setattr(
        chain,
        "build_rag_chain",
        fake_build_rag_chain,
    )

    result = chain.answer_question(
        video_reference="https://www.youtube.com/watch?v=Gfr50f6ZBvo",
        question="What happened?",
    )

    assert result["answer"] == (
        "grounded answer\n\nRelevant timestamps: 00:10–00:20"
    )
    assert result["video_id"] == "Gfr50f6ZBvo"
    assert calls["load_ready_index"] == 1
    assert calls["load_vector_store"] == 0
    assert captured_clients["query"] is configured_client
    assert captured_clients["generation"] is configured_client

@pytest.mark.regression
def test_answer_question_formats_clickable_sources_from_anchors_only(monkeypatch):
    expanded_documents = [
        doc(1, 10, 20, "answer-bearing evidence"),
        doc(2, 20, 30, "context expansion only"),
        doc(3, 30, 40, "additional anchor"),
    ]
    expanded_results = [
        (expanded_documents[0], 0.2),
        (expanded_documents[1], chain.MAX_DISTANCE),
        (expanded_documents[2], 0.5),
    ]
    context_groups = chain.merge_overlapping_results(expanded_results)
    retrieval_result = {
        "answer": (
            "The cost is about 600 dollars per hour.\n\n"
            "Relevant timestamps: 99:00–99:01"
        ),
        "retrieved_results": expanded_results,
        "source_groups": context_groups,
        "retrieval_trace": {
            "final_stage": "initial_strict",
            "final_chunk_ids": [1, 2, 3],
            "citation_chunk_ids": [1, 3],
        },
        "generation_trace": {"generated": True, "fallback_used": False},
    }

    monkeypatch.setattr(chain, "extract_video_id", lambda _reference: "Gfr50f6ZBvo")
    monkeypatch.setattr(chain, "load_ready_index", lambda _video_id: object())
    monkeypatch.setattr(chain, "create_llm_client", lambda: object())
    monkeypatch.setattr(chain, "EMBEDDING_MODEL", "test-embedding-model")
    monkeypatch.setattr(chain, "RAG_INDEX_NAMESPACE", "test-namespace")
    monkeypatch.setattr(chain, "RAG_CITATION_INCLUDE_CONTEXT_NEIGHBORS", False)

    class FakeRagChain:
        def invoke(self, _inputs):
            return retrieval_result

    monkeypatch.setattr(
        chain,
        "build_rag_chain",
        lambda **_kwargs: FakeRagChain(),
    )

    result = chain.answer_question(
        video_reference="Gfr50f6ZBvo",
        question="How much does one hour cost?",
    )

    assert result["retrieved_chunks"] == 3
    assert result["source_segments"] == 2
    assert result["retrieval_config"]["embedding_model"] == "test-embedding-model"
    assert result["retrieval_config"]["index_namespace"] == "test-namespace"
    assert result["retrieval_config"]["citation_policy"] == "selected_anchors"
    assert [source["chunk_ids"] for source in result["sources"]] == [[1], [3]]
    assert result["answer"].endswith(
        "Relevant timestamps: 00:10–00:20, 00:30–00:40"
    )
    assert "99:00–99:01" not in result["answer"]
    assert [(source["start"], source["end"]) for source in result["sources"]] == [
        (10.0, 20.0),
        (30.0, 40.0),
    ]
    retrieval_trace = result["observability"]["retrieval"]
    assert retrieval_trace["candidate_source_segments"] == 1
    assert retrieval_trace["candidate_citation_source_segments"] == 2
    assert retrieval_trace["source_timestamps"] == [
        {"source_id": 1, "start": 10.0, "end": 20.0},
        {"source_id": 2, "start": 30.0, "end": 40.0},
    ]


@pytest.mark.regression
def test_answer_question_keeps_context_neighbors_as_individual_sources(monkeypatch):
    expanded_documents = [
        doc(1, 10, 40, "selected answer anchor"),
        doc(2, 20, 50, "context-only neighbor"),
        doc(3, 60, 80, "second selected anchor"),
    ]
    expanded_results = [
        (expanded_documents[0], 0.2),
        (expanded_documents[1], chain.MAX_DISTANCE),
        (expanded_documents[2], 0.5),
    ]
    retrieval_result = {
        "answer": "A grounded answer.",
        "retrieved_results": expanded_results,
        "source_groups": chain.merge_overlapping_results(expanded_results),
        "retrieval_trace": {
            "final_stage": "initial_strict",
            "final_chunk_ids": [1, 2, 3],
            "citation_chunk_ids": [1, 3],
        },
        "generation_trace": {"generated": True, "fallback_used": False},
    }

    monkeypatch.setattr(chain, "extract_video_id", lambda _reference: "Gfr50f6ZBvo")
    monkeypatch.setattr(chain, "load_ready_index", lambda _video_id: object())
    monkeypatch.setattr(chain, "create_llm_client", lambda: object())
    monkeypatch.setattr(chain, "EMBEDDING_MODEL", "test-embedding-model")
    monkeypatch.setattr(chain, "RAG_INDEX_NAMESPACE", "test-namespace")
    monkeypatch.setattr(chain, "RAG_CITATION_INCLUDE_CONTEXT_NEIGHBORS", True)

    class FakeRagChain:
        def invoke(self, _inputs):
            return retrieval_result

    monkeypatch.setattr(
        chain,
        "build_rag_chain",
        lambda **_kwargs: FakeRagChain(),
    )

    result = chain.answer_question(
        video_reference="Gfr50f6ZBvo",
        question="What does the passage say?",
    )

    assert result["retrieval_config"]["citation_policy"] == "expanded_context_experiment"
    assert result["source_segments"] == 3
    assert [source["chunk_ids"] for source in result["sources"]] == [[1], [2], [3]]
    # Chunk 2 overlaps anchor 1 in time, but the neighbor link stays precise.
    assert [(source["start"], source["end"]) for source in result["sources"]] == [
        (10.0, 40.0),
        (20.0, 50.0),
        (60.0, 80.0),
    ]


@pytest.mark.regression
def test_answer_question_suppresses_sources_when_generation_abstains(monkeypatch):
    source_documents = [
        doc(173, 11595.279, 11675.0, "unrelated transcript section one"),
        doc(174, 11675.0, 11755.0, "unrelated transcript section two"),
        doc(175, 11755.0, 11835.16, "unrelated transcript section three"),
    ]
    retrieval_result = {
        "answer": chain.FALLBACK_ANSWER,
        "retrieved_results": [
            (source_document, 1.3)
            for source_document in source_documents
        ],
        "source_groups": [
            {
                "documents": source_documents,
                "start": 11595.279,
                "end": 11835.16,
                "distance": 1.3,
            }
        ],
        "retrieval_trace": {
            "final_stage": "rewritten_relaxed",
            "final_chunk_ids": [173, 174, 175],
        },
        "generation_trace": {
            "generated": True,
            "fallback_used": False,
        },
    }

    monkeypatch.setattr(
        chain,
        "extract_video_id",
        lambda _reference: "Gfr50f6ZBvo",
    )
    monkeypatch.setattr(
        chain,
        "load_ready_index",
        lambda _video_id: object(),
    )
    monkeypatch.setattr(
        chain,
        "create_llm_client",
        lambda: object(),
    )

    class FakeRagChain:
        def invoke(self, _inputs):
            return retrieval_result

    monkeypatch.setattr(
        chain,
        "build_rag_chain",
        lambda **_kwargs: FakeRagChain(),
    )

    result = chain.answer_question(
        video_reference="Gfr50f6ZBvo",
        question="What is the capital of Australia?",
    )

    assert result["answer"] == chain.FALLBACK_ANSWER
    assert result["sources"] == []
    assert result["retrieved_chunks"] == 0
    assert result["source_segments"] == 0

    retrieval_trace = result["observability"]["retrieval"]
    assert retrieval_trace["candidate_retrieved_chunks"] == 3
    assert retrieval_trace["candidate_source_segments"] == 1
    assert retrieval_trace["final_chunk_ids"] == [173, 174, 175]
    assert retrieval_trace["source_segments"] == 0
    assert retrieval_trace["source_timestamps"] == []
    assert retrieval_trace["response_evidence_suppressed"] is True
    assert (
        retrieval_trace["response_evidence_suppression_reason"]
        == "fallback_answer"
    )


@pytest.mark.unit
def test_answer_question_propagates_index_lifecycle_error(
    monkeypatch,
):
    monkeypatch.setattr(
        chain,
        "extract_video_id",
        lambda reference: "Gfr50f6ZBvo",
    )

    def fail(video_id):
        raise RuntimeError(
            "index creation failed"
        )

    monkeypatch.setattr(
        chain,
        "load_ready_index",
        fail,
    )

    with pytest.raises(
        RuntimeError,
        match="index creation failed",
    ):
        chain.answer_question(
            video_reference="Gfr50f6ZBvo",
            question="What happened?",
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "[SOURCE 1]\n\nThe speaker discusses protein folding.",
            "The speaker discusses protein folding.",
        ),
        (
            "The answer is supported.\n\nSources:",
            "The answer is supported.",
        ),
        (
            "   ",
            chain.FALLBACK_ANSWER,
        ),
    ],
)
def test_sanitize_generated_answer(raw, expected):
    assert chain.sanitize_generated_answer(raw) == expected



@pytest.mark.unit
def test_resolve_conversational_question_returns_standalone_question(monkeypatch):
    calls = []

    def fake_understand(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            intent="follow_up",
            standalone_question="Who are the people participating in the protest?",
            search_queries=["people participating in the protest"],
        )

    monkeypatch.setattr(
        chain,
        "understand_query",
        fake_understand,
    )

    history = [
        {
            "role": "user",
            "content": "Why are people protesting?",
        },
        {
            "role": "assistant",
            "content": "People are protesting to demand reforms.",
        },
    ]

    resolved = chain.resolve_conversational_question(
        question="Who are they?",
        conversation_history=history,
        llm_client=object(),
    )

    assert resolved == "Who are the people participating in the protest?"
    assert calls[0]["conversation_history"] == history




@pytest.mark.unit
def test_temporal_trace_bypasses_unbounded_semantic_recovery(monkeypatch):
    item = (doc(32, 1574.72, 1584.52, "ending transcript"), 1.3)
    bundle = {
        "window": {
            "mode": "explicit_range",
            "start_seconds": 1535.0,
            "end_seconds": 1584.52,
        },
        "results": [item],
        "candidates": [item],
        "coverage_ratio": 0.8,
        "covered_seconds": 39.6,
        "window_seconds": 49.52,
        "minimum_coverage": 0.5,
        "coverage_sufficient": True,
        "candidate_chunk_ids": [32],
    }
    monkeypatch.setattr(
        chain,
        "retrieve_temporal_context",
        lambda **kwargs: bundle,
    )
    monkeypatch.setattr(
        chain,
        "retrieve_question_context",
        lambda **kwargs: pytest.fail("temporal route must not use global semantic retrieval"),
    )
    monkeypatch.setattr(
        chain,
        "understand_query",
        lambda **kwargs: pytest.fail("temporal route must not invoke generic query recovery"),
    )

    results, trace = chain._retrieve_with_query_recovery_trace(
        vector_store=object(),
        question="What is said between 25:35 and 26:25?",
    )

    assert results == [item]
    assert trace["route"] == "temporal_window"
    assert trace["final_stage"] == "temporal_window"
    assert trace["planner_used"] is False
    assert trace["temporal_window"]["coverage_ratio"] == 0.8


@pytest.mark.unit
def test_temporal_trace_fails_closed_when_coverage_is_insufficient(monkeypatch):
    candidate = (doc(30, 1500, 1535.69, "partial ending"), 1.3)
    bundle = {
        "window": {
            "mode": "explicit_range",
            "start_seconds": 1535.0,
            "end_seconds": 1584.52,
        },
        "results": [],
        "candidates": [candidate],
        "coverage_ratio": 0.02,
        "covered_seconds": 0.69,
        "window_seconds": 49.52,
        "minimum_coverage": 0.5,
        "coverage_sufficient": False,
        "candidate_chunk_ids": [30],
    }
    monkeypatch.setattr(
        chain,
        "retrieve_temporal_context",
        lambda **kwargs: bundle,
    )
    monkeypatch.setattr(
        chain,
        "retrieve_question_context",
        lambda **kwargs: pytest.fail("must not bypass temporal coverage gate"),
    )
    monkeypatch.setattr(
        chain,
        "understand_query",
        lambda **kwargs: pytest.fail("must not rewrite away the requested time window"),
    )

    results, trace = chain._retrieve_with_query_recovery_trace(
        vector_store=object(),
        question="What happens during 25:35–26:25?",
    )

    assert results == []
    assert trace["final_stage"] == "temporal_window_insufficient_coverage"
    assert trace["recovery_used"] is False
    assert trace["planner_used"] is False
    assert trace["final_chunk_ids"] == []
    assert trace["attempts"][0]["retrieved_chunks"] == 1



@pytest.mark.unit
def test_temporal_trace_stops_when_requested_window_has_no_chunks(monkeypatch):
    bundle = {
        "window": {
            "mode": "point_timestamp",
            "start_seconds": 570.0,
            "end_seconds": 630.0,
        },
        "results": [],
        "candidates": [],
        "coverage_ratio": 0.0,
        "covered_seconds": 0.0,
        "window_seconds": 60.0,
        "minimum_coverage": 0.5,
        "coverage_sufficient": False,
        "candidate_chunk_ids": [],
    }
    monkeypatch.setattr(
        chain,
        "retrieve_temporal_context",
        lambda **kwargs: bundle,
    )
    monkeypatch.setattr(
        chain,
        "retrieve_question_context",
        lambda **kwargs: pytest.fail("must not retrieve from outside the requested window"),
    )
    monkeypatch.setattr(
        chain,
        "understand_query",
        lambda **kwargs: pytest.fail("must not invoke unbounded recovery for a time-specific query"),
    )

    results, trace = chain._retrieve_with_query_recovery_trace(
        vector_store=object(),
        question="What happened at 10:00?",
    )

    assert results == []
    assert trace["final_stage"] == "temporal_window_empty"
    assert trace["recovery_used"] is False
    assert trace["planner_used"] is False
