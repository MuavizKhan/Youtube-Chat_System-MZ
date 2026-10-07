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
def test_prompt_role_mapping():
    prompt=chain.RAG_PROMPT.invoke({"context":"Transcript evidence","question":"What happened?"})
    messages=chain.prompt_to_messages(prompt)
    assert [m["role"] for m in messages]==["system","user"]


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
def test_generation_rejects_malformed_response():
    fake,_=client(response=SimpleNamespace(choices=[]))
    prompt=chain.RAG_PROMPT.invoke({"context":"X","question":"Q"})
    with pytest.raises(RuntimeError,match="Could not extract"):
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
        "retrieve_with_query_recovery",
        lambda **k: [],
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


@pytest.mark.unit
def test_query_recovery_uses_relaxed_retrieval_after_strict_recovery_fails(
    monkeypatch,
):
    recovered = [
        (doc(7, 70, 80, "relaxed evidence"), 1.5)
    ]
    distances = []

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

    assert result == recovered
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
def test_create_llm_client_requires_token(monkeypatch):
    monkeypatch.setattr(chain,"HF_TOKEN",None)
    with pytest.raises(RuntimeError,match="HF_TOKEN was not found"):
        chain.create_llm_client()

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

    monkeypatch.setattr(
        chain,
        "create_llm_client",
        lambda: object(),
    )

    class FakeRagChain:
        def invoke(self, inputs):
            assert inputs["video_id"] == "Gfr50f6ZBvo"
            assert inputs["question"] == "What happened?"

            return {
                "answer": "grounded answer",
                "retrieved_results": [],
                "source_groups": [],
            }

    monkeypatch.setattr(
        chain,
        "build_rag_chain",
        lambda vector_store, llm_client: FakeRagChain(),
    )

    result = chain.answer_question(
        video_reference="https://www.youtube.com/watch?v=Gfr50f6ZBvo",
        question="What happened?",
    )

    assert result["answer"] == "grounded answer"
    assert result["video_id"] == "Gfr50f6ZBvo"
    assert calls["load_ready_index"] == 1
    assert calls["load_vector_store"] == 0

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
def test_format_conversation_history_keeps_recent_turns():
    history = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Why are people protesting?"},
        {"role": "assistant", "content": "They are protesting for reforms."},
    ]

    formatted = chain.format_conversation_history(history)

    assert "User: Why are people protesting?" in formatted
    assert "Assistant: They are protesting for reforms." in formatted


@pytest.mark.unit
def test_query_recovery_resolves_follow_up_before_original_question(monkeypatch):
    resolved = [
        (doc(8, 80, 100, "resolved follow-up evidence"), 0.5)
    ]
    attempts = []
    planner_calls = []

    monkeypatch.setattr(
        chain,
        "retrieve_question_context",
        lambda **kwargs: (
            attempts.append(kwargs["query"])
            or resolved
        ),
    )

    def fake_understand(**kwargs):
        planner_calls.append(kwargs)
        return SimpleNamespace(
            intent="follow_up",
            standalone_question="Who are the people participating in the protest?",
            search_queries=[
                "people participating in the protest",
            ],
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

    result = chain.retrieve_with_query_recovery(
        object(),
        "Who are they people?",
        object(),
        conversation_history=history,
    )

    assert result == resolved
    assert attempts == [
        "Who are the people participating in the protest?",
    ]
    assert planner_calls[0]["conversation_history"] == history


@pytest.mark.unit
def test_recovery_generation_prompt_contains_conversation_history():
    prompt = chain.RAG_PROMPT.invoke(
        {
            "context": "Transcript evidence",
            "question": "Who are they?",
            "conversation_history_text": (
                "User: Why are people protesting?\\n"
                "Assistant: They want election reforms."
            ),
        }
    )

    messages = chain.prompt_to_messages(prompt)

    assert (
        "Why are people protesting?"
        in messages[1]["content"]
    )
    assert "They want election reforms." in messages[1]["content"]
