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
    monkeypatch.setattr(chain,"retrieve_question_context",lambda **k:[])
    monkeypatch.setattr(chain,"generate_with_huggingface",lambda **k:pytest.fail("generation must not run"))
    result=chain.build_rag_chain(object(),object()).invoke({"question":"Missing?","video_id":"Gfr50f6ZBvo"})
    assert result["answer"]==chain.FALLBACK_ANSWER
    assert result["retrieved_results"]==[] and result["source_groups"]==[]


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
