import numpy as np
import pytest
from langchain_core.documents import Document
from Backend.rag_system import retrieval


class FakeDocstore:
    def __init__(self, documents):
        self.documents = documents
    def search(self, docstore_id):
        return self.documents.get(docstore_id)


class FakeEmbeddingFunction:
    def embed_query(self, query):
        return [1.0, 0.0]


class FakeIndex:
    def __init__(self, distances, indices, embeddings):
        self.ntotal=len(indices)
        self._distances=np.asarray([distances],dtype=np.float32)
        self._indices=np.asarray([indices],dtype=np.int64)
        self._embeddings=np.asarray(embeddings,dtype=np.float32)
    def search(self, query_vector, candidate_k):
        return self._distances[:,:candidate_k], self._indices[:,:candidate_k]
    def reconstruct_batch(self, ids):
        return self._embeddings[np.asarray(ids,dtype=np.int64)]


class FakeVectorStore:
    def __init__(self, documents, distances=None, indices=None, embeddings=None):
        distances=distances or [0.1,0.2,0.3]
        indices=indices or list(range(len(documents)))
        embeddings=embeddings or [[1,0],[0.8,0.6],[0,1]][:len(documents)]
        self.docstore=FakeDocstore(documents)
        self.index_to_docstore_id={i:k for i,k in enumerate(documents)}
        self.embedding_function=FakeEmbeddingFunction()
        self.index=FakeIndex(distances,indices,embeddings)


def doc(cid,start,end,text):
    return Document(page_content=text,metadata={
        "chunk_id":cid,"start":start,"end":end,"video_id":"Gfr50f6ZBvo"
    })


@pytest.mark.unit
@pytest.mark.parametrize("reference",[
    "Gfr50f6ZBvo",
    "https://www.youtube.com/watch?v=Gfr50f6ZBvo&t=20s",
    "https://www.youtube.com/shorts/Gfr50f6ZBvo",
    "https://www.youtube.com/embed/Gfr50f6ZBvo",
    "https://www.youtube.com/live/Gfr50f6ZBvo",
    "https://youtu.be/Gfr50f6ZBvo",
])
def test_extract_video_id_supported_references(reference):
    assert retrieval.extract_video_id(reference)=="Gfr50f6ZBvo"


@pytest.mark.unit
@pytest.mark.parametrize("reference",[
    "", "short", "1234567890!", "https://example.com/watch?v=Gfr50f6ZBvo",
    "https://www.youtube.com/watch", "https://www.youtube.com/watch?v=invalid",
    "https://youtu.be/not-valid",
])
def test_extract_video_id_rejects_invalid_references(reference):
    with pytest.raises(ValueError):
        retrieval.extract_video_id(reference)


@pytest.mark.unit
def test_query_terms_keep_short_domain_terms():
    assert retrieval.extract_query_terms("What is AI and RL in this video?")=={"ai","rl","video"}


@pytest.mark.unit
@pytest.mark.parametrize("query",[
    "What is this video about?","Give me an overview",
    "Summarise the video","What are the main topics?",
])
def test_overview_detection(query):
    assert retrieval.is_overview_question(query)


@pytest.mark.unit
def test_overview_detection_rejects_specific_question():
    assert not retrieval.is_overview_question("Who is Vijay Mallya?")


@pytest.mark.unit
def test_lexical_search_requires_shared_evidence():
    documents={
        "a":doc(1,0,10,"Alice works at Acme."),
        "b":doc(2,10,20,"Alice is mentioned."),
        "c":doc(3,20,30,"Acme is mentioned."),
    }
    results=retrieval.lexical_search(FakeVectorStore(documents),"Who is Alice at Acme?")
    assert [d.metadata["chunk_id"] for d,_ in results]==[1]


@pytest.mark.unit
def test_lexical_search_supports_single_term():
    documents={"a":doc(1,0,10,"Alice works at Acme."),"b":doc(2,10,20,"Bob works elsewhere.")}
    results=retrieval.lexical_search(FakeVectorStore(documents),"Alice")
    assert [d.metadata["chunk_id"] for d,_ in results]==[1]


@pytest.mark.unit
def test_overview_sampling_is_chronological():
    documents={f"d{i}":doc(i,i*10,i*10+5,f"segment {i}") for i in range(8)}
    results=retrieval.retrieve_overview(FakeVectorStore(documents),number_of_chunks=4)
    assert [d.metadata["start"] for d,_ in results]==[0,20,50,70]


@pytest.mark.unit
@pytest.mark.parametrize("kwargs,message",[
    ({"query":""},"Query cannot be empty."),
    ({"query":"x","k":0},"k must be greater than 0."),
    ({"query":"x","fetch_k":0},"fetch_k must be greater than 0."),
    ({"query":"x","k":3,"fetch_k":2},"k cannot be greater than fetch_k."),
    ({"query":"x","lambda_mult":1.1},"lambda_mult must be between 0.0 and 1.0."),
    ({"query":"x","max_distance":-0.1},"max_distance cannot be negative."),
])
def test_mmr_validates_configuration(kwargs,message):
    with pytest.raises(ValueError,match=message):
        retrieval.retrieve_mmr(vector_store=None,**kwargs)


@pytest.mark.unit
def test_mmr_filters_by_distance():
    documents={f"d{i}":doc(i,i*10,i*10+5,f"segment {i}") for i in range(3)}
    store=FakeVectorStore(documents,distances=[0.1,0.2,0.3],embeddings=[[1,0],[0.8,0.6],[0,1]])
    results=retrieval.retrieve_mmr(store,"alpha",k=3,fetch_k=3,max_distance=0.15)
    assert len(results)==1
    assert results[0][0].metadata["chunk_id"]==0


@pytest.mark.unit
def test_mmr_selects_nonduplicate_candidate():
    documents={f"d{i}":doc(i,i*10,i*10+5,f"segment {i}") for i in range(3)}
    store=FakeVectorStore(documents,distances=[0.1,0.2,0.3],embeddings=[[1,0],[0.8,0.6],[0,1]])
    results=retrieval.retrieve_mmr(store,"alpha",k=2,fetch_k=3,max_distance=1.0)
    assert len(results)==2
    assert {d.metadata["chunk_id"] for d,_ in results}=={0,1}


@pytest.mark.unit
def test_question_context_uses_overview_path(monkeypatch):
    expected=[(doc(1,0,10,"overview"),0.0)]
    monkeypatch.setattr(retrieval,"retrieve_overview",lambda *a,**k:expected)
    monkeypatch.setattr(retrieval,"retrieve_mmr",lambda *a,**k:pytest.fail("semantic path should not run"))
    assert retrieval.retrieve_question_context(object(),"What is this video about?")==expected


@pytest.mark.unit
def test_question_context_merges_and_deduplicates(monkeypatch):
    monkeypatch.setattr(retrieval,"expand_retrieval_context",lambda vector_store,retrieved_results,**kwargs:retrieved_results)
    one,two,duplicate=doc(1,0,10,"one"),doc(2,10,20,"two"),doc(1,20,30,"duplicate")
    monkeypatch.setattr(retrieval,"retrieve_mmr",lambda **k:[(one,0.4),(duplicate,0.5)])
    monkeypatch.setattr(retrieval,"lexical_search",lambda **k:[(two,0.9),(duplicate,1.0)])
    result=retrieval.retrieve_question_context(object(),"Alice Acme",k=2,max_distance=1.3)
    assert [d.metadata["chunk_id"] for d,_ in result]==[1,2]


@pytest.mark.unit
def test_question_context_rejects_weak_semantic_only_match(monkeypatch):
    monkeypatch.setattr(retrieval,"expand_retrieval_context",lambda vector_store,retrieved_results,**kwargs:retrieved_results)
    monkeypatch.setattr(retrieval,"retrieve_mmr",lambda **k:[(doc(1,0,10,"weak"),1.25)])
    monkeypatch.setattr(retrieval,"lexical_search",lambda **k:[])
    assert retrieval.retrieve_question_context(object(),"unrelated question",max_distance=1.3)==[]


@pytest.mark.unit
def test_question_context_accepts_strong_semantic_match(monkeypatch):
    monkeypatch.setattr(retrieval,"expand_retrieval_context",lambda vector_store,retrieved_results,**kwargs:retrieved_results)
    item=doc(1,0,10,"strong")
    monkeypatch.setattr(retrieval,"retrieve_mmr",lambda **k:[(item,1.0)])
    monkeypatch.setattr(retrieval,"lexical_search",lambda **k:[])
    assert retrieval.retrieve_question_context(object(),"relevant question",max_distance=1.3)==[(item,1.0)]



@pytest.mark.unit
def test_context_expansion_adds_adjacent_chunks_in_chronological_order():
    documents = {
        f"d{i}": doc(
            i,
            i * 10,
            i * 10 + 5,
            f"segment {i}",
        )
        for i in range(5)
    }
    store = FakeVectorStore(documents)

    results = retrieval.expand_retrieval_context(
        store,
        [(documents["d2"], 0.2)],
        window=1,
        max_chunks=12,
    )

    assert [item[0].metadata["chunk_id"] for item in results] == [1, 2, 3]


@pytest.mark.unit
def test_context_expansion_deduplicates_overlapping_neighbors():
    documents = {
        f"d{i}": doc(
            i,
            i * 10,
            i * 10 + 5,
            f"segment {i}",
        )
        for i in range(5)
    }
    store = FakeVectorStore(documents)

    results = retrieval.expand_retrieval_context(
        store,
        [
            (documents["d1"], 0.2),
            (documents["d2"], 0.3),
        ],
        window=1,
        max_chunks=12,
    )

    assert [item[0].metadata["chunk_id"] for item in results] == [0, 1, 2, 3]


@pytest.mark.unit
def test_context_expansion_honors_hard_maximum_and_keeps_anchors():
    documents = {
        f"d{i}": doc(
            i,
            i * 10,
            i * 10 + 5,
            f"segment {i}",
        )
        for i in range(7)
    }
    store = FakeVectorStore(documents)

    results = retrieval.expand_retrieval_context(
        store,
        [
            (documents["d1"], 0.2),
            (documents["d5"], 0.3),
        ],
        window=2,
        max_chunks=4,
    )

    chunk_ids = [
        item[0].metadata["chunk_id"]
        for item in results
    ]

    assert len(chunk_ids) == 4
    assert 1 in chunk_ids
    assert 5 in chunk_ids
    assert chunk_ids == sorted(chunk_ids)


@pytest.mark.unit
def test_context_expansion_can_be_disabled():
    documents = {
        f"d{i}": doc(
            i,
            i * 10,
            i * 10 + 5,
            f"segment {i}",
        )
        for i in range(3)
    }
    store = FakeVectorStore(documents)
    anchor = (documents["d1"], 0.2)

    assert retrieval.expand_retrieval_context(
        store,
        [anchor],
        window=0,
        max_chunks=12,
    ) == [anchor]
