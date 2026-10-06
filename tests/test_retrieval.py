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
def test_query_focus_terms_remove_speaker_question_boilerplate():
    terms = retrieval.extract_query_focus_terms(
        "What does Vijay Mallya say about the failure of Kingfisher Airlines?"
    )
    assert terms == ["failure", "kingfisher", "airlines"]


@pytest.mark.unit
def test_query_variants_keep_original_and_add_content_focus():
    variants = retrieval.build_retrieval_query_variants(
        "What does Vijay Mallya say about building the Kingfisher brand?"
    )
    assert variants[0].startswith("What does Vijay Mallya say")
    assert "building kingfisher brand" in variants



@pytest.mark.unit
def test_query_variants_add_intent_specific_facets_for_dense_questions():
    variants = retrieval.build_retrieval_query_variants(
        "Why did the company fail and what challenges caused the problems?"
    )

    assert len(variants) == 5
    assert variants[0].startswith("Why did the company fail")
    assert "company fail challenges caused problems" in variants[1]
    assert any(
        "financial crisis economic circumstances" in variant
        for variant in variants[2:]
    )
    assert any(
        "government policy regulation banks support" in variant
        for variant in variants[2:]
    )
    assert any(
        "operational challenges payments fees fuel" in variant
        for variant in variants[2:]
    )


@pytest.mark.unit
def test_evidence_facet_planner_switches_to_brand_queries():
    queries = retrieval.build_evidence_facet_queries(
        "What does Vijay Mallya say about building the Kingfisher brand?"
    )

    assert len(queries) == 3
    assert any("brand branding advertising marketing" in query for query in queries)
    assert any("business companies subsidiaries ownership" in query for query in queries)
    assert any("operational challenges payments fees" in query for query in queries)


@pytest.mark.unit
def test_policy_facet_includes_change_and_regulatory_vocabulary():
    plans = retrieval.build_evidence_facet_plan(
        "What does Vijay Mallya say about the role of Indian government policy in the problems faced by Kingfisher Airlines?"
    )

    policy_queries = [
        query
        for label, query in plans
        if label == "policy_governance"
    ]

    assert len(policy_queries) == 1
    assert "policy changes" in policy_queries[0]
    assert "regulatory changes" in policy_queries[0]


@pytest.mark.unit
def test_policy_change_facet_lexically_surfaces_shared_policy_evidence():
    documents = {
        "policy_changes": doc(
            7,
            70,
            75,
            "I asked only for policy changes.",
        ),
        "generic_policy": doc(
            8,
            80,
            85,
            "Government policy affected banks and support.",
        ),
    }

    plans = retrieval.build_evidence_facet_plan(
        "What does the role of government policy mean for this airline?"
    )
    policy_query = next(
        query
        for label, query in plans
        if label == "policy_governance"
    )

    results = retrieval.lexical_search(
        FakeVectorStore(documents),
        policy_query,
        limit=8,
    )

    assert results
    assert 7 in [
        document.metadata["chunk_id"]
        for document, _score in results
    ]


@pytest.mark.unit
def test_query_variants_do_not_add_facets_to_simple_questions():
    variants = retrieval.build_retrieval_query_variants(
        "Who is Vijay Mallya?"
    )

    assert len(variants) == 2
    assert variants[0] == "Who is Vijay Mallya?"
    assert variants[1] == "vijay mallya"
    assert not any(
        "financial economic" in variant
        or "government policy" in variant
        or "operational challenges" in variant
        for variant in variants
    )


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
def test_lexical_facet_fusion_prefers_shared_evidence():
    shared = doc(2, 10, 15, "shared financial policy evidence")
    financial = [
        (shared, 0.8),
        (doc(1, 0, 5, "financial only"), 0.9),
    ]
    policy = [
        (shared, 0.7),
        (doc(3, 20, 25, "policy only"), 0.9),
    ]

    results = retrieval.fuse_lexical_rankings(
        [financial, policy],
        rrf_k=60,
    )

    assert results[0][0].metadata["chunk_id"] == 2


@pytest.mark.unit
def test_dense_question_runs_facet_lexical_queries(monkeypatch):
    item = doc(1, 0, 10, "evidence")
    lexical_queries = []

    monkeypatch.setattr(
        retrieval,
        "retrieve_mmr",
        lambda **kwargs: [(item, 0.5)],