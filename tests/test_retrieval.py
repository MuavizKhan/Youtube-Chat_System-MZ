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
        "Why did the rocket fail and what challenges caused the problems?"
    )

    assert len(variants) == 3
    assert variants[0].startswith("Why did the rocket fail")
    assert "rocket fail challenges caused problems" in variants[1]
    assert any("causes reasons factors" in variant for variant in variants[2:])
    assert not any(
        "government policy regulation" in variant
        or "financial crisis economic circumstances" in variant
        or "fuel suppliers service costs" in variant
        for variant in variants
    )




@pytest.mark.unit
def test_evidence_facet_planner_switches_to_brand_queries():
    plans = retrieval.build_evidence_facet_plan(
        "What does Vijay Mallya say about building the Kingfisher brand?"
    )

    assert [label for label, _query in plans] == ["brand_marketing"]
    assert "brand branding advertising marketing" in plans[0][1]
    assert not any(
        label in {"financial_economic", "policy_governance", "operational_challenges"}
        for label, _query in plans
    )




@pytest.mark.unit
def test_policy_questions_add_focused_policy_change_facet():
    query = "What role does government policy play in problems faced by people?"
    plans = retrieval.build_retrieval_query_plan(query)
    labels = [label for label, _query in plans]
    assert labels[:2] == ["original", "focus"]
    assert labels[2:] == ["policy_change", "policy_governance", "causal_factors"]




@pytest.mark.unit
def test_policy_governance_facet_keeps_original_terms():
    query = "What role does government policy play in problems faced by people?"
    plans = retrieval.build_evidence_facet_plan(query)

    governance_query = next(
        query_text
        for label, query_text in plans
        if label == "policy_governance"
    )

    assert "government policy regulation banks support" in governance_query


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

    query = "What role does government policy play in these problems?"
    plans = retrieval.build_evidence_facet_plan(query)
    policy_change_query = next(
        query_text
        for label, query_text in plans
        if label == "policy_change"
    )

    assert policy_change_query.endswith(
        "policy changes regulatory changes rules regulations"
    )

    results = retrieval.lexical_search(
        FakeVectorStore(documents),
        policy_change_query,
        limit=8,
    )

    assert results
    assert results[0][0].metadata["chunk_id"] == 7


@pytest.mark.unit
@pytest.mark.parametrize(
    "query",
    [
        "Why did the airline fail and what challenges did it face?",
        "Why did the company struggle, and what problems contributed to the decline?",
    ],
)
def test_business_failure_questions_add_domain_conditioned_cause_facets(query):
    labels = [label for label, _query in retrieval.build_evidence_facet_plan(query)]

    assert "financial_economic" in labels
    assert "policy_governance" in labels
    assert "operational_challenges" in labels
    assert "causal_factors" in labels


@pytest.mark.unit
def test_nonbusiness_failure_questions_do_not_add_business_facets():
    labels = [
        label for label, _query in retrieval.build_evidence_facet_plan(
            "Why did this process fail and what caused the error?"
        )
    ]

    assert "financial_economic" not in labels
    assert "policy_governance" not in labels
    assert "operational_challenges" not in labels
    assert "causal_factors" in labels


@pytest.mark.unit
def test_non_policy_dense_questions_do_not_add_policy_change_facet():
    query = "Why did the company fail and what challenges caused the problems?"
    plans = retrieval.build_retrieval_query_plan(query)

    assert "policy_change" not in [
        label
        for label, _query in plans
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
    "What is the main topic of this video?",
    "What is the primary topic of the video?",
])
def test_overview_detection(query):
    assert retrieval.is_overview_question(query)


@pytest.mark.unit
def test_overview_detection_rejects_specific_question():
    assert not retrieval.is_overview_question("Who is Vijay Mallya?")


@pytest.mark.unit
@pytest.mark.parametrize(
    "query",
    [
        "How many episodes are in this series?",
        "How long is the interview?",
        "How old is the speaker?",
    ],
)
def test_quantitative_how_questions_do_not_trigger_dense_retrieval(query):
    assert not retrieval.is_evidence_dense_question(query)


@pytest.mark.unit
@pytest.mark.parametrize(
    "query",
    [
        "How does the engine work?",
        "Why did the rocket launch fail?",
        "What are the effects of this treatment?",
    ],
)
def test_explanatory_youtube_questions_trigger_dense_retrieval(query):
    assert retrieval.is_evidence_dense_question(query)


@pytest.mark.unit
def test_overview_can_select_one_representative_chunk():
    documents = {
        f"d{i}": doc(i, i * 10, i * 10 + 5, f"segment {i}")
        for i in range(5)
    }
    result = retrieval.retrieve_overview(
        FakeVectorStore(documents),
        number_of_chunks=1,
    )
    assert len(result) == 1
    assert result[0][0].metadata["chunk_id"] == 2


@pytest.mark.unit
@pytest.mark.parametrize("number_of_chunks", [0, -1])
def test_overview_rejects_nonpositive_chunk_budget(number_of_chunks):
    with pytest.raises(ValueError, match="positive integer"):
        retrieval.retrieve_overview(FakeVectorStore({}), number_of_chunks=number_of_chunks)


 

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
def test_lexical_search_can_retrieve_a_rare_transcript_term_alone():
    documents = {
        "rare": doc(1, 0, 5, "Photolithography is a semiconductor technique."),
        "common1": doc(2, 10, 15, "The process and work method for chips."),
        "common2": doc(3, 20, 25, "This process and work applies to chips."),
        "common3": doc(4, 30, 35, "A process works with chips and this method."),
    }
    results = retrieval.lexical_search(
        FakeVectorStore(documents),
        "Explain photolithography process work method in chips",
        limit=8,
    )
    assert 1 in [document.metadata["chunk_id"] for document, _score in results]


@pytest.mark.unit
def test_rare_term_alone_does_not_match_short_unrelated_query():
    documents = {
        "australia": doc(1, 0, 5, "The video briefly mentions Australia."),
        "capital": doc(2, 10, 15, "The speaker discusses a capital expenditure."),
        "other": doc(3, 20, 25, "A separate topic in the interview."),
    }

    results = retrieval.lexical_search(
        FakeVectorStore(documents),
        "What is the capital of Australia?",
        limit=8,
    )

    assert 1 not in [document.metadata["chunk_id"] for document, _score in results]


@pytest.mark.unit
def test_facet_support_can_rescue_evidence_diluted_by_global_fusion():
    ranked = [
        (doc(i, i * 10, i * 10 + 5, f"global candidate {i}"), 0.2 + i / 100)
        for i in range(1, 25)
    ]
    facet_target = ranked[17]

    result = retrieval.rerank_with_soft_facet_support(
        ranked,
        {"focused_facet": [facet_target]},
    )

    chunk_ids = [document.metadata["chunk_id"] for document, _score in result]
    assert chunk_ids.index(18) < 3
    assert chunk_ids.index(18) < chunk_ids.index(1)


@pytest.mark.unit
def test_lexical_search_returns_empty_for_nonpositive_limit():
    store = FakeVectorStore({
        "a": doc(1, 0, 5, "Alice works at Acme."),
        "b": doc(2, 10, 15, "Alice mentions Acme."),
    })
    assert retrieval.lexical_search(store, "Alice Acme", limit=0) == []
    assert retrieval.lexical_search(store, "Alice Acme", limit=-1) == []


@pytest.mark.unit
def test_lexical_tokenization_preserves_unicode_terms():
    assert retrieval.extract_query_terms("café naïve 東京") == {"café", "naïve", "東京"}


@pytest.mark.unit
def test_rank_fusion_does_not_collapse_chunks_without_ids_or_timestamps():
    first = Document(page_content="first transcript passage", metadata={"video_id": "videoA"})
    second = Document(page_content="second transcript passage", metadata={"video_id": "videoA"})
    fused = retrieval.fuse_ranked_results([[(first, 0.1), (second, 0.2)]])
    assert len(fused) == 2
    assert {document.page_content for document, _score in fused} == {
        "first transcript passage", "second transcript passage"
    }


@pytest.mark.unit
def test_lexical_fusion_preserves_higher_is_better_raw_scores():
    shared = doc(10, 0, 5, "specific term")
    fused = retrieval.fuse_lexical_rankings([
        [(shared, 0.2)],
        [(shared, 0.9)],
    ])
    assert len(fused) == 1
    assert fused[0][0].metadata["chunk_id"] == 10
    assert fused[0][1] == pytest.approx(0.9)


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
    )
    monkeypatch.setattr(
        retrieval,
        "lexical_search",
        lambda **kwargs: (
            lexical_queries.append(kwargs["query"])
            or []
        ),
    )
    monkeypatch.setattr(
        retrieval,
        "expand_retrieval_context",
        lambda vector_store, retrieved_results, **kwargs: retrieved_results,
    )

    results = retrieval.retrieve_question_context(
        object(),
        "Why did the rocket fail and what challenges caused the problems?",
        expand_context=False,
    )

    assert results == [(item, 0.5)]
    assert len(lexical_queries) == 2
    assert lexical_queries[0].startswith("Why did the company fail")
    assert "causes reasons factors" in lexical_queries[1]
    assert not any(
        "financial crisis economic circumstances" in query
        or "government policy regulation banks support" in query
        or "operational challenges payments fees fuel" in query
        for query in lexical_queries[1:]
    )


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
@pytest.mark.parametrize("limit", [1.5, True])
def test_anchor_selection_rejects_invalid_limit_types(limit):
    with pytest.raises(ValueError, match="limit must be a positive integer"):
        retrieval.select_diverse_retrieval_anchors([], limit=limit)


@pytest.mark.unit
@pytest.mark.parametrize("min_chunk_gap", [1.5, True])
def test_anchor_selection_rejects_invalid_gap_types(min_chunk_gap):
    with pytest.raises(ValueError, match="min_chunk_gap must be a non-negative integer"):
        retrieval.select_diverse_retrieval_anchors([], limit=2, min_chunk_gap=min_chunk_gap)


@pytest.mark.unit
def test_select_diverse_retrieval_anchors_spreads_dense_evidence():
    ranked = [
        (doc(10, 100, 105, "region A"), 0.1),
        (doc(11, 110, 115, "region A neighbor"), 0.2),
        (doc(40, 400, 405, "region B"), 0.3),
        (doc(41, 410, 415, "region B neighbor"), 0.4),
        (doc(80, 800, 805, "region C"), 0.5),
    ]

    result = retrieval.select_diverse_retrieval_anchors(
        ranked,
        limit=3,
        min_chunk_gap=3,
    )

    assert [item[0].metadata["chunk_id"] for item in result] == [10, 40, 80]


@pytest.mark.unit
def test_select_diverse_retrieval_anchors_fills_remaining_budget():
    ranked = [
        (doc(10, 100, 105, "region A"), 0.1),
        (doc(11, 110, 115, "region A neighbor"), 0.2),
    ]

    result = retrieval.select_diverse_retrieval_anchors(
        ranked,
        limit=3,
        min_chunk_gap=3,
    )

    assert [item[0].metadata["chunk_id"] for item in result] == [10, 11]


@pytest.mark.unit
def test_question_context_uses_dense_anchor_headroom(monkeypatch):
    dense = "Why did the company fail and what challenges caused the problems?"
    ranked = [
        (doc(10, 100, 105, "first region"), 0.1),
        (doc(24, 240, 245, "supporting region"), 0.2),
        (doc(26, 260, 265, "nearby supporting region"), 0.3),
    ]
    captured = {}

    monkeypatch.setattr(
        retrieval,
        "retrieve_mmr",
        lambda **kwargs: ranked,
    )
    monkeypatch.setattr(
        retrieval,
        "lexical_search",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        retrieval,
        "DENSE_ANCHOR_LIMIT",
        12,
    )
    monkeypatch.setattr(
        retrieval,
        "DENSE_ANCHOR_MIN_CHUNK_GAP",
        2,
    )
    monkeypatch.setattr(
        retrieval,
        "select_diverse_retrieval_anchors",
        lambda ranked_results, *, limit, min_chunk_gap: (
            captured.update(
                limit=limit,
                min_chunk_gap=min_chunk_gap,
            )
            or ranked_results
        ),
    )
    monkeypatch.setattr(
        retrieval,
        "expand_retrieval_context",
        lambda vector_store, retrieved_results, **kwargs: retrieved_results,
    )

    retrieval.retrieve_question_context(
        object(),
        dense,
        k=2,
        fetch_k=4,
        expand_context=False,
    )

    assert captured == {
        "limit": 12,
        "min_chunk_gap": 2,
    }


@pytest.mark.unit
def test_question_context_soft_reranks_dense_anchors(monkeypatch):
    monkeypatch.setattr(retrieval, "RAG_RERANK_ENABLED", False)

    dense = "Why did the company fail and what challenges caused the problems?"
    ranked = [
        (doc(10, 100, 105, "first region"), 0.1),
        (doc(11, 110, 115, "first region neighbor"), 0.2),
        (doc(40, 400, 405, "second region"), 0.3),
        (doc(80, 800, 805, "third region"), 0.4),
    ]

    monkeypatch.setattr(
        retrieval,
        "retrieve_mmr",
        lambda **kwargs: ranked,
    )
    monkeypatch.setattr(
        retrieval,
        "lexical_search",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        retrieval,
        "expand_retrieval_context",
        lambda vector_store, retrieved_results, **kwargs: retrieved_results,
    )

    result = retrieval.retrieve_question_context(
        object(),
        dense,
        k=2,
        fetch_k=4,
        expand_context=False,
    )

    assert [item[0].metadata["chunk_id"] for item in result[:3]] == [10, 40, 80]


@pytest.mark.unit
def test_soft_facet_rerank_boosts_facet_supported_candidate_without_hard_quota():
    ranked = [
        (doc(1, 10, 15, "global best"), 0.1),
        (doc(2, 20, 25, "global second"), 0.2),
        (doc(3, 30, 35, "global third"), 0.3),
        (doc(4, 40, 45, "facet strong"), 0.4),
        (doc(5, 50, 55, "global fifth"), 0.5),
        (doc(6, 60, 65, "global sixth"), 0.6),
    ]

    facet_rankings = {
        "financial_economic": [
            (ranked[5][0], 0.6),
        ],
    }

    reranked = retrieval.rerank_with_soft_facet_support(
        ranked,
        facet_rankings,
        facet_weight=0.05,
    )

    chunk_ids = [
        document.metadata["chunk_id"]
        for document, _ in reranked
    ]

    assert chunk_ids.index(6) < chunk_ids.index(4)
    assert chunk_ids[0] == 1


@pytest.mark.unit
def test_soft_facet_rerank_returns_global_order_without_facets():
    ranked = [
        (doc(1, 10, 15, "one"), 0.1),
        (doc(2, 20, 25, "two"), 0.2),
        (doc(3, 30, 35, "three"), 0.3),
    ]

    assert retrieval.rerank_with_soft_facet_support(
        ranked,
        {},
    ) == ranked


@pytest.mark.unit
def test_question_context_uses_overview_path(monkeypatch):
    expected=[(doc(1,0,10,"overview"),0.0)]
    monkeypatch.setattr(retrieval,"retrieve_overview",lambda *a,**k:expected)
    monkeypatch.setattr(retrieval,"retrieve_mmr",lambda *a,**k:pytest.fail("semantic path should not run"))
    assert retrieval.retrieve_question_context(object(),"What is this video about?")==expected


@pytest.mark.unit
def test_question_context_merges_and_deduplicates(monkeypatch):
    monkeypatch.setattr(retrieval, "RAG_RERANK_ENABLED", False)
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
def test_semantic_query_variant_fusion_prefers_shared_evidence():
    shared = doc(2, 10, 15, "shared topic")
    original = [
        (doc(1, 0, 5, "generic question match"), 0.2),
        (shared, 0.3),
    ]
    focused = [
        (shared, 0.25),
        (doc(3, 20, 25, "focused match"), 0.4),
    ]

    results = retrieval.fuse_semantic_rankings(
        [original, focused],
        rrf_k=60,
    )

    assert results[0][0].metadata["chunk_id"] == 2


@pytest.mark.unit
def test_lexical_search_weights_specific_terms_over_question_boilerplate():
    documents = {
        "a": doc(1, 0, 10, "Kingfisher Airlines is a company."),
        "b": doc(2, 10, 20, "Kingfisher Airlines faced a global financial crisis."),
        "c": doc(3, 20, 30, "This is a general discussion about the interview."),
    }

    results = retrieval.lexical_search(
        FakeVectorStore(documents),
        "What does Vijay Mallya say about the global financial crisis affecting Kingfisher Airlines?",
        limit=3,
    )

    assert results[0][0].metadata["chunk_id"] == 2



@pytest.mark.unit
def test_raw_faiss_diagnostic_exposes_candidate_rank():
    documents = {
        "a": doc(1, 0, 10, "Alice failed at Acme."),
        "b": doc(2, 10, 20, "Another segment."),
        "c": doc(3, 20, 30, "Different segment."),
    }
    store = FakeVectorStore(
        documents,
        distances=[0.1, 0.9, 1.4],
        indices=[0, 1, 2],
    )

    results = retrieval.retrieve_faiss_candidates_for_diagnostics(
        store,
        "alice failed",
        fetch_k=3,
        max_distance=1.0,
    )

    assert [item[0].metadata["chunk_id"] for item in results] == [1, 2, 3]
    assert [item[2] for item in results] == [1, 2, 3]




@pytest.mark.unit
def test_raw_faiss_diagnostic_keeps_distance_for_gate_analysis():
    documents = {
        "a": doc(1, 0, 10, "Alice evidence."),
    }
    store = FakeVectorStore(
        documents,
        distances=[1.5],
        indices=[0],
    )

    results = retrieval.retrieve_faiss_candidates_for_diagnostics(
        store,
        "Alice evidence",
        fetch_k=1,
        max_distance=1.0,
    )

    assert results[0][0].metadata["chunk_id"] == 1
    assert results[0][1] == pytest.approx(1.5)
    assert results[0][2] == 1
@pytest.mark.unit
def test_diagnostic_candidate_search_skips_stale_docstore_entries():
    store = FakeVectorStore({
        "valid": doc(1, 0, 5, "valid transcript candidate"),
    })
    store.index_to_docstore_id = {0: "stale-docstore-id"}

    results = retrieval.retrieve_faiss_candidates_for_diagnostics(
        store,
        "transcript question",
        fetch_k=1,
        max_distance=1.3,
    )

    assert results == []


@pytest.mark.unit
def test_diagnostic_candidate_search_rejects_nonfinite_distance_limit():
    store = FakeVectorStore({
        "valid": doc(1, 0, 5, "valid transcript candidate"),
    })
    with pytest.raises(ValueError, match="finite and non-negative"):
        retrieval.retrieve_faiss_candidates_for_diagnostics(
            store,
            "transcript question",
            fetch_k=1,
            max_distance=float("nan"),
        )


@pytest.mark.unit
def test_end_to_end_retrieval_diagnostic_exposes_all_stages():
    documents = {
        "a": doc(1, 0, 10, "Alice failed at Acme."),
        "b": doc(2, 10, 20, "Alice challenges at Acme."),
        "c": doc(3, 20, 30, "Unrelated segment."),
    }
    store = FakeVectorStore(
        documents,
        distances=[0.1, 0.2, 0.9],
        indices=[0, 1, 2],
    )

    result = retrieval.diagnose_retrieval_pipeline(
        store,
        "Why did Alice fail at Acme?",
        k=2,
        fetch_k=3,
    )

    assert result["dense_question"]
    assert result["semantic_k"] == retrieval.DENSE_SEMANTIC_K
    assert result["semantic_fetch_k"] == retrieval.DENSE_SEMANTIC_FETCH_K
    assert result["query_variants"]
    assert result["semantic_stages"]
    assert result["semantic_fused"]
    assert result["lexical_queries"]
    assert result["lexical_stages"]
    assert result["lexical"]
    assert result["facet_rankings"]
    assert result["hybrid_fused"]
    assert result["anchors"]
    assert result["final_context"]


@pytest.mark.unit
@pytest.mark.parametrize("has_lexical_evidence", [True, False])
def test_diagnostics_match_production_anchor_and_context_budgets(
    monkeypatch,
    has_lexical_evidence,
):
    evidence = doc(10, 0, 5, "relevant evidence")
    semantic_result = [(evidence, 0.1)]
    lexical_result = [(evidence, 0.2)] if has_lexical_evidence else []
    observed_anchor_gaps = []
    observed_context_budgets = []

    monkeypatch.setattr(
        retrieval,
        "is_overview_question",
        lambda _query: False,
    )
    monkeypatch.setattr(
        retrieval,
        "is_evidence_dense_question",
        lambda _query: True,
    )
    monkeypatch.setattr(
        retrieval,
        "build_retrieval_query_plan",
        lambda _query: [("original", "test question")],
    )
    monkeypatch.setattr(
        retrieval,
        "build_evidence_facet_plan",
        lambda _query: [],
    )
    monkeypatch.setattr(
        retrieval,
        "retrieve_faiss_candidates_for_diagnostics",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        retrieval,
        "retrieve_mmr",
        lambda **kwargs: semantic_result,
    )
    monkeypatch.setattr(
        retrieval,
        "fuse_semantic_rankings",
        lambda _rankings: semantic_result,
    )
    monkeypatch.setattr(
        retrieval,
        "lexical_search",
        lambda **kwargs: lexical_result,
    )
    monkeypatch.setattr(
        retrieval,
        "fuse_lexical_rankings",
        lambda _rankings: lexical_result,
    )
    monkeypatch.setattr(
        retrieval,
        "fuse_semantic_and_lexical_results",
        lambda **kwargs: semantic_result,
    )
    monkeypatch.setattr(
        retrieval,
        "rerank_with_cross_encoder",
        lambda **kwargs: semantic_result,
    )

    def record_anchor_selection(ranked_results, *, limit, min_chunk_gap=3):
        observed_anchor_gaps.append(min_chunk_gap)
        return list(ranked_results[:limit])

    def record_context_expansion(
        _vector_store,
        ranked_results,
        *,
        max_chunks,
    ):
        observed_context_budgets.append(max_chunks)
        return list(ranked_results[:max_chunks])

    monkeypatch.setattr(
        retrieval,
        "select_diverse_retrieval_anchors",
        record_anchor_selection,
    )
    monkeypatch.setattr(
        retrieval,
        "expand_retrieval_context",
        record_context_expansion,
    )

    result = retrieval.diagnose_retrieval_pipeline(
        object(),
        "test question",
    )

    expected_context_budget = retrieval.DENSE_CONTEXT_MAX_CHUNKS
    assert observed_anchor_gaps == [
        retrieval.DENSE_ANCHOR_MIN_CHUNK_GAP
    ]
    assert observed_context_budgets == [expected_context_budget]
    assert result["anchors"] == semantic_result
    assert result["final_context"] == semantic_result


@pytest.mark.unit
@pytest.mark.parametrize("window", [1.5, True])
def test_context_expansion_rejects_invalid_window_types(window):
    document = doc(1, 0, 5, "transcript")
    store = FakeVectorStore({"one": document})
    with pytest.raises(ValueError, match="window must be a non-negative integer"):
        retrieval.expand_retrieval_context(store, [(document, 0.1)], window=window)


@pytest.mark.unit
@pytest.mark.parametrize("max_chunks", [1.5, True])
def test_context_expansion_rejects_invalid_budget_types(max_chunks):
    document = doc(1, 0, 5, "transcript")
    store = FakeVectorStore({"one": document})
    with pytest.raises(ValueError, match="max_chunks must be a positive integer"):
        retrieval.expand_retrieval_context(store, [(document, 0.1)], max_chunks=max_chunks)


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


@pytest.mark.unit
def test_question_context_can_return_raw_retrieval_without_expansion(monkeypatch):
    anchor = doc(2, 20, 25, "anchor")
    neighbor = doc(3, 30, 35, "neighbor")

    monkeypatch.setattr(
        retrieval,
        "retrieve_mmr",
        lambda **kwargs: [(anchor, 0.2)],
    )
    monkeypatch.setattr(
        retrieval,
        "lexical_search",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        retrieval,
        "expand_retrieval_context",
        lambda *args, **kwargs: [(neighbor, 0.2), (anchor, 0.2)],
    )

    raw = retrieval.retrieve_question_context(
        object(),
        "relevant question",
        expand_context=False,
    )
    expanded = retrieval.retrieve_question_context(
        object(),
        "relevant question",
        expand_context=True,
    )

    assert raw == [(anchor, 0.2)]
    assert expanded == [(neighbor, 0.2), (anchor, 0.2)]



@pytest.mark.unit
def test_reciprocal_rank_fusion_combines_semantic_and_lexical_rankings():
    semantic = [
        (doc(1, 0, 5, "semantic one"), 0.1),
        (doc(2, 5, 10, "semantic two"), 0.2),
        (doc(3, 10, 15, "semantic three"), 0.3),
        (doc(4, 15, 20, "semantic four"), 0.4),
    ]
    lexical = [
        (doc(5, 20, 25, "exact lexical evidence"), 1.0),
    ]

    results = retrieval.fuse_semantic_and_lexical_results(
        semantic,
        lexical,
        lexical_distance=1.3,
        rrf_k=60,
    )

    chunk_ids = [document.metadata["chunk_id"] for document, _ in results]

    assert chunk_ids[0] == 1
    assert chunk_ids[1] == 5
    assert chunk_ids.index(5) < chunk_ids.index(4)


@pytest.mark.unit
def test_reciprocal_rank_fusion_prefers_documents_supported_by_both_signals():
    shared = doc(2, 5, 10, "shared evidence")
    semantic = [
        (doc(1, 0, 5, "semantic only"), 0.1),
        (shared, 0.2),
    ]
    lexical = [
        (shared, 1.0),
        (doc(3, 10, 15, "lexical only"), 1.0),
    ]

    results = retrieval.fuse_semantic_and_lexical_results(
        semantic,
        lexical,
        lexical_distance=1.3,
        rrf_k=60,
    )

    assert results[0][0].metadata["chunk_id"] == 2


@pytest.mark.unit
def test_weighted_hybrid_rrf_can_retain_lower_ranked_lexical_evidence():
    shared = [
        (doc(100 + rank, rank * 10, rank * 10 + 5, f"shared evidence {rank}"), 0.05 + rank / 100)
        for rank in range(1, 7)
    ]
    semantic_only = [
        (doc(200 + rank, 1000 + rank * 10, 1005 + rank * 10, f"semantic {rank}"), 0.1 + rank / 100)
        for rank in range(7, 13)
    ]
    lexical_only = [
        (doc(300 + rank, 2000 + rank * 10, 2005 + rank * 10, f"lexical {rank}"), 1.0)
        for rank in range(7, 13)
    ]
    semantic = shared + semantic_only
    lexical = [(document, 1.0) for document, _distance in shared] + lexical_only
    target_chunk_id = lexical_only[-1][0].metadata["chunk_id"]

    equal_weight = retrieval.fuse_semantic_and_lexical_results(
        semantic,
        lexical,
        lexical_distance=1.3,
        lexical_weight=1.0,
        rrf_k=60,
    )
    weighted = retrieval.fuse_semantic_and_lexical_results(
        semantic,
        lexical,
        lexical_distance=1.3,
        lexical_weight=1.25,
        rrf_k=60,
    )

    equal_ids = [document.metadata["chunk_id"] for document, _distance in equal_weight]
    weighted_ids = [document.metadata["chunk_id"] for document, _distance in weighted]

    # Six chunks have support from both branches. With equal source weights,
    # the rank-12 lexical-only candidate falls outside the dense anchor budget;
    # the bounded lexical contribution lets it survive alongside shared evidence.
    assert equal_ids.index(target_chunk_id) >= 12
    assert weighted_ids.index(target_chunk_id) == 11

    target_distance = next(
        distance
        for document, distance in weighted
        if document.metadata["chunk_id"] == target_chunk_id
    )
    assert target_distance == pytest.approx(1.3)


@pytest.mark.unit
@pytest.mark.parametrize("lexical_weight", [0.0, -1.0, float("nan"), float("inf")])
def test_hybrid_rrf_rejects_invalid_lexical_weight(lexical_weight):
    with pytest.raises(ValueError, match="lexical_weight"):
        retrieval.fuse_semantic_and_lexical_results(
            [],
            [],
            lexical_weight=lexical_weight,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("query", "dense", "expected_calls"),
    [
        ("Why did the company fail and what challenges caused the problems?", True, 2),
        ("Who is Vijay Mallya?", False, 1),
    ],
)
def test_question_context_weights_lexical_rrf_only_for_dense_questions(
    monkeypatch,
    query,
    dense,
    expected_calls,
):
    item = doc(1, 0, 10, "policy evidence")
    ranked = [(item, 0.2)]
    observed_weights = []
    observed_lexical_limits = []

    monkeypatch.setattr(retrieval, "is_overview_question", lambda _query: False)
    monkeypatch.setattr(retrieval, "is_evidence_dense_question", lambda _query: dense)
    monkeypatch.setattr(
        retrieval,
        "build_retrieval_query_plan",
        lambda _query: (
            [("original", query), ("policy_governance", "government policy support")]
            if dense
            else [("original", query)]
        ),
    )
    monkeypatch.setattr(
        retrieval,
        "build_evidence_facet_plan",
        lambda _query: [("policy_governance", "government policy support")]
        if dense
        else [],
    )
    monkeypatch.setattr(retrieval, "retrieve_mmr", lambda **_kwargs: ranked)
    monkeypatch.setattr(retrieval, "fuse_semantic_rankings", lambda _rankings: ranked)
    def fake_lexical_search(**kwargs):
        observed_lexical_limits.append(kwargs["limit"])
        return ranked

    monkeypatch.setattr(retrieval, "lexical_search", fake_lexical_search)
    monkeypatch.setattr(retrieval, "fuse_lexical_rankings", lambda _rankings: ranked)

    def capture_fusion(**kwargs):
        observed_weights.append(kwargs["lexical_weight"])
        return ranked

    monkeypatch.setattr(
        retrieval,
        "fuse_semantic_and_lexical_results",
        capture_fusion,
    )
    monkeypatch.setattr(retrieval, "rerank_with_cross_encoder", lambda **kwargs: ranked)
    monkeypatch.setattr(retrieval, "rerank_with_soft_facet_support", lambda results, _facets: results)
    monkeypatch.setattr(
        retrieval,
        "select_diverse_retrieval_anchors",
        lambda results, **_kwargs: results,
    )
    monkeypatch.setattr(
        retrieval,
        "expand_retrieval_context",
        lambda _store, results, **_kwargs: results,
    )

    retrieval.retrieve_question_context(
        object(),
        query,
        expand_context=False,
    )

    expected_weight = retrieval.DENSE_LEXICAL_RRF_WEIGHT if dense else 1.0
    assert len(observed_weights) == expected_calls
    assert observed_weights == [expected_weight] * expected_calls
    if dense:
        assert observed_lexical_limits
        assert min(observed_lexical_limits) >= retrieval.DENSE_ANCHOR_LIMIT


@pytest.mark.unit
def test_cross_encoder_reranking_reorders_candidates_and_preserves_distances(monkeypatch):
    documents = [
        doc(1, 0, 10, "first candidate"),
        doc(2, 10, 20, "second candidate"),
        doc(3, 20, 30, "third candidate"),
    ]
    ranked_results = [
        (documents[0], 0.2),
        (documents[1], 0.3),
        (documents[2], 0.4),
    ]

    class FakeReranker:
        def predict(self, pairs, batch_size, show_progress_bar):
            assert len(pairs) == 2
            assert batch_size == 4
            assert show_progress_bar is False
            return np.asarray([0.1, 0.9], dtype=np.float32)

    monkeypatch.setattr(retrieval, "RAG_RERANK_ENABLED", True)
    monkeypatch.setattr(
        retrieval,
        "RAG_RERANK_BATCH_SIZE",
        4,
    )
    monkeypatch.setattr(
        retrieval,
        "create_cross_encoder_reranker",
        lambda: FakeReranker(),
    )

    result = retrieval.rerank_with_cross_encoder(
        query="Which candidate is relevant?",
        ranked_results=ranked_results,
        candidate_k=2,
    )

    assert [item[0].metadata["chunk_id"] for item in result] == [2, 1, 3]
    assert [item[1] for item in result] == [0.3, 0.2, 0.4]


@pytest.mark.unit
def test_cross_encoder_reranking_rejects_noninteger_candidate_budget():
    with pytest.raises(ValueError, match="candidate_k must be a positive integer"):
        retrieval.rerank_with_cross_encoder(
            query="test",
            ranked_results=[(doc(1, 0, 1, "candidate"), 0.1), (doc(2, 1, 2, "candidate"), 0.2)],
            candidate_k=1.5,
        )


@pytest.mark.unit
def test_cross_encoder_reranking_is_fail_open_on_nonfinite_scores(monkeypatch):
    documents = [
        doc(1, 0, 10, "first"),
        doc(2, 10, 20, "second"),
    ]
    ranked_results = [(documents[0], 0.2), (documents[1], 0.3)]

    class NonfiniteReranker:
        def predict(self, pairs, batch_size, show_progress_bar):
            return np.asarray([np.nan, 0.9], dtype=np.float32)

    monkeypatch.setattr(retrieval, "RAG_RERANK_ENABLED", True)
    monkeypatch.setattr(
        retrieval, "create_cross_encoder_reranker", lambda: NonfiniteReranker()
    )

    assert retrieval.rerank_with_cross_encoder(
        query="test question",
        ranked_results=ranked_results,
    ) == ranked_results


@pytest.mark.unit
def test_cross_encoder_reranking_is_fail_open(monkeypatch):
    documents = [
        doc(1, 0, 10, "first"),
        doc(2, 10, 20, "second"),
    ]
    ranked_results = [
        (documents[0], 0.2),
        (documents[1], 0.3),
    ]

    monkeypatch.setattr(retrieval, "RAG_RERANK_ENABLED", True)

    def fail_to_load():
        raise RuntimeError("reranker unavailable")

    monkeypatch.setattr(
        retrieval,
        "create_cross_encoder_reranker",
        fail_to_load,
    )

    result = retrieval.rerank_with_cross_encoder(
        query="test",
        ranked_results=ranked_results,
    )

    assert result == ranked_results


@pytest.mark.unit
def test_cross_encoder_reranking_disabled_skips_model(monkeypatch):
    documents = [
        doc(1, 0, 10, "first"),
        doc(2, 10, 20, "second"),
    ]
    ranked_results = [
        (documents[0], 0.2),
        (documents[1], 0.3),
    ]

    monkeypatch.setattr(retrieval, "RAG_RERANK_ENABLED", False)

    monkeypatch.setattr(
        retrieval,
        "create_cross_encoder_reranker",
        lambda: pytest.fail("reranker should not load"),
    )

    result = retrieval.rerank_with_cross_encoder(
        query="test",
        ranked_results=ranked_results,
    )

    assert result == ranked_results


@pytest.mark.unit
def test_retrieve_question_context_applies_cross_encoder_before_anchor_selection(
    monkeypatch,
):
    documents = [
        doc(1, 0, 10, "first"),
        doc(2, 10, 20, "second"),
    ]
    semantic_results = [
        (documents[0], 0.2),
        (documents[1], 0.3),
    ]

    monkeypatch.setattr(
        retrieval,
        "retrieve_mmr",
        lambda **kwargs: semantic_results,
    )
    monkeypatch.setattr(
        retrieval,
        "lexical_search",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        retrieval,
        "RAG_RERANK_ENABLED",
        True,
    )
    monkeypatch.setattr(
        retrieval,
        "rerank_with_cross_encoder",
        lambda **kwargs: [
            semantic_results[1],
            semantic_results[0],
        ],
    )

    result = retrieval.retrieve_question_context(
        FakeVectorStore(documents),
        "Who is relevant?",
        k=2,
        fetch_k=2,
        expand_context=False,
    )

    assert [item[0].metadata["chunk_id"] for item in result] == [2, 1]



@pytest.mark.unit
def test_temporal_query_parser_reads_explicit_time_range():
    window = retrieval.parse_temporal_query_window(
        "What is said from 25:35–26:25?",
        video_duration_seconds=2000,
    )

    assert window == {
        "mode": "explicit_range",
        "start_seconds": 1535.0,
        "end_seconds": 1585.0,
    }


@pytest.mark.unit
def test_temporal_query_parser_resolves_tail_and_video_end():
    tail = retrieval.parse_temporal_query_window(
        "What happens during the final 50 seconds?",
        video_duration_seconds=210,
    )
    ending = retrieval.parse_temporal_query_window(
        "What happens at the end of the video?",
        video_duration_seconds=210,
    )

    assert tail == {
        "mode": "relative_tail",
        "start_seconds": 160.0,
        "end_seconds": 210.0,
    }
    assert ending == {
        "mode": "video_end",
        "start_seconds": 120.0,
        "end_seconds": 210.0,
    }


@pytest.mark.unit
def test_temporal_route_is_not_used_for_general_questions():
    assert not retrieval.is_temporal_question("What is this video about?")
    assert retrieval.retrieve_temporal_context(
        object(),
        "What is this video about?",
    ) is None


@pytest.mark.unit
def test_temporal_retrieval_only_returns_chunks_overlapping_requested_window():
    documents = {
        "before": doc(0, 0, 60, "opening"),
        "first_overlap": doc(1, 60, 120, "first target segment"),
        "second_overlap": doc(2, 120, 180, "second target segment"),
        "after": doc(3, 180, 240, "closing"),
    }
    bundle = retrieval.retrieve_temporal_context(
        FakeVectorStore(documents),
        "What is said between 01:10 and 02:10?",
    )

    assert bundle is not None
    assert bundle["window"] == {
        "mode": "explicit_range",
        "start_seconds": 70.0,
        "end_seconds": 130.0,
    }
    assert [document.metadata["chunk_id"] for document, _ in bundle["results"]] == [1, 2]
    assert bundle["coverage_sufficient"] is True
    assert bundle["coverage_ratio"] == 1.0


@pytest.mark.unit
def test_temporal_retrieval_fails_closed_when_window_coverage_is_too_sparse():
    documents = {
        "old": doc(30, 1500, 1535.69, "earlier transcript"),
        "empty_gap": doc(31, 1535.69, 1574.72, ""),
        "ending": doc(32, 1574.72, 1584.52, "ending transcript"),
    }
    bundle = retrieval.retrieve_temporal_context(
        FakeVectorStore(documents),
        "What happens during the final 50 seconds, approximately 25:35–26:25?",
    )

    assert bundle is not None
    assert bundle["window"]["mode"] == "explicit_range"
    assert bundle["coverage_sufficient"] is False
    assert bundle["coverage_ratio"] < retrieval.TEMPORAL_MIN_WINDOW_COVERAGE
    assert bundle["results"] == []
    assert {
        document.metadata["chunk_id"]
        for document, _ in bundle["candidates"]
    } == {32}


@pytest.mark.unit
def test_temporal_end_query_selects_chronological_tail_chunks():
    documents = {
        "opening": doc(0, 0, 60, "opening"),
        "middle": doc(1, 60, 120, "middle"),
        "ending": doc(2, 120, 180, "ending"),
    }
    bundle = retrieval.retrieve_temporal_context(
        FakeVectorStore(documents),
        "What is said at the end of the video?",
    )

    assert bundle is not None
    assert bundle["window"]["mode"] == "video_end"
    assert [document.metadata["chunk_id"] for document, _ in bundle["results"]] == [1, 2]
    assert bundle["coverage_ratio"] == 1.0



@pytest.mark.unit
def test_temporal_parser_supports_hhmmss_and_point_timestamps():
    clock_range = retrieval.parse_temporal_query_window(
        "What happened from 1:02:30 to 1:03:40?",
        video_duration_seconds=5000,
    )
    point = retrieval.parse_temporal_query_window(
        "What happened at 10:00?",
        video_duration_seconds=1200,
    )

    assert clock_range == {
        "mode": "explicit_range",
        "start_seconds": 3750.0,
        "end_seconds": 3820.0,
    }
    assert point == {
        "mode": "point_timestamp",
        "start_seconds": 570.0,
        "end_seconds": 630.0,
    }


@pytest.mark.unit
def test_temporal_retrieval_returns_no_evidence_for_a_window_past_video_end():
    documents = {
        "opening": doc(0, 0, 60, "opening"),
        "middle": doc(1, 60, 120, "middle"),
        "ending": doc(2, 120, 180, "ending"),
    }
    bundle = retrieval.retrieve_temporal_context(
        FakeVectorStore(documents),
        "What happened at 10:00?",
    )

    assert bundle is not None
    assert bundle["results"] == []
    assert bundle["candidates"] == []
    assert bundle["coverage_ratio"] == 0.0
    assert bundle["coverage_sufficient"] is False



@pytest.mark.unit
def test_temporal_retrieval_prefers_focused_chunk_for_narrow_window():
    # Reproduces the observed index: chunk 31's coarse timestamp envelope spans
    # the full video ending, while chunk 32 contains the focused final segment.
    documents = {
        "before_window": doc(30, 1471.92, 1535.69, "earlier transcript"),
        "overlapping_window": doc(31, 1525.08, 1584.52, "broad chunk including the ending"),
        "end_window": doc(32, 1574.72, 1584.52, "focused final transcript"),
    }

    wide = retrieval.retrieve_temporal_context(
        FakeVectorStore(documents),
        "What is said between 25:35 and 26:25?",
    )
    narrow = retrieval.retrieve_temporal_context(
        FakeVectorStore(documents),
        "What is said between 26:14 and 26:24?",
    )

    assert wide is not None
    assert [document.metadata["chunk_id"] for document, _ in wide["results"]] == [31, 32]
    assert wide["coverage_ratio"] == 1.0

    assert narrow is not None
    assert [document.metadata["chunk_id"] for document, _ in narrow["results"]] == [32]
    assert narrow["candidate_chunk_ids"] == [32]
    assert narrow["coverage_ratio"] == pytest.approx(0.928, abs=0.001)
    assert narrow["coverage_sufficient"] is True


@pytest.mark.unit
def test_temporal_coverage_is_measured_after_context_budget():
    documents = {
        "first": doc(0, 0, 30, "first section"),
        "middle": doc(1, 30, 60, "middle section"),
        "last": doc(2, 60, 90, "last section"),
    }

    bundle = retrieval.retrieve_temporal_context(
        FakeVectorStore(documents),
        "What happened between 00:00 and 01:30?",
        max_chunks=1,
    )

    assert bundle is not None
    assert bundle["coverage_sufficient"] is False
    assert bundle["coverage_ratio"] == pytest.approx(1 / 3, abs=0.01)
    assert bundle["results"] == []
    assert [document.metadata["chunk_id"] for document, _ in bundle["candidates"]] == [1]
