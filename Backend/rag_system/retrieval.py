"""
retrieval.py

Retrieval layer for the YouTube RAG system.

Responsibilities
----------------
1. Extract YouTube video IDs.
2. Create the embedding model.
3. Load cached FAISS vector stores.
4. Retrieve candidates from FAISS.
5. Apply distance filtering.
6. Select final chunks using MMR and bounded soft facet reranking.

This file does NOT:
- fetch YouTube transcripts
- create vector stores
- build prompts
- call the generation model
- handle FastAPI
"""
import math
import re
from typing import Any
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


from .config import (
    CONTEXT_EXPANSION_CHUNKS,
    CONTEXT_MAX_CHUNKS,
    DENSE_ANCHOR_LIMIT,
    DENSE_ANCHOR_MIN_CHUNK_GAP,
    DENSE_CONTEXT_MAX_CHUNKS,
    DENSE_FACET_RERANK_WEIGHT,
    DENSE_SEMANTIC_FETCH_K,
    DENSE_SEMANTIC_K,
    EMBEDDING_MODEL,
    MAX_DISTANCE,
    MMR_FETCH_K,
    MMR_LAMBDA,
    RRF_K,
    TOP_K,
    VECTOR_STORE_ROOT,
)


# ============================================================
# YOUTUBE VIDEO ID VALIDATION
# ============================================================

YOUTUBE_VIDEO_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9_-]{11}$"
)


def validate_video_id(
    video_id: str,
) -> str:
    """
    Validate and return a canonical YouTube video ID.

    YouTube video IDs are expected to contain exactly
    11 characters using letters, digits, underscore, or hyphen.
    """

    video_id = video_id.strip()

    if not YOUTUBE_VIDEO_ID_PATTERN.fullmatch(video_id):

        raise ValueError(
            "Invalid YouTube video ID."
        )

    return video_id



# ============================================================
# 1. EXTRACT VIDEO ID
# ============================================================

def extract_video_id(
    video_reference: str,
) -> str:
    """
    Accept either a YouTube video ID or common YouTube URL
    formats and return the video ID.
    """

    if not video_reference:

        raise ValueError(
            "Video reference cannot be empty."
        )

    video_reference = video_reference.strip()

    if not video_reference:

        raise ValueError(
            "Video reference cannot be empty."
        )


    # --------------------------------------------------------
    # Raw video ID
    # --------------------------------------------------------

    if "://" not in video_reference:

        return validate_video_id(
            video_reference
        )


    # --------------------------------------------------------
    # Parse URL
    # --------------------------------------------------------

    parsed_url = urlparse(
        video_reference
    )

    hostname = (
        parsed_url.hostname or ""
    ).lower()


    # --------------------------------------------------------
    # youtube.com/watch?v=...
    # --------------------------------------------------------

    if hostname in {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
    }:

        query_parameters = parse_qs(
            parsed_url.query
        )

        video_ids = query_parameters.get(
            "v"
        )

        if video_ids:

            return validate_video_id(
                video_ids[0]
            )


    # --------------------------------------------------------
    # youtube.com/shorts/...
    # youtube.com/embed/...
    # youtube.com/live/...
    # --------------------------------------------------------

    if hostname in {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
    }:

        path_parts = [
            part
            for part in parsed_url.path.split("/")
            if part
        ]

        if len(path_parts) >= 2:

            if path_parts[0] in {
                "shorts",
                "embed",
                "live",
            }:

                return validate_video_id(
                    path_parts[1]
                )


    # --------------------------------------------------------
    # youtu.be/...
    # --------------------------------------------------------

    if hostname in {
        "youtu.be",
        "www.youtu.be",
    }:

        path_parts = [
            part
            for part in parsed_url.path.split("/")
            if part
        ]

        if path_parts:

            return validate_video_id(
                path_parts[0]
            )


    raise ValueError(
        "Could not extract a YouTube video ID from:\n"
        f"{video_reference}"
    )


# ============================================================
# 2. CREATE EMBEDDING MODEL
# ============================================================

@lru_cache(maxsize=1)
def create_embedding_model():

    return HuggingFaceEmbeddings(

        model_name=EMBEDDING_MODEL,

        model_kwargs={
            "device": "cpu",
        },

        encode_kwargs={
            "normalize_embeddings": True,
        },
    )


# ============================================================
# 3. LOAD VECTOR STORE
# ============================================================

@lru_cache(maxsize=8)
def load_vector_store(
    video_id: str,
):
    """
    Load one video's FAISS vector store.

    The result is cached so repeated questions for the same
    video do not repeatedly load the same FAISS index.
    """

    if not video_id:

        raise ValueError(
            "video_id cannot be empty."
        )

    vector_store_path = (
        VECTOR_STORE_ROOT / video_id
    )


    if not vector_store_path.exists():

        raise FileNotFoundError(
            "\nNo vector store found for video:\n"
            f"{video_id}\n\n"
            "Expected location:\n"
            f"{vector_store_path}\n\n"
            "Run the indexing pipeline first."
        )


    embeddings = create_embedding_model()


    return FAISS.load_local(

        str(vector_store_path),

        embeddings,

        # Required by the current local FAISS persistence
        # mechanism. Only load indexes created by this project
        # or another trusted source.
        allow_dangerous_deserialization=True,
    )


# ============================================================
# 4. RETRIEVE WITH MMR
# ============================================================

def retrieve_mmr(
    vector_store,
    query: str,
    k: int = TOP_K,
    fetch_k: int = MMR_FETCH_K,
    lambda_mult: float = MMR_LAMBDA,
    max_distance: float = MAX_DISTANCE,
    expand_context: bool = True,
):
    """
    Retrieve relevant transcript chunks using MMR.

    Returns:

        [
            (Document, distance),
            ...
        ]

    Lower FAISS distance means greater similarity.
    """

    # --------------------------------------------------------
    # Validate query
    # --------------------------------------------------------

    if not query or not query.strip():

        raise ValueError(
            "Query cannot be empty."
        )


    # --------------------------------------------------------
    # Validate MMR configuration
    # --------------------------------------------------------

    if k <= 0:

        raise ValueError(
            "k must be greater than 0."
        )


    if fetch_k <= 0:

        raise ValueError(
            "fetch_k must be greater than 0."
        )


    if k > fetch_k:

        raise ValueError(
            "k cannot be greater than fetch_k."
        )


    if not 0.0 <= lambda_mult <= 1.0:

        raise ValueError(
            "lambda_mult must be between 0.0 and 1.0."
        )


    if max_distance < 0:

        raise ValueError(
            "max_distance cannot be negative."
        )


    # --------------------------------------------------------
    # 1. Create query embedding
    # --------------------------------------------------------

    query_embedding = (
        vector_store
        .embedding_function
        .embed_query(
            query.strip()
        )
    )


    query_vector = np.asarray(
        query_embedding,
        dtype=np.float32,
    ).reshape(
        1,
        -1,
    )


    # --------------------------------------------------------
    # 2. Determine how many vectors exist
    # --------------------------------------------------------

    total_vectors = (
        vector_store.index.ntotal
    )


    if total_vectors == 0:

        return []


    candidate_k = min(
        fetch_k,
        total_vectors,
    )


    # --------------------------------------------------------
    # 3. Search FAISS
    # --------------------------------------------------------

    distances, indices = (
        vector_store.index.search(

            np.ascontiguousarray(
                query_vector
            ),

            candidate_k,
        )
    )


    candidate_indices = indices[0]

    candidate_distances = distances[0]


    # --------------------------------------------------------
    # 4. Apply distance threshold
    # --------------------------------------------------------

    eligible_positions = []


    for position, distance in enumerate(
        candidate_distances
    ):

        index_id = int(
            candidate_indices[position]
        )


        if index_id < 0:
            continue


        if float(distance) <= max_distance:

            eligible_positions.append(
                position
            )


    if not eligible_positions:

        return []


    # --------------------------------------------------------
    # 5. Reconstruct candidate embeddings
    # --------------------------------------------------------

    faiss_candidate_ids = [

        int(
            candidate_indices[position]
        )

        for position in eligible_positions
    ]


    if hasattr(
        vector_store.index,
        "reconstruct_batch",
    ):

        candidate_embeddings = (
            vector_store.index.reconstruct_batch(

                np.asarray(
                    faiss_candidate_ids,
                    dtype=np.int64,
                )
            )
        )

    else:

        candidate_embeddings = np.vstack(
            [
                vector_store.index.reconstruct(
                    vector_id
                )
                for vector_id in faiss_candidate_ids
            ]
        )


    candidate_embeddings = np.asarray(
        candidate_embeddings,
        dtype=np.float32,
    )


    # --------------------------------------------------------
    # 6. Normalize query and candidates
    # --------------------------------------------------------

    query_norm = np.linalg.norm(
        query_vector,
        axis=1,
        keepdims=True,
    )

    query_norm = np.maximum(
        query_norm,
        1e-12,
    )


    query_normalized = (
        query_vector / query_norm
    )


    candidate_norms = np.linalg.norm(
        candidate_embeddings,
        axis=1,
        keepdims=True,
    )

    candidate_norms = np.maximum(
        candidate_norms,
        1e-12,
    )


    candidate_normalized = (
        candidate_embeddings / candidate_norms
    )


    # --------------------------------------------------------
    # 7. Query-to-document similarity
    # --------------------------------------------------------

    query_similarities = (

        candidate_normalized
        @ query_normalized.T

    ).reshape(-1)


    # --------------------------------------------------------
    # 8. MMR selection
    # --------------------------------------------------------

    number_to_select = min(
        k,
        len(eligible_positions),
    )


    selected_positions = []

    remaining_positions = list(
        range(
            len(eligible_positions)
        )
    )


    # First document:
    # choose the most relevant candidate.

    first_position = int(
        np.argmax(
            query_similarities
        )
    )


    selected_positions.append(
        first_position
    )

    remaining_positions.remove(
        first_position
    )


    # Remaining documents:
    # maximize relevance while reducing redundancy.

    while (

        remaining_positions

        and len(selected_positions)
        < number_to_select

    ):

        best_position = None
        best_score = -np.inf


        selected_embeddings = (
            candidate_normalized[
                selected_positions
            ]
        )


        for candidate_position in (
            remaining_positions
        ):

            relevance = float(
                query_similarities[
                    candidate_position
                ]
            )


            candidate_embedding = (
                candidate_normalized[
                    candidate_position
                ]
            )


            redundancy = float(

                np.max(

                    selected_embeddings
                    @ candidate_embedding

                )
            )


            mmr_score = (

                lambda_mult * relevance

                -

                (
                    1.0 - lambda_mult
                )
                * redundancy

            )


            if mmr_score > best_score:

                best_score = mmr_score

                best_position = (
                    candidate_position
                )


        if best_position is None:

            break


        selected_positions.append(
            best_position
        )

        remaining_positions.remove(
            best_position
        )


    # --------------------------------------------------------
    # 9. Convert FAISS IDs back into Documents
    # --------------------------------------------------------

    results = []


    for selected_position in (
        selected_positions
    ):

        original_position = (
            eligible_positions[
                selected_position
            ]
        )


        vector_id = int(

            candidate_indices[
                original_position
            ]
        )


        distance = float(

            candidate_distances[
                original_position
            ]
        )


        docstore_id = (
            vector_store
            .index_to_docstore_id
            .get(vector_id)
        )


        if docstore_id is None:
            continue


        document = (
            vector_store
            .docstore
            .search(
                docstore_id
            )
        )


        if document is None:
            continue


        results.append(
            (
                document,
                distance,
            )
        )


    return results


def retrieve_faiss_candidates_for_diagnostics(
    vector_store,
    query: str,
    *,
    fetch_k: int,
    max_distance: float,
):
    """Return raw FAISS candidates before MMR for diagnostic inspection only.

    Each tuple is (document, distance, faiss_rank). The production retriever
    performs the same FAISS search internally before MMR; this helper exposes
    that otherwise-hidden candidate stage without changing production output.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")
    if fetch_k <= 0:
        raise ValueError("fetch_k must be greater than 0.")
    if max_distance < 0:
        raise ValueError("max_distance cannot be negative.")

    query_embedding = vector_store.embedding_function.embed_query(
        query.strip()
    )
    query_vector = np.asarray(
        query_embedding,
        dtype=np.float32,
    ).reshape(1, -1)

    total_vectors = vector_store.index.ntotal
    if total_vectors == 0:
        return []

    candidate_k = min(fetch_k, total_vectors)
    distances, indices = vector_store.index.search(
        np.ascontiguousarray(query_vector),
        candidate_k,
    )

    results = []
    for position, distance in enumerate(distances[0]):
        vector_id = int(indices[0][position])
        faiss_rank = position + 1

        if vector_id < 0:
            continue

        docstore_id = vector_store.index_to_docstore_id.get(vector_id)
        if docstore_id is None:
            continue

        document = vector_store.docstore.search(docstore_id)
        if document is None:
            continue

        results.append((document, float(distance), faiss_rank))

    return results


def diagnose_retrieval_pipeline(
    vector_store,
    query: str,
    *,
    k: int = TOP_K,
    fetch_k: int = MMR_FETCH_K,
    lambda_mult: float = MMR_LAMBDA,
    max_distance: float = MAX_DISTANCE,
):
    """Capture every retrieval stage for end-to-end debugging.

    This function is diagnostic-only. It does not alter the production
    retrieval path. It exposes raw FAISS candidates, per-variant MMR output,
    semantic RRF, lexical retrieval, hybrid RRF, anchor selection, and final
    context expansion so a missing gold chunk can be localized precisely.
    """

    if is_overview_question(query):
        raise ValueError(
            "End-to-end retrieval diagnostics are not defined for overview questions."
        )

    dense_question = is_evidence_dense_question(query)
    semantic_k = (
        DENSE_SEMANTIC_K
        if dense_question
        else k
    )
    semantic_fetch_k = (
        DENSE_SEMANTIC_FETCH_K
        if dense_question
        else max(fetch_k, semantic_k * 2)
    )
    lexical_limit = (
        max(k * 2, 8)
        if dense_question
        else max(k, min(k * 2, 8))
    )

    query_plan = build_retrieval_query_plan(query)
    semantic_stages = []
    semantic_rankings = []
    semantic_rankings_by_label = {}

    for label, query_variant in query_plan:
        raw_candidates = retrieve_faiss_candidates_for_diagnostics(
            vector_store,
            query_variant,
            fetch_k=semantic_fetch_k,
            max_distance=max_distance,
        )
        mmr_results = retrieve_mmr(
            vector_store=vector_store,
            query=query_variant,
            k=semantic_k,
            fetch_k=semantic_fetch_k,
            lambda_mult=lambda_mult,
            max_distance=max_distance,
        )
        semantic_stages.append({
            "label": label,
            "query": query_variant,
            "raw_candidates": raw_candidates,
            "mmr_results": mmr_results,
        })
        semantic_rankings.append(mmr_results)
        if label in EVIDENCE_FACET_QUERY_DEFINITIONS:
            semantic_rankings_by_label[label] = mmr_results

    semantic_results = fuse_semantic_rankings(semantic_rankings)

    lexical_queries = [("original", query)]
    if dense_question:
        lexical_queries.extend(
            build_evidence_facet_plan(query)
        )

    lexical_stages = []
    lexical_rankings = []
    lexical_rankings_by_label = {}

    for label, lexical_query in lexical_queries:
        lexical_results_for_query = lexical_search(
            vector_store=vector_store,
            query=lexical_query,
            limit=lexical_limit,
        )
        lexical_stages.append({
            "label": label,
            "query": lexical_query,
            "results": lexical_results_for_query,
        })
        lexical_rankings.append(lexical_results_for_query)
        if label in EVIDENCE_FACET_QUERY_DEFINITIONS:
            lexical_rankings_by_label[label] = lexical_results_for_query

    lexical_results = fuse_lexical_rankings(lexical_rankings)

    combined_results = fuse_semantic_and_lexical_results(
        semantic_results=semantic_results,
        lexical_results=lexical_results,
        lexical_distance=max_distance,
    )

    facet_rankings = {}
    for facet_name in EVIDENCE_FACET_QUERY_DEFINITIONS:
        semantic_facet = semantic_rankings_by_label.get(facet_name, [])
        lexical_facet = lexical_rankings_by_label.get(facet_name, [])

        if semantic_facet or lexical_facet:
            facet_rankings[facet_name] = (
                fuse_semantic_and_lexical_results(
                    semantic_results=semantic_facet,
                    lexical_results=lexical_facet,
                    lexical_distance=max_distance,
                )
            )

    context_max_chunks = (
        DENSE_CONTEXT_MAX_CHUNKS
        if dense_question
        else CONTEXT_MAX_CHUNKS
    )
    anchor_limit = (
        min(DENSE_ANCHOR_LIMIT, context_max_chunks)
        if dense_question
        else min(k + 2, context_max_chunks)
    )

    reranked_results = combined_results

    if lexical_results:
        reranked_results = (
            rerank_with_soft_facet_support(
                combined_results,
                facet_rankings,
            )
            if dense_question and facet_rankings
            else combined_results
        )
        anchor_results = (
            select_diverse_retrieval_anchors(
                reranked_results,
                limit=anchor_limit,
            )
            if dense_question
            else reranked_results[:anchor_limit]
        )
    else:
        semantic_distances = [
            distance
            for _document, distance in semantic_results
        ]
        strict_semantic_limit = max_distance * 0.92
        if not semantic_distances or min(semantic_distances) > strict_semantic_limit:
            anchor_results = []
        else:
            reranked_results = (
                rerank_with_soft_facet_support(
                    combined_results,
                    facet_rankings,
                )
                if dense_question and facet_rankings
                else combined_results
            )
            anchor_results = (
                select_diverse_retrieval_anchors(
                    reranked_results,
                    limit=anchor_limit,
                )
                if dense_question
                else reranked_results[:anchor_limit]
            )

    context_results = expand_retrieval_context(
        vector_store,
        anchor_results,
        max_chunks=context_max_chunks,
    )

    return {
        "query": query,
        "dense_question": dense_question,
        "semantic_k": semantic_k,
        "semantic_fetch_k": semantic_fetch_k,
        "lexical_limit": lexical_limit,
        "anchor_limit": anchor_limit,
        "query_plan": query_plan,
        "query_variants": [
            query_text
            for _label, query_text in query_plan
        ],
        "semantic_stages": semantic_stages,
        "semantic_fused": semantic_results,
        "lexical_queries": [
            {
                "label": label,
                "query": lexical_query,
            }
            for label, lexical_query in lexical_queries
        ],
        "lexical_stages": lexical_stages,
        "lexical": lexical_results,
        "facet_rankings": facet_rankings,
        "facet_soft_reranked": reranked_results,
        "hybrid_fused": combined_results,
        "anchors": anchor_results,
        "final_context": context_results,
    }

# ============================================================
# 9. QUESTION-AWARE RETRIEVAL
# ============================================================

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "about",
    "as",
    "at",
    "be",
    "by",
    "can",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "who",
    "why",
    "with",
    "you",
}



# Question boilerplate is useful for human language but is weak retrieval
# evidence. These terms are removed only from the focus-query path; the
# original user query is always retained as a semantic retrieval variant.
QUERY_FOCUS_STOPWORDS = STOPWORDS | {
    "he", "she", "his", "her", "their", "them", "me", "my", "your", "our",
    "we", "i", "say", "says", "said", "tell", "tells", "told",
    "discuss", "discusses", "discussed", "describe", "describes",
    "described", "explain", "explains", "explained", "talk", "talks",
    "talking", "mention", "mentions", "mentioned", "give", "gives",
    "given", "provide", "provides", "provided", "does", "did",
    "video", "interview",
}

SPEAKER_ATTRIBUTION_PATTERN = re.compile(
    r"\b(?:does|did|is|are|was|were|can|could|would|will|has|have|had)"
    r"\s+[a-z]+(?:\s+[a-z]+){0,2}\s+"
    r"(?:say|says|said|tell|tells|told|discuss|discusses|discussed|"
    r"describe|describes|described|explain|explains|explained|"
    r"talk|talks|mention|mentions|mentioned)\b"
)


def extract_query_focus_terms(
    query: str,
) -> list[str]:
    """Extract content-bearing terms after removing speaker/question boilerplate."""

    normalized_query = " ".join(query.strip().lower().split())
    normalized_query = SPEAKER_ATTRIBUTION_PATTERN.sub(
        " ",
        normalized_query,
        count=1,
    )
    normalized_query = re.sub(
        r"^\s*(?:what|how|why|where|when|who|which)\b",
        " ",
        normalized_query,
        count=1,
    )

    words = re.findall(r"[a-zA-Z0-9]+", normalized_query)
    return [
        word
        for word in words
        if len(word) >= 2 and word not in QUERY_FOCUS_STOPWORDS
    ]


EVIDENCE_FACET_QUERY_DEFINITIONS = {
    "financial_economic": (
        "financial crisis economic circumstances cash flow money debt "
        "losses costs"
    ),
    "policy_governance": (
        "government policy regulation banks support approval restrictions"
    ),
    "operational_challenges": (
        "operational challenges payments fees fuel suppliers service costs"
    ),
    "brand_marketing": (
        "brand branding advertising marketing personality identity"
    ),
    "business_corporate": (
        "business companies subsidiaries ownership investments"
    ),
}


def build_evidence_facet_plan(
    query: str,
) -> list[tuple[str, str]]:
    """Build ordered, intent-specific evidence facets for a dense question."""

    original = " ".join(query.strip().split())
    if not original:
        return []

    focus_terms = extract_query_focus_terms(original)
    if len(focus_terms) < 2:
        return []

    focus_query = " ".join(focus_terms)
    normalized = original.casefold()

    if re.search(
        r"\b(?:brand|branding|advertis|marketing|personality)\b",
        normalized,
    ):
        facet_names = (
            "brand_marketing",
            "business_corporate",
            "operational_challenges",
        )
    elif re.search(
        r"\b(?:policy|government|regulation|banks?|support|approval|"
        r"restrictions?)\b",
        normalized,
    ):
        facet_names = (
            "policy_governance",
            "operational_challenges",
            "financial_economic",
        )
    elif re.search(
        r"\b(?:challenge|challenges|problems?|problem|operations?|"
        r"operational|running)\b",
        normalized,
    ):
        facet_names = (
            "operational_challenges",
            "financial_economic",
            "policy_governance",
        )
    elif re.search(
        r"\b(?:why|reasons?|cause|causes?|failure|failed|struggled|"
        r"impact)\b",
        normalized,
    ):
        facet_names = (
            "financial_economic",
            "policy_governance",
            "operational_challenges",
        )
    elif re.search(
        r"\b(?:businesses|companies|company|corporate|subsidiaries|"
        r"ownership|investments?)\b",
        normalized,
    ):
        facet_names = (
            "business_corporate",
            "brand_marketing",
            "financial_economic",
        )
    else:
        facet_names = (
            "financial_economic",
            "policy_governance",
            "operational_challenges",
        )

    return [
        (
            facet_name,
            f"{focus_query} "
            f"{EVIDENCE_FACET_QUERY_DEFINITIONS[facet_name]}",
        )
        for facet_name in facet_names
    ]


def build_evidence_facet_queries(
    query: str,
) -> list[str]:
    """Return planned facet query strings."""

    return [
        facet_query
        for _facet_name, facet_query in build_evidence_facet_plan(query)
    ]


def build_retrieval_query_plan(
    query: str,
) -> list[tuple[str, str]]:
    """Build labeled query variants for retrieval and diagnostics."""

    original = " ".join(query.strip().split())
    if not original:
        return []

    plan = [("original", original)]
    focus_terms = extract_query_focus_terms(original)

    if len(focus_terms) < 2:
        return plan

    focus_query = " ".join(focus_terms)

    if focus_query.casefold() != original.casefold():
        plan.append(("focus", focus_query))

    if is_evidence_dense_question(original):
        plan.extend(build_evidence_facet_plan(original))

    return plan


def build_retrieval_query_variants(
    query: str,
) -> list[str]:
    """Build deterministic query strings while preserving the public API."""

    return [
        query_text
        for _label, query_text in build_retrieval_query_plan(query)
    ]


OVERVIEW_PATTERNS = (
    r"\bwhat is this video about\b",
    r"\bwhat's this video about\b",
    r"\bwhat are they talking about\b",
    r"\bwhat are they discussing\b",
    r"\bwhat is being discussed\b",
    r"\bwhat is discussed in this video\b",
    r"\bgive me an overview\b",
    r"\boverview of this video\b",
    r"\bsummarize this video\b",
    r"\bsummarise this video\b",
    r"\bsummarize the video\b",
    r"\bsummarise the video\b",
    r"\bwhat are the main topics\b",
    r"\bwhat topics are covered\b",
)


def is_overview_question(
    query: str,
) -> bool:
    """
    Detect questions that ask for the overall topic
    or summary of the video.
    """

    normalized_query = (
        query.strip()
        .lower()
    )

    return any(
        re.search(
            pattern,
            normalized_query,
        )
        for pattern in OVERVIEW_PATTERNS
    )


def extract_query_terms(
    query: str,
) -> set[str]:
    """
    Extract meaningful terms from the user's question.

    Stopwords are removed so that entity/name questions such as:

        "Who is Demis Hassabis?"

    become approximately:

        {"demis", "hassabis"}
    """

    words = re.findall(
        r"[a-zA-Z0-9]+",
        query.lower(),
    )

    return {
        word
        for word in words
        if (
            len(word) >= 2
            and word not in STOPWORDS
        )
    }


def get_all_documents(
    vector_store,
) -> list:
    """
    Read the documents stored inside the FAISS docstore.

    This is used only for lightweight lexical fallback
    retrieval. It does not rebuild the vector database.
    """

    documents = []

    seen_docstore_ids = set()

    for docstore_id in (
        vector_store
        .index_to_docstore_id
        .values()
    ):

        if docstore_id in seen_docstore_ids:
            continue

        seen_docstore_ids.add(
            docstore_id
        )

        document = (
            vector_store
            .docstore
            .search(docstore_id)
        )

        if document is not None:

            documents.append(
                document
            )

    return documents


def lexical_search(
    vector_store,
    query: str,
    limit: int = 8,
):
    """Search transcript chunks with content-aware lexical evidence.

    The lexical path uses query-focus terms plus lightweight inverse-document
    frequency weighting. This prevents generic question words from outranking
    rare domain terms and allows a single distinctive term to surface a
    relevant chunk even when the exact wording differs.
    """

    focus_terms = extract_query_focus_terms(query)
    if not focus_terms:
        focus_terms = list(extract_query_terms(query))

    if not focus_terms:
        return []

    documents = get_all_documents(vector_store)
    if not documents:
        return []

    normalized_focus = " ".join(focus_terms)
    unique_terms = list(dict.fromkeys(focus_terms))
    document_frequency = {term: 0 for term in unique_terms}

    for document in documents:
        text = document.page_content.casefold()
        for term in unique_terms:
            if re.search(rf"\b{re.escape(term)}\b", text):
                document_frequency[term] += 1

    total_documents = len(documents)
    term_weights = {
        term: (
            math.log((total_documents + 1) / (frequency + 1)) + 1.0
        )
        for term, frequency in document_frequency.items()
    }

    total_query_weight = sum(term_weights.values()) or 1.0
    scored_documents = []

    for document in documents:
        text = document.page_content.casefold()
        matched_terms = [
            term
            for term in unique_terms
            if re.search(rf"\b{re.escape(term)}\b", text)
        ]

        if not matched_terms:
            continue

        weighted_coverage = (
            sum(term_weights[term] for term in matched_terms)
            / total_query_weight
        )

        # Multi-term lexical matches require shared evidence in the same
        # chunk. This prevents one generic/query-side term from becoming
        # enough evidence on its own. Single-term queries remain supported.
        if len(unique_terms) > 1 and len(matched_terms) < 2:
            continue

        raw_coverage = len(matched_terms) / len(unique_terms)

        phrase_bonus = 0.0
        for left, right in zip(focus_terms, focus_terms[1:]):
            if f"{left} {right}" in text:
                phrase_bonus += 0.12

        if normalized_focus in text and len(unique_terms) >= 2:
            phrase_bonus += 0.5

        score = (
            weighted_coverage
            + (0.20 * raw_coverage)
            + phrase_bonus
        )

        scored_documents.append((document, float(score)))

    scored_documents.sort(
        key=lambda item: (
            -item[1],
            float(item[0].metadata.get("start", 0.0)),
            (
                item[0].metadata.get("chunk_id")
                if item[0].metadata.get("chunk_id") is not None
                else 10**9
            ),
        )
    )

    return scored_documents[:limit]


def fuse_ranked_results(
    ranked_results: list[list[tuple[Any, float]]],
    *,
    rrf_k: int = RRF_K,
):
    """Fuse independently ranked result lists with Reciprocal Rank Fusion."""

    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than 0.")

    fused = {}

    def identity(document):
        chunk_id = document.metadata.get("chunk_id")
        if chunk_id is not None:
            return ("chunk", chunk_id)

        return (
            "time",
            float(document.metadata.get("start", 0.0)),
            float(document.metadata.get("end", 0.0)),
        )

    for results in ranked_results:
        for rank, (document, score) in enumerate(results, start=1):
            key = identity(document)
            entry = fused.get(key)

            if entry is None:
                entry = {
                    "document": document,
                    "score": float(score),
                    "rrf_score": 0.0,
                }
                fused[key] = entry
            else:
                entry["score"] = min(
                    entry["score"],
                    float(score),
                )

            entry["rrf_score"] += 1.0 / (rrf_k + rank)

    ranked = sorted(
        fused.values(),
        key=lambda entry: (
            -entry["rrf_score"],
            entry["score"],
            float(entry["document"].metadata.get("start", 0.0)),
            (
                entry["document"].metadata.get("chunk_id")
                if entry["document"].metadata.get("chunk_id") is not None
                else 10**9
            ),
        ),
    )

    return [
        (entry["document"], entry["score"])
        for entry in ranked
    ]


def fuse_semantic_rankings(
    ranked_results: list[list[tuple[Any, float]]],
    *,
    rrf_k: int = RRF_K,
):
    """Fuse semantic query variants with Reciprocal Rank Fusion."""

    return fuse_ranked_results(
        ranked_results,
        rrf_k=rrf_k,
    )


def fuse_lexical_rankings(
    ranked_results: list[list[tuple[Any, float]]],
    *,
    rrf_k: int = RRF_K,
):
    """Fuse lexical facet query results with Reciprocal Rank Fusion."""

    return fuse_ranked_results(
        ranked_results,
        rrf_k=rrf_k,
    )


def fuse_semantic_and_lexical_results(
    semantic_results,
    lexical_results,
    *,
    lexical_distance: float = MAX_DISTANCE,
    rrf_k: int = RRF_K,
):
    """
    Fuse semantic and lexical rankings with Reciprocal Rank Fusion.

    Semantic FAISS distances and lexical scores live on different scales,
    so directly sorting their raw values would be misleading. RRF uses only
    rank position and therefore combines the two retrieval signals without
    pretending their scores are comparable.

    The returned tuple keeps the original FAISS distance when a document was
    retrieved semantically. Lexical-only documents receive MAX_DISTANCE so
    downstream semantic-distance contracts remain conservative.
    """

    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than 0.")

    fused = {}

    def identity(document):
        chunk_id = document.metadata.get("chunk_id")
        if chunk_id is not None:
            return ("chunk", chunk_id)

        return (
            "time",
            float(document.metadata.get("start", 0.0)),
            float(document.metadata.get("end", 0.0)),
        )

    def add_ranked_result(document, distance, rank):
        key = identity(document)
        entry = fused.get(key)

        if entry is None:
            entry = {
                "document": document,
                "distance": float(distance),
                "score": 0.0,
            }
            fused[key] = entry

        else:
            # Prefer the real FAISS distance when the same chunk is present
            # in both branches.
            entry["distance"] = min(
                entry["distance"],
                float(distance),
            )

        entry["score"] += 1.0 / (rrf_k + rank)

    for rank, (document, distance) in enumerate(
        semantic_results,
        start=1,
    ):
        add_ranked_result(
            document,
            distance,
            rank,
        )

    for rank, (document, _lexical_score) in enumerate(
        lexical_results,
        start=1,
    ):
        add_ranked_result(
            document,
            lexical_distance,
            rank,
        )

    ranked = sorted(
        fused.values(),
        key=lambda entry: (
            -entry["score"],
            entry["distance"],
            float(entry["document"].metadata.get("start", 0.0)),
            (
                entry["document"].metadata.get("chunk_id")
                if entry["document"].metadata.get("chunk_id") is not None
                else 10**9
            ),
        ),
    )

    return [
        (
            entry["document"],
            entry["distance"],
        )
        for entry in ranked
    ]


def retrieve_overview(
    vector_store,
    number_of_chunks: int = 6,
):
    """
    Retrieve representative transcript sections across
    the entire video.

    This is used for broad questions such as:

        "What are they talking about?"
        "What is this video about?"

    It deliberately samples the video chronologically
    instead of relying on semantic similarity to a vague
    question.
    """

    documents = (
        get_all_documents(
            vector_store
        )
    )

    if not documents:

        return []

    documents.sort(
        key=lambda document: float(
            document.metadata.get(
                "start",
                0.0,
            )
        )
    )

    if len(documents) <= number_of_chunks:

        selected_documents = (
            documents
        )

    else:

        selected_indices = []

        for i in range(
            number_of_chunks
        ):

            position = round(
                i
                *
                (
                    (
                        len(documents)
                        - 1
                    )
                    /
                    (
                        number_of_chunks
                        - 1
                    )
                )
            )

            selected_indices.append(
                position
            )

        selected_documents = [
            documents[index]
            for index in selected_indices
        ]

    # Overview retrieval does not use a meaningful
    # semantic distance, so we use MAX_DISTANCE as
    # a neutral API value.
    return [
        (
            document,
            float(MAX_DISTANCE),
        )
        for document in selected_documents
    ]


def expand_retrieval_context(
    vector_store,
    retrieved_results,
    window: int = CONTEXT_EXPANSION_CHUNKS,
    max_chunks: int = CONTEXT_MAX_CHUNKS,
):
    """
    Add bounded chronological neighbors around retrieved evidence.

    The original retrieved chunks remain anchors. Neighboring chunks are
    selected by their sequential chunk_id, then the final set is returned
    in transcript order. The hard max keeps prompt/context growth bounded.
    """

    if not retrieved_results:
        return []

    if window < 0:
        raise ValueError("window cannot be negative.")

    if max_chunks <= 0:
        raise ValueError("max_chunks must be greater than 0.")

    if window == 0:
        return retrieved_results[:max_chunks]

    documents = get_all_documents(vector_store)

    if not documents:
        return retrieved_results[:max_chunks]

    ordered_documents = sorted(
        documents,
        key=lambda document: (
            float(document.metadata.get("start", 0.0)),
            float(document.metadata.get("end", 0.0)),
            document.metadata.get("chunk_id", 0),
        ),
    )

    positions = {}
    for position, document in enumerate(ordered_documents):
        chunk_id = document.metadata.get("chunk_id")
        if chunk_id is not None:
            positions[chunk_id] = position

    selected = {}
    anchor_keys = set()
    anchor_positions = []

    def result_key(document):
        chunk_id = document.metadata.get("chunk_id")
        if chunk_id is not None:
            return ("chunk", chunk_id)

        return (
            "time",
            float(document.metadata.get("start", 0.0)),
            float(document.metadata.get("end", 0.0)),
        )

    def add_document(document, distance, is_anchor=False):
        key = result_key(document)

        if is_anchor:
            anchor_keys.add(key)

        if key not in selected:
            selected[key] = (
                document,
                float(distance),
            )

    for anchor_document, anchor_distance in retrieved_results:
        add_document(
            anchor_document,
            anchor_distance,
            is_anchor=True,
        )

        chunk_id = anchor_document.metadata.get("chunk_id")
        position = positions.get(chunk_id)

        if position is None:
            continue

        anchor_positions.append(position)

        start = max(0, position - window)
        end = min(len(ordered_documents), position + window + 1)

        for neighbor_position in range(start, end):
            neighbor = ordered_documents[neighbor_position]
            add_document(
                neighbor,
                anchor_distance,
                is_anchor=(neighbor_position == position),
            )

    if len(selected) > max_chunks:
        def priority(item):
            key, (document, _distance) = item
            position = positions.get(
                document.metadata.get("chunk_id"),
                10**9,
            )

            if anchor_positions:
                neighbor_distance = min(
                    abs(position - anchor_position)
                    for anchor_position in anchor_positions
                )
            else:
                neighbor_distance = 10**9

            return (
                0 if key in anchor_keys else 1,
                neighbor_distance,
                float(document.metadata.get("start", 0.0)),
            )

        selected_items = sorted(
            selected.items(),
            key=priority,
        )[:max_chunks]
    else:
        selected_items = list(selected.items())

    expanded_results = [
        item[1]
        for item in selected_items
    ]

    expanded_results.sort(
        key=lambda item: (
            float(item[0].metadata.get("start", 0.0)),
            float(item[0].metadata.get("end", 0.0)),
            item[0].metadata.get("chunk_id", 0),
        )
    )

    return expanded_results


# ============================================================
# QUESTION-AWARE RETRIEVAL
# ============================================================

def is_evidence_dense_question(query: str) -> bool:
    """Detect questions that are likely to require several evidence facets."""

    normalized = " ".join(query.strip().lower().split())
    focus_terms = extract_query_focus_terms(query)

    if re.search(
        r"\b(?:why|how|reasons?|challenges?|problems?|failure|causes?|"
        r"role of|impact|building .*\b(?:brand|business)\b|"
        r"policy|government|businesses|companies|multiple|several)\b",
        normalized,
    ):
        return True

    return bool(
        len(focus_terms) >= 6
        and re.search(r"\b(?:and|or|both)\b", normalized)
    )



def _retrieval_document_identity(document):
    chunk_id = document.metadata.get("chunk_id")
    if chunk_id is not None:
        return ("chunk", chunk_id)

    return (
        "time",
        float(document.metadata.get("start", 0.0)),
        float(document.metadata.get("end", 0.0)),
    )


def rerank_with_soft_facet_support(
    ranked_results,
    facet_rankings: dict[str, list[tuple[Any, float]]],
    *,
    rrf_k: int = RRF_K,
    facet_weight: float = DENSE_FACET_RERANK_WEIGHT,
):
    """Softly boost candidates supported strongly by evidence facets.

    The global fused ranking remains the primary signal. Facet support adds a
    bounded, rank-based bonus with diminishing returns across multiple facets.
    Unlike hard facet quotas, this can only reorder candidates; it cannot force
    a low-confidence candidate into the anchor set.
    """

    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than 0.")

    if facet_weight < 0:
        raise ValueError("facet_weight cannot be negative.")

    if not ranked_results or not facet_rankings or facet_weight == 0:
        return list(ranked_results)

    global_scores = {
        _retrieval_document_identity(item[0]): 1.0 / (rrf_k + rank)
        for rank, item in enumerate(ranked_results, start=1)
    }

    facet_scores: dict[tuple, list[float]] = {}

    for candidates in facet_rankings.values():
        for rank, (document, _distance) in enumerate(
            candidates,
            start=1,
        ):
            identity = _retrieval_document_identity(document)
            facet_scores.setdefault(identity, []).append(
                1.0 / (rrf_k + rank)
            )

    scored = []

    for global_rank, item in enumerate(
        ranked_results,
        start=1,
    ):
        identity = _retrieval_document_identity(item[0])
        base_score = global_scores[identity]

        support_scores = sorted(
            facet_scores.get(identity, []),
            reverse=True,
        )

        # Diminishing returns prevent a chunk shared by several facets from
        # overwhelming the global retrieval signal.
        facet_support = 0.0
        decay = 1.0

        for support_score in support_scores[:3]:
            facet_support += decay * support_score
            decay *= 0.5

        soft_score = (
            base_score
            + facet_weight * facet_support
        )

        scored.append(
            (
                soft_score,
                global_rank,
                item,
            )
        )

    scored.sort(
        key=lambda entry: (
            -entry[0],
            entry[1],
        )
    )

    return [
        item
        for _soft_score, _global_rank, item in scored
    ]


def select_diverse_retrieval_anchors(
    ranked_results,
    *,
    limit: int,
    min_chunk_gap: int = 3,
):
    """Select high-ranked anchors while spreading them across the transcript."""

    if limit <= 0 or not ranked_results:
        return []

    if min_chunk_gap < 0:
        raise ValueError("min_chunk_gap cannot be negative.")

    selected = []
    selected_chunk_ids = []
    remaining = list(ranked_results)

    def chunk_id(document):
        value = document.metadata.get("chunk_id")
        return int(value) if value is not None else None

    for item in ranked_results:
        if len(selected) >= limit:
            break

        document = item[0]
        current_id = chunk_id(document)

        if current_id is None:
            selected.append(item)
            continue

        if all(
            abs(current_id - previous_id) >= min_chunk_gap
            for previous_id in selected_chunk_ids
        ):
            selected.append(item)
            selected_chunk_ids.append(current_id)
            remaining.remove(item)

    for item in remaining:
        if len(selected) >= limit:
            break
        if item not in selected:
            selected.append(item)

    return selected[:limit]


def retrieve_question_context(
    vector_store,
    query: str,
    k: int = TOP_K,
    fetch_k: int = MMR_FETCH_K,
    lambda_mult: float = MMR_LAMBDA,
    max_distance: float = MAX_DISTANCE,
    expand_context: bool = True,
):
    """Main question-aware retrieval entry point.

    Dense explanation and multi-part questions receive a wider candidate
    budget so multiple supporting transcript regions can survive ranking.
    Final context size remains bounded by CONTEXT_MAX_CHUNKS.
    """

    if is_overview_question(query):
        return retrieve_overview(vector_store)

    dense_question = is_evidence_dense_question(query)

    semantic_k = (
        DENSE_SEMANTIC_K
        if dense_question
        else k
    )
    semantic_fetch_k = (
        DENSE_SEMANTIC_FETCH_K
        if dense_question
        else max(
            fetch_k,
            semantic_k * 2,
        )
    )
    lexical_limit = (
        max(k * 2, 8)
        if dense_question
        else max(k, min(k * 2, 8))
    )

    query_plan = build_retrieval_query_plan(query)
    semantic_rankings = []
    semantic_rankings_by_label = {}

    for label, query_variant in query_plan:
        ranking = retrieve_mmr(
            vector_store=vector_store,
            query=query_variant,
            k=semantic_k,
            fetch_k=semantic_fetch_k,
            lambda_mult=lambda_mult,
            max_distance=max_distance,
        )
        semantic_rankings.append(ranking)
        if label in EVIDENCE_FACET_QUERY_DEFINITIONS:
            semantic_rankings_by_label[label] = ranking

    semantic_results = fuse_semantic_rankings(semantic_rankings)

    lexical_queries = [("original", query)]
    if dense_question:
        lexical_queries.extend(
            build_evidence_facet_plan(query)
        )

    lexical_rankings = []
    lexical_rankings_by_label = {}

    for label, lexical_query in lexical_queries:
        ranking = lexical_search(
            vector_store=vector_store,
            query=lexical_query,
            limit=lexical_limit,
        )
        lexical_rankings.append(ranking)
        if label in EVIDENCE_FACET_QUERY_DEFINITIONS:
            lexical_rankings_by_label[label] = ranking

    lexical_results = fuse_lexical_rankings(
        lexical_rankings
    )

    combined_results = fuse_semantic_and_lexical_results(
        semantic_results=semantic_results,
        lexical_results=lexical_results,
        lexical_distance=max_distance,
    )

    facet_rankings = {}
    for facet_name in EVIDENCE_FACET_QUERY_DEFINITIONS:
        semantic_facet = semantic_rankings_by_label.get(facet_name, [])
        lexical_facet = lexical_rankings_by_label.get(facet_name, [])

        if semantic_facet or lexical_facet:
            facet_rankings[facet_name] = (
                fuse_semantic_and_lexical_results(
                    semantic_results=semantic_facet,
                    lexical_results=lexical_facet,
                    lexical_distance=max_distance,
                )
            )

    if not combined_results:
        return []

    has_lexical_evidence = bool(lexical_results)

    context_max_chunks = (
        DENSE_CONTEXT_MAX_CHUNKS
        if dense_question
        else CONTEXT_MAX_CHUNKS
    )
    anchor_limit = (
        min(DENSE_ANCHOR_LIMIT, context_max_chunks)
        if dense_question
        else min(k + 2, context_max_chunks)
    )

    if has_lexical_evidence:
        if dense_question:
            reranked_results = (
                rerank_with_soft_facet_support(
                    combined_results,
                    facet_rankings,
                )
                if facet_rankings
                else combined_results
            )
            raw_results = select_diverse_retrieval_anchors(
                reranked_results,
                limit=anchor_limit,
                min_chunk_gap=(
                    DENSE_ANCHOR_MIN_CHUNK_GAP
                    if dense_question
                    else 3
                ),
            )
        else:
            raw_results = combined_results[:anchor_limit]

        if not expand_context:
            return raw_results

        return expand_retrieval_context(
            vector_store,
            raw_results,
            max_chunks=context_max_chunks,
        )

    semantic_distances = [
        distance
        for _document, distance in semantic_results
    ]

    if not semantic_distances:
        return []

    best_distance = min(semantic_distances)
    strict_semantic_limit = max_distance * 0.92

    if best_distance > strict_semantic_limit:
        return []

    if dense_question:
        reranked_results = (
            rerank_with_soft_facet_support(
                combined_results,
                facet_rankings,
            )
            if facet_rankings
            else combined_results
        )
        raw_results = select_diverse_retrieval_anchors(
            reranked_results,
            limit=anchor_limit,
            min_chunk_gap=(
                DENSE_ANCHOR_MIN_CHUNK_GAP
                if dense_question
                else 3
            ),
        )
    else:
        raw_results = combined_results[:anchor_limit]

    if not expand_context:
        return raw_results

    return expand_retrieval_context(
        vector_store,
        raw_results,
    )

