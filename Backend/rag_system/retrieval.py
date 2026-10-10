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
import logging
import math
import re
from typing import Any
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from sentence_transformers import CrossEncoder

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


from .config import (
    CONTEXT_EXPANSION_CHUNKS,
    CONTEXT_MAX_CHUNKS,
    DENSE_ANCHOR_LIMIT,
    DENSE_ANCHOR_MIN_CHUNK_GAP,
    DENSE_CONTEXT_MAX_CHUNKS,
    DENSE_FACET_RERANK_WEIGHT,
    DENSE_LEXICAL_RRF_WEIGHT,
    RAG_RERANK_BATCH_SIZE,
    RAG_RERANK_CANDIDATE_K,
    RAG_RERANK_ENABLED,
    RAG_RERANK_MAX_LENGTH,
    RAG_RERANK_MODEL,
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


logger = logging.getLogger(__name__)


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


def _validate_retrieval_parameters(
    query: str,
    k: int,
    fetch_k: int,
    lambda_mult: float,
    max_distance: float,
) -> tuple[float, float]:
    """Validate caller-controlled retrieval parameters before doing any work."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query cannot be empty.")

    for name, value in (("k", k), ("fetch_k", fetch_k)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be a positive integer.")
        if value <= 0:
            raise ValueError(f"{name} must be greater than 0.")

    try:
        lambda_value = float(lambda_mult)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("lambda_mult must be between 0.0 and 1.0.") from error
    if not math.isfinite(lambda_value) or not 0.0 <= lambda_value <= 1.0:
        raise ValueError("lambda_mult must be between 0.0 and 1.0.")

    try:
        distance_limit = float(max_distance)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("max_distance must be finite and non-negative.") from error
    if not math.isfinite(distance_limit):
        raise ValueError("max_distance must be finite and non-negative.")
    if distance_limit < 0:
        raise ValueError("max_distance cannot be negative.")

    return lambda_value, distance_limit


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
    """Retrieve timestamped YouTube transcript chunks with distance-gated MMR.

    The expand_context argument is retained for API compatibility. Context
    expansion is owned by retrieve_question_context so budgets remain consistent.
    """
    lambda_mult, max_distance = _validate_retrieval_parameters(
        query, k, fetch_k, lambda_mult, max_distance
    )
    if k > fetch_k:
        raise ValueError("k cannot be greater than fetch_k.")

    query_embedding = vector_store.embedding_function.embed_query(query.strip())
    query_vector = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
    if query_vector.shape[1] == 0 or not np.all(np.isfinite(query_vector)):
        logger.warning("Skipping retrieval because the query embedding is empty or non-finite.")
        return []

    index = vector_store.index
    index_dimension = getattr(index, "d", None)
    if isinstance(index_dimension, int) and index_dimension != query_vector.shape[1]:
        raise ValueError(
            "Query embedding dimension does not match the stored FAISS index "
            f"({query_vector.shape[1]} != {index_dimension})."
        )

    total_vectors = int(getattr(index, "ntotal", 0))
    if total_vectors <= 0:
        return []

    candidate_k = min(fetch_k, total_vectors)
    distances, indices = index.search(
        np.ascontiguousarray(query_vector),
        candidate_k,
    )
    candidate_indices = np.asarray(indices[0], dtype=np.int64)
    candidate_distances = np.asarray(distances[0], dtype=np.float64)
    eligible_positions = [
        position
        for position, distance in enumerate(candidate_distances)
        if (
            int(candidate_indices[position]) >= 0
            and math.isfinite(float(distance))
            and float(distance) <= max_distance
        )
    ]
    if not eligible_positions:
        return []

    faiss_candidate_ids = [
        int(candidate_indices[position]) for position in eligible_positions
    ]
    try:
        if hasattr(index, "reconstruct_batch"):
            candidate_embeddings = index.reconstruct_batch(
                np.asarray(faiss_candidate_ids, dtype=np.int64)
            )
        else:
            candidate_embeddings = np.vstack(
                [index.reconstruct(vector_id) for vector_id in faiss_candidate_ids]
            )
    except Exception:
        logger.exception("Unable to reconstruct candidate embeddings from the FAISS index.")
        return []

    candidate_embeddings = np.asarray(candidate_embeddings, dtype=np.float32)
    if candidate_embeddings.ndim == 1 and len(faiss_candidate_ids) == 1:
        candidate_embeddings = candidate_embeddings.reshape(1, -1)
    if (
        candidate_embeddings.ndim != 2
        or candidate_embeddings.shape[0] != len(eligible_positions)
        or candidate_embeddings.shape[1] != query_vector.shape[1]
    ):
        logger.error(
            "FAISS returned candidate embeddings with an unexpected shape: %s.",
            candidate_embeddings.shape,
        )
        return []

    candidate_norms = np.linalg.norm(candidate_embeddings, axis=1)
    valid_rows = np.all(np.isfinite(candidate_embeddings), axis=1)
    valid_rows &= np.isfinite(candidate_norms) & (candidate_norms > 1e-12)
    if not np.all(valid_rows):
        candidate_embeddings = candidate_embeddings[valid_rows]
        eligible_positions = [
            position
            for position, is_valid in zip(eligible_positions, valid_rows)
            if bool(is_valid)
        ]
    if not eligible_positions:
        return []

    query_norm = float(np.linalg.norm(query_vector))
    if not math.isfinite(query_norm) or query_norm <= 1e-12:
        logger.warning("Skipping retrieval because the query embedding has zero norm.")
        return []

    query_normalized = query_vector / query_norm
    candidate_norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
    candidate_normalized = candidate_embeddings / np.maximum(candidate_norms, 1e-12)
    query_similarities = (candidate_normalized @ query_normalized.T).reshape(-1)

    number_to_select = min(k, len(eligible_positions))
    selected_positions = [int(np.argmax(query_similarities))]
    remaining_positions = [
        position for position in range(len(eligible_positions))
        if position != selected_positions[0]
    ]
    while remaining_positions and len(selected_positions) < number_to_select:
        best_position = None
        best_score = -np.inf
        selected_embeddings = candidate_normalized[selected_positions]
        for candidate_position in remaining_positions:
            relevance = float(query_similarities[candidate_position])
            redundancy = float(
                np.max(selected_embeddings @ candidate_normalized[candidate_position])
            )
            mmr_score = (
                lambda_mult * relevance
                - (1.0 - lambda_mult) * redundancy
            )
            if mmr_score > best_score:
                best_score = mmr_score
                best_position = candidate_position
        if best_position is None:
            break
        selected_positions.append(best_position)
        remaining_positions.remove(best_position)

    results = []
    docstore = getattr(vector_store, "docstore", None)
    index_to_docstore_id = getattr(vector_store, "index_to_docstore_id", {})
    if docstore is None:
        return []
    for selected_position in selected_positions:
        original_position = eligible_positions[selected_position]
        vector_id = int(candidate_indices[original_position])
        docstore_id = index_to_docstore_id.get(vector_id)
        if docstore_id is None:
            continue
        try:
            document = docstore.search(docstore_id)
        except Exception:
            logger.warning("Could not load FAISS document %r from the docstore.", docstore_id)
            continue
        if (
            not isinstance(getattr(document, "page_content", None), str)
            or not isinstance(getattr(document, "metadata", None), dict)
        ):
            logger.warning("Skipping invalid or stale FAISS docstore entry %r.", docstore_id)
            continue
        results.append((document, float(candidate_distances[original_position])))
    return results




def retrieve_faiss_candidates_for_diagnostics(
    vector_store,
    query: str,
    *,
    fetch_k: int,
    max_distance: float,
):
    """Return raw, finite FAISS candidates before applying the production gate.

    Each tuple is (document, distance, faiss_rank). Candidates outside
    max_distance are intentionally retained here so the diagnostic can explain
    why production rejected them; invalid vectors and stale docstore entries
    are not returned.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query cannot be empty.")
    if isinstance(fetch_k, bool) or not isinstance(fetch_k, int) or fetch_k <= 0:
        raise ValueError("fetch_k must be greater than 0.")
    try:
        distance_limit = float(max_distance)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("max_distance must be finite and non-negative.") from error
    if not math.isfinite(distance_limit):
        raise ValueError("max_distance must be finite and non-negative.")
    if distance_limit < 0:
        raise ValueError("max_distance cannot be negative.")

    query_embedding = vector_store.embedding_function.embed_query(query.strip())
    query_vector = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
    if query_vector.shape[1] == 0 or not np.all(np.isfinite(query_vector)):
        logger.warning("Skipping diagnostics because the query embedding is empty or non-finite.")
        return []

    index = vector_store.index
    index_dimension = getattr(index, "d", None)
    if isinstance(index_dimension, int) and index_dimension != query_vector.shape[1]:
        raise ValueError(
            "Query embedding dimension does not match the stored FAISS index "
            f"({query_vector.shape[1]} != {index_dimension})."
        )
    total_vectors = int(getattr(index, "ntotal", 0))
    if total_vectors <= 0:
        return []

    candidate_k = min(fetch_k, total_vectors)
    distances, indices = index.search(
        np.ascontiguousarray(query_vector),
        candidate_k,
    )
    candidate_distances = np.asarray(distances[0], dtype=np.float64)
    candidate_indices = np.asarray(indices[0], dtype=np.int64)
    docstore = getattr(vector_store, "docstore", None)
    index_to_docstore_id = getattr(vector_store, "index_to_docstore_id", {})
    if docstore is None or not hasattr(index_to_docstore_id, "get"):
        return []

    results = []
    for position, distance in enumerate(candidate_distances):
        vector_id = int(candidate_indices[position])
        if vector_id < 0 or not math.isfinite(float(distance)):
            continue
        docstore_id = index_to_docstore_id.get(vector_id)
        if docstore_id is None:
            continue
        try:
            document = docstore.search(docstore_id)
        except Exception:
            logger.warning(
                "Could not load raw FAISS diagnostic candidate %r from the docstore.",
                docstore_id,
            )
            continue
        if (
            not isinstance(getattr(document, "page_content", None), str)
            or not isinstance(getattr(document, "metadata", None), dict)
        ):
            logger.warning(
                "Skipping invalid or stale raw FAISS diagnostic candidate %r.",
                docstore_id,
            )
            continue
        results.append((document, float(distance), position + 1))
    return results




def _select_final_retrieval_candidates(
    *,
    query: str,
    semantic_results,
    lexical_results,
    combined_results,
    facet_rankings,
    facet_candidate_rankings,
    dense_question: bool,
    k: int,
    max_distance: float,
) -> dict[str, Any]:
    """Apply the same evidence gate, reranking, and anchor policy in all routes."""
    context_max_chunks = (
        DENSE_CONTEXT_MAX_CHUNKS if dense_question else CONTEXT_MAX_CHUNKS
    )
    anchor_limit = (
        min(DENSE_ANCHOR_LIMIT, context_max_chunks)
        if dense_question
        else min(k + 2, context_max_chunks)
    )
    if not combined_results:
        return {
            "cross_encoder_results": [],
            "facet_soft_reranked": [],
            "anchors": [],
            "context_max_chunks": context_max_chunks,
            "anchor_limit": anchor_limit,
            "semantic_gate_passed": False,
        }

    if not lexical_results:
        semantic_distances = []
        for _document, distance in semantic_results:
            try:
                value = float(distance)
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(value):
                semantic_distances.append(value)
        strict_semantic_limit = max_distance * 0.92
        if not semantic_distances or min(semantic_distances) > strict_semantic_limit:
            # Reject weak semantic-only matches before attempting to load a
            # reranker. Diagnostic and production now share this exact gate.
            return {
                "cross_encoder_results": list(combined_results),
                "facet_soft_reranked": list(combined_results),
                "anchors": [],
                "context_max_chunks": context_max_chunks,
                "anchor_limit": anchor_limit,
                "semantic_gate_passed": False,
            }

    cross_encoder_results = rerank_with_cross_encoder(
        query=query,
        ranked_results=combined_results,
    )
    facet_soft_reranked = (
        rerank_with_soft_facet_support(cross_encoder_results, facet_rankings)
        if dense_question and facet_rankings
        else cross_encoder_results
    )
    anchors = (
        select_diverse_retrieval_anchors(
            facet_soft_reranked,
            limit=anchor_limit,
            min_chunk_gap=DENSE_ANCHOR_MIN_CHUNK_GAP,
            facet_candidate_rankings=facet_candidate_rankings,
        )
        if dense_question
        else facet_soft_reranked[:anchor_limit]
    )
    return {
        "cross_encoder_results": cross_encoder_results,
        "facet_soft_reranked": facet_soft_reranked,
        "anchors": anchors,
        "context_max_chunks": context_max_chunks,
        "anchor_limit": anchor_limit,
        "semantic_gate_passed": True,
    }


def _build_facet_rankings(
    semantic_rankings_by_label,
    lexical_rankings_by_label,
    *,
    lexical_distance: float,
    lexical_weight: float,
):
    """Build merged facet rankings plus source-level lists for anchor reserves."""
    facet_rankings = {}
    facet_candidate_rankings = {}
    for facet_name in EVIDENCE_FACET_QUERY_DEFINITIONS:
        semantic_facet = semantic_rankings_by_label.get(facet_name, [])
        lexical_facet = lexical_rankings_by_label.get(facet_name, [])
        if semantic_facet:
            facet_candidate_rankings[f"{facet_name}::semantic"] = semantic_facet
        if lexical_facet:
            facet_candidate_rankings[f"{facet_name}::lexical"] = lexical_facet
        if semantic_facet or lexical_facet:
            facet_rankings[facet_name] = fuse_semantic_and_lexical_results(
                semantic_results=semantic_facet,
                lexical_results=lexical_facet,
                lexical_distance=lexical_distance,
                lexical_weight=lexical_weight,
            )
    return facet_rankings, facet_candidate_rankings


def diagnose_retrieval_pipeline(
    vector_store,
    query: str,
    *,
    k: int = TOP_K,
    fetch_k: int = MMR_FETCH_K,
    lambda_mult: float = MMR_LAMBDA,
    max_distance: float = MAX_DISTANCE,
):
    """Expose retrieval stages while using production's exact final selection path."""
    lambda_mult, max_distance = _validate_retrieval_parameters(
        query, k, fetch_k, lambda_mult, max_distance
    )
    if is_overview_question(query):
        raise ValueError(
            "End-to-end retrieval diagnostics are not defined for overview questions."
        )

    dense_question = is_evidence_dense_question(query)
    lexical_rrf_weight = DENSE_LEXICAL_RRF_WEIGHT if dense_question else 1.0
    semantic_k = DENSE_SEMANTIC_K if dense_question else k
    semantic_fetch_k = (
        DENSE_SEMANTIC_FETCH_K if dense_question else max(fetch_k, semantic_k * 2)
    )
    lexical_limit = (
        max(k * 2, DENSE_ANCHOR_LIMIT, 8)
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
        lexical_queries.extend(build_evidence_facet_plan(query))
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
        lexical_weight=lexical_rrf_weight,
    )
    facet_rankings, facet_candidate_rankings = _build_facet_rankings(
        semantic_rankings_by_label,
        lexical_rankings_by_label,
        lexical_distance=max_distance,
        lexical_weight=lexical_rrf_weight,
    )

    selection = _select_final_retrieval_candidates(
        query=query,
        semantic_results=semantic_results,
        lexical_results=lexical_results,
        combined_results=combined_results,
        facet_rankings=facet_rankings,
        facet_candidate_rankings=facet_candidate_rankings,
        dense_question=dense_question,
        k=k,
        max_distance=max_distance,
    )
    context_results = expand_retrieval_context(
        vector_store,
        selection["anchors"],
        max_chunks=selection["context_max_chunks"],
    )
    return {
        "query": query,
        "dense_question": dense_question,
        "reranking_enabled": RAG_RERANK_ENABLED,
        "lexical_rrf_weight": lexical_rrf_weight,
        "facet_rerank_weight": DENSE_FACET_RERANK_WEIGHT,
        "semantic_k": semantic_k,
        "semantic_fetch_k": semantic_fetch_k,
        "lexical_limit": lexical_limit,
        "anchor_limit": selection["anchor_limit"],
        "query_plan": query_plan,
        "query_variants": [query_text for _label, query_text in query_plan],
        "semantic_stages": semantic_stages,
        "semantic_fused": semantic_results,
        "lexical_queries": [
            {"label": label, "query": lexical_query}
            for label, lexical_query in lexical_queries
        ],
        "lexical_stages": lexical_stages,
        "lexical": lexical_results,
        "facet_rankings": facet_rankings,
        "facet_soft_reranked": selection["facet_soft_reranked"],
        # When disabled, this field is a pass-through, not an actual model result.
        "cross_encoder_reranked": selection["cross_encoder_results"],
        "hybrid_fused": combined_results,
        "anchors": selection["anchors"],
        "final_context": context_results,
        "semantic_gate_passed": selection["semantic_gate_passed"],
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

    normalized_query = " ".join(query.strip().casefold().split())
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

    words = re.findall(r"[^\W_]+", normalized_query, flags=re.UNICODE)
    return [
        word
        for word in words
        if len(word) >= 2 and word not in QUERY_FOCUS_STOPWORDS
    ]


EVIDENCE_FACET_QUERY_DEFINITIONS = {
    # General reasoning facets for a wide range of YouTube content.
    "causal_factors": (
        "causes reasons factors because due to led to contributed to "
        "challenges problems constraints barriers limits"
    ),
    "mechanism_process": (
        "process mechanism how it works steps method procedure implementation"
    ),
    "impact_outcomes": (
        "effects impact consequences outcomes results changes"
    ),
    "comparison_tradeoffs": (
        "compared versus differences similarities advantages disadvantages tradeoffs"
    ),
    "examples_evidence": (
        "examples evidence demonstration showed explained instance case study"
    ),
    "sequence_timeline": (
        "before after first then next later eventually timeline sequence"
    ),
    # Domain-specific facets are selected only when the question signals that domain.
    "financial_economic": (
        "financial crisis economic circumstances cash flow money debt losses costs"
    ),
    "policy_governance": (
        "government policy regulation banks support approval restrictions"
    ),
    "policy_change": (
        "policy changes regulatory changes rules regulations"
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
    """Build topic-aware facet queries without injecting unrelated domains."""
    original = " ".join(query.strip().split())
    if not original:
        return []

    focus_terms = extract_query_focus_terms(original)
    if len(focus_terms) < 2:
        return []

    focus_query = " ".join(focus_terms)
    normalized = original.casefold()
    facet_names = []

    def add_facet(name: str) -> None:
        if name not in facet_names:
            facet_names.append(name)

    if re.search(r"\b(?:brand|branding|advertis\w*|marketing|personality)\b", normalized):
        add_facet("brand_marketing")
    if re.search(
        r"\b(?:policy|government|governance|regulation\w*|regulatory|bank|banks|approval|restriction\w*)\b",
        normalized,
    ):
        add_facet("policy_change")
        add_facet("policy_governance")
    if re.search(
        r"\b(?:financial|finance|economic|economy|money|debt|cash|cost\w*|revenue|profit\w*|loss\w*|funding|budget|price\w*|expense\w*)\b",
        normalized,
    ):
        add_facet("financial_economic")
    if re.search(
        r"\b(?:operations?|operational|fuel|suppliers?|payments?|fees|service|maintenance|production|logistics?)\b",
        normalized,
    ):
        add_facet("operational_challenges")
    if re.search(
        r"\b(?:businesses|companies|subsidiaries|ownership|investments?|corporate structure|organizations?)\b",
        normalized,
    ):
        add_facet("business_corporate")

    # A question about a business/airline struggling, failing, or facing
    # challenges should explore the common cause categories for that domain.
    # This is conditional on explicit business context; unrelated science,
    # software, education, travel, or sports questions do not inherit these
    # financial/policy/operations facets.
    business_context = bool(re.search(
        r"\b(?:business\w*|companies|company|corporate|corporation|firms?|"
        r"airlines?|startups?|banks?|banking)\b",
        normalized,
    ))
    business_failure_context = bool(re.search(
        r"\b(?:fail\w*|struggl\w*|challenge\w*|problem\w*|difficult\w*|"
        r"bankrupt\w*|crisis|declin\w*|surviv\w*)\b",
        normalized,
    ))
    if business_context and business_failure_context:
        add_facet("financial_economic")
        add_facet("policy_governance")
        add_facet("operational_challenges")

    if re.search(
        r"\b(?:why|reason\w*|caus\w*|because|fail\w*|problem\w*|challenge\w*|issue\w*|factor\w*|role of)\b",
        normalized,
    ):
        add_facet("causal_factors")
    if re.search(
        r"\b(?:how|process|mechanism|steps?|method|procedure|work\w*|made|built|created|developed|implemented|calculated)\b",
        normalized,
    ) and not re.match(
        r"^how\s+(?:many|much|long|old|far|tall|wide|high|heavy|often|soon|fast)\b",
        normalized,
    ):
        add_facet("mechanism_process")
    if re.search(
        r"\b(?:impact|effects?|affect\w*|consequence\w*|outcome\w*|results?)\b",
        normalized,
    ):
        add_facet("impact_outcomes")
    if re.search(
        r"\b(?:compare|compared|comparison|versus|vs\.?|differences?|similarities|advantages?|disadvantages?|tradeoffs?)\b",
        normalized,
    ):
        add_facet("comparison_tradeoffs")
    if re.search(
        r"\b(?:examples?|evidence|demonstrat\w*|case stud\w*|proof|show(?:s|ed|ing)?)\b",
        normalized,
    ):
        add_facet("examples_evidence")
    if re.search(
        r"\b(?:before|after|first|then|next|later|eventually|timeline|sequence|over time)\b",
        normalized,
    ):
        add_facet("sequence_timeline")

    return [
        (name, f"{focus_query} {EVIDENCE_FACET_QUERY_DEFINITIONS[name]}")
        for name in facet_names[:4]
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
    r"\bwhat is the main topic of (?:this|the) video\b",
    r"\bwhat is the primary topic of (?:this|the) video\b",
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
        r"[^\W_]+",
        query.casefold(),
        flags=re.UNICODE,
    )

    return {
        word
        for word in words
        if (
            len(word) >= 2
            and word not in STOPWORDS
        )
    }


def _normalise_chunk_id(value):
    """Normalize numeric and textual chunk identifiers without mixed-type sorting."""
    if value is None:
        return None
    try:
        numeric = float(value)
        if math.isfinite(numeric) and numeric.is_integer():
            return ("number", int(numeric))
    except (TypeError, ValueError, OverflowError):
        pass
    return ("text", str(value))


def _chunk_id_sort_key(value):
    normalized = _normalise_chunk_id(value)
    if normalized is None:
        return (2, "")
    if normalized[0] == "number":
        return (0, normalized[1])
    return (1, normalized[1])


def _safe_metadata_float(metadata, key, default=0.0):
    try:
        value = float(metadata.get(key, default))
    except (AttributeError, TypeError, ValueError, OverflowError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _document_sort_key(document):
    metadata = getattr(document, "metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    start = _safe_metadata_float(metadata, "start", 0.0)
    end = _safe_metadata_float(metadata, "end", start)
    return (start, end, _chunk_id_sort_key(metadata.get("chunk_id")))


def get_all_documents(
    vector_store,
) -> list:
    """Return valid transcript Documents, skipping stale persisted docstore IDs."""
    docstore = getattr(vector_store, "docstore", None)
    index_to_docstore_id = getattr(vector_store, "index_to_docstore_id", None)
    if docstore is None or not isinstance(index_to_docstore_id, dict):
        return []

    documents = []
    seen_docstore_ids = set()
    for docstore_id in index_to_docstore_id.values():
        try:
            if docstore_id in seen_docstore_ids:
                continue
            seen_docstore_ids.add(docstore_id)
        except TypeError:
            logger.warning("Skipping an invalid FAISS docstore ID: %r.", docstore_id)
            continue

        try:
            document = docstore.search(docstore_id)
        except Exception:
            logger.warning("Could not load FAISS document %r from the docstore.", docstore_id)
            continue
        if (
            not isinstance(getattr(document, "page_content", None), str)
            or not isinstance(getattr(document, "metadata", None), dict)
        ):
            logger.warning("Skipping invalid or stale FAISS docstore entry %r.", docstore_id)
            continue
        documents.append(document)
    return documents


def lexical_search(
    vector_store,
    query: str,
    limit: int = 8,
):
    """Find transcript evidence with Unicode-aware tokens and IDF weighting.

    A rare technical term or proper noun can independently surface a chunk.
    Common single-token overlap is not enough evidence for a multi-term query.
    """
    if not isinstance(query, str) or not query.strip():
        return []
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        return []

    focus_terms = extract_query_focus_terms(query)
    if not focus_terms:
        focus_terms = list(extract_query_terms(query))
    if not focus_terms:
        return []

    documents = get_all_documents(vector_store)
    if not documents:
        return []

    unique_terms = list(dict.fromkeys(term.casefold() for term in focus_terms))
    tokenized_documents = []
    for document in documents:
        text = document.page_content.casefold()
        tokens = set(re.findall(r"[^\W_]+", text, flags=re.UNICODE))
        tokenized_documents.append((document, text, tokens))

    document_frequency = {term: 0 for term in unique_terms}
    for _document, _text, tokens in tokenized_documents:
        for term in unique_terms:
            if term in tokens:
                document_frequency[term] += 1

    total_documents = len(tokenized_documents)
    term_weights = {
        term: math.log((total_documents + 1) / (frequency + 1)) + 1.0
        for term, frequency in document_frequency.items()
    }
    total_query_weight = sum(term_weights.values()) or 1.0
    rare_term_frequency_limit = max(1, math.ceil(total_documents * 0.005))
    normalized_focus = " ".join(focus_terms).casefold()

    scored_documents = []
    for document, text, tokens in tokenized_documents:
        matched_terms = [term for term in unique_terms if term in tokens]
        if not matched_terms:
            continue
        if len(unique_terms) > 1 and len(matched_terms) < 2:
            has_rare_match = any(
                document_frequency[term] <= rare_term_frequency_limit
                for term in matched_terms
            )
            # A lone rare term can rescue a long, technical question, but a
            # short multi-concept query needs corroboration. Otherwise an
            # incidental mention (e.g. "Australia" in a video) can make an
            # unrelated question look answerable to lexical retrieval.
            allow_rare_single_match = (
                len(unique_terms) >= 3 and has_rare_match
            )
            if not allow_rare_single_match:
                continue

        weighted_coverage = (
            sum(term_weights[term] for term in matched_terms)
            / total_query_weight
        )
        raw_coverage = len(matched_terms) / len(unique_terms)
        phrase_bonus = 0.0
        for left, right in zip(focus_terms, focus_terms[1:]):
            if f"{left} {right}".casefold() in text:
                phrase_bonus += 0.12
        if normalized_focus in text and len(unique_terms) >= 2:
            phrase_bonus += 0.5
        score = weighted_coverage + (0.20 * raw_coverage) + phrase_bonus
        scored_documents.append((document, float(score)))

    scored_documents.sort(
        key=lambda item: (-item[1], *_document_sort_key(item[0]))
    )
    return scored_documents[:limit]




def fuse_ranked_results(
    ranked_results: list[list[tuple[Any, float]]],
    *,
    rrf_k: int = RRF_K,
    score_direction: str = "lower",
):
    """Fuse ranked lists using RRF, preserving the input score semantics."""
    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than 0.")
    if score_direction not in {"lower", "higher"}:
        raise ValueError("score_direction must be 'lower' or 'higher'.")

    fused = {}
    for results in ranked_results:
        for rank, (document, score) in enumerate(results, start=1):
            key = _retrieval_document_identity(document)
            value = float(score)
            entry = fused.get(key)
            if entry is None:
                entry = {
                    "document": document,
                    "rrf_score": 0.0,
                    "best_rank": rank,
                    "rank_sum": 0,
                    "raw_score": value,
                }
                fused[key] = entry
            elif math.isfinite(value):
                current = entry["raw_score"]
                if not math.isfinite(current):
                    entry["raw_score"] = value
                elif score_direction == "lower":
                    entry["raw_score"] = min(current, value)
                else:
                    entry["raw_score"] = max(current, value)

            entry["rrf_score"] += 1.0 / (rrf_k + rank)
            entry["best_rank"] = min(entry["best_rank"], rank)
            entry["rank_sum"] += rank

    ranked = sorted(
        fused.values(),
        key=lambda entry: (
            -entry["rrf_score"],
            entry["best_rank"],
            entry["rank_sum"],
            *_document_sort_key(entry["document"]),
        ),
    )
    return [(entry["document"], entry["raw_score"]) for entry in ranked]




def fuse_semantic_rankings(
    ranked_results: list[list[tuple[Any, float]]],
    *,
    rrf_k: int = RRF_K,
):
    """Fuse semantic query variants with Reciprocal Rank Fusion."""

    return fuse_ranked_results(
        ranked_results,
        rrf_k=rrf_k,
        score_direction="lower",
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
        score_direction="higher",
    )


def fuse_semantic_and_lexical_results(
    semantic_results,
    lexical_results,
    *,
    lexical_distance: float = MAX_DISTANCE,
    lexical_weight: float = 1.0,
    rrf_k: int = RRF_K,
):
    """Fuse ranks and preserve actual FAISS distances for semantic hits.

    FAISS distances and lexical scores are on different scales. RRF combines
    rank positions; lexical-only chunks receive a conservative distance.
    """
    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than 0.")
    try:
        lexical_weight = float(lexical_weight)
        lexical_distance = float(lexical_distance)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("lexical_weight and lexical_distance must be numeric.") from error
    if not math.isfinite(lexical_weight) or lexical_weight <= 0:
        raise ValueError("lexical_weight must be finite and greater than 0.")
    if not math.isfinite(lexical_distance) or lexical_distance < 0:
        raise ValueError("lexical_distance must be finite and non-negative.")

    fused = {}

    def add_ranked_result(document, rank, *, is_semantic, raw_score, weight=1.0):
        key = _retrieval_document_identity(document)
        entry = fused.get(key)
        if entry is None:
            entry = {
                "document": document,
                "distance": lexical_distance,
                "has_semantic": False,
                "lexical_score": float("-inf"),
                "score": 0.0,
            }
            fused[key] = entry

        if is_semantic:
            distance = float(raw_score)
            if math.isfinite(distance):
                if not entry["has_semantic"] or distance < entry["distance"]:
                    entry["distance"] = distance
                entry["has_semantic"] = True
        else:
            lexical_score = float(raw_score)
            if math.isfinite(lexical_score):
                entry["lexical_score"] = max(entry["lexical_score"], lexical_score)
        entry["score"] += weight / (rrf_k + rank)

    for rank, (document, distance) in enumerate(semantic_results, start=1):
        add_ranked_result(document, rank, is_semantic=True, raw_score=distance)
    for rank, (document, lexical_score) in enumerate(lexical_results, start=1):
        add_ranked_result(
            document,
            rank,
            is_semantic=False,
            raw_score=lexical_score,
            weight=lexical_weight,
        )

    ranked = sorted(
        fused.values(),
        key=lambda entry: (
            -entry["score"],
            0 if entry["has_semantic"] else 1,
            entry["distance"],
            -entry["lexical_score"],
            *_document_sort_key(entry["document"]),
        ),
    )
    return [(entry["document"], entry["distance"]) for entry in ranked]




def retrieve_overview(
    vector_store,
    number_of_chunks: int = 6,
):
    """Sample representative transcript sections in chronological order."""
    if (
        isinstance(number_of_chunks, bool)
        or not isinstance(number_of_chunks, int)
        or number_of_chunks <= 0
    ):
        raise ValueError("number_of_chunks must be a positive integer.")

    documents = get_all_documents(vector_store)
    if not documents:
        return []
    documents.sort(key=_document_sort_key)
    if len(documents) <= number_of_chunks:
        selected_documents = documents
    elif number_of_chunks == 1:
        selected_documents = [documents[(len(documents) - 1) // 2]]
    else:
        selected_indices = [
            round(i * (len(documents) - 1) / (number_of_chunks - 1))
            for i in range(number_of_chunks)
        ]
        selected_documents = [documents[index] for index in selected_indices]
    return [(document, float(MAX_DISTANCE)) for document in selected_documents]




def expand_retrieval_context(
    vector_store,
    retrieved_results,
    window: int = CONTEXT_EXPANSION_CHUNKS,
    max_chunks: int = CONTEXT_MAX_CHUNKS,
):
    """Add nearby transcript chunks while preserving anchors and context bounds."""
    if isinstance(window, bool) or not isinstance(window, int):
        raise ValueError("window must be a non-negative integer.")
    if window < 0:
        raise ValueError("window cannot be negative.")
    if isinstance(max_chunks, bool) or not isinstance(max_chunks, int):
        raise ValueError("max_chunks must be a positive integer.")
    if max_chunks <= 0:
        raise ValueError("max_chunks must be greater than 0.")
    if not retrieved_results:
        return []
    if window == 0:
        return retrieved_results[:max_chunks]

    documents = get_all_documents(vector_store)
    if not documents:
        return retrieved_results[:max_chunks]

    ordered_documents = sorted(documents, key=_document_sort_key)
    positions = {}
    for position, document in enumerate(ordered_documents):
        normalized_id = _normalise_chunk_id(document.metadata.get("chunk_id"))
        if normalized_id is not None:
            positions[normalized_id] = position

    selected = {}
    anchor_keys = set()
    anchor_positions = []

    def add_document(document, distance, *, is_anchor=False):
        key = _retrieval_document_identity(document)
        if is_anchor:
            if key not in anchor_keys:
                selected[key] = (document, float(distance))
                anchor_keys.add(key)
            else:
                prior_document, prior_distance = selected[key]
                selected[key] = (prior_document, min(prior_distance, float(distance)))
        elif key not in selected:
            # This is a neighbor, not a scored retrieval result.
            selected[key] = (document, float(MAX_DISTANCE))

    for anchor_document, anchor_distance in retrieved_results:
        add_document(anchor_document, anchor_distance, is_anchor=True)
        normalized_id = _normalise_chunk_id(anchor_document.metadata.get("chunk_id"))
        position = positions.get(normalized_id)
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
            normalized_id = _normalise_chunk_id(document.metadata.get("chunk_id"))
            position = positions.get(normalized_id, 10**9)
            neighbor_distance = (
                min(abs(position - anchor_position) for anchor_position in anchor_positions)
                if anchor_positions
                else 10**9
            )
            return (
                0 if key in anchor_keys else 1,
                neighbor_distance,
                *_document_sort_key(document),
            )
        selected_items = sorted(selected.items(), key=priority)[:max_chunks]
    else:
        selected_items = list(selected.items())

    expanded_results = [item[1] for item in selected_items]
    expanded_results.sort(key=lambda item: _document_sort_key(item[0]))
    return expanded_results




# ============================================================
# QUESTION-AWARE RETRIEVAL
# ============================================================

def is_evidence_dense_question(query: str) -> bool:
    """Detect multi-evidence questions without treating every 'how' as dense."""
    if not isinstance(query, str) or not query.strip():
        return False

    normalized = " ".join(query.casefold().split())
    focus_terms = extract_query_focus_terms(query)
    quantitative_how = re.match(
        r"^how\s+(?:many|much|long|old|far|tall|wide|high|heavy|often|soon|fast|big|large|small)\b",
        normalized,
    )
    compound_or_explanatory = re.search(
        r"\b(?:and|or|both|why|reason\w*|caus\w*|because|impact|effect\w*|compare|difference)\b",
        normalized,
    )
    if quantitative_how and not compound_or_explanatory:
        return False

    if re.search(
        r"\b(?:why|reason\w*|caus\w*|because|challenges?|problems?|problem|failure|failed|fail|impact|effects?|affect|role of|multiple|several)\b",
        normalized,
    ):
        return True

    if re.search(r"\b(?:policy|government|regulation\w*|brand|branding)\b", normalized) and re.search(
        r"\b(?:role|impact|problem\w*|challenge\w*|reason\w*|cost\w*|effect\w*)\b",
        normalized,
    ):
        return True

    if re.search(r"\bhow\b", normalized) and re.search(
        r"\b(?:work\w*|process|mechanism|steps?|method|procedure|made|built|created|developed|implemented|calculated|operat\w*)\b",
        normalized,
    ):
        return True

    if re.search(r"\b(?:building|growing)\b.*\b(?:brand|business)\b", normalized):
        return True

    return bool(
        len(focus_terms) >= 6
        and re.search(r"\b(?:and|or|both)\b", normalized)
    )




def _retrieval_document_identity(document):
    """Return a stable identity for deduplication across query variants."""
    metadata = getattr(document, "metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    video_id = str(metadata.get("video_id", "") or "")
    chunk_id = _normalise_chunk_id(metadata.get("chunk_id"))
    if chunk_id is not None:
        return ("chunk", video_id, chunk_id)

    start = _safe_metadata_float(metadata, "start", float("nan"))
    end = _safe_metadata_float(metadata, "end", float("nan"))
    if math.isfinite(start) and math.isfinite(end):
        return ("time", video_id, start, end)

    # Legacy or external indexes may omit IDs and timestamps; full text keeps
    # distinct transcript passages distinct instead of collapsing them to (0, 0).
    return ("content", video_id, str(getattr(document, "page_content", "")))




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

    if not math.isfinite(facet_weight) or facet_weight < 0:
        raise ValueError("facet_weight must be finite and non-negative.")

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


@lru_cache(maxsize=1)
def create_cross_encoder_reranker():
    """Lazily load the configured Cross-Encoder reranker once per process."""

    if not RAG_RERANK_ENABLED:
        return None

    return CrossEncoder(
        RAG_RERANK_MODEL,
        device="cpu",
        max_length=RAG_RERANK_MAX_LENGTH,
    )


def rerank_with_cross_encoder(
    query: str,
    ranked_results,
    *,
    candidate_k: int = RAG_RERANK_CANDIDATE_K,
):
    """Reorder existing candidates with a Cross-Encoder relevance score.

    The reranker never creates new candidates and never changes their stored
    FAISS distance. If the optional reranker cannot be loaded or queried, the
    original ranking is returned unchanged so retrieval remains fail-open.
    """

    if not ranked_results:
        return []

    if not query or not query.strip():
        raise ValueError("query cannot be empty.")

    if isinstance(candidate_k, bool) or not isinstance(candidate_k, int):
        raise ValueError("candidate_k must be a positive integer.")
    if candidate_k <= 0:
        raise ValueError("candidate_k must be greater than 0.")

    if not RAG_RERANK_ENABLED:
        return list(ranked_results)

    candidates = list(ranked_results[:candidate_k])
    remainder = list(ranked_results[candidate_k:])

    if len(candidates) <= 1:
        return candidates + remainder

    try:
        reranker = create_cross_encoder_reranker()

        if reranker is None:
            return list(ranked_results)

        pairs = [
            (
                query.strip(),
                document.page_content,
            )
            for document, _distance in candidates
        ]

        scores = np.asarray(
            reranker.predict(
                pairs,
                batch_size=RAG_RERANK_BATCH_SIZE,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        ).reshape(-1)

        if len(scores) != len(candidates):
            raise RuntimeError(
                "Cross-Encoder returned an unexpected number of scores."
            )
        if not np.all(np.isfinite(scores)):
            raise RuntimeError("Cross-Encoder returned non-finite relevance scores.")

        order = np.argsort(-scores, kind="stable")

        reranked = [
            candidates[int(position)]
            for position in order
        ]

        return reranked + remainder

    except Exception:
        logger.exception(
            "Cross-Encoder reranking failed; preserving original retrieval ranking."
        )
        return list(ranked_results)


def select_diverse_retrieval_anchors(
    ranked_results,
    *,
    limit: int,
    min_chunk_gap: int = 3,
    facet_candidate_rankings: dict[str, list[tuple[Any, float]]] | None = None,
    facet_reserve_limit: int = 2,
    facet_rank_cutoff: int = 24,
):
    """Select diverse anchors, then reserve a small budget for facet evidence.

    Facet reserves are limited to candidates already in the fused ranking, within
    three times the anchor budget, and near the top of a source-level facet list.
    """

    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError("limit must be a positive integer.")
    if isinstance(min_chunk_gap, bool) or not isinstance(min_chunk_gap, int):
        raise ValueError("min_chunk_gap must be a non-negative integer.")
    if isinstance(facet_reserve_limit, bool) or not isinstance(facet_reserve_limit, int):
        raise ValueError("facet_reserve_limit must be a non-negative integer.")
    if isinstance(facet_rank_cutoff, bool) or not isinstance(facet_rank_cutoff, int):
        raise ValueError("facet_rank_cutoff must be a positive integer.")
    if facet_reserve_limit < 0:
        raise ValueError("facet_reserve_limit cannot be negative.")
    if facet_rank_cutoff <= 0:
        raise ValueError("facet_rank_cutoff must be greater than 0.")
    if limit <= 0 or not ranked_results:
        return []
    if min_chunk_gap < 0:
        raise ValueError("min_chunk_gap cannot be negative.")

    selected = []
    selected_chunk_ids = []
    remaining = list(ranked_results)

    def chunk_id(document):
        normalized = _normalise_chunk_id(document.metadata.get("chunk_id"))
        if normalized is None or normalized[0] != "number":
            return None
        return normalized[1]

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

    selected = selected[:limit]
    if not facet_candidate_rankings or facet_reserve_limit == 0 or len(selected) < 2:
        return selected

    rank_by_identity = {}
    item_by_identity = {}
    for global_rank, item in enumerate(ranked_results, start=1):
        identity = _retrieval_document_identity(item[0])
        rank_by_identity.setdefault(identity, global_rank)
        item_by_identity.setdefault(identity, item)

    selected_identities = {
        _retrieval_document_identity(item[0]) for item in selected
    }
    proposals = {}
    for source_name, facet_results in facet_candidate_rankings.items():
        facet_name = source_name.rsplit("::", 1)[0]
        for facet_rank, facet_item in enumerate(
            facet_results[:facet_rank_cutoff],
            start=1,
        ):
            identity = _retrieval_document_identity(facet_item[0])
            global_rank = rank_by_identity.get(identity)
            if (
                global_rank is None
                or global_rank <= limit
                or global_rank > limit * 3
                or identity in selected_identities
            ):
                continue

            proposal = {
                "item": item_by_identity[identity],
                "global_rank": global_rank,
                "facet_name": facet_name,
                "facet_rank": facet_rank,
                "rescue_delta": global_rank - facet_rank,
            }
            previous = proposals.get(identity)
            if previous is None or (
                proposal["rescue_delta"],
                -proposal["facet_rank"],
            ) > (
                previous["rescue_delta"],
                -previous["facet_rank"],
            ):
                proposals[identity] = proposal

    def document_chunk_id(item):
        metadata = getattr(item[0], "metadata", {})
        if not isinstance(metadata, dict):
            return None
        normalized = _normalise_chunk_id(metadata.get("chunk_id"))
        if normalized is None or normalized[0] != "number":
            return None
        return normalized[1]

    reserved_identities = set()
    reserved_facets = set()
    reserve_budget = min(facet_reserve_limit, len(selected) - 1)
    candidates = sorted(
        proposals.items(),
        key=lambda pair: (
            -pair[1]["rescue_delta"],
            pair[1]["facet_rank"],
            pair[1]["global_rank"],
        ),
    )

    for identity, proposal in candidates:
        if len(reserved_identities) >= reserve_budget:
            break
        if identity in reserved_identities or proposal["facet_name"] in reserved_facets:
            continue

        candidate_item = proposal["item"]
        candidate_chunk_id = document_chunk_id(candidate_item)
        replacement_indices = sorted(
            range(len(selected)),
            key=lambda index: rank_by_identity.get(
                _retrieval_document_identity(selected[index][0]),
                len(ranked_results) + index,
            ),
            reverse=True,
        )
        replacement_index = None
        for index in replacement_indices:
            current_identity = _retrieval_document_identity(selected[index][0])
            if current_identity in reserved_identities:
                continue
            remaining_ids = [
                document_chunk_id(item)
                for item_index, item in enumerate(selected)
                if item_index != index
            ]
            if candidate_chunk_id is not None and any(
                previous_id is not None
                and abs(candidate_chunk_id - previous_id) < min_chunk_gap
                for previous_id in remaining_ids
            ):
                continue
            replacement_index = index
            break

        if replacement_index is None:
            continue

        selected.pop(replacement_index)
        selected.append(candidate_item)
        reserved_identities.add(identity)
        reserved_facets.add(proposal["facet_name"])

    return selected



# ============================================================
# TEMPORAL QUERY ROUTING
# ============================================================

# Timestamp-constrained questions must not be answered from arbitrary
# transcript regions. If the transcript metadata does not cover enough of
# the requested interval, the pipeline returns the normal evidence fallback.
TEMPORAL_TAIL_WINDOW_SECONDS = 90.0
TEMPORAL_MIN_WINDOW_COVERAGE = 0.50
# Ignore chunks that barely intersect a requested window. The comparison uses
# the smaller interval to retain genuinely useful chunks at either boundary.
TEMPORAL_MIN_CHUNK_OVERLAP_RATIO = 0.10
# If a focused chunk starting inside the requested interval covers nearly all
# of it, prefer it over a much larger, earlier-starting envelope chunk.
TEMPORAL_FOCUSED_WINDOW_COVERAGE = 0.90
TEMPORAL_REDUNDANT_CHUNK_DURATION_RATIO = 2.0

_TEMPORAL_TIME_TOKEN = r"(?<!\d)(?:\d{1,2}:)?\d{1,2}:\d{2}(?!\d)"
_TEMPORAL_RANGE_PATTERN = re.compile(
    rf"(?P<start>{_TEMPORAL_TIME_TOKEN})\s*"
    r"(?:-|–|—|to|through|until|and)\s*"
    rf"(?P<end>{_TEMPORAL_TIME_TOKEN})",
    re.IGNORECASE,
)
_TEMPORAL_DURATION_PATTERN = re.compile(
    r"\b(?:last|final|past|previous|preceding)\s+"
    r"(?:(?P<amount>\d+(?:\.\d+)?)\s*)?"
    r"(?P<unit>seconds?|secs?|minutes?|mins?)\b",
    re.IGNORECASE,
)
_TEMPORAL_POINT_PATTERN = re.compile(
    rf"\b(?:at|around|near)\s+(?:approximately\s+)?"
    rf"(?P<time>{_TEMPORAL_TIME_TOKEN})\b",
    re.IGNORECASE,
)
_TEMPORAL_END_PATTERN = re.compile(
    r"\b(?:at|near|toward|towards|in)\s+(?:the\s+)?(?:very\s+)?"
    r"(?:end|ending|closing)\b|"
    r"\b(?:the\s+)?(?:final|last|closing)\s+"
    r"(?:part|section|portion|moments?|segment)\b|"
    r"\b(?:the\s+)?(?:ending|outro)\b",
    re.IGNORECASE,
)


def _parse_timestamp_token(value: str) -> float | None:
    """Parse MM:SS or HH:MM:SS into seconds."""
    try:
        parts = [int(part) for part in value.split(":")]
    except (TypeError, ValueError):
        return None

    if len(parts) == 2:
        minutes, seconds = parts
        if seconds >= 60:
            return None
        return float(minutes * 60 + seconds)

    if len(parts) == 3:
        hours, minutes, seconds = parts
        if minutes >= 60 or seconds >= 60:
            return None
        return float(hours * 3600 + minutes * 60 + seconds)

    return None


def is_temporal_question(query: str) -> bool:
    """Return True for explicit time ranges, timestamp points, or tail queries."""
    if not query or not query.strip():
        return False

    return any(
        pattern.search(query)
        for pattern in (
            _TEMPORAL_RANGE_PATTERN,
            _TEMPORAL_DURATION_PATTERN,
            _TEMPORAL_POINT_PATTERN,
            _TEMPORAL_END_PATTERN,
        )
    )


def parse_temporal_query_window(
    query: str,
    video_duration_seconds: float | None = None,
) -> dict[str, float | str | None] | None:
    """Resolve temporal wording to a bounded transcript interval.

    Explicit ranges take precedence. Relative tail queries use indexed
    duration metadata. Questions saying only "at the end" use a conservative
    90-second window. Point timestamps use a 30-second window on each side.
    """
    if not query or not query.strip():
        return None

    range_match = _TEMPORAL_RANGE_PATTERN.search(query)
    if range_match:
        start = _parse_timestamp_token(range_match.group("start"))
        end = _parse_timestamp_token(range_match.group("end"))
        if start is None or end is None:
            return None

        start, end = min(start, end), max(start, end)
        if start == end:
            start = max(0.0, start - 30.0)
            end = end + 30.0
            mode = "point_timestamp"
        else:
            mode = "explicit_range"

        if (
            video_duration_seconds is not None
            and video_duration_seconds > 0
            and start < float(video_duration_seconds)
        ):
            end = min(end, float(video_duration_seconds))

        return {
            "mode": mode,
            "start_seconds": float(start),
            "end_seconds": float(end),
        }

    duration_match = _TEMPORAL_DURATION_PATTERN.search(query)
    if duration_match:
        amount = float(duration_match.group("amount") or 1.0)
        unit = duration_match.group("unit").lower()
        requested_seconds = amount * (
            60.0 if unit.startswith("min") else 1.0
        )

        if video_duration_seconds is None or video_duration_seconds <= 0:
            return {
                "mode": "relative_tail",
                "start_seconds": None,
                "end_seconds": None,
                "window_seconds": requested_seconds,
            }

        end = float(video_duration_seconds)
        return {
            "mode": "relative_tail",
            "start_seconds": max(0.0, end - requested_seconds),
            "end_seconds": end,
        }

    point_match = _TEMPORAL_POINT_PATTERN.search(query)
    if point_match:
        point = _parse_timestamp_token(point_match.group("time"))
        if point is None:
            return None

        start = max(0.0, point - 30.0)
        end = point + 30.0
        if (
            video_duration_seconds is not None
            and video_duration_seconds > 0
            and point <= float(video_duration_seconds)
        ):
            end = min(end, float(video_duration_seconds))

        return {
            "mode": "point_timestamp",
            "start_seconds": start,
            "end_seconds": end,
        }

    if _TEMPORAL_END_PATTERN.search(query):
        if video_duration_seconds is None or video_duration_seconds <= 0:
            return {
                "mode": "video_end",
                "start_seconds": None,
                "end_seconds": None,
                "window_seconds": TEMPORAL_TAIL_WINDOW_SECONDS,
            }

        end = float(video_duration_seconds)
        return {
            "mode": "video_end",
            "start_seconds": max(0.0, end - TEMPORAL_TAIL_WINDOW_SECONDS),
            "end_seconds": end,
        }

    return None


def _temporal_window_coverage(
    documents,
    start_seconds: float,
    end_seconds: float,
) -> tuple[float, float]:
    """Return union coverage ratio and covered seconds for the target interval."""
    window_seconds = max(0.0, end_seconds - start_seconds)
    if window_seconds <= 0:
        return 0.0, 0.0

    intervals = []
    for document in documents:
        try:
            doc_start = float(document.metadata["start"])
            doc_end = float(document.metadata["end"])
        except (KeyError, TypeError, ValueError):
            continue

        overlap_start = max(start_seconds, doc_start)
        overlap_end = min(end_seconds, doc_end)
        if overlap_end > overlap_start:
            intervals.append((overlap_start, overlap_end))

    if not intervals:
        return 0.0, 0.0

    intervals.sort()
    merged_start, merged_end = intervals[0]
    covered_seconds = 0.0

    for interval_start, interval_end in intervals[1:]:
        if interval_start <= merged_end:
            merged_end = max(merged_end, interval_end)
        else:
            covered_seconds += merged_end - merged_start
            merged_start, merged_end = interval_start, interval_end

    covered_seconds += merged_end - merged_start
    covered_seconds = min(window_seconds, covered_seconds)

    return covered_seconds / window_seconds, covered_seconds


def retrieve_temporal_context(
    vector_store,
    query: str,
    *,
    max_chunks: int = DENSE_CONTEXT_MAX_CHUNKS,
) -> dict[str, Any] | None:
    """Retrieve only transcript chunks overlapping a requested time window.

    None means the question has no recognized temporal intent. A returned
    dictionary means the temporal route owns the request, even when no
    sufficient evidence exists; the caller must not fall back to unbounded
    semantic retrieval in that case.
    """
    if not is_temporal_question(query):
        return None

    documents = get_all_documents(vector_store)
    valid_documents = []

    for document in documents:
        if not document.page_content or not document.page_content.strip():
            continue

        try:
            start = float(document.metadata["start"])
            end = float(document.metadata["end"])
        except (KeyError, TypeError, ValueError):
            continue

        if end > start:
            valid_documents.append(document)

    video_duration = max(
        (
            float(document.metadata["end"])
            for document in valid_documents
        ),
        default=None,
    )
    window = parse_temporal_query_window(query, video_duration)

    if window is None:
        return {
            "window": None,
            "results": [],
            "candidates": [],
            "coverage_ratio": 0.0,
            "covered_seconds": 0.0,
            "window_seconds": 0.0,
            "minimum_coverage": TEMPORAL_MIN_WINDOW_COVERAGE,
            "coverage_sufficient": False,
            "candidate_chunk_ids": [],
        }

    start_seconds = window.get("start_seconds")
    end_seconds = window.get("end_seconds")

    if (
        not isinstance(start_seconds, (int, float))
        or not isinstance(end_seconds, (int, float))
        or float(end_seconds) <= float(start_seconds)
    ):
        return {
            "window": window,
            "results": [],
            "candidates": [],
            "coverage_ratio": 0.0,
            "covered_seconds": 0.0,
            "window_seconds": 0.0,
            "minimum_coverage": TEMPORAL_MIN_WINDOW_COVERAGE,
            "coverage_sufficient": False,
            "candidate_chunk_ids": [],
        }

    start_seconds = float(start_seconds)
    end_seconds = float(end_seconds)
    window_seconds = end_seconds - start_seconds

    candidates = []
    for document in valid_documents:
        doc_start = float(document.metadata["start"])
        doc_end = float(document.metadata["end"])
        overlap_start = max(start_seconds, doc_start)
        overlap_end = min(end_seconds, doc_end)
        overlap_seconds = overlap_end - overlap_start

        if overlap_seconds <= 0:
            continue

        # Chunks can straddle window boundaries. Ignore only negligible
        # intersection; focused-window pruning below handles broad chunks that
        # merely envelope a narrow request.
        chunk_seconds = doc_end - doc_start
        comparison_seconds = min(chunk_seconds, window_seconds)
        if comparison_seconds <= 0:
            continue

        overlap_ratio = overlap_seconds / comparison_seconds
        if overlap_ratio < TEMPORAL_MIN_CHUNK_OVERLAP_RATIO:
            continue

        candidates.append(document)

    candidates.sort(
        key=lambda document: (
            float(document.metadata["start"]),
            float(document.metadata["end"]),
            document.metadata.get("chunk_id", 0),
        )
    )

    # Timestamp envelopes can be much wider than the text that best represents
    # a narrow interval. If a shorter chunk begins inside the window and covers
    # at least 90% of it, remove a substantially longer chunk that starts before
    # the window and contributes only redundant overlap. Do not apply this to
    # broad windows where the focused tail chunk covers only a small fraction.
    focused_candidates = []
    for document in candidates:
        doc_start = float(document.metadata["start"])
        doc_end = float(document.metadata["end"])
        overlap_seconds = max(
            0.0,
            min(end_seconds, doc_end) - max(start_seconds, doc_start),
        )
        if (
            start_seconds <= doc_start < end_seconds
            and overlap_seconds / window_seconds
            >= TEMPORAL_FOCUSED_WINDOW_COVERAGE
        ):
            focused_candidates.append(document)

    if focused_candidates:
        pruned_candidates = []
        for document in candidates:
            doc_start = float(document.metadata["start"])
            doc_end = float(document.metadata["end"])
            doc_seconds = doc_end - doc_start
            doc_overlap_start = max(start_seconds, doc_start)
            doc_overlap_end = min(end_seconds, doc_end)
            doc_overlap = max(0.0, doc_overlap_end - doc_overlap_start)

            redundant = False
            if doc_start < start_seconds and doc_end >= end_seconds:
                for focused in focused_candidates:
                    if focused is document:
                        continue
                    focused_start = float(focused.metadata["start"])
                    focused_end = float(focused.metadata["end"])
                    focused_seconds = focused_end - focused_start
                    shared_seconds = max(
                        0.0,
                        min(doc_overlap_end, focused_end)
                        - max(doc_overlap_start, focused_start),
                    )
                    if (
                        focused_seconds > 0
                        and doc_seconds
                        >= focused_seconds * TEMPORAL_REDUNDANT_CHUNK_DURATION_RATIO
                        and doc_overlap > 0
                        and shared_seconds / doc_overlap
                        >= TEMPORAL_FOCUSED_WINDOW_COVERAGE
                    ):
                        redundant = True
                        break

            if not redundant:
                pruned_candidates.append(document)

        candidates = pruned_candidates

    # Preserve coverage across long windows while keeping prompt context bounded.
    selected_candidates = candidates
    if max_chunks <= 0:
        raise ValueError("max_chunks must be greater than 0.")
    if len(candidates) > max_chunks:
        if max_chunks == 1:
            selected_candidates = [candidates[len(candidates) // 2]]
        else:
            selected_indices = {
                round(
                    index * (len(candidates) - 1) / (max_chunks - 1)
                )
                for index in range(max_chunks)
            }
            selected_candidates = [
                candidates[index]
                for index in sorted(selected_indices)
            ]

    candidate_results = [
        (document, float(MAX_DISTANCE))
        for document in selected_candidates
    ]

    # Coverage must describe the chunks that actually enter the model context,
    # not every possible candidate before the bounded context budget is applied.
    coverage_ratio, covered_seconds = _temporal_window_coverage(
        selected_candidates,
        start_seconds,
        end_seconds,
    )
    coverage_sufficient = (
        coverage_ratio >= TEMPORAL_MIN_WINDOW_COVERAGE
    )
    results = candidate_results if coverage_sufficient else []

    candidate_chunk_ids = []
    for document in candidates:
        chunk_id = document.metadata.get("chunk_id")
        if chunk_id is not None and chunk_id not in candidate_chunk_ids:
            candidate_chunk_ids.append(chunk_id)

    return {
        "window": window,
        "results": results,
        "candidates": candidate_results,
        "coverage_ratio": round(coverage_ratio, 4),
        "covered_seconds": round(covered_seconds, 2),
        "window_seconds": round(window_seconds, 2),
        "minimum_coverage": TEMPORAL_MIN_WINDOW_COVERAGE,
        "coverage_sufficient": coverage_sufficient,
        "candidate_chunk_ids": candidate_chunk_ids[:50],
    }


def retrieve_question_context(
    vector_store,
    query: str,
    k: int = TOP_K,
    fetch_k: int = MMR_FETCH_K,
    lambda_mult: float = MMR_LAMBDA,
    max_distance: float = MAX_DISTANCE,
    expand_context: bool = True,
):
    """Retrieve transcript evidence for a question about the current YouTube video."""
    lambda_mult, max_distance = _validate_retrieval_parameters(
        query, k, fetch_k, lambda_mult, max_distance
    )
    if is_overview_question(query):
        return retrieve_overview(vector_store)

    dense_question = is_evidence_dense_question(query)
    lexical_rrf_weight = DENSE_LEXICAL_RRF_WEIGHT if dense_question else 1.0
    semantic_k = DENSE_SEMANTIC_K if dense_question else k
    semantic_fetch_k = (
        DENSE_SEMANTIC_FETCH_K if dense_question else max(fetch_k, semantic_k * 2)
    )
    # The dense anchor budget is 12 by default. A lexical cutoff of eight could
    # discard otherwise useful facet evidence before anchor selection.
    lexical_limit = (
        max(k * 2, DENSE_ANCHOR_LIMIT, 8)
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
        lexical_queries.extend(build_evidence_facet_plan(query))
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
    lexical_results = fuse_lexical_rankings(lexical_rankings)

    combined_results = fuse_semantic_and_lexical_results(
        semantic_results=semantic_results,
        lexical_results=lexical_results,
        lexical_distance=max_distance,
        lexical_weight=lexical_rrf_weight,
    )
    facet_rankings = {}
    for facet_name in EVIDENCE_FACET_QUERY_DEFINITIONS:
        semantic_facet = semantic_rankings_by_label.get(facet_name, [])
        lexical_facet = lexical_rankings_by_label.get(facet_name, [])
        if semantic_facet or lexical_facet:
            facet_rankings[facet_name] = fuse_semantic_and_lexical_results(
                semantic_results=semantic_facet,
                lexical_results=lexical_facet,
                lexical_distance=max_distance,
                lexical_weight=lexical_rrf_weight,
            )

    selection = _select_final_retrieval_candidates(
        query=query,
        semantic_results=semantic_results,
        lexical_results=lexical_results,
        combined_results=combined_results,
        facet_rankings=facet_rankings,
        dense_question=dense_question,
        k=k,
        max_distance=max_distance,
    )
    raw_results = selection["anchors"]
    if not expand_context:
        return raw_results
    return expand_retrieval_context(
        vector_store,
        raw_results,
        max_chunks=selection["context_max_chunks"],
    )

