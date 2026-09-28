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
6. Select final chunks using MMR.

This file does NOT:
- fetch YouTube transcripts
- create vector stores
- build prompts
- call the generation model
- handle FastAPI
"""

from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from .config import (
    EMBEDDING_MODEL,
    MAX_DISTANCE,
    MMR_FETCH_K,
    MMR_LAMBDA,
    TOP_K,
    VECTOR_STORE_ROOT,
)


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

        return video_reference


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

            return video_ids[0]


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

                return path_parts[1]


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

            return path_parts[0]


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