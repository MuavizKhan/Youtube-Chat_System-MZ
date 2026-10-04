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
import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


from .config import (
    CONTEXT_EXPANSION_CHUNKS,
    CONTEXT_MAX_CHUNKS,
    EMBEDDING_MODEL,
    MAX_DISTANCE,
    MMR_FETCH_K,
    MMR_LAMBDA,
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
    limit: int = 4,
):
    """
    Search transcript chunks using actual words from
    the user's question.

    This is especially useful for:

        - names
        - organizations
        - products
        - technical terms
        - exact phrases

    Returns:

        [(Document, score), ...]

    The second value is a lexical relevance score,
    not a FAISS distance.
    """

    query_terms = (
        extract_query_terms(
            query
        )
    )

    if not query_terms:

        return []

    normalized_query = (
        " ".join(
            query.lower()
            .split()
        )
    )

    documents = (
        get_all_documents(
            vector_store
        )
    )

    scored_documents = []

    for document in documents:

        text = (
            document.page_content
            .lower()
        )

        matched_terms = sum(
            1
            for term in query_terms
            if re.search(
            rf"\b{re.escape(term)}\b",
                text,
            )
        )

        # ----------------------------------------------------
        # Exact phrase match
        # ----------------------------------------------------

        exact_phrase_match = (
            normalized_query in text
            and len(query_terms) >= 2
        )

        # ----------------------------------------------------
        # Evidence rule
        #
        # For a multi-term query, require at least two
        # meaningful query terms to appear in the same chunk.
        #
        # This is especially important for names such as:
        #
        #     "Who is Demis Hassabis?"
        #
        # A chunk containing both "demis" and "hassabis"
        # is strong lexical evidence.
        # ----------------------------------------------------

        if exact_phrase_match:

            score = 1.0

        elif len(query_terms) == 1:

            if matched_terms != 1:
                continue

            score = 1.0

        else:

            if matched_terms < 2:
                continue

            score = (
                matched_terms
                /
                len(query_terms)
            )

        scored_documents.append(
            (
                document,
                float(score),
            )
        )

    scored_documents.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return scored_documents[:limit]


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

def retrieve_question_context(
    vector_store,
    query: str,
    k: int = TOP_K,
    fetch_k: int = MMR_FETCH_K,
    lambda_mult: float = MMR_LAMBDA,
    max_distance: float = MAX_DISTANCE,
):
    """
    Main question-aware retrieval entry point.

    Strategy:

        Overview question
            ↓
        representative video sections

        Otherwise
            ↓
        MMR semantic retrieval
            +
        lexical/name retrieval

        If neither gives sufficient evidence:
            ↓
        return []
    """

    # --------------------------------------------------------
    # 1. Broad overview question
    # --------------------------------------------------------

    if is_overview_question(
        query
    ):

        return retrieve_overview(
            vector_store
        )

    # --------------------------------------------------------
    # 2. Semantic retrieval
    # --------------------------------------------------------

    semantic_results = retrieve_mmr(
        vector_store=vector_store,
        query=query,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
        max_distance=max_distance,
    )

    # --------------------------------------------------------
    # 3. Lexical retrieval
    # --------------------------------------------------------

    lexical_results = lexical_search(
        vector_store=vector_store,
        query=query,
        limit=k,
    )

    # --------------------------------------------------------
    # 4. Merge semantic + lexical results
    # --------------------------------------------------------

    combined_results = []

    seen_chunk_ids = set()

    def add_result(
        document,
        distance,
    ):

        chunk_id = (
            document
            .metadata
            .get("chunk_id")
        )

        identity = (
            chunk_id
            if chunk_id is not None
            else (
                document
                .metadata
                .get("start"),
                document
                .metadata
                .get("end"),
            )
        )

        if identity in seen_chunk_ids:

            return

        seen_chunk_ids.add(
            identity
        )

        combined_results.append(
            (
                document,
                float(distance),
            )
        )

    # Semantic results first.
    for (
        document,
        distance,
    ) in semantic_results:

        add_result(
            document,
            distance,
        )

    # Lexical results next.
    #
    # We use max_distance as a neutral value for these
    # because lexical score is not a FAISS distance.
    for (
        document,
        lexical_score,
    ) in lexical_results:

        add_result(
            document,
            max_distance,
        )

    # --------------------------------------------------------
    # 5. Evidence gate
    # --------------------------------------------------------

    if not combined_results:

        return []

    has_lexical_evidence = (
        len(lexical_results) > 0
    )

    if has_lexical_evidence:

        return expand_retrieval_context(
            vector_store,
            combined_results[:k + 2],
        )

    # No lexical evidence.
    # Require stronger semantic evidence before allowing
    # the LLM to answer.
    semantic_distances = [
        distance
        for (
            _document,
            distance,
        )
        in semantic_results
    ]

    if not semantic_distances:

        return []

    best_distance = min(
        semantic_distances
    )

    strict_semantic_limit = (
        max_distance * 0.92
    )

    if (
        best_distance
        > strict_semantic_limit
    ):

        return []

    return expand_retrieval_context(
        vector_store,
        combined_results[:k],
    )