"""
retriever.py

RAG - Retrieval Stage

Pipeline:

YouTube URL / Video ID
        ↓
Extract Video ID
        ↓
Load FAISS Vector Store
        ↓
Embed User Query
        ↓
Candidate Search
        ↓
MMR Selection
        ↓
Distance Filtering
        ↓
Relevant Documents
"""


from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

VECTOR_STORE_ROOT = (
    Path(__file__).resolve().parent
    / "vector_stores"
)

# Number of final chunks returned by retrieval.
DEFAULT_K = 4

# Number of candidate chunks considered before MMR
# selects the final results.
DEFAULT_MMR_FETCH_K = 10

# MMR relevance/diversity balance.
#
# 1.0 = prioritize relevance
# 0.0 = prioritize diversity
#
# 0.5 = balanced starting point.
DEFAULT_MMR_LAMBDA = 0.5

# Experimental distance threshold for the current setup.
#
# FAISS returns raw distances:
# lower = more similar
#
# This is NOT a universal semantic relevance score.
MAX_DISTANCE = 1.30


# ============================================================
# 1. EXTRACT VIDEO ID
# ============================================================

def extract_video_id(
    video_reference: str
) -> str:

    if not video_reference:
        raise ValueError(
            "Video reference cannot be empty."
        )

    video_reference = video_reference.strip()

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

        params = parse_qs(
            parsed_url.query
        )

        video_ids = params.get("v")

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
        f"Could not extract YouTube video ID from:\n"
        f"{video_reference}"
    )


# ============================================================
# 2. CREATE EMBEDDING MODEL
# ============================================================

def create_embedding_model():

    return HuggingFaceEmbeddings(

        model_name=EMBEDDING_MODEL,

        model_kwargs={
            "device": "cpu"
        },

        encode_kwargs={
            "normalize_embeddings": True
        }
    )


# ============================================================
# 3. LOAD VECTOR STORE
# ============================================================

def load_vector_store(
    video_id: str,
    embeddings
):

    vector_store_path = (
        VECTOR_STORE_ROOT
        / video_id
    )

    if not vector_store_path.exists():

        raise FileNotFoundError(
            "\nNo vector store found for video:\n"
            f"{video_id}\n\n"
            "Expected location:\n"
            f"{vector_store_path}\n\n"
            "Run indexing.py first."
        )

    return FAISS.load_local(

        str(vector_store_path),

        embeddings,

        allow_dangerous_deserialization=True
    )


# ============================================================
# 4. RAW SIMILARITY SEARCH
# ============================================================

def similarity_search(
    vector_store,
    query: str,
    k: int = DEFAULT_K
):
    """
    Return top-k chunks together with their raw FAISS
    query-document distances.
    """

    if not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    return (
        vector_store
        .similarity_search_with_score(
            query,
            k=k
        )
    )


# ============================================================
# 5. DISTANCE FILTER
# ============================================================

def filter_by_distance(
    results,
    max_distance: float = MAX_DISTANCE
):
    """
    Keep only documents whose FAISS distance is <=
    max_distance.

    Lower distance = more similar.

    IMPORTANT:
    This is an experimental threshold, not a universal
    semantic relevance score.
    """

    filtered_results = [

        (
            document,
            float(distance)
        )

        for document, distance in results

        if float(distance) <= max_distance
    ]

    return filtered_results


# ============================================================
# 6. DISPLAY RESULTS
# ============================================================

def display_results(
    results,
    max_distance: float
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FILTERED RETRIEVAL RESULTS"
    )

    print(
        "=" * 70
    )

    if not results:

        print(
            "\nNo sufficiently similar chunks were found."
        )

        print(
            f"Maximum allowed distance: "
            f"{max_distance:.3f}"
        )

        return

    for rank, (
        document,
        distance
    ) in enumerate(
        results,
        start=1
    ):

        metadata = (
            document.metadata
        )

        print(
            f"\n--- Result {rank} ---"
        )

        print(
            f"Distance: {distance:.6f}"
        )

        print(
            f"Chunk ID: "
            f"{metadata.get('chunk_id')}"
        )

        print(
            f"Timestamp: "
            f"{metadata.get('start', 0):.2f}s"
            f" → "
            f"{metadata.get('end', 0):.2f}s"
        )

        print(
            f"Duration: "
            f"{metadata.get('duration', 0):.2f}s"
        )

        print(
            "\nText:\n"
            f"{document.page_content}"
        )


# ============================================================
# 7. STANDARD RETRIEVAL PIPELINE
# ============================================================

def retrieve(
    vector_store,
    query: str,
    k: int = DEFAULT_K,
    max_distance: float = MAX_DISTANCE
):
    """
    Standard similarity-based retrieval followed by
    distance filtering.
    """

    raw_results = similarity_search(
        vector_store,
        query,
        k=k
    )

    filtered_results = filter_by_distance(
        raw_results,
        max_distance=max_distance
    )

    return filtered_results


# ============================================================
# 8. MMR RETRIEVAL
# ============================================================

def retrieve_mmr(
    vector_store,
    query: str,
    k: int = DEFAULT_K,
    fetch_k: int = DEFAULT_MMR_FETCH_K,
    lambda_mult: float = DEFAULT_MMR_LAMBDA,
    max_distance: float = MAX_DISTANCE,
):
    """
    Retrieve documents using Maximum Marginal Relevance.

    MMR balances:

        relevance to the query
                  +
        diversity between retrieved chunks

    Process:

        Query
          ↓
        Query embedding
          ↓
        Fetch candidate chunks
          ↓
        Distance threshold
          ↓
        MMR selection
          ↓
        Final documents

    This implementation is intentionally independent of
    LangChain's optional MMR-with-score method because the
    installed FAISS wrapper does not expose that method.
    """

    # --------------------------------------------------------
    # Validate query
    # --------------------------------------------------------

    if not query.strip():

        raise ValueError(
            "Query cannot be empty."
        )

    # --------------------------------------------------------
    # Validate MMR parameters
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
        .embed_query(query)
    )

    query_vector = np.asarray(
        query_embedding,
        dtype=np.float32
    ).reshape(1, -1)

    # --------------------------------------------------------
    # 2. Determine candidate count
    #
    #    Don't request more vectors than actually exist.
    # --------------------------------------------------------

    total_vectors = (
        vector_store.index.ntotal
    )

    if total_vectors == 0:

        return []

    candidate_k = min(
        fetch_k,
        total_vectors
    )

    # --------------------------------------------------------
    # 3. Search FAISS for candidate chunks
    #
    #    FAISS requires a NumPy array with shape:
    #
    #        (number_of_queries, embedding_dimension)
    # --------------------------------------------------------

    distances, indices = (
        vector_store.index.search(
            np.ascontiguousarray(
                query_vector
            ),
            candidate_k
        )
    )

    candidate_indices = (
        indices[0]
    )

    candidate_distances = (
        distances[0]
    )

    # --------------------------------------------------------
    # 4. Keep only candidates inside our validated
    #    distance threshold.
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

    # --------------------------------------------------------
    # No candidates passed the threshold.
    # --------------------------------------------------------

    if not eligible_positions:

        return []

    # --------------------------------------------------------
    # 5. Reconstruct candidate embeddings
    #
    #    These are the actual vectors stored in FAISS.
    # --------------------------------------------------------

    faiss_candidate_ids = (
        [
            int(candidate_indices[position])
            for position in eligible_positions
        ]
    )

    if hasattr(
        vector_store.index,
        "reconstruct_batch"
    ):

        candidate_embeddings = (
            vector_store.index.reconstruct_batch(
                np.asarray(
                    faiss_candidate_ids,
                    dtype=np.int64
                )
            )
        )

    else:

        candidate_embeddings = np.vstack(
            [
                vector_store.index.reconstruct(
                    vector_id
                )

                for vector_id
                in faiss_candidate_ids
            ]
        )

    candidate_embeddings = np.asarray(
        candidate_embeddings,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # 6. Normalize embeddings for cosine similarity
    #
    #    Our indexing pipeline already normalizes embeddings,
    #    but doing it again here makes the MMR calculation
    #    explicit and robust.
    # --------------------------------------------------------

    query_norm = np.linalg.norm(
        query_vector,
        axis=1,
        keepdims=True
    )

    query_norm = np.maximum(
        query_norm,
        1e-12
    )

    query_normalized = (
        query_vector
        / query_norm
    )

    candidate_norms = np.linalg.norm(
        candidate_embeddings,
        axis=1,
        keepdims=True
    )

    candidate_norms = np.maximum(
        candidate_norms,
        1e-12
    )

    candidate_normalized = (
        candidate_embeddings
        / candidate_norms
    )

    # --------------------------------------------------------
    # 7. Calculate query-to-candidate similarity
    # --------------------------------------------------------

    query_similarities = (
        candidate_normalized
        @ query_normalized.T
    ).reshape(-1)

    # --------------------------------------------------------
    # 8. MMR selection
    #
    #    MMR(candidate) =
    #
    #        lambda * relevance
    #
    #        -
    #
    #        (1 - lambda) * redundancy
    # --------------------------------------------------------

    number_to_select = min(
        k,
        len(
            eligible_positions
        )
    )

    selected_positions = []

    remaining_positions = list(
        range(
            len(
                eligible_positions
            )
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

    # Remaining selections:
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

            # Similarity to the most similar
            # already-selected document.
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

        selected_positions.append(
            best_position
        )

        remaining_positions.remove(
            best_position
        )

    # --------------------------------------------------------
    # 9. Convert selected FAISS vectors back to Documents
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
            .search(docstore_id)
        )

        if document is None:
            continue

        results.append(
            (
                document,
                distance
            )
        )

    return results


# ============================================================
# 9. MAIN TEST
# ============================================================

if __name__ == "__main__":

    video = "Gfr50f6ZBvo"

    questions = [

        "What does Demis Hassabis think about the Turing test?",

        "What is DeepMind?",

        "What is AlphaFold?",

        "How does Demis Hassabis describe artificial intelligence?",

        "What is the capital of Australia?"
    ]

    # --------------------------------------------------------
    # Load embedding model ONCE
    # --------------------------------------------------------

    print(
        "\nLoading embedding model..."
    )

    embeddings = (
        create_embedding_model()
    )

    # --------------------------------------------------------
    # Extract video ID ONCE
    # --------------------------------------------------------

    video_id = extract_video_id(
        video
    )

    # --------------------------------------------------------
    # Load FAISS ONCE
    # --------------------------------------------------------

    print(
        "Loading FAISS vector store..."
    )

    vector_store = (
        load_vector_store(
            video_id,
            embeddings
        )
    )

    # --------------------------------------------------------
    # Process questions
    # --------------------------------------------------------

    for question in questions:

        print(
            "\n"
            + "#"
            * 80
        )

        print(
            f"QUESTION: {question}"
        )

        print(
            "#"
            * 80
        )

        results = retrieve_mmr(

            vector_store=vector_store,

            query=question,

            k=DEFAULT_K,

            fetch_k=DEFAULT_MMR_FETCH_K,

            lambda_mult=DEFAULT_MMR_LAMBDA,

            max_distance=MAX_DISTANCE
        )

        display_results(
            results,
            MAX_DISTANCE
        )