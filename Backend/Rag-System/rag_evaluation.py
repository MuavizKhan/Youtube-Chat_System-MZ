"""
rag_evaluation.py

RAG Evaluation - MMR Retrieval

This script evaluates retrieval quality for a YouTube
video using Maximum Marginal Relevance (MMR).

It does NOT call the generation model.

That means this stage does not consume Hugging Face
generation inference.

Evaluation flow:

Question
   ↓
Query embedding
   ↓
Candidate retrieval
   ↓
MMR selection
   ↓
Distance filtering
   ↓
Top-K chunks
   ↓
Inspect:
    - relevance
    - diversity
    - timestamps
    - distance
    - retrieved text
"""


from retriever import (
    create_embedding_model,
    extract_video_id,
    load_vector_store,
    retrieve_mmr,
)


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_REFERENCE = (
    "https://www.youtube.com/watch?v=MdeQMVBuGgY"
)

TOP_K = 4

MAX_DISTANCE = 1.30

MMR_FETCH_K = 10
MMR_LAMBDA = 0.7


# ============================================================
# EVALUATION QUESTIONS
# ============================================================

TEST_QUESTIONS = [

    # --------------------------------------------------------
    # Direct factual
    # --------------------------------------------------------

    {
        "id": "Q01",
        "type": "direct_fact",
        "question": (
            "What is Kingfisher Airlines?"
        ),
    },

    {
        "id": "Q02",
        "type": "direct_fact",
        "question": (
            "What connection does Vijay Mallya discuss "
            "with Formula One?"
        ),
    },

    {
        "id": "Q03",
        "type": "direct_fact",
        "question": (
            "What businesses or companies does Vijay Mallya "
            "discuss in the interview?"
        ),
    },


    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    {
        "id": "Q04",
        "type": "explanation",
        "question": (
            "Why does Vijay Mallya say Kingfisher Airlines "
            "struggled or failed?"
        ),
    },

    {
        "id": "Q05",
        "type": "explanation",
        "question": (
            "How does Vijay Mallya describe the challenges "
            "he faced while running Kingfisher Airlines?"
        ),
    },

    {
        "id": "Q06",
        "type": "explanation",
        "question": (
            "What does Vijay Mallya say about building "
            "the Kingfisher brand?"
        ),
    },


    # --------------------------------------------------------
    # Specific detail
    # --------------------------------------------------------

    {
        "id": "Q07",
        "type": "specific_detail",
        "question": (
            "What does Vijay Mallya say about the amount "
            "of debt or money involved in the Kingfisher "
            "situation?"
        ),
    },

    {
        "id": "Q08",
        "type": "specific_detail",
        "question": (
            "What does Vijay Mallya say about banks and "
            "the recovery of money?"
        ),
    },

    {
        "id": "Q09",
        "type": "specific_detail",
        "question": (
            "What does Vijay Mallya say about the role of "
            "Indian government policy in the problems "
            "faced by Kingfisher Airlines?"
        ),
    },


    # --------------------------------------------------------
    # Multi-part
    # --------------------------------------------------------

    {
        "id": "Q10",
        "type": "multi_part",
        "question": (
            "What does Vijay Mallya say about the failure "
            "of Kingfisher Airlines, and what reasons does "
            "he give for it?"
        ),
    },


    # --------------------------------------------------------
    # Temporal
    # --------------------------------------------------------

    {
        "id": "Q11",
        "type": "temporal",
        "question": (
            "Where in the video does Vijay Mallya discuss "
            "Kingfisher Airlines, and what is discussed "
            "there?"
        ),
    },


    # --------------------------------------------------------
    # Negative / unanswerable
    # --------------------------------------------------------

    {
        "id": "Q12",
        "type": "unanswerable",
        "question": (
            "What programming language does Vijay Mallya "
            "say is his favorite?"
        ),
    },

    {
        "id": "Q13",
        "type": "unrelated",
        "question": (
            "What is the capital of Australia?"
        ),
    },
]


# ============================================================
# 1. LOAD VIDEO INDEX
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "RAG RETRIEVAL EVALUATION"
)

print(
    "=" * 80
)


video_id = extract_video_id(
    VIDEO_REFERENCE
)


print(
    f"\nVideo ID: {video_id}"
)


# ------------------------------------------------------------
# Load embedding model ONCE
# ------------------------------------------------------------

print(
    "\nLoading embedding model..."
)


embeddings = (
    create_embedding_model()
)


# ------------------------------------------------------------
# Load vector store ONCE
# ------------------------------------------------------------

print(
    "Loading FAISS vector store..."
)


vector_store = (
    load_vector_store(

        video_id,

        embeddings
    )
)


# ============================================================
# 2. RUN EVALUATION
# ============================================================

evaluation_results = []


for test_case in TEST_QUESTIONS:

    question_id = test_case["id"]

    question_type = test_case["type"]

    question = test_case["question"]


    print(
        "\n"
        + "#" * 80
    )

    print(
        f"{question_id} | {question_type}"
    )

    print(
        f"QUESTION: {question}"
    )

    print(
        "#" * 80
    )


    # --------------------------------------------------------
    # Retrieve candidates
    # --------------------------------------------------------

    results = retrieve_mmr(

    vector_store=vector_store,

    query=question,

    k=TOP_K,

    fetch_k=MMR_FETCH_K,

    lambda_mult=MMR_LAMBDA,

    max_distance=MAX_DISTANCE
)


    # --------------------------------------------------------
    # No retrieval
    # --------------------------------------------------------

    if not results:

        print(
            "\nNO RELEVANT DOCUMENTS RETRIEVED."
        )


        evaluation_results.append(
            {
                "id": question_id,

                "type": question_type,

                "question": question,

                "retrieved": 0,

                "relevant": None,
            }
        )


        continue


    # --------------------------------------------------------
    # Display retrieved chunks
    # --------------------------------------------------------

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
            f"Distance: "
            f"{float(distance):.4f}"
        )


        print(
            f"Chunk: "
            f"{metadata.get('chunk_id')}"
        )


        print(
            f"Timestamp: "
            f"{metadata.get('start', 0):.2f}s"
            f" → "
            f"{metadata.get('end', 0):.2f}s"
        )


        print(
            "\nTEXT:"
        )


        print(
            document.page_content
        )


    # --------------------------------------------------------
    # Record result
    # --------------------------------------------------------

    evaluation_results.append(
        {
            "id": question_id,

            "type": question_type,

            "question": question,

            "retrieved": len(results),

            "relevant": None,
        }
    )


# ============================================================
# 3. SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "RETRIEVAL EVALUATION SUMMARY"
)

print(
    "=" * 80
)


for result in evaluation_results:

    print(
        f"\n{result['id']} | "
        f"{result['type']}"
    )

    print(
        f"Retrieved: "
        f"{result['retrieved']}"
    )