"""
generation.py

YouTube RAG - Generation Stage

Complete RAG flow:

YouTube URL / Video ID
        ↓
        Retrieval
        ↓
Relevant Documents
        ↓
        Augmentation
        ↓
Structured Chat Prompt
        ↓
Hugging Face InferenceClient
        ↓
        Generation
        ↓
Final Answer + Sources


Responsibilities
----------------
This file is responsible for:

1. Loading Hugging Face configuration
2. Loading the embedding model
3. Loading the video's FAISS vector store
4. Retrieving relevant transcript chunks using MMR
5. Building the augmented prompt
6. Calling the Hugging Face generation model
7. Returning the final answer
8. Returning source/timestamp information

This file does NOT create the vector store.

Indexing belongs to:
    indexing.py

Retrieval belongs to:
    retriever.py

Prompt construction belongs to:
    augmentation.py
"""


# ============================================================
# IMPORTS
# ============================================================

import os
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from retriever import (
    create_embedding_model,
    extract_video_id,
    load_vector_store,
    retrieve_mmr,
)

from augmentation import (
    create_augmented_prompt,
)


# ============================================================
# 1. LOAD ENVIRONMENT
# ============================================================

load_dotenv()


HF_TOKEN = os.getenv(
    "HF_TOKEN"
)


if not HF_TOKEN:

    raise EnvironmentError(
        "\nHF_TOKEN was not found.\n\n"
        "Create a .env file in the project root:\n\n"
        "HF_TOKEN=hf_your_token_here\n\n"
        "Then run generation.py again."
    )


# ============================================================
# 2. MODEL CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Hugging Face generation model
# ------------------------------------------------------------

MODEL_ID = os.getenv(
    "HF_MODEL_ID",
    "openai/gpt-oss-20b"
)


# ------------------------------------------------------------
# Hugging Face provider policy
# ------------------------------------------------------------

HF_PROVIDER = os.getenv(
    "HF_PROVIDER",
    "auto"
)


# ------------------------------------------------------------
# Maximum answer length
# ------------------------------------------------------------

MAX_TOKENS = int(
    os.getenv(
        "HF_MAX_TOKENS",
        "500"
    )
)


# ------------------------------------------------------------
# Low temperature for grounded RAG answers
# ------------------------------------------------------------

TEMPERATURE = float(
    os.getenv(
        "HF_TEMPERATURE",
        "0.1"
    )
)


# ============================================================
# 3. RETRIEVAL CONFIGURATION
# ============================================================

# Number of final chunks returned by MMR.
#
# This matches the retrieval evaluation.
TOP_K = int(
    os.getenv(
        "RAG_TOP_K",
        "4"
    )
)


# Number of candidates considered before MMR
# selects the final chunks.
#
# This matches the retrieval evaluation.
MMR_FETCH_K = int(
    os.getenv(
        "RAG_MMR_FETCH_K",
        "10"
    )
)


# MMR relevance/diversity balance.
#
# 1.0 -> prioritize relevance
# 0.0 -> prioritize diversity
#
# 0.7 is our current experimental value.
MMR_LAMBDA = float(
    os.getenv(
        "RAG_MMR_LAMBDA",
        "0.7"
    )
)


# Experimental FAISS distance threshold.
#
# Lower distance = more similar.
#
# 1.30 is the value currently validated against
# our retrieval evaluation setup.
#
# IMPORTANT:
# This is not a universal threshold for every video.
MAX_DISTANCE = float(
    os.getenv(
        "RAG_MAX_DISTANCE",
        "1.30"
    )
)


# ============================================================
# 4. CREATE HUGGING FACE CLIENT
# ============================================================

def create_llm_client() -> InferenceClient:
    """
    Create the Hugging Face Inference Client.

    provider="auto" lets Hugging Face select an
    available inference provider.
    """

    client = InferenceClient(

        provider=HF_PROVIDER,

        api_key=HF_TOKEN,
    )

    return client


# ============================================================
# 5. CONVERT LANGCHAIN PROMPT TO HF CHAT MESSAGES
# ============================================================

def prompt_to_messages(
    prompt_value
) -> list[dict[str, str]]:
    """
    Convert LangChain's ChatPromptValue into the
    message structure expected by Hugging Face
    Chat Completions.

    LangChain roles:
        system
        human
        ai

    Hugging Face/OpenAI-style roles:
        system
        user
        assistant
    """

    messages = []


    for message in prompt_value.messages:

        role = message.type

        # ----------------------------------------------------
        # LangChain → Chat Completion role mapping
        # ----------------------------------------------------

        if role == "human":

            role = "user"

        elif role == "ai":

            role = "assistant"


        content = message.content


        # ----------------------------------------------------
        # Normalize content
        # ----------------------------------------------------

        if not isinstance(
            content,
            str
        ):

            content = str(
                content
            )


        messages.append(
            {
                "role": role,
                "content": content,
            }
        )


    return messages


# ============================================================
# 6. CALL HUGGING FACE GENERATION MODEL
# ============================================================

def generate_with_huggingface(
    client: InferenceClient,
    prompt_value,
) -> Any:
    """
    Send the augmented prompt to the Hugging Face
    generation model.
    """

    messages = prompt_to_messages(
        prompt_value
    )


    try:

        response = (
            client
            .chat
            .completions
            .create(

                model=MODEL_ID,

                messages=messages,

                max_tokens=MAX_TOKENS,

                temperature=TEMPERATURE,
            )
        )


    except Exception as error:

        error_text = str(
            error
        )


        # ----------------------------------------------------
        # Helpful provider/model error
        # ----------------------------------------------------

        if (
            "model_not_supported"
            in error_text.lower()
        ):

            raise RuntimeError(
                "\nHugging Face could not find an enabled "
                "Inference Provider for the selected model.\n\n"

                f"Model: {MODEL_ID}\n"
                f"Provider policy: {HF_PROVIDER}\n\n"

                "Check the model's available Inference Providers "
                "on Hugging Face or change HF_MODEL_ID in .env."
            ) from error


        # ----------------------------------------------------
        # Authentication error
        # ----------------------------------------------------

        if (
            "401"
            in error_text
            or "unauthorized"
            in error_text.lower()
        ):

            raise RuntimeError(
                "\nHugging Face authentication failed.\n\n"
                "Check that HF_TOKEN exists in .env and that "
                "the token has permission to make Inference "
                "Provider calls."
            ) from error


        # ----------------------------------------------------
        # Permission / access error
        # ----------------------------------------------------

        if (
            "403"
            in error_text
            or "forbidden"
            in error_text.lower()
        ):

            raise RuntimeError(
                "\nHugging Face rejected the inference request.\n\n"
                "Check your token permissions and the selected "
                "model/provider."
            ) from error


        # ----------------------------------------------------
        # Re-raise unknown error
        # ----------------------------------------------------

        raise


    return response


# ============================================================
# 7. EXTRACT FINAL ANSWER
# ============================================================

def extract_answer(
    response
) -> str:
    """
    Extract the generated assistant message.
    """

    try:

        answer = (
            response
            .choices[0]
            .message
            .content
        )

    except (
        AttributeError,
        IndexError,
        TypeError,
    ) as error:

        raise RuntimeError(
            "Could not extract the generated answer "
            "from the Hugging Face response."
        ) from error


    if not answer:

        raise RuntimeError(
            "The Hugging Face model returned an empty answer."
        )


    if isinstance(
        answer,
        str
    ):

        return answer.strip()


    return str(
        answer
    ).strip()


# ============================================================
# 8. FORMAT SOURCE METADATA
# ============================================================

def format_sources(
    retrieved_results
) -> list[dict]:
    """
    Convert:

        (Document, distance)

    into clean source information.

    This metadata will eventually be consumed by
    the Chrome extension.
    """

    sources = []


    for document, distance in (
        retrieved_results
    ):

        metadata = (
            document.metadata
        )


        sources.append(
            {
                "chunk_id": metadata.get(
                    "chunk_id"
                ),

                "video_id": metadata.get(
                    "video_id"
                ),

                "start": metadata.get(
                    "start"
                ),

                "end": metadata.get(
                    "end"
                ),

                "duration": metadata.get(
                    "duration"
                ),

                "distance": float(
                    distance
                ),
            }
        )


    return sources


# ============================================================
# 9. FORMAT TIMESTAMP
# ============================================================

def format_timestamp(
    seconds: float
) -> str:
    """
    Convert seconds into:

        MM:SS

    or:

        HH:MM:SS
    """

    total_seconds = max(
        0,
        int(seconds)
    )


    hours = (
        total_seconds // 3600
    )


    minutes = (
        total_seconds % 3600
    ) // 60


    remaining_seconds = (
        total_seconds % 60
    )


    if hours > 0:

        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{remaining_seconds:02d}"
        )


    return (
        f"{minutes:02d}:"
        f"{remaining_seconds:02d}"
    )


# ============================================================
# 10. DISPLAY SOURCES
# ============================================================

def display_sources(
    sources: list[dict]
):
    """
    Display source timestamps.

    These will later become clickable timestamps
    in the Chrome extension.
    """

    print(
        "\n"
        + "-" * 80
    )


    print(
        "VIDEO SOURCES"
    )


    print(
        "-" * 80
    )


    if not sources:

        print(
            "No sources were retrieved."
        )

        return


    for source in sources:

        start = format_timestamp(
            source["start"]
        )


        end = format_timestamp(
            source["end"]
        )


        print(
            f"Chunk {source['chunk_id']}: "
            f"{start} → {end}"
        )


# ============================================================
# 11. COMPLETE RAG PIPELINE
# ============================================================

def generate_answer(
    video_reference: str,
    question: str,
) -> dict:
    """
    Execute the complete RAG pipeline.

    Flow:

        Video URL / ID
              ↓
           Video ID
              ↓
          Embeddings
              ↓
        FAISS vector store
              ↓
         MMR Retrieval
              ↓
        Distance filtering
              ↓
       Relevant Documents
              ↓
         Augmentation
              ↓
       Hugging Face LLM
              ↓
          Final Answer
              ↓
         Answer + Sources
    """

    # ========================================================
    # VALIDATION
    # ========================================================

    if not video_reference:

        raise ValueError(
            "video_reference cannot be empty."
        )


    if not question or not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )


    question = question.strip()


    # ========================================================
    # HEADER
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )


    print(
        "YOUTUBE RAG GENERATION"
    )


    print(
        "=" * 80
    )


    # ========================================================
    # STEP 1 — VIDEO ID
    # ========================================================

    video_id = extract_video_id(
        video_reference
    )


    print(
        f"\nVideo ID: {video_id}"
    )


    print(
        f"Question: {question}"
    )


    # ========================================================
    # STEP 2 — EMBEDDING MODEL
    # ========================================================

    print(
        "\n[1/5] Loading embedding model..."
    )


    embeddings = (
        create_embedding_model()
    )


    # ========================================================
    # STEP 3 — LOAD VECTOR STORE
    # ========================================================

    print(
        "\n[2/5] Loading FAISS vector store..."
    )


    vector_store = (
        load_vector_store(
            video_id,
            embeddings
        )
    )


    # ========================================================
    # STEP 4 — MMR RETRIEVAL
    # ========================================================

    print(
        "\n[3/5] Retrieving relevant context..."
    )


    print(
        f"MMR configuration: "
        f"k={TOP_K}, "
        f"fetch_k={MMR_FETCH_K}, "
        f"lambda={MMR_LAMBDA}, "
        f"max_distance={MAX_DISTANCE}"
    )


    retrieved_results = retrieve_mmr(

        vector_store=vector_store,

        query=question,

        k=TOP_K,

        fetch_k=MMR_FETCH_K,

        lambda_mult=MMR_LAMBDA,

        max_distance=MAX_DISTANCE,
    )


    # --------------------------------------------------------
    # No relevant context
    # --------------------------------------------------------

    if not retrieved_results:

        fallback_answer = (
            "I couldn't find enough information "
            "about that in the video."
        )


        print(
            "\nNo sufficiently relevant transcript "
            "chunks were found."
        )


        return {

            "answer": fallback_answer,

            "video_id": video_id,

            "sources": [],

            "retrieved_chunks": 0,

            "model": MODEL_ID,

            "retrieval_method": "mmr",

            "retrieval_config": {
                "top_k": TOP_K,
                "fetch_k": MMR_FETCH_K,
                "lambda_mult": MMR_LAMBDA,
                "max_distance": MAX_DISTANCE,
            },
        }


    # --------------------------------------------------------
    # Extract Documents from:
    #
    # [(Document, distance), ...]
    # --------------------------------------------------------

    retrieved_documents = [

        document

        for document, distance
        in retrieved_results

    ]


    print(
        f"Retrieved chunks: "
        f"{len(retrieved_documents)}"
    )


    # ========================================================
    # STEP 5 — AUGMENTATION
    # ========================================================

    print(
        "\n[4/5] Building augmented prompt..."
    )


    prompt_value = (
        create_augmented_prompt(

            question=question,

            retrieved_documents=
                retrieved_documents,
        )
    )


    # ========================================================
    # STEP 6 — GENERATION
    # ========================================================

    print(
        "\n[5/5] Generating answer with "
        "Hugging Face..."
    )


    client = (
        create_llm_client()
    )


    response = (
        generate_with_huggingface(

            client=client,

            prompt_value=prompt_value,
        )
    )


    # --------------------------------------------------------
    # Extract answer
    # --------------------------------------------------------

    answer = extract_answer(
        response
    )


    # --------------------------------------------------------
    # Format source information
    # --------------------------------------------------------

    sources = format_sources(
        retrieved_results
    )

    sources = sorted(
        sources,
        key=lambda source: source["start"]
    )


    # ========================================================
    # FINAL RESULT
    # ========================================================

    return {

        "answer": answer,

        "video_id": video_id,

        "sources": sources,

        "retrieved_chunks":
            len(retrieved_documents),

        "model": MODEL_ID,

        "retrieval_method": "mmr",

        "retrieval_config": {
            "top_k": TOP_K,
            "fetch_k": MMR_FETCH_K,
            "lambda_mult": MMR_LAMBDA,
            "max_distance": MAX_DISTANCE,
        },
    }


# ============================================================
# 12. DISPLAY FINAL RESULT
# ============================================================

def display_result(
    result: dict
):
    """
    Display the final generated answer and
    source timestamps.
    """

    print(
        "\n"
        + "=" * 80
    )


    print(
        "FINAL ANSWER"
    )


    print(
        "=" * 80
    )


    print(
        "\n"
        + result["answer"]
    )


    print(
        f"\nModel: "
        f"{result['model']}"
    )


    print(
        f"Retrieval method: "
        f"{result.get('retrieval_method', 'unknown')}"
    )


    print(
        f"Retrieved chunks: "
        f"{result['retrieved_chunks']}"
    )


    display_sources(
        result["sources"]
    )


# ============================================================
# 13. DEVELOPMENT TEST
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Video ID
    # --------------------------------------------------------

    video = "Gfr50f6ZBvo"


    # --------------------------------------------------------
    # Example question
    # --------------------------------------------------------

    question = (
        "Is the topic of nuclear fusion discussed "
        "in this video? If yes, what was discussed?"
    )


    result = generate_answer(

        video_reference=video,

        question=question,
    )


    display_result(
        result
    )