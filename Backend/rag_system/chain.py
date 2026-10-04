"""
chain.py

End-to-end LangChain RAG pipeline.

Flow:

Question
    ↓
MMR Retrieval
    ↓
Retrieved Documents
    ↓
Context Formatting
    ↓
ChatPromptTemplate
    ↓
Hugging Face Generation
    ↓
StrOutputParser
    ↓
Final Answer + Sources
"""


import time
import argparse
import re
from typing import Any

from huggingface_hub import InferenceClient

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from .config import (
    HF_MAX_TOKENS,
    HF_MODEL_ID,
    HF_PROVIDER,
    HF_REASONING_EFFORT,
    HF_TEMPERATURE,
    HF_TOKEN,
    MAX_DISTANCE,
    MMR_FETCH_K,
    MMR_LAMBDA,
    SOURCE_MERGE_GAP_SECONDS,
    TOP_K,
    HF_MAX_RETRIES,
    HF_RETRY_DELAY_SECONDS,
)

from .retrieval import (
    extract_video_id,
    retrieve_question_context,
)

from .index_service import load_ready_index


# ============================================================
# CONSTANTS
# ============================================================

FALLBACK_ANSWER = (
    "I couldn't find enough information about that "
    "in the video. It may not be covered by this video."
)


# ============================================================
# 1. SYSTEM INSTRUCTIONS
# ============================================================

SYSTEM_INSTRUCTIONS = """
You are a YouTube video question-answering assistant.

Your task is to answer the user's question using ONLY the
transcript evidence provided in the context.

GROUNDING RULES
---------------

1. Use only information supported by the provided transcript.

2. Do not use outside knowledge.

3. Do not invent facts, names, events, explanations,
   conclusions, or relationships.

4. If the transcript context does not contain enough evidence
   to answer the question, respond exactly with:

   "I couldn't find enough information about that in the video. It may not be covered by this video."

5. The transcript may contain speech-recognition errors,
   incomplete sentences, repetitions, or informal wording.
   Do not silently repair missing facts using outside knowledge.

6. If a statement cannot be supported by the provided context,
   leave it out.

7. Distinguish between what the speaker said and your own
   interpretation. Do not add interpretation unless the
   transcript clearly supports it.

8. Do not make claims about a person's importance, reputation,
   intelligence, brilliance, influence, or impact unless the
   transcript explicitly supports that claim.

ANSWER STYLE
------------

9. Answer the user's actual question directly.

10. Prefer a concise paragraph for a simple factual question.

11. Use bullet points only when the question naturally contains
    multiple distinct points.

12. Do not create empty bullet points.

13. Do not repeat the same information.

14. Use clear, natural language.

15. Use Markdown only when it improves readability.

SOURCE / TIMESTAMP RULES
------------------------

16. Do NOT output source labels such as:
    [SOURCE 1]
    [Source 1]
    Source 1
    [1]

17. Do NOT output citations, references, or timestamp ranges.

18. Do NOT mention the retrieval system, vector database,
    embeddings, prompts, or internal implementation.

19. The application displays video sources and clickable
    timestamps separately. Therefore, the answer itself must
    contain ONLY the substantive answer.

20. Never fabricate a source, timestamp, citation, or reference.

FINAL RULE
----------

21. Return only the answer to the user's question.
"""


# ============================================================
# 2. PROMPT
# ============================================================

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYSTEM_INSTRUCTIONS,
        ),

        (
            "human",
            """
VIDEO CONTEXT
=============

{context}


USER QUESTION
=============

{question}


Answer the user's question using only the transcript evidence
provided above.

Return only the answer.
Do not include source labels, citations, references,
timestamps, or metadata.
""",
        ),
    ]
)


# ============================================================
# 3. TIMESTAMP FORMATTER
# ============================================================

def format_timestamp(
    seconds: float,
) -> str:
    """
    Convert seconds into MM:SS or HH:MM:SS.
    """

    total_seconds = max(
        0,
        int(seconds),
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

def merge_overlapping_results(
    retrieved_results,
    merge_gap_seconds: float = SOURCE_MERGE_GAP_SECONDS,
) -> list[dict]:
    """
    Convert retrieval chunks into chronological source segments.

    Retrieval chunks may overlap because the indexing pipeline
    intentionally uses chunk overlap.

    Example:

        Chunk A: 10s -> 50s
        Chunk B: 40s -> 80s
        Chunk C: 70s -> 100s

    becomes:

        Source 1: 10s -> 100s

    Separate regions remain separate:

        Chunk A: 10s -> 50s
        Chunk B: 80s -> 120s

    becomes:

        Source 1: 10s -> 50s
        Source 2: 80s -> 120s
    """

    if not retrieved_results:
        return []


    # --------------------------------------------------------
    # Convert raw retrieval results into sortable records
    # --------------------------------------------------------

    candidates = []


    for document, distance in retrieved_results:

        metadata = document.metadata


        start = float(
            metadata.get(
                "start",
                0.0,
            )
        )


        end = float(
            metadata.get(
                "end",
                start,
            )
        )


        if end < start:

            start, end = end, start


        candidates.append(
            {
                "documents": [
                    document
                ],

                "start": start,

                "end": end,

                # Keep the strongest/lowest distance for the group.
                "distance": float(distance),
            }
        )


    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    candidates.sort(
        key=lambda item: (
            item["start"],
            item["end"],
        )
    )


    # --------------------------------------------------------
    # Merge overlapping intervals
    # --------------------------------------------------------

    groups = []


    for candidate in candidates:

        if not groups:

            groups.append(
                candidate
            )

            continue


        current = groups[-1]


        # Overlap or permitted small gap.
        overlaps_or_is_close = (
            candidate["start"]
            <=
            current["end"]
            +
            merge_gap_seconds
        )


        if overlaps_or_is_close:

            current["documents"].extend(
                candidate["documents"]
            )


            current["start"] = min(
                current["start"],
                candidate["start"],
            )


            current["end"] = max(
                current["end"],
                candidate["end"],
            )


            current["distance"] = min(
                current["distance"],
                candidate["distance"],
            )


        else:

            groups.append(
                candidate
            )


    return groups


# Add overlap-aware text reconstruction
def merge_overlapping_text(
    texts: list[str],
    min_overlap_chars: int = 40,
    max_overlap_chars: int = 400,
) -> str:
    """
    Reconstruct transcript text from overlapping chunks.

    The indexing pipeline uses overlapping text chunks.
    This function removes duplicated boundary text while
    preserving the original chronological content.

    Example:

        Chunk A:
            "...AlphaGo used reinforcement learning..."

        Chunk B:
            "reinforcement learning...self-play..."

    becomes:

        "...AlphaGo used reinforcement learning...self-play..."
    """

    if not texts:
        return ""

    merged = texts[0].strip()

    for text in texts[1:]:

        current = text.strip()

        if not current:
            continue

        # ----------------------------------------------------
        # Exact suffix/prefix overlap detection
        # ----------------------------------------------------

        max_possible_overlap = min(
            len(merged),
            len(current),
            max_overlap_chars,
        )

        overlap_found = 0

        for overlap_size in range(
            max_possible_overlap,
            min_overlap_chars - 1,
            -1,
        ):

            if (
                merged[-overlap_size:]
                ==
                current[:overlap_size]
            ):

                overlap_found = overlap_size
                break

        # ----------------------------------------------------
        # Merge
        # ----------------------------------------------------

        if overlap_found > 0:

            merged += current[overlap_found:]

        else:

            merged += "\n\n" + current

    return merged


# ============================================================
# 4. FORMAT ONE DOCUMENT
# ============================================================

def format_source_group(
    source_group: dict,
    source_number: int,
) -> str:
    """
    Convert one merged source segment into LLM context.

    A source segment may contain multiple retrieval chunks
    that overlap in time.
    """

    documents = (
        source_group["documents"]
    )


    video_id = (
        documents[0]
        .metadata
        .get(
            "video_id",
            "unknown",
        )
    )


    start = float(
        source_group["start"]
    )


    end = float(
        source_group["end"]
    )


    # --------------------------------------------------------
    # Combine transcript text while avoiding duplicate
    # chunk entries.
    # --------------------------------------------------------

    transcript_parts = []

    seen_chunk_ids = set()

    for document in documents:

        chunk_id = document.metadata.get(
            "chunk_id"
        )

        if chunk_id in seen_chunk_ids:
            continue

        seen_chunk_ids.add(
            chunk_id
        )

        text = (
            document.page_content
            .strip()
        )

        if text:

            transcript_parts.append(
                text
            )

    transcript_text = merge_overlapping_text(
        transcript_parts
    )

    return (
        f"[SOURCE {source_number}]\n"
        f"Video ID: {video_id}\n"
        f"Timestamp: "
        f"{format_timestamp(start)} - "
        f"{format_timestamp(end)}\n"
        f"Transcript:\n"
        f"{transcript_text}"
    )



# ============================================================
# 5. BUILD CONTEXT
# ============================================================

def build_context(
    source_groups: list[dict],
) -> str:
    """
    Convert merged source segments into one structured
    context block for the generation model.
    """

    if not source_groups:

        return (
            "No relevant transcript context was retrieved."
        )


    formatted_sources = []


    for source_number, source_group in enumerate(
        source_groups,
        start=1,
    ):

        formatted_sources.append(
            format_source_group(
                source_group=source_group,
                source_number=source_number,
            )
        )


    return "\n\n".join(
        formatted_sources
    )


# ============================================================
# 6. HUGGING FACE CLIENT
# ============================================================

def create_llm_client() -> InferenceClient:
    """
    Create the Hugging Face InferenceClient.
    """

    if not HF_TOKEN:

        raise RuntimeError(
            "HF_TOKEN was not found.\n\n"
            "Create a .env file in the project root "
            "with:\n\n"
            "HF_TOKEN=hf_your_token_here"
        )


    return InferenceClient(

        provider=HF_PROVIDER,

        api_key=HF_TOKEN,
    )


# ============================================================
# 7. LANGCHAIN PROMPT → HUGGING FACE MESSAGES
# ============================================================

def prompt_to_messages(
    prompt_value,
) -> list[dict[str, str]]:
    """
    Convert LangChain ChatPromptValue into the message format
    expected by Hugging Face Chat Completions.
    """

    messages = []


    for message in prompt_value.messages:

        role = message.type


        ROLE_MAP = {
            "system": "system",
            "human": "user",
            "ai": "assistant",
        }

        try:
            role = ROLE_MAP[message.type]

        except KeyError as error:

            raise RuntimeError(
                "Unsupported LangChain message role: "
                f"{message.type!r}"
            ) from error


        content = message.content


        if not isinstance(
            content,
            str,
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

# response extractor
def extract_huggingface_answer(response) -> str:
    """
    Extract the user-visible answer from a Hugging Face
    Chat Completions response.

    Raises RuntimeError when the response structure is invalid
    or contains no visible answer.
    """

    try:
        choice = response.choices[0]
        message = choice.message
    except (
        AttributeError,
        IndexError,
        TypeError,
    ) as error:

        raise RuntimeError(
            "Hugging Face returned an invalid chat completion response."
        ) from error

    content = getattr(
        message,
        "content",
        None,
    )

    if content is None:
        return ""

    if not isinstance(content, str):
        content = str(content)

    return content.strip()

# capture response diagnostics
def get_huggingface_response_diagnostics(
    response,
) -> dict[str, object]:
    """
    Extract safe diagnostic metadata from a Hugging Face
    Chat Completions response.

    This function intentionally excludes prompt content,
    transcript content, reasoning content, and credentials.
    """

    try:
        choice = response.choices[0]
        message = choice.message

    except (
        AttributeError,
        IndexError,
        TypeError,
    ):

        return {
            "finish_reason": None,
            "content_length": 0,
            "reasoning_length": 0,
        }

    content = getattr(
        message,
        "content",
        None,
    )

    reasoning = getattr(
        message,
        "reasoning",
        None,
    )

    return {
        "finish_reason":
            getattr(
                choice,
                "finish_reason",
                None,
            ),

        "content_length":
            len(content or ""),

        "reasoning_length":
            len(reasoning or ""),
    }


# ============================================================
# 8. OUTPUT SANITIZATION
# ============================================================

def sanitize_generated_answer(answer: str) -> str:
    """Remove accidental source/citation scaffolding from model output."""

    if not answer or not answer.strip():
        return FALLBACK_ANSWER

    cleaned = answer.strip()

    # Keep the API contract defensive if the provider ignores the prompt
    # and echoes internal source labels.
    cleaned = re.sub(
        r"(?im)^\s*\[?source\s+\d+\]?\s*$",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"(?im)^\s*(?:sources?|citations?)\s*:\s*$",
        "",
        cleaned,
    )
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    return cleaned or FALLBACK_ANSWER


# ============================================================
# 9. HUGGING FACE GENERATION
# ============================================================

def generate_with_huggingface(
    client: InferenceClient,
    prompt_value,
) -> str:
    """
    Send the LangChain prompt to Hugging Face and return
    the generated text.

    Empty model responses are retried because the inference
    provider can occasionally return an empty content field.
    Configuration/request errors are translated and raised
    immediately.
    """

    messages = prompt_to_messages(prompt_value)

    for attempt in range(HF_MAX_RETRIES + 1):

        try:

            response = (
                client
                .chat
                .completions
                .create(
                    model=HF_MODEL_ID,
                    messages=messages,
                    max_tokens=HF_MAX_TOKENS,
                    temperature=HF_TEMPERATURE,
                    extra_body={
                        "reasoning_effort": HF_REASONING_EFFORT,
                    },
                )
            )

        except Exception as error:

            error_text = str(error).lower()

            if "model_not_supported" in error_text:

                raise RuntimeError(
                    "Hugging Face could not find an enabled "
                    "Inference Provider for the selected model.\n\n"
                    f"Model: {HF_MODEL_ID}\n"
                    f"Provider policy: {HF_PROVIDER}\n\n"
                    "Check the selected model/provider in Hugging Face."
                ) from error

            if (
                "401" in error_text
                or "unauthorized" in error_text
            ):

                raise RuntimeError(
                    "Hugging Face authentication failed.\n\n"
                    "Check HF_TOKEN in .env."
                ) from error

            if (
                "403" in error_text
                or "forbidden" in error_text
            ):

                raise RuntimeError(
                    "Hugging Face rejected the inference request.\n\n"
                    "Check token permissions and the selected "
                    "model/provider."
                ) from error

            raise RuntimeError(
                f"Hugging Face generation failed: {error}"
            ) from error

        # --------------------------------------------------------
        # Extract visible model answer
        # --------------------------------------------------------

        answer = extract_huggingface_answer(response)

        if answer:
            return sanitize_generated_answer(answer)

        # --------------------------------------------------------
        # Empty response diagnostics
        # --------------------------------------------------------

        diagnostics = get_huggingface_response_diagnostics(
            response
        )

        # --------------------------------------------------------
        # Retry if attempts remain
        # --------------------------------------------------------

        if attempt < HF_MAX_RETRIES:

            time.sleep(
                HF_RETRY_DELAY_SECONDS
            )

            continue

        # --------------------------------------------------------
        # All attempts exhausted
        # --------------------------------------------------------

        raise RuntimeError(
            "The Hugging Face model returned an empty answer "
            f"after {HF_MAX_RETRIES + 1} attempts. "
            f"Diagnostics: {diagnostics}"
        )

# ============================================================
# 9. FORMAT SOURCES
# ============================================================

def format_sources(
    source_groups: list[dict],
) -> list[dict[str, Any]]:
    """
    Convert merged source segments into API-friendly
    metadata.

    source_id corresponds to the SOURCE number given to
    the LLM.
    """

    sources = []


    for source_id, source_group in enumerate(
        source_groups,
        start=1,
    ):

        documents = (
            source_group["documents"]
        )


        video_id = (
            documents[0]
            .metadata
            .get(
                "video_id"
            )
        )


        chunk_ids = []


        for document in documents:

            chunk_id = (
                document.metadata.get(
                    "chunk_id"
                )
            )


            if (
                chunk_id is not None
                and
                chunk_id not in chunk_ids
            ):

                chunk_ids.append(
                    chunk_id
                )


        start = float(
            source_group["start"]
        )


        end = float(
            source_group["end"]
        )


        sources.append(
            {
                "source_id":
                    source_id,

                "chunk_ids":
                    chunk_ids,

                "video_id":
                    video_id,

                "start":
                    start,

                "end":
                    end,

                "duration":
                    max(
                        0.0,
                        end - start,
                    ),

                "distance":
                    float(
                        source_group["distance"]
                    ),
            }
        )


    return sources

# ============================================================
# 10. BUILD THE LANGCHAIN RAG CHAIN
# ============================================================

def build_rag_chain(
    vector_store,
    llm_client: InferenceClient,
):
    """
    Build the executable LangChain RAG pipeline.

    Flow:

    Input
        ↓
    MMR retrieval
        ↓
    Retrieved chunks
        ↓
    Merge overlapping chunks into source segments
        ↓
    Build context
        ↓
    Prompt
        ↓
    Hugging Face
        ↓
    Parsed answer
    """

    # --------------------------------------------------------
    # 1. Retrieval
    # --------------------------------------------------------

    retrieval_runnable = RunnableLambda(
    lambda inputs: retrieve_question_context(
        vector_store=vector_store,
        query=inputs["question"],
        k=TOP_K,
        fetch_k=MMR_FETCH_K,
        lambda_mult=MMR_LAMBDA,
        max_distance=MAX_DISTANCE,
    )
)


    # --------------------------------------------------------
    # 2. Group overlapping retrieval chunks
    # --------------------------------------------------------

    source_group_runnable = RunnableLambda(
        lambda inputs: merge_overlapping_results(
            inputs["retrieved_results"]
        )
    )


    # --------------------------------------------------------
    # 3. Build context from source groups
    # --------------------------------------------------------

    context_runnable = RunnableLambda(
        lambda inputs: build_context(
            inputs["source_groups"]
        )
    )


    # --------------------------------------------------------
    # 4. Prepare chain inputs
    # --------------------------------------------------------

    prepared_chain = (

        RunnablePassthrough

        .assign(
            retrieved_results=
                retrieval_runnable
        )

        .assign(
            source_groups=
                source_group_runnable
        )

        .assign(
            context=
                context_runnable
        )
    )


    # --------------------------------------------------------
    # 5. Generation chain
    # --------------------------------------------------------

    generation_chain = (

        RAG_PROMPT

        |

        RunnableLambda(
            lambda prompt_value:
                generate_with_huggingface(
                    client=llm_client,
                    prompt_value=prompt_value,
                )
        )

        |

        StrOutputParser()
    )


    # --------------------------------------------------------
    # 6. Answer branch
    # --------------------------------------------------------

    def generate_or_fallback(
        inputs,
    ) -> str:

        if not inputs["retrieved_results"]:

            return FALLBACK_ANSWER


        return generation_chain.invoke(
            inputs
        )


    # --------------------------------------------------------
    # 7. Final chain
    # --------------------------------------------------------

    final_chain = (

        prepared_chain

        |

        RunnableParallel(

            answer=
                RunnableLambda(
                    generate_or_fallback
                ),

            retrieved_results=
                RunnableLambda(
                    lambda inputs:
                        inputs[
                            "retrieved_results"
                        ]
                ),

            source_groups=
                RunnableLambda(
                    lambda inputs:
                        inputs[
                            "source_groups"
                        ]
                ),
        )
    )


    return final_chain


# ============================================================
# 11. PUBLIC RAG FUNCTION
# ============================================================

def answer_question(
    video_reference: str,
    question: str,
) -> dict:
    """
    Main application entry point.

    Performs:

        video reference → video ID
        ↓
        cached FAISS
        ↓
        MMR retrieval
        ↓
        LangChain prompt
        ↓
        Hugging Face generation
        ↓
        answer + source metadata
    """

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not video_reference:

        raise ValueError(
            "video_reference cannot be empty."
        )


    if not question or not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )


    question = question.strip()


    # --------------------------------------------------------
    # Video ID
    # --------------------------------------------------------

    video_id = extract_video_id(
        video_reference
    )


    # --------------------------------------------------------
    # Chat requires an index prepared through POST /index.
    # It must not perform expensive transcript indexing inline.
    # --------------------------------------------------------

    vector_store = load_ready_index(
        video_id
    )


    # --------------------------------------------------------
    # Create Hugging Face client
    # --------------------------------------------------------

    llm_client = create_llm_client()


    # --------------------------------------------------------
    # Build chain
    # --------------------------------------------------------

    rag_chain = build_rag_chain(

        vector_store=vector_store,

        llm_client=llm_client,
    )


    # --------------------------------------------------------
    # Invoke chain
    # --------------------------------------------------------

    result = rag_chain.invoke(

        {
            "question": question,
            "video_id": video_id,
        }
    )


    # --------------------------------------------------------
    # Retrieved results
    # --------------------------------------------------------

    retrieved_results = result[
        "retrieved_results"
    ]

    source_groups = result[
    "source_groups"
    ]

    sources = format_sources(
    source_groups
    )

    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    return {
    "answer":
        result["answer"],

    "video_id":
        video_id,

    "sources":
        sources,

    "retrieved_chunks":
        len(retrieved_results),

    "source_segments":
        len(source_groups),

    "model":
        HF_MODEL_ID,

    "retrieval_method": 
        "question_aware_mmr_lexical",

    "retrieval_config":
        {
            "top_k":
                TOP_K,

            "fetch_k":
                MMR_FETCH_K,

            "lambda_mult":
                MMR_LAMBDA,

            "max_distance":
                MAX_DISTANCE,
        },
}


# ============================================================
# 12. COMMAND-LINE TEST
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run the complete YouTube RAG chain."
        )
    )


    parser.add_argument(
        "video",
        help=(
            "YouTube video ID or YouTube URL."
        ),
    )


    parser.add_argument(
        "question",
        help=(
            "Question to ask about the video."
        ),
    )


    args = parser.parse_args()


    result = answer_question(

        video_reference=args.video,

        question=args.question,
    )


    print(
        "\n" + "=" * 80
    )

    print(
        "ANSWER"
    )

    print(
        "=" * 80
    )

    print(
        f"\n{result['answer']}"
    )


    print(
        "\nSOURCES"
    )


    for source in result["sources"]:

        print(
            f"- Source {source['source_id']}: "
            f"{format_timestamp(source['start'])}"
            f" → "
            f"{format_timestamp(source['end'])} "
            f"(chunks: {source['chunk_ids']})"
        )


if __name__ == "__main__":

    main()