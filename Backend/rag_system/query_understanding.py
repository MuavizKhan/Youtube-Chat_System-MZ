"""
query_understanding.py

Adaptive query understanding for retrieval and follow-up resolution.

The normal retrieval path remains the fast path for ordinary questions. This
module can also use a short conversation history to resolve follow-up
references such as "they", "them", "this", or "that" before retrieval.
"""

from __future__ import annotations

import json
import re
from typing import Any

from huggingface_hub import InferenceClient
from pydantic import BaseModel, Field


class QueryUnderstandingError(RuntimeError):
    """Raised when the query-understanding model returns unusable output."""


class QueryPlan(BaseModel):
    """Structured representation of an unfamiliar user question."""

    intent: str = "general"
    standalone_question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
    )
    search_queries: list[str] = Field(
        default_factory=list,
        max_length=3,
    )


SUPPORTED_INTENTS = {
    "overview",
    "factual",
    "causal",
    "comparison",
    "entity",
    "temporal",
    "follow_up",
    "opinion",
    "general",
}


QUERY_UNDERSTANDING_SYSTEM_PROMPT = """
You are a query-understanding component for a YouTube transcript RAG system.

Your job is NOT to answer the user's question.

Instead, transform the user's wording into a small search plan that another
retrieval system can use to find evidence in the video's transcript.

Return JSON with exactly these fields:
- intent: one of overview, factual, causal, comparison, entity, temporal,
  follow_up, opinion, general
- standalone_question: a clear version of the user's question suitable for
  retrieval
- search_queries: up to 3 short alternative search queries

Rules:
1. Preserve the user's meaning.
2. Do not invent names, events, facts, relationships, or conclusions.
3. When conversation history is provided, use it only to resolve references
   in the current question.
4. Never invent a referent when the conversation history does not establish
   one. Preserve the ambiguity instead.
5. Do not treat conversation history as transcript evidence.
6. Search queries should focus on concepts likely to appear in a transcript.
7. For broad overview requests, set intent to "overview".
8. Do not answer the question.
9. Return JSON only.
"""


def _format_conversation_history(
    conversation_history: list[dict[str, str]] | None,
) -> str:
    """Format a bounded conversation history for the query planner."""

    if not conversation_history:
        return "No previous conversation."

    lines: list[str] = []

    for turn in conversation_history[-6:]:
        role = str(turn.get("role", "")).strip().lower()
        content = str(turn.get("content", "")).strip()

        if role not in {"user", "assistant"} or not content:
            continue

        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content}")

    return "\n".join(lines) or "No previous conversation."


def is_likely_follow_up(question: str) -> bool:
    """Return True for lightweight linguistic signals of a follow-up."""

    normalized = " ".join(question.lower().split())

    if not normalized:
        return False

    follow_up_patterns = (
        r"\bthey\b",
        r"\bthem\b",
        r"\btheir\b",
        r"\bhe\b",
        r"\bshe\b",
        r"\bhis\b",
        r"\bher\b",
        r"\bit\b",
        r"\bthis\b",
        r"\bthat\b",
        r"\bthese\b",
        r"\bthose\b",
        r"\bhere\b",
        r"\bthere\b",
    )

    return any(
        re.search(pattern, normalized)
        for pattern in follow_up_patterns
    )


def _extract_response_content(response: Any) -> str:
    try:
        choice = response.choices[0]
        content = choice.message.content
    except (AttributeError, IndexError, TypeError) as error:
        raise QueryUnderstandingError(
            "Hugging Face returned an invalid query-understanding response."
        ) from error

    if content is None:
        return ""

    if not isinstance(content, str):
        content = str(content)

    return content.strip()


def _extract_json_object(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    fence = "\x60\x60\x60"

    if cleaned.startswith(fence):
        cleaned = re.sub(
            r"^\x60\x60\x60(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\s*\x60\x60\x60$",
            "",
            cleaned,
        )

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)

    if not match:
        raise QueryUnderstandingError(
            "Query-understanding model did not return a JSON object."
        )

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as error:
        raise QueryUnderstandingError(
            "Query-understanding model returned invalid JSON."
        ) from error

    if not isinstance(parsed, dict):
        raise QueryUnderstandingError(
            "Query-understanding model returned a JSON value instead of an object."
        )

    return parsed


def _normalize_query_plan(plan: QueryPlan) -> QueryPlan:
    intent = plan.intent.strip().lower()

    if intent not in SUPPORTED_INTENTS:
        intent = "general"

    standalone_question = " ".join(
        plan.standalone_question.split()
    ).strip()

    queries: list[str] = []
    seen: set[str] = set()

    candidates = [
        standalone_question,
        *plan.search_queries,
    ]

    for query in candidates:
        normalized = " ".join(query.split()).strip()
        key = normalized.casefold()

        if not normalized or key in seen:
            continue

        seen.add(key)
        queries.append(normalized)

        if len(queries) >= 3:
            break

    if not standalone_question:
        raise QueryUnderstandingError(
            "Query-understanding model returned an empty standalone question."
        )

    return QueryPlan(
        intent=intent,
        standalone_question=standalone_question,
        search_queries=queries,
    )


def understand_query(
    client: InferenceClient,
    question: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> QueryPlan:
    """
    Ask the Hugging Face model to convert an unfamiliar question into a
    structured retrieval plan.

    This function intentionally does not call the vector store. It is only
    responsible for understanding, resolving short follow-ups, and
    reformulating the user's wording.
    """

    question = question.strip()

    if not question:
        raise ValueError("question cannot be empty.")

    from .config import HF_REASONING_EFFORT

    response = client.chat.completions.create(
        model=_get_model_id(),
        messages=[
            {
                "role": "system",
                "content": QUERY_UNDERSTANDING_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    "CONVERSATION HISTORY\n"
                    "====================\n\n"
                    f"{_format_conversation_history(conversation_history)}\n\n"
                    "CURRENT QUESTION\n"
                    "=================\n\n"
                    f"{question}"
                ),
            },
        ],
        max_tokens=256,
        temperature=0.0,
        extra_body={
            "reasoning_effort": HF_REASONING_EFFORT,
        },
    )

    content = _extract_response_content(response)

    if not content:
        raise QueryUnderstandingError(
            "Query-understanding model returned an empty response."
        )

    raw_plan = _extract_json_object(content)

    try:
        plan = QueryPlan.model_validate(raw_plan)
    except Exception as error:
        raise QueryUnderstandingError(
            "Query-understanding model returned an invalid query plan."
        ) from error

    return _normalize_query_plan(plan)


def _get_model_id() -> str:
    """
    Import the configured model lazily so unit tests can exercise this module
    without importing the application configuration at module import time.
    """

    from .config import HF_MODEL_ID

    if not HF_MODEL_ID:
        raise QueryUnderstandingError(
            "HF_MODEL_ID is not configured."
        )

    return HF_MODEL_ID
