"""Privacy-conscious user feedback intake for Phase 11.0E.

This module is deliberately outside the RAG answer-generation path. Feedback is
an unverified user signal, never a ground-truth label or an automatic model update.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

from dotenv import load_dotenv
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


VIDEO_ID_PATTERN = r"^[A-Za-z0-9_-]{11}$"
ANSWER_ID_PATTERN = r"^[A-Za-z0-9_-]{16,80}$"
REQUEST_ID_PATTERN = r"^[A-Za-z0-9_-]{8,80}$"
MAX_FEEDBACK_SOURCES = 16


class FeedbackReason(StrEnum):
    """Structured reasons for a not-helpful answer report."""

    ANSWER_INCORRECT_OR_INCOMPLETE = "answer_incorrect_or_incomplete"
    QUESTION_OR_CONTEXT_MISUNDERSTOOD = "question_or_context_misunderstood"
    TRANSCRIPT_MISSING_OR_INACCURATE = "transcript_missing_or_inaccurate"
    SOURCE_OR_TIMESTAMP_ISSUE = "source_or_timestamp_issue"
    LANGUAGE_OR_TRANSLATION_ISSUE = "language_or_translation_issue"
    OTHER = "other"


class FeedbackSource(BaseModel):
    """Source timing metadata. Transcript text is intentionally not accepted."""

    model_config = ConfigDict(extra="forbid")

    source_id: int = Field(ge=0)
    video_id: str = Field(min_length=11, max_length=11, pattern=VIDEO_ID_PATTERN)
    start: float = Field(ge=0, le=86_400)
    end: float = Field(ge=0, le=86_400)
    chunk_ids: list[int] = Field(default_factory=list, max_length=16)

    @field_validator("chunk_ids", mode="before")
    @classmethod
    def validate_chunk_ids(cls, value: Any) -> Any:
        if not isinstance(value, list):
            raise ValueError("chunk_ids must be an array.")
        if any(type(item) is not int or item < 0 for item in value):
            raise ValueError("chunk_ids must contain non-negative integers.")
        if len(value) != len(set(value)):
            raise ValueError("chunk_ids must not contain duplicates.")
        return value

    @model_validator(mode="after")
    def validate_interval(self) -> "FeedbackSource":
        if self.end < self.start:
            raise ValueError("source end must be greater than or equal to start.")
        return self


class FeedbackSubmission(BaseModel):
    """One rating for one rendered answer, with explicitly consented context."""

    model_config = ConfigDict(extra="forbid")

    answer_id: str = Field(min_length=16, max_length=80, pattern=ANSWER_ID_PATTERN)
    video_id: str = Field(min_length=11, max_length=11, pattern=VIDEO_ID_PATTERN)
    helpful: bool
    reason: FeedbackReason | None = None
    comment: str | None = Field(default=None, max_length=1_000)
    share_context: bool = False
    question: str | None = Field(default=None, min_length=1, max_length=1_000)
    answer: str | None = Field(default=None, min_length=1, max_length=12_000)
    sources: list[FeedbackSource] = Field(
        default_factory=list,
        max_length=MAX_FEEDBACK_SOURCES,
    )
    request_id: str | None = Field(
        default=None,
        min_length=8,
        max_length=80,
        pattern=REQUEST_ID_PATTERN,
    )
    model: str | None = Field(default=None, max_length=160)
    retrieval_method: str | None = Field(default=None, max_length=80)
    retrieval_config: dict[str, str | int | float | bool] = Field(
        default_factory=dict,
        max_length=24,
    )

    @field_validator("comment")
    @classmethod
    def normalize_comment(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @model_validator(mode="after")
    def validate_feedback_context(self) -> "FeedbackSubmission":
        if not self.helpful and self.reason is None:
            raise ValueError("a not-helpful rating requires a feedback reason.")

        if self.share_context:
            if not self.question or not self.answer:
                raise ValueError(
                    "sharing context requires both the question and answer."
                )
        elif self.question is not None or self.answer is not None or self.sources:
            raise ValueError(
                "question, answer, and source timings require share_context=true."
            )

        if any(source.video_id != self.video_id for source in self.sources):
            raise ValueError("all feedback sources must match the feedback video_id.")

        return self


class FeedbackReceipt(BaseModel):
    """Acknowledgement returned after a feedback row is persisted."""

    feedback_id: str
    status: Literal["received", "already_received"]


load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

def default_feedback_database_path() -> Path:
    """Resolve the configurable SQLite path without creating it at import time."""
    configured = os.getenv("RAG_FEEDBACK_DB_PATH")
    if configured:
        return Path(configured).expanduser()
    repository_root = Path(__file__).resolve().parents[2]
    return repository_root / "data" / "feedback.sqlite3"


class FeedbackStore:
    """Small SQLite-backed intake store; deployment persistence is a Phase 12 gate."""

    def __init__(self, database_path: str | Path | None = None) -> None:
        self.path = Path(database_path) if database_path is not None else default_feedback_database_path()

    def _connect(self) -> sqlite3.Connection:
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.path), timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS answer_feedback (
                feedback_id TEXT PRIMARY KEY,
                answer_id TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                helpful INTEGER NOT NULL,
                reason TEXT,
                comment TEXT,
                video_id TEXT NOT NULL,
                request_id TEXT,
                context_shared INTEGER NOT NULL,
                question TEXT,
                answer TEXT,
                sources_json TEXT NOT NULL,
                model TEXT,
                retrieval_method TEXT,
                retrieval_config_json TEXT NOT NULL,
                review_status TEXT NOT NULL DEFAULT 'new'
            )
            """
        )
        connection.commit()
        return connection

    def save(self, submission: FeedbackSubmission) -> FeedbackReceipt:
        """Persist one answer rating; repeated submissions for an answer are idempotent."""
        feedback_id = uuid.uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        context_shared = bool(submission.share_context)
        sources = [
            source.model_dump(mode="json") for source in submission.sources
        ]

        values = (
            feedback_id,
            submission.answer_id,
            created_at,
            int(submission.helpful),
            submission.reason.value if submission.reason is not None else None,
            submission.comment,
            submission.video_id,
            submission.request_id,
            int(context_shared),
            submission.question if context_shared else None,
            submission.answer if context_shared else None,
            json.dumps(sources if context_shared else [], ensure_ascii=False),
            submission.model,
            submission.retrieval_method,
            json.dumps(submission.retrieval_config, ensure_ascii=False, sort_keys=True),
        )

        connection = self._connect()
        try:
            with connection:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO answer_feedback (
                        feedback_id, answer_id, created_at, helpful, reason,
                        comment, video_id, request_id, context_shared, question,
                        answer, sources_json, model, retrieval_method,
                        retrieval_config_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
                if cursor.rowcount == 1:
                    return FeedbackReceipt(
                        feedback_id=feedback_id,
                        status="received",
                    )

                existing = connection.execute(
                    "SELECT feedback_id FROM answer_feedback WHERE answer_id = ?",
                    (submission.answer_id,),
                ).fetchone()
                if existing is None:
                    raise RuntimeError("feedback insert was ignored without an existing answer_id.")
                return FeedbackReceipt(
                    feedback_id=str(existing["feedback_id"]),
                    status="already_received",
                )
        finally:
            connection.close()

    def list_records(self) -> list[dict[str, Any]]:
        """Return feedback rows for a local review/export command."""
        if str(self.path) != ":memory:" and not self.path.exists():
            return []

        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT feedback_id, answer_id, created_at, helpful, reason,
                       comment, video_id, request_id, context_shared, question,
                       answer, sources_json, model, retrieval_method,
                       retrieval_config_json, review_status
                FROM answer_feedback
                ORDER BY created_at ASC, feedback_id ASC
                """
            ).fetchall()
            return [
                {
                    "feedback_id": row["feedback_id"],
                    "answer_id": row["answer_id"],
                    "created_at": row["created_at"],
                    "helpful": bool(row["helpful"]),
                    "reason": row["reason"],
                    "comment": row["comment"],
                    "video_id": row["video_id"],
                    "request_id": row["request_id"],
                    "context_shared": bool(row["context_shared"]),
                    "question": row["question"],
                    "answer": row["answer"],
                    "sources": json.loads(row["sources_json"]),
                    "model": row["model"],
                    "retrieval_method": row["retrieval_method"],
                    "retrieval_config": json.loads(row["retrieval_config_json"]),
                    "review_status": row["review_status"],
                }
                for row in rows
            ]
        finally:
            connection.close()

    def summary(self) -> dict[str, Any]:
        """Aggregate feedback without exposing question/answer/comment content."""
        records = self.list_records()
        reasons: dict[str, int] = {}
        statuses: dict[str, int] = {}
        for record in records:
            if record["reason"]:
                reasons[record["reason"]] = reasons.get(record["reason"], 0) + 1
            status = record["review_status"]
            statuses[status] = statuses.get(status, 0) + 1

        return {
            "total_feedback": len(records),
            "helpful": sum(1 for item in records if item["helpful"]),
            "not_helpful": sum(1 for item in records if not item["helpful"]),
            "context_shared": sum(1 for item in records if item["context_shared"]),
            "reason_counts": dict(sorted(reasons.items())),
            "review_status_counts": dict(sorted(statuses.items())),
        }
