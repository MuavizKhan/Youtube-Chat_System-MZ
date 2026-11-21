"""Privacy-conscious SQLite storage for user-submitted answer feedback.

The public API only writes feedback. Review and export operations are local CLI
operations so an unauthenticated public client cannot read the feedback queue.
Raw transcript text is never accepted or stored by this module.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sqlite3
import uuid
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DB_PATH = Path(__file__).with_name("feedback.sqlite3")
_CONFIGURED_DB_PATH = Path(
    os.getenv("FEEDBACK_DB_PATH", str(_DEFAULT_DB_PATH))
).expanduser()
FEEDBACK_DB_PATH = (
    _CONFIGURED_DB_PATH
    if _CONFIGURED_DB_PATH.is_absolute()
    else PROJECT_ROOT / _CONFIGURED_DB_PATH
)

REVIEW_STATUSES = (
    "new",
    "triaged",
    "verified_issue",
    "not_reproducible",
    "transcript_issue",
    "needs_context",
    "evaluation_candidate",
    "closed",
)
FEEDBACK_VOTES = ("helpful", "not_helpful")
FEEDBACK_REASONS = (
    "incorrect_or_unsupported",
    "missing_information",
    "wrong_attribution",
    "wrong_timestamp_or_source",
    "transcript_problem",
    "question_context_misunderstood",
    "uncertain_context",
    "other",
)
_REVIEWER_ID_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9_-]{1,31}$")
_MAX_CONTEXT_JSON_CHARS = 24_000


class FeedbackStoreError(ValueError):
    """Base error for invalid feedback-store operations."""


class DuplicateFeedbackError(FeedbackStoreError):
    """Raised when an answer already has a feedback record."""


class FeedbackNotFoundError(FeedbackStoreError):
    """Raised when a requested feedback record does not exist."""


def _connect() -> sqlite3.Connection:
    path = Path(FEEDBACK_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 15000")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS user_feedback (
            feedback_id TEXT PRIMARY KEY,
            request_id TEXT NOT NULL UNIQUE,
            video_id TEXT NOT NULL,
            vote TEXT NOT NULL CHECK (vote IN ('helpful', 'not_helpful')),
            reason TEXT,
            comment TEXT,
            context_shared INTEGER NOT NULL DEFAULT 0
                CHECK (context_shared IN (0, 1)),
            context_json TEXT,
            review_status TEXT NOT NULL DEFAULT 'new',
            reviewer_id TEXT,
            review_notes TEXT,
            created_at TEXT NOT NULL,
            reviewed_at TEXT
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_user_feedback_status_created
        ON user_feedback(review_status, created_at)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_user_feedback_video
        ON user_feedback(video_id)
        """
    )
    connection.commit()


def _serialize_context(
    context: Mapping[str, Any] | None,
    *,
    share_context: bool,
) -> str | None:
    if not share_context:
        return None
    if not isinstance(context, Mapping):
        raise FeedbackStoreError(
            "context is required when share_context is enabled."
        )

    # Explicit allowlist: never persist arbitrary client-supplied fields,
    # transcript text, conversation history, or hidden model reasoning.
    allowed = {
        "question",
        "answer",
        "sources",
        "model",
        "retrieval_method",
        "retrieval_config",
    }
    clean = {key: context[key] for key in allowed if key in context}
    if not isinstance(clean.get("question"), str) or not clean["question"].strip():
        raise FeedbackStoreError("shared context must include the question.")
    if not isinstance(clean.get("answer"), str) or not clean["answer"].strip():
        raise FeedbackStoreError("shared context must include the answer.")

    sources = clean.get("sources", [])
    if not isinstance(sources, list) or len(sources) > 20:
        raise FeedbackStoreError("shared context may contain at most 20 sources.")

    # Rebuild each source from an explicit field allowlist. Source metadata is
    # useful for diagnosis; the transcript passage itself is intentionally absent.
    safe_sources: list[dict[str, Any]] = []
    source_fields = {
        "source_id",
        "chunk_ids",
        "video_id",
        "start",
        "end",
        "duration",
        "distance",
    }
    for source in sources:
        if not isinstance(source, Mapping):
            raise FeedbackStoreError("each source must be an object.")
        safe_sources.append(
            {key: source[key] for key in source_fields if key in source}
        )
    clean["sources"] = safe_sources

    if "model" in clean and not isinstance(clean["model"], str):
        raise FeedbackStoreError("model must be text.")
    if "retrieval_method" in clean and not isinstance(
        clean["retrieval_method"], str
    ):
        raise FeedbackStoreError("retrieval_method must be text.")
    if "retrieval_config" in clean and not isinstance(
        clean["retrieval_config"], Mapping
    ):
        raise FeedbackStoreError("retrieval_config must be an object.")

    encoded = json.dumps(clean, ensure_ascii=False, separators=(",", ":"))
    if len(encoded) > _MAX_CONTEXT_JSON_CHARS:
        raise FeedbackStoreError("shared context exceeds the storage size limit.")
    return encoded


def create_feedback_record(
    *,
    request_id: str,
    video_id: str,
    vote: str,
    reason: str | None = None,
    comment: str | None = None,
    share_context: bool = False,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist one rating; context is retained only after explicit opt-in."""
    if not isinstance(request_id, str) or not request_id.strip():
        raise FeedbackStoreError("request_id must be non-empty text.")
    request_id = request_id.strip()
    if len(request_id) > 64:
        raise FeedbackStoreError("request_id is too long.")
    if not isinstance(video_id, str) or not video_id.strip():
        raise FeedbackStoreError("video_id must be non-empty text.")
    video_id = video_id.strip()
    if vote not in FEEDBACK_VOTES:
        raise FeedbackStoreError("vote must be helpful or not_helpful.")
    if reason is not None and reason not in FEEDBACK_REASONS:
        raise FeedbackStoreError("reason is not a supported feedback category.")
    if comment is not None:
        if not isinstance(comment, str) or len(comment) > 1200:
            raise FeedbackStoreError("comment must be text of at most 1200 characters.")
        comment = comment.strip() or None
    if not isinstance(share_context, bool):
        raise FeedbackStoreError("share_context must be a boolean.")

    context_json = _serialize_context(
        context,
        share_context=share_context,
    )
    feedback_id = uuid.uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()

    try:
        with _connect() as connection:
            _ensure_schema(connection)
            connection.execute(
                """
                INSERT INTO user_feedback (
                    feedback_id, request_id, video_id, vote, reason, comment,
                    context_shared, context_json, review_status, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'new', ?)
                """,
                (
                    feedback_id,
                    request_id,
                    video_id,
                    vote,
                    reason,
                    comment,
                    int(share_context),
                    context_json,
                    created_at,
                ),
            )
    except sqlite3.IntegrityError as error:
        raise DuplicateFeedbackError(
            "Feedback has already been submitted for this answer."
        ) from error

    return {
        "feedback_id": feedback_id,
        "request_id": request_id,
        "video_id": video_id,
        "vote": vote,
        "review_status": "new",
        "context_shared": share_context,
        "created_at": created_at,
    }


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["context_shared"] = bool(item["context_shared"])
    item["context"] = (
        json.loads(item.pop("context_json"))
        if item.get("context_json")
        else None
    )
    return item


def list_feedback_records(
    *,
    statuses: tuple[str, ...] | list[str] | None = None,
    context_shared_only: bool = False,
) -> list[dict[str, Any]]:
    """Return records for the local operator CLI only."""
    if statuses is not None:
        invalid = set(statuses) - set(REVIEW_STATUSES)
        if invalid:
            raise FeedbackStoreError(
                "Unsupported review status: " + ", ".join(sorted(invalid))
            )

    with _connect() as connection:
        _ensure_schema(connection)
        clauses: list[str] = []
        parameters: list[Any] = []
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            clauses.append(f"review_status IN ({placeholders})")
            parameters.extend(statuses)
        if context_shared_only:
            clauses.append("context_shared = 1")
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = connection.execute(
            "SELECT * FROM user_feedback" + where + " ORDER BY created_at, feedback_id",
            parameters,
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


def update_feedback_review(
    *,
    feedback_id: str,
    status: str,
    reviewer_id: str,
    notes: str | None = None,
) -> dict[str, Any]:
    """Record an operator's triage outcome; it does not create a gold label."""
    if status not in REVIEW_STATUSES or status == "new":
        raise FeedbackStoreError("status must be a valid reviewed state.")
    if not isinstance(reviewer_id, str) or not _REVIEWER_ID_PATTERN.fullmatch(
        reviewer_id
    ):
        raise FeedbackStoreError(
            "reviewer_id must be a pseudonymous uppercase ID (for example, R01)."
        )
    if notes is not None:
        if not isinstance(notes, str) or len(notes) > 2000:
            raise FeedbackStoreError("notes must be at most 2000 characters.")
        notes = notes.strip() or None

    reviewed_at = datetime.now(timezone.utc).isoformat()
    with _connect() as connection:
        _ensure_schema(connection)
        cursor = connection.execute(
            """
            UPDATE user_feedback
            SET review_status = ?, reviewer_id = ?, review_notes = ?, reviewed_at = ?
            WHERE feedback_id = ?
            """,
            (status, reviewer_id, notes, reviewed_at, feedback_id),
        )
        if cursor.rowcount != 1:
            raise FeedbackNotFoundError("Feedback record was not found.")
        row = connection.execute(
            "SELECT * FROM user_feedback WHERE feedback_id = ?",
            (feedback_id,),
        ).fetchone()
    return _row_to_dict(row)


def summarize_feedback() -> dict[str, Any]:
    records = list_feedback_records()
    votes = Counter(item["vote"] for item in records)
    reasons = Counter(item["reason"] or "unspecified" for item in records)
    statuses = Counter(item["review_status"] for item in records)
    videos = Counter(item["video_id"] for item in records)
    return {
        "total_feedback": len(records),
        "votes": dict(sorted(votes.items())),
        "reasons": dict(sorted(reasons.items())),
        "review_statuses": dict(sorted(statuses.items())),
        "top_videos": [
            {"video_id": video_id, "feedback_count": count}
            for video_id, count in videos.most_common(10)
        ],
        "context_shared_count": sum(
            1 for item in records if item["context_shared"]
        ),
        "evaluation_candidate_count": sum(
            1 for item in records
            if item["review_status"] == "evaluation_candidate"
            and item["context_shared"]
        ),
    }


def export_evaluation_candidates(output_path: str | Path) -> int:
    """Export only human-approved, context-consented candidates as JSONL.

    This is a curation handoff, not a Phase 11.0C dataset: no labels, exact
    transcript passages, or source-verification attestations are created here.
    """
    private_root = (PROJECT_ROOT / "evaluation_data" / "private").resolve()
    target = Path(output_path).expanduser()
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    target = target.resolve()
    if not target.is_relative_to(private_root):
        raise FeedbackStoreError(
            "candidate exports must stay under evaluation_data/private/."
        )

    records = list_feedback_records(
        statuses=("evaluation_candidate",),
        context_shared_only=True,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        for item in records:
            payload = {
                "schema_version": "1.0.0",
                "candidate_type": "user_feedback",
                "candidate_id": item["feedback_id"],
                "video_id": item["video_id"],
                "vote": item["vote"],
                "user_reason": item["reason"],
                "user_comment": item["comment"],
                "question": item["context"]["question"],
                "answer": item["context"]["answer"],
                "sources": item["context"].get("sources", []),
                "model": item["context"].get("model", ""),
                "retrieval_method": item["context"].get("retrieval_method", ""),
                "retrieval_config": item["context"].get("retrieval_config", {}),
                "submitted_at": item["created_at"],
                "reviewed_by": item["reviewer_id"],
                "review_notes": item["review_notes"],
                "requires_source_and_transcript_verification": True,
                "is_gold_label": False,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return len(records)
