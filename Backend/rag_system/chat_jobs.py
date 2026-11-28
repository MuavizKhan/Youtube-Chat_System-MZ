"""Durable coordination for long-running whole-video chat jobs.

Only the request payload and final API result are persisted. Transcript chunks,
prompts, provider credentials, and intermediate model output are never stored.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import uuid
from typing import Any, Iterator


QUEUED = "queued"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
_ACTIVE = (QUEUED, RUNNING)


@dataclass(frozen=True)
class ChatJob:
    job_id: str
    fingerprint: str
    video_id: str
    question: str
    conversation_history: list[dict[str, str]]
    status: str
    result: dict[str, Any] | None
    error_code: str | None
    error_message: str | None
    retry_after_seconds: int | None
    created_at: float
    updated_at: float
    lease_until: float
    worker_id: str | None
    attempt: int


class ChatJobStore:
    """SQLite-backed job lifecycle with atomic claims and expiring worker leases."""

    def __init__(self, db_path: str | Path, lease_seconds: int = 7200) -> None:
        if isinstance(lease_seconds, bool) or int(lease_seconds) < 60:
            raise ValueError("lease_seconds must be at least 60.")
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.lease_seconds = int(lease_seconds)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @contextmanager
    def _managed_connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._managed_connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_jobs (
                    job_id TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL,
                    video_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    conversation_history_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT,
                    error_code TEXT,
                    error_message TEXT,
                    retry_after_seconds INTEGER,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    lease_until REAL NOT NULL DEFAULT 0,
                    worker_id TEXT,
                    attempt INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_jobs_status_created
                ON chat_jobs(status, created_at)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_jobs_fingerprint_status
                ON chat_jobs(fingerprint, status)
                """
            )

    @staticmethod
    def _fingerprint(
        video_id: str,
        question: str,
        conversation_history: list[dict[str, str]],
    ) -> str:
        payload = json.dumps(
            {
                "video_id": video_id,
                "question": question,
                "conversation_history": conversation_history,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _row_to_job(row: sqlite3.Row | None) -> ChatJob | None:
        if row is None:
            return None
        result = json.loads(row["result_json"]) if row["result_json"] else None
        history = json.loads(row["conversation_history_json"])
        return ChatJob(
            job_id=row["job_id"],
            fingerprint=row["fingerprint"],
            video_id=row["video_id"],
            question=row["question"],
            conversation_history=history,
            status=row["status"],
            result=result,
            error_code=row["error_code"],
            error_message=row["error_message"],
            retry_after_seconds=row["retry_after_seconds"],
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            lease_until=float(row["lease_until"]),
            worker_id=row["worker_id"],
            attempt=int(row["attempt"]),
        )

    def enqueue(
        self,
        video_id: str,
        question: str,
        conversation_history: list[dict[str, str]],
    ) -> ChatJob:
        history_json = json.dumps(conversation_history, ensure_ascii=False)
        fingerprint = self._fingerprint(video_id, question, conversation_history)
        now = time.time()
        with self._managed_connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT * FROM chat_jobs
                WHERE fingerprint = ? AND status IN (?, ?)
                ORDER BY created_at DESC LIMIT 1
                """,
                (fingerprint, QUEUED, RUNNING),
            ).fetchone()
            if existing is not None:
                return self._row_to_job(existing)

            job_id = uuid.uuid4().hex
            connection.execute(
                """
                INSERT INTO chat_jobs (
                    job_id, fingerprint, video_id, question,
                    conversation_history_json, status, result_json,
                    error_code, error_message, retry_after_seconds,
                    created_at, updated_at, lease_until, worker_id, attempt
                )
                VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, ?, ?, 0, NULL, 0)
                """,
                (
                    job_id,
                    fingerprint,
                    video_id,
                    question,
                    history_json,
                    QUEUED,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM chat_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return self._row_to_job(row)

    def get(self, job_id: str) -> ChatJob | None:
        with self._managed_connection() as connection:
            row = connection.execute(
                "SELECT * FROM chat_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return self._row_to_job(row)

    def list_queued(self, limit: int = 32) -> list[ChatJob]:
        if limit <= 0:
            return []
        with self._managed_connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM chat_jobs WHERE status = ?
                ORDER BY created_at ASC LIMIT ?
                """,
                (QUEUED, int(limit)),
            ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def claim(self, job_id: str, worker_id: str) -> ChatJob | None:
        now = time.time()
        lease_until = now + self.lease_seconds
        with self._managed_connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE chat_jobs
                SET status = ?, attempt = attempt + 1, updated_at = ?,
                    lease_until = ?, worker_id = ?, error_code = NULL,
                    error_message = NULL, retry_after_seconds = NULL
                WHERE job_id = ? AND status = ?
                """,
                (RUNNING, now, lease_until, worker_id, job_id, QUEUED),
            )
            if cursor.rowcount != 1:
                return None
            row = connection.execute(
                "SELECT * FROM chat_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return self._row_to_job(row)

    def renew_lease(self, job_id: str, worker_id: str) -> bool:
        now = time.time()
        with self._managed_connection() as connection:
            cursor = connection.execute(
                """
                UPDATE chat_jobs SET updated_at = ?, lease_until = ?
                WHERE job_id = ? AND worker_id = ? AND status = ?
                """,
                (now, now + self.lease_seconds, job_id, worker_id, RUNNING),
            )
        return cursor.rowcount == 1

    def complete(self, job_id: str, worker_id: str, result: dict[str, Any]) -> bool:
        result_json = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        now = time.time()
        with self._managed_connection() as connection:
            cursor = connection.execute(
                """
                UPDATE chat_jobs
                SET status = ?, result_json = ?, error_code = NULL,
                    error_message = NULL, retry_after_seconds = NULL,
                    updated_at = ?, lease_until = 0, worker_id = NULL
                WHERE job_id = ? AND worker_id = ? AND status = ?
                """,
                (COMPLETED, result_json, now, job_id, worker_id, RUNNING),
            )
        return cursor.rowcount == 1

    def fail(
        self,
        job_id: str,
        worker_id: str,
        error_code: str,
        error_message: str,
        retry_after_seconds: int | None = None,
    ) -> bool:
        now = time.time()
        safe_code = str(error_code or "chat_job_failed")[:80]
        safe_message = str(error_message or "The video answer could not be completed.")[:500]
        retry_after = (
            max(1, int(retry_after_seconds))
            if isinstance(retry_after_seconds, (int, float))
            and not isinstance(retry_after_seconds, bool)
            else None
        )
        with self._managed_connection() as connection:
            cursor = connection.execute(
                """
                UPDATE chat_jobs
                SET status = ?, error_code = ?, error_message = ?,
                    retry_after_seconds = ?, updated_at = ?,
                    lease_until = 0, worker_id = NULL
                WHERE job_id = ? AND worker_id = ? AND status = ?
                """,
                (
                    FAILED, safe_code, safe_message, retry_after, now,
                    job_id, worker_id, RUNNING,
                ),
            )
        return cursor.rowcount == 1

    def recover_expired(self) -> int:
        now = time.time()
        with self._managed_connection() as connection:
            cursor = connection.execute(
                """
                UPDATE chat_jobs
                SET status = ?, updated_at = ?, lease_until = 0,
                    worker_id = NULL, error_code = NULL, error_message = NULL,
                    retry_after_seconds = NULL
                WHERE status = ? AND lease_until > 0 AND lease_until <= ?
                """,
                (QUEUED, now, RUNNING, now),
            )
        return int(cursor.rowcount)
