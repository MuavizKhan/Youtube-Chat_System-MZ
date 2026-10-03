"""
index_jobs.py

SQLite-backed coordination for asynchronous video-index preparation.

The store is intentionally small and dependency-free so the free/local
deployment can coordinate multiple FastAPI worker processes on a shared
filesystem. Vector-store persistence remains in indexing.py.

This module stores job metadata and leases; it does not perform indexing.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import time
import uuid


JOB_QUEUED = "queued"
JOB_BUILDING = "building"
JOB_READY = "ready"
JOB_FAILED = "failed"

_ACTIVE_STATUSES = {
    JOB_QUEUED,
    JOB_BUILDING,
}


@dataclass(frozen=True)
class IndexJob:
    job_id: str
    video_id: str
    status: str
    action: str
    languages: list[str] | None
    attempt: int
    error: str | None
    created_at: float
    updated_at: float
    lease_until: float
    worker_id: str | None


class IndexJobStore:
    """Persistent index-job state backed by SQLite."""

    def __init__(
        self,
        db_path: str | Path,
        lease_seconds: int = 1800,
    ) -> None:
        self.db_path = Path(db_path).expanduser()
        self.lease_seconds = int(lease_seconds)

        if self.lease_seconds <= 0:
            raise ValueError("lease_seconds must be greater than zero.")

        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.db_path,
            timeout=30.0,
        )

        connection.row_factory = sqlite3.Row

        connection.execute(
            "PRAGMA journal_mode=WAL"
        )
        connection.execute(
            "PRAGMA synchronous=NORMAL"
        )
        connection.execute(
            "PRAGMA busy_timeout=30000"
        )

        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS index_jobs (
                    video_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    action TEXT NOT NULL,
                    languages_json TEXT,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    lease_until REAL NOT NULL DEFAULT 0,
                    worker_id TEXT
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_index_jobs_status
                ON index_jobs(status)
                """
            )

    @staticmethod
    def _row_to_job(row: sqlite3.Row | None) -> IndexJob | None:
        if row is None:
            return None

        languages = None

        if row["languages_json"]:
            languages = json.loads(
                row["languages_json"]
            )

        return IndexJob(
            job_id=row["job_id"],
            video_id=row["video_id"],
            status=row["status"],
            action=row["action"],
            languages=languages,
            attempt=row["attempt"],
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            lease_until=row["lease_until"],
            worker_id=row["worker_id"],
        )

    def get(self, video_id: str) -> IndexJob | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM index_jobs
                WHERE video_id = ?
                """,
                (video_id,),
            ).fetchone()

        return self._row_to_job(row)

    def enqueue(
        self,
        video_id: str,
        action: str,
        languages: list[str] | None = None,
    ) -> IndexJob:
        """
        Create a new queued job, or return an already-active job.

        A failed/previously completed record may be reused for a retry,
        which gives every retry a fresh job ID and attempt counter.
        """
        now = time.time()
        serialized_languages = (
            json.dumps(languages)
            if languages is not None
            else None
        )

        with self._connect() as connection:
            connection.execute(
                "BEGIN IMMEDIATE"
            )

            existing = connection.execute(
                """
                SELECT *
                FROM index_jobs
                WHERE video_id = ?
                """,
                (video_id,),
            ).fetchone()

            if (
                existing is not None
                and existing["status"] in _ACTIVE_STATUSES
            ):
                return self._row_to_job(existing)

            job = IndexJob(
                job_id=uuid.uuid4().hex,
                video_id=video_id,
                status=JOB_QUEUED,
                action=action,
                languages=(
                    list(languages)
                    if languages is not None
                    else None
                ),
                attempt=0,
                error=None,
                created_at=now,
                updated_at=now,
                lease_until=0.0,
                worker_id=None,
            )

            connection.execute(
                """
                INSERT INTO index_jobs (
                    video_id,
                    job_id,
                    status,
                    action,
                    languages_json,
                    attempt,
                    error,
                    created_at,
                    updated_at,
                    lease_until,
                    worker_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(video_id)
                DO UPDATE SET
                    job_id = excluded.job_id,
                    status = excluded.status,
                    action = excluded.action,
                    languages_json = excluded.languages_json,
                    attempt = excluded.attempt,
                    error = excluded.error,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at,
                    lease_until = excluded.lease_until,
                    worker_id = excluded.worker_id
                """,
                (
                    job.video_id,
                    job.job_id,
                    job.status,
                    job.action,
                    serialized_languages,
                    job.attempt,
                    job.error,
                    job.created_at,
                    job.updated_at,
                    job.lease_until,
                    job.worker_id,
                ),
            )

        return job

    def list_queued(
        self,
        limit: int = 32,
    ) -> list[IndexJob]:
        if limit <= 0:
            return []

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM index_jobs
                WHERE status = ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (JOB_QUEUED, limit),
            ).fetchall()

        return [
            self._row_to_job(row)
            for row in rows
        ]

    def recover_expired_jobs(self) -> int:
        """Return expired building leases to the queued state."""
        now = time.time()

        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE index_jobs
                SET
                    status = ?,
                    updated_at = ?,
                    lease_until = 0,
                    worker_id = NULL,
                    error = NULL
                WHERE
                    status = ?
                    AND lease_until > 0
                    AND lease_until <= ?
                """,
                (
                    JOB_QUEUED,
                    now,
                    JOB_BUILDING,
                    now,
                ),
            )

            return int(cursor.rowcount)

    def claim(
        self,
        video_id: str,
        worker_id: str,
    ) -> IndexJob | None:
        """
        Atomically claim one queued/expired job.

        SQLite's IMMEDIATE transaction prevents two worker processes
        from claiming the same video job.
        """
        now = time.time()
        lease_until = now + self.lease_seconds

        with self._connect() as connection:
            connection.execute(
                "BEGIN IMMEDIATE"
            )

            row = connection.execute(
                """
                SELECT *
                FROM index_jobs
                WHERE video_id = ?
                """,
                (video_id,),
            ).fetchone()

            if row is None:
                return None

            active_building = (
                row["status"] == JOB_BUILDING
                and row["lease_until"] > now
            )

            if (
                row["status"] != JOB_QUEUED
                and active_building
            ):
                return None

            if row["status"] not in {
                JOB_QUEUED,
                JOB_BUILDING,
            }:
                return None

            connection.execute(
                """
                UPDATE index_jobs
                SET
                    status = ?,
                    attempt = attempt + 1,
                    updated_at = ?,
                    lease_until = ?,
                    worker_id = ?,
                    error = NULL
                WHERE video_id = ?
                """,
                (
                    JOB_BUILDING,
                    now,
                    lease_until,
                    worker_id,
                    video_id,
                ),
            )

            claimed = connection.execute(
                """
                SELECT *
                FROM index_jobs
                WHERE video_id = ?
                """,
                (video_id,),
            ).fetchone()

        return self._row_to_job(claimed)

    def renew_lease(
        self,
        job_id: str,
        worker_id: str,
    ) -> bool:
        """
        Extend an active worker lease.

        A heartbeat prevents a legitimately long indexing operation from
        being reclaimed by another worker before it finishes.
        """
        now = time.time()
        lease_until = now + self.lease_seconds

        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE index_jobs
                SET
                    updated_at = ?,
                    lease_until = ?
                WHERE
                    job_id = ?
                    AND worker_id = ?
                    AND status = ?
                """,
                (
                    now,
                    lease_until,
                    job_id,
                    worker_id,
                    JOB_BUILDING,
                ),
            )

        return cursor.rowcount == 1


    def mark_ready(
        self,
        job_id: str,
        worker_id: str,
        action: str,
    ) -> bool:
        now = time.time()

        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE index_jobs
                SET
                    status = ?,
                    action = ?,
                    updated_at = ?,
                    lease_until = 0,
                    worker_id = NULL,
                    error = NULL
                WHERE
                    job_id = ?
                    AND worker_id = ?
                    AND status = ?
                """,
                (
                    JOB_READY,
                    action,
                    now,
                    job_id,
                    worker_id,
                    JOB_BUILDING,
                ),
            )

        return cursor.rowcount == 1

    def mark_failed(
        self,
        job_id: str,
        worker_id: str,
        error: str,
    ) -> bool:
        now = time.time()

        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE index_jobs
                SET
                    status = ?,
                    updated_at = ?,
                    lease_until = 0,
                    worker_id = NULL,
                    error = ?
                WHERE
                    job_id = ?
                    AND worker_id = ?
                    AND status = ?
                """,
                (
                    JOB_FAILED,
                    now,
                    error[:500],
                    job_id,
                    worker_id,
                    JOB_BUILDING,
                ),
            )

        return cursor.rowcount == 1
