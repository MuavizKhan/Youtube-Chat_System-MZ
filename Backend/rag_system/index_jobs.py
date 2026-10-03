"""
index_jobs.py

Durable, idempotent index-preparation jobs backed by SQLite.

The job store solves the Phase 5 limitation where queued/building/failed
state and index work were process-local.

Design
------
* SQLite persists job state across API requests and process restarts.
* A partial unique index guarantees at most one queued/building job
  per video.
* Each API process may run one lightweight daemon worker.
* Workers coordinate through SQLite transactions, so two processes
  cannot claim the same job.
* A stale building job is returned to the queue after the configured
  timeout, allowing recovery after a worker crash.
* Internal exception details are stored only for server diagnostics;
  the API exposes sanitized state.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import sqlite3
import threading
import time
from typing import Iterable
from uuid import uuid4

from .config import (
    INDEX_JOB_DB_PATH,
    INDEX_JOB_POLL_SECONDS,
    INDEX_JOB_STALE_SECONDS,
)
from .indexing import (
    IndexAction,
    IndexState,
    ensure_index_with_action,
    get_index_state,
)

logger = logging.getLogger(__name__)


# ============================================================
# TYPES
# ============================================================

ACTIVE_STATES = ("queued", "building")


@dataclass(frozen=True)
class IndexJob:
    """Public job representation."""

    job_id: str
    video_id: str
    state: str
    action: str
    languages: tuple[str, ...]
    error_type: str | None
    created_at: float
    updated_at: float
    started_at: float | None
    completed_at: float | None


# ============================================================
# WORKER STATE
# ============================================================

_WORKER_LOCK = threading.Lock()
_WORKER_THREAD: threading.Thread | None = None
_SCHEMA_LOCK = threading.Lock()
_SCHEMA_INITIALIZED_PATH: Path | None = None


# ============================================================
# DATABASE
# ============================================================

def _database_path() -> Path:
    path = Path(INDEX_JOB_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(
        _database_path(),
        timeout=30.0,
        isolation_level=None,
        check_same_thread=False,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def initialize_job_store() -> None:
    """Create the durable job schema when needed."""

    global _SCHEMA_INITIALIZED_PATH

    database_path = _database_path()

    with _SCHEMA_LOCK:
        if _SCHEMA_INITIALIZED_PATH == database_path:
            return

        with _connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS index_jobs (
                    job_id TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    action TEXT NOT NULL,
                    languages_json TEXT NOT NULL DEFAULT '[]',
                    error_type TEXT,
                    error_message TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    started_at REAL,
                    completed_at REAL
                );

                CREATE INDEX IF NOT EXISTS idx_index_jobs_video_created
                ON index_jobs(video_id, created_at DESC);

                CREATE INDEX IF NOT EXISTS idx_index_jobs_state_updated
                ON index_jobs(state, updated_at);

                CREATE UNIQUE INDEX IF NOT EXISTS idx_index_jobs_active_video
                ON index_jobs(video_id)
                WHERE state IN ('queued', 'building');
                """
            )

        _SCHEMA_INITIALIZED_PATH = database_path


def _row_to_job(row: sqlite3.Row | None) -> IndexJob | None:
    if row is None:
        return None

    languages_raw = row["languages_json"] or "[]"
    try:
        languages = tuple(
            str(item)
            for item in json.loads(languages_raw)
            if isinstance(item, str)
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        languages = ()

    return IndexJob(
        job_id=str(row["job_id"]),
        video_id=str(row["video_id"]),
        state=str(row["state"]),
        action=str(row["action"]),
        languages=languages,
        error_type=(
            str(row["error_type"])
            if row["error_type"] is not None
            else None
        ),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        started_at=(
            float(row["started_at"])
            if row["started_at"] is not None
            else None
        ),
        completed_at=(
            float(row["completed_at"])
            if row["completed_at"] is not None
            else None
        ),
    )


def _recover_stale_jobs(connection: sqlite3.Connection) -> None:
    cutoff = time.time() - INDEX_JOB_STALE_SECONDS
    now = time.time()

    connection.execute(
        """
        UPDATE index_jobs
        SET
            state = 'queued',
            updated_at = ?,
            started_at = NULL,
            error_type = NULL,
            error_message = NULL
        WHERE state = 'building'
          AND updated_at < ?
        """,
        (now, cutoff),
    )


# ============================================================
# JOB QUERIES / COMMANDS
# ============================================================

def get_active_job(video_id: str) -> IndexJob | None:
    initialize_job_store()

    with _connect() as connection:
        _recover_stale_jobs(connection)
        row = connection.execute(
            """
            SELECT *
            FROM index_jobs
            WHERE video_id = ?
              AND state IN ('queued', 'building')
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (video_id,),
        ).fetchone()

    return _row_to_job(row)


def get_latest_job(video_id: str) -> IndexJob | None:
    initialize_job_store()

    with _connect() as connection:
        _recover_stale_jobs(connection)
        row = connection.execute(
            """
            SELECT *
            FROM index_jobs
            WHERE video_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (video_id,),
        ).fetchone()

    return _row_to_job(row)


def enqueue_index_job(
    video_id: str,
    action: str,
    languages: Iterable[str] | None = None,
) -> IndexJob:
    """
    Create or reuse the active preparation job for a video.

    At most one queued/building job can exist for a video.
    """

    initialize_job_store()
    requested_languages = tuple(languages or ())
    now = time.time()
    job_id = uuid4().hex

    with _connect() as connection:
        _recover_stale_jobs(connection)

        row = connection.execute(
            """
            SELECT *
            FROM index_jobs
            WHERE video_id = ?
              AND state IN ('queued', 'building')
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (video_id,),
        ).fetchone()

        if row is not None:
            job = _row_to_job(row)
        else:
            try:
                connection.execute(
                    """
                    INSERT INTO index_jobs (
                        job_id,
                        video_id,
                        state,
                        action,
                        languages_json,
                        created_at,
                        updated_at
                    )
                    VALUES (?, ?, 'queued', ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        video_id,
                        action,
                        json.dumps(
                            requested_languages,
                            separators=(",", ":"),
                        ),
                        now,
                        now,
                    ),
                )
            except sqlite3.IntegrityError:
                # Another process won the race between SELECT and INSERT.
                row = connection.execute(
                    """
                    SELECT *
                    FROM index_jobs
                    WHERE video_id = ?
                      AND state IN ('queued', 'building')
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (video_id,),
                ).fetchone()

                if row is None:
                    raise

                job = _row_to_job(row)
            else:
                row = connection.execute(
                    """
                    SELECT *
                    FROM index_jobs
                    WHERE job_id = ?
                    """,
                    (job_id,),
                ).fetchone()
                job = _row_to_job(row)

    if job is None:
        raise RuntimeError("Failed to create index job.")

    _ensure_worker_started()
    return job


def claim_next_job() -> IndexJob | None:
    """Atomically move one queued job to building."""

    initialize_job_store()

    with _connect() as connection:
        _recover_stale_jobs(connection)

        connection.execute("BEGIN IMMEDIATE")

        row = connection.execute(
            """
            SELECT *
            FROM index_jobs
            WHERE state = 'queued'
            ORDER BY created_at ASC
            LIMIT 1
            """
        ).fetchone()

        if row is None:
            connection.execute("COMMIT")
            return None

        job_id = str(row["job_id"])
        now = time.time()

        connection.execute(
            """
            UPDATE index_jobs
            SET
                state = 'building',
                updated_at = ?,
                started_at = COALESCE(started_at, ?),
                error_type = NULL,
                error_message = NULL
            WHERE job_id = ?
              AND state = 'queued'
            """,
            (now, now, job_id),
        )

        updated = connection.execute(
            "SELECT * FROM index_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()

        connection.execute("COMMIT")

    return _row_to_job(updated)


def mark_job_ready(
    job_id: str,
    action: str,
) -> None:
    initialize_job_store()
    now = time.time()

    with _connect() as connection:
        connection.execute(
            """
            UPDATE index_jobs
            SET
                state = 'ready',
                action = ?,
                updated_at = ?,
                completed_at = ?,
                error_type = NULL,
                error_message = NULL
            WHERE job_id = ?
            """,
            (action, now, now, job_id),
        )


def mark_job_failed(
    job_id: str,
    error: BaseException,
) -> None:
    initialize_job_store()
    now = time.time()

    with _connect() as connection:
        connection.execute(
            """
            UPDATE index_jobs
            SET
                state = 'failed',
                updated_at = ?,
                completed_at = ?,
                error_type = ?,
                error_message = ?
            WHERE job_id = ?
            """,
            (
                now,
                now,
                type(error).__name__,
                str(error)[:1000],
                job_id,
            ),
        )


# ============================================================
# WORKER
# ============================================================

def _action_value(action: object) -> str:
    value = getattr(action, "value", action)
    return str(value)


def _touch_job(job_id: str) -> None:
    initialize_job_store()

    with _connect() as connection:
        connection.execute(
            """
            UPDATE index_jobs
            SET updated_at = ?
            WHERE job_id = ?
              AND state = 'building'
            """,
            (time.time(), job_id),
        )


def _run_job(job: IndexJob) -> None:
    heartbeat_stop = threading.Event()
    heartbeat_interval = max(
        1.0,
        min(INDEX_JOB_STALE_SECONDS / 3.0, 60.0),
    )

    def heartbeat() -> None:
        while not heartbeat_stop.wait(heartbeat_interval):
            try:
                _touch_job(job.job_id)
            except Exception:
                logger.exception(
                    "index job heartbeat failed job_id=%s",
                    job.job_id,
                )

    heartbeat_thread = threading.Thread(
        target=heartbeat,
        name=f"index-heartbeat-{job.job_id[:8]}",
        daemon=True,
    )
    heartbeat_thread.start()

    try:
        languages = list(job.languages) or None

        _vector_store, actual_action = ensure_index_with_action(
            job.video_id,
            languages=languages,
        )

        final_state = get_index_state(job.video_id)

        if final_state is not IndexState.VALID:
            raise RuntimeError(
                "Index preparation completed without a valid final state."
            )

        mark_job_ready(
            job.job_id,
            _action_value(actual_action),
        )

        logger.info(
            "index job completed job_id=%s video_id=%s action=%s",
            job.job_id,
            job.video_id,
            _action_value(actual_action),
        )

    except Exception as error:
        mark_job_failed(job.job_id, error)

        logger.exception(
            "index job failed job_id=%s video_id=%s",
            job.job_id,
            job.video_id,
        )
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=1.0)


def _worker_loop() -> None:
    while True:
        try:
            job = claim_next_job()

            if job is None:
                time.sleep(INDEX_JOB_POLL_SECONDS)
                continue

            _run_job(job)

        except Exception:
            logger.exception("index job worker loop failed")
            time.sleep(INDEX_JOB_POLL_SECONDS)


def _ensure_worker_started() -> None:
    global _WORKER_THREAD

    with _WORKER_LOCK:
        if (
            _WORKER_THREAD is not None
            and _WORKER_THREAD.is_alive()
        ):
            return

        _WORKER_THREAD = threading.Thread(
            target=_worker_loop,
            name="youtube-index-worker",
            daemon=True,
        )
        _WORKER_THREAD.start()


def ensure_worker_started() -> None:
    """Ensure this API process participates in job processing."""

    initialize_job_store()
    _ensure_worker_started()
