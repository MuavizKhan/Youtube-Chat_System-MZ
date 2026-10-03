"""
index_service.py

Application-level service for asynchronous video-index preparation.

Responsibilities
----------------
1. Expose index preparation as a durable job operation.
2. Keep /chat independent from index creation.
3. Persist queued/building/ready/failed states in SQLite.
4. Coordinate multiple FastAPI worker processes through job leases.
5. Prevent chat from using stale, invalid, or in-progress indexes.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
import os
import socket
import threading
import uuid

from .config import (
    INDEX_JOB_DB_PATH,
    INDEX_JOB_LEASE_SECONDS,
    INDEX_JOB_WORKERS,
)
from .index_jobs import (
    JOB_BUILDING,
    JOB_FAILED,
    JOB_QUEUED,
    JOB_READY,
    IndexJobStore,
)
from .indexing import (
    IndexAction,
    IndexState,
    ensure_index_with_action,
    get_index_action,
    get_index_state,
)
from .retrieval import (
    load_vector_store,
    validate_video_id,
)


logger = logging.getLogger(__name__)


# ============================================================
# RESULT TYPES
# ============================================================

@dataclass(frozen=True)
class IndexStatus:
    """Public service representation of index state."""

    video_id: str
    state: str
    action: str
    ready: bool


@dataclass(frozen=True)
class IndexPreparationResult:
    """Result returned after synchronous index preparation."""

    video_id: str
    state: str
    action: str
    ready: bool


class IndexNotReadyError(FileNotFoundError):
    """
    Raised when chat is requested before a usable index exists.

    state is the public lifecycle state, including queued/building
    runtime states introduced by the durable job system.
    """

    def __init__(
        self,
        video_id: str,
        state: str,
    ) -> None:
        self.video_id = video_id
        self.state = state

        super().__init__(
            f"Video index is not ready for '{video_id}'. "
            f"Current state: {state}."
        )


# ============================================================
# DURABLE JOB COORDINATION
# ============================================================

_JOB_STORE = IndexJobStore(
    INDEX_JOB_DB_PATH,
    lease_seconds=INDEX_JOB_LEASE_SECONDS,
)

_EXECUTOR: ThreadPoolExecutor | None = None
_EXECUTOR_LOCK = threading.Lock()
_DISPATCHED_VIDEOS: set[str] = set()
_DISPATCHED_LOCK = threading.Lock()

_WORKER_ID = (
    f"{socket.gethostname()}:"
    f"{os.getpid()}:"
    f"{uuid.uuid4().hex[:12]}"
)


def _get_executor() -> ThreadPoolExecutor:
    global _EXECUTOR

    with _EXECUTOR_LOCK:
        if _EXECUTOR is None:
            _EXECUTOR = ThreadPoolExecutor(
                max_workers=INDEX_JOB_WORKERS,
                thread_name_prefix="index-worker",
            )

        return _EXECUTOR


def _on_worker_finished(video_id: str, _future) -> None:
    with _DISPATCHED_LOCK:
        _DISPATCHED_VIDEOS.discard(video_id)


def _dispatch_video(video_id: str) -> None:
    """
    Schedule one local worker for a video.

    The durable database remains the source of truth. The in-memory
    set only prevents repeatedly submitting the same queued job to
    the same process while a worker future is already pending.
    """

    with _DISPATCHED_LOCK:
        if video_id in _DISPATCHED_VIDEOS:
            return

        _DISPATCHED_VIDEOS.add(video_id)

    future = _get_executor().submit(
        _run_index_job,
        video_id,
    )

    future.add_done_callback(
        lambda completed: _on_worker_finished(
            video_id,
            completed,
        )
    )


def _dispatch_available_jobs() -> None:
    for job in _JOB_STORE.list_queued(
        limit=max(INDEX_JOB_WORKERS * 2, 1)
    ):
        _dispatch_video(job.video_id)


def _run_index_job(video_id: str) -> None:
    job = _JOB_STORE.claim(
        video_id,
        worker_id=_WORKER_ID,
    )

    if job is None:
        return

    heartbeat_stop = threading.Event()
    heartbeat_interval = max(
        1.0,
        min(INDEX_JOB_LEASE_SECONDS / 3.0, 60.0),
    )

    def heartbeat() -> None:
        while not heartbeat_stop.wait(heartbeat_interval):
            try:
                renewed = _JOB_STORE.renew_lease(
                    job.job_id,
                    worker_id=_WORKER_ID,
                )

                if not renewed:
                    logger.warning(
                        "Index job lease was lost video_id=%s job_id=%s",
                        job.video_id,
                        job.job_id,
                    )
                    return

            except Exception:
                logger.exception(
                    "Index job heartbeat failed video_id=%s job_id=%s",
                    job.video_id,
                    job.job_id,
                )

    heartbeat_thread = threading.Thread(
        target=heartbeat,
        name=f"index-heartbeat-{job.job_id[:8]}",
        daemon=True,
    )
    heartbeat_thread.start()

    try:
        _vector_store, actual_action = ensure_index_with_action(
            job.video_id,
            languages=job.languages,
        )

        final_state = get_index_state(
            job.video_id
        )

        if final_state is not IndexState.VALID:
            raise RuntimeError(
                "Index preparation completed, but the persisted "
                f"index is not valid. Final state: "
                f"{final_state.value}"
            )

        marked_ready = _JOB_STORE.mark_ready(
            job.job_id,
            worker_id=_WORKER_ID,
            action=actual_action.value,
        )

        if not marked_ready:
            logger.warning(
                "Index job completion could not be recorded "
                "because its lease changed. video_id=%s job_id=%s",
                job.video_id,
                job.job_id,
            )

        logger.info(
            "Index job completed video_id=%s action=%s attempt=%s",
            job.video_id,
            actual_action.value,
            job.attempt,
        )

    except Exception:
        logger.exception(
            "Index job failed video_id=%s job_id=%s attempt=%s",
            job.video_id,
            job.job_id,
            job.attempt,
        )

        marked_failed = _JOB_STORE.mark_failed(
            job.job_id,
            worker_id=_WORKER_ID,
            error="Index preparation failed.",
        )

        if not marked_failed:
            logger.warning(
                "Index job failure could not be recorded because "
                "its lease changed. video_id=%s job_id=%s",
                job.video_id,
                job.job_id,
            )
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=1.0)


def recover_index_jobs() -> None:
    """
    Recover expired worker leases and dispatch queued jobs.

    This is safe to call on startup and from status requests.
    """

    _JOB_STORE.recover_expired_jobs()
    _dispatch_available_jobs()


def shutdown_index_workers() -> None:
    """Stop local worker threads during application shutdown."""

    global _EXECUTOR

    with _EXECUTOR_LOCK:
        executor = _EXECUTOR
        _EXECUTOR = None

    if executor is not None:
        executor.shutdown(
            wait=False,
            cancel_futures=False,
        )


# ============================================================
# STATUS
# ============================================================

def get_index_status(
    video_id: str,
) -> IndexStatus:
    """
    Return durable job state when present, otherwise persisted index state.

    queued and building are durable across FastAPI worker
    processes. failed is also retained until the next retry.
    """

    canonical_video_id = validate_video_id(
        video_id
    )

    recover_index_jobs()

    job = _JOB_STORE.get(
        canonical_video_id
    )

    if job is not None:
        if job.status == JOB_QUEUED:
            return IndexStatus(
                video_id=canonical_video_id,
                state="queued",
                action=job.action,
                ready=False,
            )

        if job.status == JOB_BUILDING:
            return IndexStatus(
                video_id=canonical_video_id,
                state="building",
                action=job.action,
                ready=False,
            )

        if job.status == JOB_FAILED:
            persisted_state = get_index_state(
                canonical_video_id
            )

            if persisted_state is IndexState.VALID:
                return IndexStatus(
                    video_id=canonical_video_id,
                    state="ready",
                    action=IndexAction.REUSE.value,
                    ready=True,
                )

            return IndexStatus(
                video_id=canonical_video_id,
                state="failed",
                action="retry",
                ready=False,
            )

        if job.status == JOB_READY:
            persisted_state = get_index_state(
                canonical_video_id
            )

            if persisted_state is IndexState.VALID:
                return IndexStatus(
                    video_id=canonical_video_id,
                    state="ready",
                    action=job.action,
                    ready=True,
                )

    persisted_state = get_index_state(
        canonical_video_id
    )

    action = get_index_action(
        persisted_state
    )

    public_state = (
        "ready"
        if persisted_state is IndexState.VALID
        else persisted_state.value
    )

    return IndexStatus(
        video_id=canonical_video_id,
        state=public_state,
        action=action.value,
        ready=(persisted_state is IndexState.VALID),
    )


# ============================================================
# ASYNCHRONOUS PREPARATION
# ============================================================

def submit_index(
    video_id: str,
    languages: list[str] | None = None,
) -> IndexStatus:
    """
    Create, reuse, or rebuild a video index through a durable background job.

    A valid index is returned immediately as ready. Otherwise the
    job is queued and a worker process claims it using a SQLite lease.
    """

    canonical_video_id = validate_video_id(
        video_id
    )

    recover_index_jobs()

    persisted_state = get_index_state(
        canonical_video_id
    )

    if persisted_state is IndexState.VALID:
        return IndexStatus(
            video_id=canonical_video_id,
            state="ready",
            action=IndexAction.REUSE.value,
            ready=True,
        )

    action = get_index_action(
        persisted_state
    )

    job = _JOB_STORE.enqueue(
        canonical_video_id,
        action=action.value,
        languages=languages,
    )

    _dispatch_video(
        canonical_video_id
    )

    status = get_index_status(
        canonical_video_id
    )

    logger.info(
        "Index job submitted video_id=%s job_id=%s state=%s action=%s",
        canonical_video_id,
        job.job_id,
        status.state,
        status.action,
    )

    return status


# ============================================================
# SYNCHRONOUS PREPARATION (SERVICE COMPATIBILITY)
# ============================================================

def prepare_index(
    video_id: str,
    languages: list[str] | None = None,
) -> IndexPreparationResult:
    """
    Synchronously create, reuse, or rebuild a video index.

    This remains available to internal callers and tests. The HTTP API
    uses submit_index so request latency is independent of indexing.
    """

    canonical_video_id = validate_video_id(
        video_id
    )

    _vector_store, action = ensure_index_with_action(
        canonical_video_id,
        languages=languages,
    )

    final_state = get_index_state(
        canonical_video_id
    )

    if final_state is not IndexState.VALID:
        raise RuntimeError(
            "Index preparation completed, but the index "
            f"is not ready. Final state: {final_state.value}"
        )

    return IndexPreparationResult(
        video_id=canonical_video_id,
        state="ready",
        action=action.value,
        ready=True,
    )


# ============================================================
# READY INDEX
# ============================================================

def load_ready_index(
    video_id: str,
):
    """
    Load a vector store only when its lifecycle state is VALID and
    no durable index job says the video is still being prepared.

    This is the boundary used by /chat. Chat never creates or rebuilds
    indexes.
    """

    canonical_video_id = validate_video_id(
        video_id
    )

    status = get_index_status(
        canonical_video_id
    )

    if not status.ready:
        raise IndexNotReadyError(
            canonical_video_id,
            status.state,
        )

    return load_vector_store(
        canonical_video_id
    )
