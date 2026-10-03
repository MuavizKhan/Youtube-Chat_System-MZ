"""
index_service.py

Application-level service for video index preparation and readiness.

Phase 6 responsibilities
------------------------
1. Expose index preparation as an asynchronous, idempotent operation.
2. Persist job state so queued/building/failed state survives requests
   and can be coordinated by multiple API processes on one shared store.
3. Keep /chat independent from index creation.
4. Prevent chat from using missing, invalid, or stale indexes.
"""

from __future__ import annotations

from dataclasses import dataclass

from .index_jobs import (
    enqueue_index_job,
    get_active_job,
    get_latest_job,
    ensure_worker_started,
)
from .indexing import (
    IndexAction,
    IndexState,
    get_index_action,
    get_index_state,
)
from .retrieval import (
    load_vector_store,
    validate_video_id,
)


# ============================================================
# RESULT TYPES
# ============================================================

@dataclass(frozen=True)
class IndexStatus:
    """Public service representation of index/job state."""

    video_id: str
    state: str
    action: str
    ready: bool
    job_id: str | None = None


@dataclass(frozen=True)
class IndexPreparationResult:
    """Result returned after a preparation request is accepted."""

    video_id: str
    state: str
    action: str
    ready: bool
    job_id: str | None = None


class IndexNotReadyError(FileNotFoundError):
    """
    Raised when chat is requested before a usable index exists.

    The state and job_id make the error actionable for the API/client
    without exposing implementation paths or provider details.
    """

    def __init__(
        self,
        video_id: str,
        state: str,
        job_id: str | None = None,
    ) -> None:
        self.video_id = video_id
        self.state = state
        self.job_id = job_id

        super().__init__(
            f"Video index is not ready for '{video_id}'. "
            f"Current state: {state}."
        )


# ============================================================
# STATUS
# ============================================================

def _persisted_public_state(
    state: IndexState,
) -> tuple[str, str, bool]:
    public_state = (
        "ready"
        if state is IndexState.VALID
        else state.value
    )

    return (
        public_state,
        get_index_action(state).value,
        state is IndexState.VALID,
    )


def get_index_status(
    video_id: str,
) -> IndexStatus:
    """
    Return durable job state when preparation is active, otherwise
    return the persisted lifecycle state.

    A failed job does not hide a valid previously persisted index:
    chat remains available while the latest failed rebuild is retained
    for diagnostics.
    """

    canonical_video_id = validate_video_id(
        video_id
    )

    ensure_worker_started()

    active_job = get_active_job(
        canonical_video_id
    )

    if active_job is not None:
        return IndexStatus(
            video_id=canonical_video_id,
            state=active_job.state,
            action=active_job.action,
            ready=False,
            job_id=active_job.job_id,
        )

    persisted_state = get_index_state(
        canonical_video_id
    )

    public_state, action, ready = _persisted_public_state(
        persisted_state
    )

    latest_job = get_latest_job(
        canonical_video_id
    )

    if (
        not ready
        and latest_job is not None
        and latest_job.state == "failed"
    ):
        return IndexStatus(
            video_id=canonical_video_id,
            state="failed",
            action="retry",
            ready=False,
            job_id=latest_job.job_id,
        )

    return IndexStatus(
        video_id=canonical_video_id,
        state=public_state,
        action=action,
        ready=ready,
        job_id=(
            latest_job.job_id
            if ready
            and latest_job is not None
            and latest_job.state == "ready"
            else None
        ),
    )


# ============================================================
# PREPARATION
# ============================================================

def prepare_index(
    video_id: str,
    languages: list[str] | None = None,
) -> IndexPreparationResult:
    """
    Enqueue or reuse an idempotent preparation job.

    Lifecycle:
        MISSING         -> queued(create)
        VALID           -> ready(reuse)
        INVALID / STALE -> queued(rebuild)

    Actual transcript/embedding work is performed by the durable
    background worker in index_jobs.py.
    """

    canonical_video_id = validate_video_id(
        video_id
    )

    ensure_worker_started()

    current_state = get_index_state(
        canonical_video_id
    )

    if current_state is IndexState.VALID:
        return IndexPreparationResult(
            video_id=canonical_video_id,
            state="ready",
            action=IndexAction.REUSE.value,
            ready=True,
            job_id=None,
        )

    action = get_index_action(
        current_state
    )

    job = enqueue_index_job(
        canonical_video_id,
        action.value,
        languages=languages,
    )

    return IndexPreparationResult(
        video_id=canonical_video_id,
        state=job.state,
        action=job.action,
        ready=False,
        job_id=job.job_id,
    )


# ============================================================
# READY INDEX
# ============================================================

def load_ready_index(
    video_id: str,
):
    """
    Load a vector store only when its persisted lifecycle state
    is VALID.

    This is the boundary used by /chat. Chat never creates or
    rebuilds indexes.
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
            status.job_id,
        )

    return load_vector_store(
        canonical_video_id
    )
