"""
index_service.py

Application-level service for video index preparation and readiness.

Responsibilities
----------------
1. Expose the index lifecycle as application-friendly operations.
2. Keep /chat independent from index creation.
3. Report persisted index readiness.
4. Prevent chat from using stale, invalid, or missing indexes.
"""

from dataclasses import dataclass

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


# ============================================================
# RESULT TYPES
# ============================================================

@dataclass(frozen=True)
class IndexStatus:
    """Public service representation of persisted index state."""

    video_id: str
    state: IndexState
    action: IndexAction
    ready: bool


@dataclass(frozen=True)
class IndexPreparationResult:
    """Result returned after a successful preparation request."""

    video_id: str
    state: IndexState
    action: IndexAction
    ready: bool


class IndexNotReadyError(FileNotFoundError):
    """
    Raised when chat is requested before a usable index exists.

    This is intentionally distinct from an unexpected persistence
    failure so the API can return a client-actionable response.
    """

    def __init__(
        self,
        video_id: str,
        state: IndexState,
    ) -> None:
        self.video_id = video_id
        self.state = state

        super().__init__(
            f"Video index is not ready for '{video_id}'. "
            f"Current state: {state.value}."
        )


# ============================================================
# STATUS
# ============================================================

def get_index_status(
    video_id: str,
) -> IndexStatus:
    """
    Return the persisted lifecycle state for one canonical video ID.

    The status reflects durable index state only. There is no
    synthetic 'building' state because preparation is synchronous:
    the caller receives a response only after the lifecycle operation
    completes.
    """

    canonical_video_id = validate_video_id(
        video_id
    )

    state = get_index_state(
        canonical_video_id
    )

    action = get_index_action(
        state
    )

    return IndexStatus(
        video_id=canonical_video_id,
        state=state,
        action=action,
        ready=(
            state is IndexState.VALID
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
    Create, reuse, or rebuild the video index as required.

    Lifecycle:
        MISSING         -> CREATE
        VALID           -> REUSE
        INVALID / STALE -> REBUILD

    The underlying ensure_index() operation owns lifecycle locking
    and final-state verification.
    """

    canonical_video_id = validate_video_id(
        video_id
    )

    _vector_store, action = ensure_index_with_action(
        canonical_video_id,
        languages=languages,
    )

    final_status = get_index_status(
        canonical_video_id
    )

    if not final_status.ready:
        raise RuntimeError(
            "Index preparation completed, but the index "
            f"is not ready. Final state: "
            f"{final_status.state.value}"
        )

    return IndexPreparationResult(
        video_id=canonical_video_id,
        state=final_status.state,
        action=action,
        ready=True,
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
        )

    return load_vector_store(
        canonical_video_id
    )
