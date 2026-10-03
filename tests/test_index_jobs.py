import sqlite3
import time

import pytest

from Backend.rag_system import index_jobs
from Backend.rag_system.indexing import IndexState


VIDEO_ID = "Gfr50f6ZBvo"


@pytest.fixture
def isolated_job_store(tmp_path, monkeypatch):
    db_path = tmp_path / "index_jobs.sqlite3"

    monkeypatch.setattr(
        index_jobs,
        "INDEX_JOB_DB_PATH",
        db_path,
    )

    monkeypatch.setattr(
        index_jobs,
        "_ensure_worker_started",
        lambda: None,
    )

    index_jobs.initialize_job_store()

    return db_path


@pytest.mark.unit
def test_enqueue_is_idempotent_per_video(isolated_job_store):
    first = index_jobs.enqueue_index_job(
        VIDEO_ID,
        "create",
        languages=["en"],
    )
    second = index_jobs.enqueue_index_job(
        VIDEO_ID,
        "rebuild",
        languages=["de"],
    )

    assert first.job_id == second.job_id
    assert second.state == "queued"
    assert second.action == "create"
    assert second.languages == ("en",)


@pytest.mark.unit
def test_claim_moves_job_to_building(isolated_job_store):
    created = index_jobs.enqueue_index_job(
        VIDEO_ID,
        "create",
    )

    claimed = index_jobs.claim_next_job()

    assert claimed is not None
    assert claimed.job_id == created.job_id
    assert claimed.state == "building"
    assert claimed.started_at is not None

    active = index_jobs.get_active_job(
        VIDEO_ID
    )

    assert active is not None
    assert active.state == "building"


@pytest.mark.unit
def test_successful_job_is_terminal(isolated_job_store):
    created = index_jobs.enqueue_index_job(
        VIDEO_ID,
        "create",
    )

    index_jobs.mark_job_ready(
        created.job_id,
        "create",
    )

    latest = index_jobs.get_latest_job(
        VIDEO_ID
    )

    assert latest is not None
    assert latest.state == "ready"
    assert latest.action == "create"
    assert latest.completed_at is not None
    assert index_jobs.get_active_job(
        VIDEO_ID
    ) is None


@pytest.mark.unit
def test_failed_job_is_terminal_and_sanitizes_type(
    isolated_job_store,
):
    created = index_jobs.enqueue_index_job(
        VIDEO_ID,
        "create",
    )

    index_jobs.mark_job_failed(
        created.job_id,
        RuntimeError("private provider secret"),
    )

    latest = index_jobs.get_latest_job(
        VIDEO_ID
    )

    assert latest is not None
    assert latest.state == "failed"
    assert latest.error_type == "RuntimeError"
    assert latest.completed_at is not None


@pytest.mark.unit
def test_stale_building_job_is_requeued(
    isolated_job_store,
    monkeypatch,
):
    created = index_jobs.enqueue_index_job(
        VIDEO_ID,
        "create",
    )

    claimed = index_jobs.claim_next_job()
    assert claimed is not None

    stale_time = (
        time.time()
        - index_jobs.INDEX_JOB_STALE_SECONDS
        - 1
    )

    with sqlite3.connect(isolated_job_store) as connection:
        connection.execute(
            """
            UPDATE index_jobs
            SET updated_at = ?
            WHERE job_id = ?
            """,
            (stale_time, created.job_id),
        )

    recovered = index_jobs.get_active_job(
        VIDEO_ID
    )

    assert recovered is not None
    assert recovered.job_id == created.job_id
    assert recovered.state == "queued"


@pytest.mark.unit
def test_worker_completes_job_when_index_becomes_valid(
    isolated_job_store,
    monkeypatch,
):
    created = index_jobs.enqueue_index_job(
        VIDEO_ID,
        "create",
    )

    claimed = index_jobs.claim_next_job()
    assert claimed is not None

    monkeypatch.setattr(
        index_jobs,
        "ensure_index_with_action",
        lambda video_id, languages=None: (
            object(),
            type(
                "Action",
                (),
                {"value": "create"},
            )(),
        ),
    )

    monkeypatch.setattr(
        index_jobs,
        "get_index_state",
        lambda video_id: IndexState.VALID,
    )

    index_jobs._run_job(
        claimed
    )

    latest = index_jobs.get_latest_job(
        VIDEO_ID
    )

    assert latest is not None
    assert latest.state == "ready"
    assert latest.action == "create"


@pytest.mark.unit
def test_worker_marks_failed_job_on_index_exception(
    isolated_job_store,
    monkeypatch,
):
    created = index_jobs.enqueue_index_job(
        VIDEO_ID,
        "create",
    )

    claimed = index_jobs.claim_next_job()
    assert claimed is not None

    monkeypatch.setattr(
        index_jobs,
        "ensure_index_with_action",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("secret provider failure")
        ),
    )

    index_jobs._run_job(
        claimed
    )

    latest = index_jobs.get_latest_job(
        VIDEO_ID
    )

    assert latest is not None
    assert latest.state == "failed"
    assert latest.error_type == "RuntimeError"
