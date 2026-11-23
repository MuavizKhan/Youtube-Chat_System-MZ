import sqlite3
import threading
import time

import pytest

from Backend.rag_system.index_jobs import (
    ActiveIndexJobLanguageConflictError,
    JOB_BUILDING,
    JOB_FAILED,
    JOB_QUEUED,
    JOB_READY,
    IndexJobStore,
)


VIDEO_ID = "Gfr50f6ZBvo"


@pytest.mark.unit
def test_job_store_closes_connection_after_successful_operation(
    tmp_path,
    monkeypatch,
):
    store = IndexJobStore(tmp_path / "jobs.sqlite3", lease_seconds=60)
    original_connect = store._connect
    connections = []

    def tracked_connect():
        connection = original_connect()
        connections.append(connection)
        return connection

    monkeypatch.setattr(store, "_connect", tracked_connect)

    assert store.get(VIDEO_ID) is None
    assert len(connections) == 1

    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connections[0].execute("SELECT 1")


@pytest.mark.unit
def test_job_store_closes_connection_and_rolls_back_after_error(tmp_path):
    store = IndexJobStore(tmp_path / "jobs.sqlite3", lease_seconds=60)
    connection = None

    with pytest.raises(RuntimeError, match="simulated operation failure"):
        with store._managed_connection() as managed_connection:
            connection = managed_connection
            managed_connection.execute("BEGIN IMMEDIATE")
            managed_connection.execute(
                """
                INSERT INTO index_jobs (
                    video_id, job_id, status, action, languages_json,
                    attempt, error, created_at, updated_at, lease_until, worker_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    VIDEO_ID,
                    "test-job",
                    JOB_QUEUED,
                    "create",
                    None,
                    0,
                    None,
                    time.time(),
                    time.time(),
                    0.0,
                    None,
                ),
            )
            raise RuntimeError("simulated operation failure")

    assert connection is not None
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")
    assert store.get(VIDEO_ID) is None


@pytest.mark.unit
def test_job_store_enqueue_and_read(tmp_path):
    store = IndexJobStore(tmp_path / "jobs.sqlite3", lease_seconds=60)

    job = store.enqueue(VIDEO_ID, action="create", languages=["en"])
    loaded = store.get(VIDEO_ID)

    assert loaded == job
    assert loaded.status == JOB_QUEUED
    assert loaded.action == "create"
    assert loaded.languages == ["en"]
    assert loaded.attempt == 0


@pytest.mark.unit
def test_job_store_deduplicates_active_jobs(tmp_path):
    store = IndexJobStore(tmp_path / "jobs.sqlite3", lease_seconds=60)

    first = store.enqueue(VIDEO_ID, action="create")
    second = store.enqueue(VIDEO_ID, action="create")

    assert second.job_id == first.job_id
    assert store.get(VIDEO_ID).job_id == first.job_id


@pytest.mark.unit
def test_job_store_rejects_active_job_with_different_language_preferences(tmp_path):
    store = IndexJobStore(tmp_path / "jobs.sqlite3", lease_seconds=60)
    original = store.enqueue(
        VIDEO_ID,
        action="create",
        languages=["en"],
    )

    with pytest.raises(
        ActiveIndexJobLanguageConflictError,
        match="different transcript-language preferences",
    ):
        store.enqueue(
            VIDEO_ID,
            action="rebuild",
            languages=["ar"],
        )

    assert store.get(VIDEO_ID).job_id == original.job_id


@pytest.mark.unit
def test_job_store_allows_only_one_worker_to_claim(tmp_path):
    store = IndexJobStore(tmp_path / "jobs.sqlite3", lease_seconds=60)
    store.enqueue(VIDEO_ID, action="create")

    results = []
    barrier = threading.Barrier(2)

    def claim(worker_id):
        barrier.wait()
        results.append(store.claim(VIDEO_ID, worker_id))

    threads = [
        threading.Thread(target=claim, args=("worker-a",)),
        threading.Thread(target=claim, args=("worker-b",)),
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    successful = [item for item in results if item is not None]

    assert len(successful) == 1
    assert successful[0].status == JOB_BUILDING
    assert successful[0].attempt == 1


@pytest.mark.unit
def test_job_store_recovers_expired_building_job(tmp_path):
    store = IndexJobStore(tmp_path / "jobs.sqlite3", lease_seconds=60)
    store.enqueue(VIDEO_ID, action="create")
    store.claim(VIDEO_ID, "worker-a")

    with store._managed_connection() as connection:
        connection.execute(
            "UPDATE index_jobs SET lease_until = ?, updated_at = ? WHERE video_id = ?",
            (time.time() - 1, time.time(), VIDEO_ID),
        )

    recovered = store.recover_expired_jobs()
    job = store.get(VIDEO_ID)

    assert recovered == 1
    assert job.status == JOB_QUEUED
    assert job.worker_id is None
    assert job.lease_until == 0


@pytest.mark.unit
def test_job_store_renews_active_lease(tmp_path):
    store = IndexJobStore(
        tmp_path / "jobs.sqlite3",
        lease_seconds=60,
    )
    store.enqueue(
        VIDEO_ID,
        action="create",
    )

    claimed = store.claim(
        VIDEO_ID,
        "worker-a",
    )
    assert claimed is not None

    before = store.get(VIDEO_ID).lease_until

    assert store.renew_lease(
        claimed.job_id,
        "worker-a",
    ) is True

    after = store.get(VIDEO_ID).lease_until

    assert after > before
    assert store.get(VIDEO_ID).worker_id == "worker-a"
    assert store.get(VIDEO_ID).status == JOB_BUILDING


@pytest.mark.unit
def test_job_store_records_ready_and_failed_states(tmp_path):
    store = IndexJobStore(tmp_path / "jobs.sqlite3", lease_seconds=60)

    ready = store.enqueue(VIDEO_ID, action="create")
    claimed = store.claim(VIDEO_ID, "worker-a")

    assert store.mark_ready(ready.job_id, "worker-a", "create") is True
    assert store.get(VIDEO_ID).status == JOB_READY

    retry = store.enqueue(VIDEO_ID, action="rebuild")
    store.claim(VIDEO_ID, "worker-b")

    assert store.mark_failed(
        retry.job_id,
        "worker-b",
        "Index preparation failed.",
    ) is True
    assert store.get(VIDEO_ID).status == JOB_FAILED
    assert claimed is not None

@pytest.mark.unit
def test_separate_store_instances_coordinate_claims(tmp_path):
    database = tmp_path / "jobs.sqlite3"
    first_store = IndexJobStore(database, lease_seconds=60)
    second_store = IndexJobStore(database, lease_seconds=60)

    first_store.enqueue(VIDEO_ID, action="create")

    barrier = threading.Barrier(2)
    results = []

    def claim(store, worker_id):
        barrier.wait()
        results.append(store.claim(VIDEO_ID, worker_id))

    threads = [
        threading.Thread(target=claim, args=(first_store, "worker-a")),
        threading.Thread(target=claim, args=(second_store, "worker-b")),
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    successful = [item for item in results if item is not None]

    assert len(successful) == 1
    assert successful[0].status == JOB_BUILDING




@pytest.mark.unit
def test_browser_transcript_revision_blocks_stale_ready_transition(tmp_path):
    store = IndexJobStore(tmp_path / "jobs.sqlite3", lease_seconds=60)
    first = store.enqueue(VIDEO_ID, action="create")
    assert store.claim(VIDEO_ID, "worker-a") is not None

    updated = store.enqueue_browser_transcript(
        VIDEO_ID,
        action="create",
        fingerprint="b" * 64,
    )
    assert updated.job_id == first.job_id
    assert updated.status == JOB_BUILDING
    assert updated.browser_transcript_fingerprint == "b" * 64

    assert store.mark_ready(
        first.job_id,
        "worker-a",
        "create",
        indexed_transcript_fingerprint="a" * 64,
    ) is False
    assert store.get(VIDEO_ID).status == JOB_BUILDING

    assert store.mark_ready(
        first.job_id,
        "worker-a",
        "create",
        indexed_transcript_fingerprint="b" * 64,
    ) is True
    assert store.get(VIDEO_ID).status == JOB_READY


@pytest.mark.unit
def test_browser_transcript_upload_after_completion_queues_rebuild(tmp_path):
    store = IndexJobStore(tmp_path / "jobs.sqlite3", lease_seconds=60)
    first = store.enqueue(VIDEO_ID, action="create")
    assert store.claim(VIDEO_ID, "worker-a") is not None
    assert store.mark_ready(
        first.job_id,
        "worker-a",
        "create",
        indexed_transcript_fingerprint=None,
    ) is True

    replacement = store.enqueue_browser_transcript(
        VIDEO_ID,
        action="rebuild",
        fingerprint="c" * 64,
    )

    assert replacement.job_id != first.job_id
    assert replacement.status == JOB_QUEUED
    assert replacement.action == "rebuild"
    assert replacement.browser_transcript_fingerprint == "c" * 64


@pytest.mark.unit
def test_browser_transcript_revision_prevents_failure_from_clobbering_newer_request(tmp_path):
    store = IndexJobStore(tmp_path / "jobs.sqlite3", lease_seconds=60)
    job = store.enqueue(VIDEO_ID, action="create")
    assert store.claim(VIDEO_ID, "worker-a") is not None
    store.enqueue_browser_transcript(
        VIDEO_ID,
        action="create",
        fingerprint="e" * 64,
    )

    assert store.mark_failed(
        job.job_id,
        "worker-a",
        "stale failure",
        expected_browser_transcript_fingerprint=None,
    ) is False
    assert store.requeue_for_browser_transcript(job.job_id, "worker-a") is True
    assert store.get(VIDEO_ID).status == JOB_QUEUED


@pytest.mark.unit
def test_legacy_job_database_is_migrated_with_browser_fingerprint_column(tmp_path):
    database = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute(
        """
        CREATE TABLE index_jobs (
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
    connection.commit()
    connection.close()

    store = IndexJobStore(database, lease_seconds=60)
    job = store.enqueue(VIDEO_ID, action="create")
    assert job.browser_transcript_fingerprint is None

    with store._managed_connection() as managed:
        columns = {
            row["name"]
            for row in managed.execute("PRAGMA table_info(index_jobs)").fetchall()
        }
    assert "browser_transcript_fingerprint" in columns
