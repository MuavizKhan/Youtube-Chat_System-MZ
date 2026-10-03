import threading
import time

import pytest

from Backend.rag_system.index_jobs import (
    JOB_BUILDING,
    JOB_FAILED,
    JOB_QUEUED,
    JOB_READY,
    IndexJobStore,
)


VIDEO_ID = "Gfr50f6ZBvo"


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

    with store._connect() as connection:
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
