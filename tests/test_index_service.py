import pytest

from Backend.rag_system import index_service
from Backend.rag_system.index_jobs import (
    JOB_BUILDING,
    JOB_FAILED,
    JOB_QUEUED,
    IndexJob,
)
from Backend.rag_system.indexing import IndexAction, IndexState


VIDEO_ID = "Gfr50f6ZBvo"


class FakeJobStore:
    def __init__(self, job=None):
        self.job = job
        self.enqueued = []

    def get(self, video_id):
        return self.job

    def enqueue(self, video_id, action, languages=None):
        self.job = IndexJob(
            job_id="job-1",
            video_id=video_id,
            status=JOB_QUEUED,
            action=action,
            languages=languages,
            attempt=0,
            error=None,
            created_at=0.0,
            updated_at=0.0,
            lease_until=0.0,
            worker_id=None,
        )
        self.enqueued.append((video_id, action, languages))
        return self.job


@pytest.mark.unit
def test_status_reports_missing_index(monkeypatch):
    monkeypatch.setattr(index_service, "recover_index_jobs", lambda: None)
    monkeypatch.setattr(index_service, "_JOB_STORE", FakeJobStore())
    monkeypatch.setattr(
        index_service,
        "get_index_state",
        lambda video_id: IndexState.MISSING,
    )

    status = index_service.get_index_status(VIDEO_ID)

    assert status.video_id == VIDEO_ID
    assert status.state == "missing"
    assert status.action == IndexAction.CREATE.value
    assert status.ready is False


@pytest.mark.unit
def test_status_reports_valid_index(monkeypatch):
    monkeypatch.setattr(index_service, "recover_index_jobs", lambda: None)
    monkeypatch.setattr(index_service, "_JOB_STORE", FakeJobStore())
    monkeypatch.setattr(
        index_service,
        "get_index_state",
        lambda video_id: IndexState.VALID,
    )

    status = index_service.get_index_status(VIDEO_ID)

    assert status.state == "ready"
    assert status.action == IndexAction.REUSE.value
    assert status.ready is True


@pytest.mark.unit
def test_submit_index_queues_missing_video(monkeypatch):
    store = FakeJobStore()
    dispatched = []

    monkeypatch.setattr(index_service, "recover_index_jobs", lambda: None)
    monkeypatch.setattr(index_service, "_JOB_STORE", store)
    monkeypatch.setattr(
        index_service,
        "_dispatch_video",
        lambda video_id: dispatched.append(video_id),
    )
    monkeypatch.setattr(
        index_service,
        "get_index_state",
        lambda video_id: IndexState.MISSING,
    )

    status = index_service.submit_index(VIDEO_ID, languages=["en"])

    assert status.video_id == VIDEO_ID
    assert status.state == "queued"
    assert status.action == IndexAction.CREATE.value
    assert status.ready is False
    assert store.enqueued == [
        (VIDEO_ID, IndexAction.CREATE.value, ["en"])
    ]
    assert dispatched == [VIDEO_ID]


@pytest.mark.unit
def test_submit_index_reuses_valid_index_without_queueing(monkeypatch):
    store = FakeJobStore()
    dispatched = []

    monkeypatch.setattr(index_service, "recover_index_jobs", lambda: None)
    monkeypatch.setattr(index_service, "_JOB_STORE", store)
    monkeypatch.setattr(
        index_service,
        "_dispatch_video",
        lambda video_id: dispatched.append(video_id),
    )
    monkeypatch.setattr(
        index_service,
        "get_index_state",
        lambda video_id: IndexState.VALID,
    )

    status = index_service.submit_index(VIDEO_ID)

    assert status.state == "ready"
    assert status.action == IndexAction.REUSE.value
    assert status.ready is True
    assert store.enqueued == []
    assert dispatched == []


@pytest.mark.unit
def test_load_ready_index_rejects_queued_index(monkeypatch):
    queued_job = IndexJob(
        job_id="job-1",
        video_id=VIDEO_ID,
        status=JOB_QUEUED,
        action=IndexAction.CREATE.value,
        languages=None,
        attempt=0,
        error=None,
        created_at=0.0,
        updated_at=0.0,
        lease_until=0.0,
        worker_id=None,
    )
    monkeypatch.setattr(index_service, "recover_index_jobs", lambda: None)
    monkeypatch.setattr(index_service, "_JOB_STORE", FakeJobStore(queued_job))
    monkeypatch.setattr(
        index_service,
        "get_index_state",
        lambda video_id: IndexState.MISSING,
    )
    monkeypatch.setattr(
        index_service,
        "load_vector_store",
        lambda video_id: pytest.fail("A queued index must not be loaded."),
    )

    with pytest.raises(index_service.IndexNotReadyError) as error:
        index_service.load_ready_index(VIDEO_ID)

    assert error.value.video_id == VIDEO_ID
    assert error.value.state == "queued"


@pytest.mark.unit
def test_load_ready_index_loads_valid_index(monkeypatch):
    vector_store = object()

    monkeypatch.setattr(index_service, "recover_index_jobs", lambda: None)
    monkeypatch.setattr(index_service, "_JOB_STORE", FakeJobStore())
    monkeypatch.setattr(
        index_service,
        "get_index_state",
        lambda video_id: IndexState.VALID,
    )
    monkeypatch.setattr(
        index_service,
        "load_vector_store",
        lambda video_id: vector_store,
    )

    assert index_service.load_ready_index(VIDEO_ID) is vector_store


@pytest.mark.unit
@pytest.mark.parametrize("video_id", ["", "not-valid", "short"])
def test_service_rejects_invalid_video_ids(video_id):
    with pytest.raises(ValueError):
        index_service.get_index_status(video_id)


@pytest.mark.unit
def test_status_reports_building_from_durable_job(monkeypatch):
    job = IndexJob(
        job_id="job-2",
        video_id=VIDEO_ID,
        status=JOB_BUILDING,
        action=IndexAction.REBUILD.value,
        languages=["en"],
        attempt=1,
        error=None,
        created_at=0.0,
        updated_at=0.0,
        lease_until=9999999999.0,
        worker_id="worker-1",
    )
    monkeypatch.setattr(index_service, "recover_index_jobs", lambda: None)
    monkeypatch.setattr(index_service, "_JOB_STORE", FakeJobStore(job))

    status = index_service.get_index_status(VIDEO_ID)

    assert status.state == "building"
    assert status.action == IndexAction.REBUILD.value
    assert status.ready is False


@pytest.mark.unit
def test_status_reports_failed_preparation_from_durable_job(monkeypatch):
    job = IndexJob(
        job_id="job-3",
        video_id=VIDEO_ID,
        status=JOB_FAILED,
        action=IndexAction.REBUILD.value,
        languages=["en"],
        attempt=1,
        error="Index preparation failed.",
        created_at=0.0,
        updated_at=0.0,
        lease_until=0.0,
        worker_id=None,
    )
    monkeypatch.setattr(index_service, "recover_index_jobs", lambda: None)
    monkeypatch.setattr(index_service, "_JOB_STORE", FakeJobStore(job))
    # This test is about durable job state. Keep the persisted-index state
    # explicit so a real local FAISS index cannot affect the unit test.
    monkeypatch.setattr(
        index_service,
        "get_index_state",
        lambda video_id: IndexState.MISSING,
    )

    status = index_service.get_index_status(VIDEO_ID)

    assert status.state == "failed"
    assert status.action == "retry"
    assert status.ready is False

@pytest.mark.unit
def test_failed_job_does_not_hide_a_valid_persisted_index(monkeypatch):
    job = IndexJob(
        job_id="job-4",
        video_id=VIDEO_ID,
        status=JOB_FAILED,
        action=IndexAction.REBUILD.value,
        languages=["en"],
        attempt=1,
        error="Index preparation failed.",
        created_at=0.0,
        updated_at=0.0,
        lease_until=0.0,
        worker_id=None,
    )
    monkeypatch.setattr(index_service, "recover_index_jobs", lambda: None)
    monkeypatch.setattr(index_service, "_JOB_STORE", FakeJobStore(job))
    monkeypatch.setattr(
        index_service,
        "get_index_state",
        lambda video_id: IndexState.VALID,
    )

    status = index_service.get_index_status(VIDEO_ID)

    assert status.state == "ready"
    assert status.action == IndexAction.REUSE.value
    assert status.ready is True

@pytest.mark.unit
def test_submit_index_returns_durable_job_id(monkeypatch):
    store = FakeJobStore()
    monkeypatch.setattr(index_service, "recover_index_jobs", lambda: None)
    monkeypatch.setattr(index_service, "_JOB_STORE", store)
    monkeypatch.setattr(
        index_service,
        "_dispatch_video",
        lambda video_id: None,
    )
    monkeypatch.setattr(
        index_service,
        "get_index_state",
        lambda video_id: IndexState.MISSING,
    )

    status = index_service.submit_index(VIDEO_ID)

    assert status.state == "queued"
    assert status.job_id == "job-1"


@pytest.mark.unit
def test_not_ready_error_carries_durable_job_id():
    error = index_service.IndexNotReadyError(
        VIDEO_ID,
        "building",
        "job-123",
    )

    assert error.video_id == VIDEO_ID
    assert error.state == "building"
    assert error.job_id == "job-123"

