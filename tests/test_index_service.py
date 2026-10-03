import pytest

from types import SimpleNamespace

from Backend.rag_system import index_service
from Backend.rag_system.indexing import IndexAction, IndexState


VIDEO_ID = "Gfr50f6ZBvo"


def disable_worker(monkeypatch):
    monkeypatch.setattr(
        index_service,
        "ensure_worker_started",
        lambda: None,
    )


@pytest.mark.unit
def test_status_reports_missing_index(monkeypatch):
    disable_worker(monkeypatch)
    monkeypatch.setattr(
        index_service,
        "get_active_job",
        lambda video_id: None,
    )
    monkeypatch.setattr(
        index_service,
        "get_latest_job",
        lambda video_id: None,
    )
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
    assert status.job_id is None


@pytest.mark.unit
def test_status_reports_valid_index(monkeypatch):
    disable_worker(monkeypatch)
    monkeypatch.setattr(
        index_service,
        "get_active_job",
        lambda video_id: None,
    )
    monkeypatch.setattr(
        index_service,
        "get_latest_job",
        lambda video_id: None,
    )
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
def test_status_reports_active_job(monkeypatch):
    disable_worker(monkeypatch)

    active_job = SimpleNamespace(
        job_id="job-123",
        state="building",
        action="create",
    )

    monkeypatch.setattr(
        index_service,
        "get_active_job",
        lambda video_id: active_job,
    )

    status = index_service.get_index_status(VIDEO_ID)

    assert status.state == "building"
    assert status.action == "create"
    assert status.ready is False
    assert status.job_id == "job-123"


@pytest.mark.unit
def test_status_reports_failed_job_when_index_is_not_ready(monkeypatch):
    disable_worker(monkeypatch)

    failed_job = SimpleNamespace(
        job_id="job-failed",
        state="failed",
        action="rebuild",
    )

    monkeypatch.setattr(
        index_service,
        "get_active_job",
        lambda video_id: None,
    )
    monkeypatch.setattr(
        index_service,
        "get_latest_job",
        lambda video_id: failed_job,
    )
    monkeypatch.setattr(
        index_service,
        "get_index_state",
        lambda video_id: IndexState.STALE,
    )

    status = index_service.get_index_status(VIDEO_ID)

    assert status.state == "failed"
    assert status.action == "retry"
    assert status.ready is False
    assert status.job_id == "job-failed"


@pytest.mark.unit
@pytest.mark.parametrize(
    "state,expected_action",
    [
        (IndexState.MISSING, IndexAction.CREATE),
        (IndexState.INVALID, IndexAction.REBUILD),
        (IndexState.STALE, IndexAction.REBUILD),
    ],
)
def test_prepare_index_enqueues_lifecycle_action(
    monkeypatch,
    state,
    expected_action,
):
    disable_worker(monkeypatch)
    monkeypatch.setattr(
        index_service,
        "get_index_state",
        lambda video_id: state,
    )

    expected_job = SimpleNamespace(
        job_id="job-created",
        state="queued",
        action=expected_action.value,
    )

    calls = []

    def fake_enqueue(video_id, action, languages=None):
        calls.append((video_id, action, languages))
        return expected_job

    monkeypatch.setattr(
        index_service,
        "enqueue_index_job",
        fake_enqueue,
    )

    result = index_service.prepare_index(
        VIDEO_ID,
        languages=["en"],
    )

    assert result.video_id == VIDEO_ID
    assert result.state == "queued"
    assert result.action == expected_action.value
    assert result.ready is False
    assert result.job_id == "job-created"
    assert calls == [(VIDEO_ID, expected_action.value, ["en"])]


@pytest.mark.unit
def test_prepare_index_reuses_valid_index(monkeypatch):
    disable_worker(monkeypatch)
    monkeypatch.setattr(
        index_service,
        "get_index_state",
        lambda video_id: IndexState.VALID,
    )

    result = index_service.prepare_index(VIDEO_ID)

    assert result.state == "ready"
    assert result.action == IndexAction.REUSE.value
    assert result.ready is True
    assert result.job_id is None


@pytest.mark.unit
def test_load_ready_index_rejects_missing_index(monkeypatch):
    disable_worker(monkeypatch)

    monkeypatch.setattr(
        index_service,
        "get_index_status",
        lambda video_id: index_service.IndexStatus(
            video_id=VIDEO_ID,
            state="missing",
            action="create",
            ready=False,
            job_id=None,
        ),
    )

    monkeypatch.setattr(
        index_service,
        "load_vector_store",
        lambda video_id: pytest.fail(
            "A not-ready index must not be loaded."
        ),
    )

    with pytest.raises(index_service.IndexNotReadyError) as error:
        index_service.load_ready_index(VIDEO_ID)

    assert error.value.video_id == VIDEO_ID
    assert error.value.state == "missing"
    assert error.value.job_id is None


@pytest.mark.unit
def test_load_ready_index_loads_valid_index(monkeypatch):
    vector_store = object()

    monkeypatch.setattr(
        index_service,
        "get_index_status",
        lambda video_id: index_service.IndexStatus(
            video_id=VIDEO_ID,
            state="ready",
            action="reuse",
            ready=True,
            job_id=None,
        ),
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
