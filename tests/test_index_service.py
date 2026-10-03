import pytest

from Backend.rag_system import index_service
from Backend.rag_system.indexing import IndexAction, IndexState


VIDEO_ID = "Gfr50f6ZBvo"


@pytest.mark.unit
def test_status_reports_missing_index(monkeypatch):
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
@pytest.mark.parametrize(
    "state,expected_action",
    [
        (IndexState.MISSING, IndexAction.CREATE),
        (IndexState.INVALID, IndexAction.REBUILD),
        (IndexState.STALE, IndexAction.REBUILD),
    ],
)
def test_prepare_index_delegates_lifecycle(
    monkeypatch,
    state,
    expected_action,
):
    current_state = {"value": state}
    ensure_calls = []

    monkeypatch.setattr(
        index_service,
        "get_index_state",
        lambda video_id: current_state["value"],
    )

    def fake_ensure_index_with_action(video_id, languages=None):
        ensure_calls.append((video_id, languages))
        current_state["value"] = IndexState.VALID
        return object(), expected_action

    monkeypatch.setattr(
        index_service,
        "ensure_index_with_action",
        fake_ensure_index_with_action,
    )

    result = index_service.prepare_index(
        VIDEO_ID,
        languages=["en"],
    )

    assert result.video_id == VIDEO_ID
    assert result.state == "ready"
    assert result.action == expected_action.value
    assert result.ready is True
    assert ensure_calls == [(VIDEO_ID, ["en"])]


@pytest.mark.unit
def test_prepare_index_reuses_valid_index(monkeypatch):
    monkeypatch.setattr(
        index_service,
        "get_index_state",
        lambda video_id: IndexState.VALID,
    )

    monkeypatch.setattr(
        index_service,
        "ensure_index_with_action",
        lambda *args, **kwargs: (object(), IndexAction.REUSE),
    )

    result = index_service.prepare_index(VIDEO_ID)

    assert result.action == IndexAction.REUSE.value
    assert result.ready is True


@pytest.mark.unit
def test_load_ready_index_rejects_missing_index(monkeypatch):
    monkeypatch.setattr(
        index_service,
        "get_index_state",
        lambda video_id: IndexState.MISSING,
    )

    monkeypatch.setattr(
        index_service,
        "load_vector_store",
        lambda video_id: pytest.fail(
            "A missing index must not be loaded."
        ),
    )

    with pytest.raises(index_service.IndexNotReadyError) as error:
        index_service.load_ready_index(VIDEO_ID)

    assert error.value.video_id == VIDEO_ID
    assert error.value.state is IndexState.MISSING


@pytest.mark.unit
def test_load_ready_index_loads_valid_index(monkeypatch):
    vector_store = object()

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
def test_status_reports_building_without_reading_disk_state(monkeypatch):
    monkeypatch.setattr(
        index_service,
        "_set_runtime_status",
        lambda video_id, state: index_service._RUNTIME_STATUS.__setitem__(video_id, state)
        if state is not None
        else index_service._RUNTIME_STATUS.pop(video_id, None),
    )
    index_service._set_runtime_status(VIDEO_ID, "building")
    monkeypatch.setattr(
        index_service,
        "get_index_state",
        lambda video_id: pytest.fail(
            "Building state should be served from runtime status."
        ),
    )

    try:
        status = index_service.get_index_status(VIDEO_ID)
        assert status.state == "building"
        assert status.action == "building"
        assert status.ready is False
    finally:
        index_service._set_runtime_status(VIDEO_ID, None)


@pytest.mark.unit
def test_status_reports_failed_preparation(monkeypatch):
    index_service._set_runtime_status(VIDEO_ID, "failed")
    try:
        status = index_service.get_index_status(VIDEO_ID)
        assert status.state == "failed"
        assert status.action == "retry"
        assert status.ready is False
    finally:
        index_service._set_runtime_status(VIDEO_ID, None)
