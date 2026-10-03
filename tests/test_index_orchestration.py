import pytest
import threading
import time

from Backend.rag_system import indexing


class FakeVectorStore:
    pass


@pytest.fixture
def disable_cache_clear(monkeypatch):
    calls = []

    monkeypatch.setattr(
        indexing,
        "clear_vector_store_cache",
        lambda: calls.append("clear"),
    )

    return calls

def test_ensure_index_rejects_empty_video_id():
    with pytest.raises(
        ValueError,
        match="video_id cannot be empty",
    ):
        indexing.ensure_index("")


def test_ensure_index_rejects_whitespace_video_id():
    with pytest.raises(
        ValueError,
        match="video_id cannot be empty",
    ):
        indexing.ensure_index("   ")
        
        
def test_ensure_index_reuses_valid_index(
    monkeypatch,
    disable_cache_clear,
):
    vector_store = FakeVectorStore()

    calls = {
        "create": 0,
        "load": 0,
    }

    monkeypatch.setattr(
        indexing,
        "get_index_state",
        lambda video_id: indexing.IndexState.VALID,
    )

    monkeypatch.setattr(
        indexing,
        "create_index",
        lambda *args, **kwargs: (
            calls.__setitem__(
                "create",
                calls["create"] + 1,
            )
        ),
    )

    def fake_load(video_id):
        calls["load"] += 1
        return vector_store

    monkeypatch.setattr(
        indexing,
        "load_vector_store",
        fake_load,
    )

    result = indexing.ensure_index(
        "Gfr50f6ZBvo"
    )

    assert result is vector_store
    assert calls["create"] == 0
    assert calls["load"] == 1
    assert disable_cache_clear == []


def test_ensure_index_serializes_same_video_creation(
    monkeypatch,
):
    video_id = "Gfr50f6ZBvo"
    vector_store = FakeVectorStore()

    create_started = threading.Event()
    allow_create_to_finish = threading.Event()

    state_lock = threading.Lock()

    states = {
        video_id: indexing.IndexState.MISSING,
    }

    create_calls = 0
    create_calls_lock = threading.Lock()

    def fake_get_index_state(requested_video_id):
        with state_lock:
            return states[requested_video_id]

    def fake_create_index(
        requested_video_id,
        languages=None,
    ):
        nonlocal create_calls

        with create_calls_lock:
            create_calls += 1

        create_started.set()

        if not allow_create_to_finish.wait(
            timeout=2
        ):
            raise RuntimeError(
                "Timed out waiting for test release."
            )

        with state_lock:
            states[requested_video_id] = (
                indexing.IndexState.VALID
            )

        return vector_store

    monkeypatch.setattr(
        indexing,
        "get_index_state",
        fake_get_index_state,
    )

    monkeypatch.setattr(
        indexing,
        "create_index",
        fake_create_index,
    )

    monkeypatch.setattr(
        indexing,
        "clear_vector_store_cache",
        lambda: None,
    )

    monkeypatch.setattr(
        indexing,
        "load_vector_store",
        lambda requested_video_id: vector_store,
    )

    results = []
    errors = []

    def run():
        try:
            results.append(
                indexing.ensure_index(
                    video_id
                )
            )
        except Exception as error:
            errors.append(error)

    first = threading.Thread(
        target=run
    )

    second = threading.Thread(
        target=run
    )

    first.start()

    assert create_started.wait(
        timeout=2
    )

    second.start()

    # The second request must still be waiting for the
    # same video's lifecycle lock.
    assert second.is_alive()

    allow_create_to_finish.set()

    first.join(timeout=2)
    second.join(timeout=2)

    assert not errors
    assert create_calls == 1
    assert len(results) == 2
    assert all(
        result is vector_store
        for result in results
    )


def test_ensure_index_allows_different_video_ids_to_progress(
    monkeypatch,
):
    first_video = "Gfr50f6ZBvo"
    second_video = "dQw4w9WgXcQ"

    vector_store = FakeVectorStore()

    first_started = threading.Event()
    second_started = threading.Event()
    release_creates = threading.Event()

    states = {
        first_video: indexing.IndexState.MISSING,
        second_video: indexing.IndexState.MISSING,
    }

    def fake_get_index_state(video_id):
        return states[video_id]

    def fake_create_index(
        video_id,
        languages=None,
    ):
        if video_id == first_video:
            first_started.set()
        else:
            second_started.set()

        if not release_creates.wait(
            timeout=2
        ):
            raise RuntimeError(
                "Timed out waiting for test release."
            )

        states[video_id] = (
            indexing.IndexState.VALID
        )

        return vector_store

    monkeypatch.setattr(
        indexing,
        "get_index_state",
        fake_get_index_state,
    )

    monkeypatch.setattr(
        indexing,
        "create_index",
        fake_create_index,
    )

    monkeypatch.setattr(
        indexing,
        "clear_vector_store_cache",
        lambda: None,
    )

    monkeypatch.setattr(
        indexing,
        "load_vector_store",
        lambda video_id: vector_store,
    )

    errors = []

    def run(video_id):
        try:
            indexing.ensure_index(video_id)
        except Exception as error:
            errors.append(error)

    first = threading.Thread(
        target=run,
        args=(first_video,),
    )

    second = threading.Thread(
        target=run,
        args=(second_video,),
    )

    first.start()
    second.start()

    assert first_started.wait(
        timeout=2
    )

    assert second_started.wait(
        timeout=2
    )

    release_creates.set()

    first.join(timeout=2)
    second.join(timeout=2)
    assert not errors


def test_ensure_index_rejects_invalid_final_state(
    monkeypatch,
    disable_cache_clear,
):
    state_sequence = iter(
        [
            indexing.IndexState.MISSING,
            indexing.IndexState.INVALID,
        ]
    )

    monkeypatch.setattr(
        indexing,
        "get_index_state",
        lambda video_id: next(state_sequence),
    )

    monkeypatch.setattr(
        indexing,
        "create_index",
        lambda video_id, languages=None: None,
    )

    with pytest.raises(
        RuntimeError,
        match="persisted index is not valid",
    ):
        indexing.ensure_index(
            "Gfr50f6ZBvo"
        )

    assert disable_cache_clear == [
        "clear",
        "clear",
    ]



def test_ensure_index_creates_missing_index(
    monkeypatch,
    disable_cache_clear,
):
    vector_store = FakeVectorStore()

    state_sequence = iter(
        [
            indexing.IndexState.MISSING,
            indexing.IndexState.VALID,
        ]
    )

    calls = {
        "create": 0,
        "load": 0,
    }

    monkeypatch.setattr(
        indexing,
        "get_index_state",
        lambda video_id: next(state_sequence),
    )

    def fake_create(
        video_id,
        languages=None,
    ):
        calls["create"] += 1
        return vector_store

    def fake_load(video_id):
        calls["load"] += 1
        return vector_store

    monkeypatch.setattr(
        indexing,
        "create_index",
        fake_create,
    )

    monkeypatch.setattr(
        indexing,
        "load_vector_store",
        fake_load,
    )

    result = indexing.ensure_index(
        "Gfr50f6ZBvo"
    )

    assert result is vector_store
    assert calls["create"] == 1
    assert calls["load"] == 1
    assert disable_cache_clear == [
        "clear",
        "clear",
    ]


def test_ensure_index_rebuilds_invalid_index(
    monkeypatch,
    disable_cache_clear,
):
    vector_store = FakeVectorStore()

    state_sequence = iter(
        [
            indexing.IndexState.INVALID,
            indexing.IndexState.VALID,
        ]
    )

    calls = {
        "create": 0,
        "load": 0,
    }

    monkeypatch.setattr(
        indexing,
        "get_index_state",
        lambda video_id: next(state_sequence),
    )

    monkeypatch.setattr(
        indexing,
        "create_index",
        lambda video_id, languages=None: (
            calls.__setitem__(
                "create",
                calls["create"] + 1,
            )
            or vector_store
        ),
    )

    monkeypatch.setattr(
        indexing,
        "load_vector_store",
        lambda video_id: (
            calls.__setitem__(
                "load",
                calls["load"] + 1,
            )
            or vector_store
        ),
    )

    result = indexing.ensure_index(
        "Gfr50f6ZBvo"
    )

    assert result is vector_store
    assert calls["create"] == 1
    assert calls["load"] == 1
    assert disable_cache_clear == [
        "clear",
        "clear",
    ]


def test_ensure_index_rebuilds_stale_index(
    monkeypatch,
    disable_cache_clear,
):
    vector_store = FakeVectorStore()

    state_sequence = iter(
        [
            indexing.IndexState.STALE,
            indexing.IndexState.VALID,
        ]
    )

    calls = {
        "create": 0,
        "load": 0,
    }

    monkeypatch.setattr(
        indexing,
        "get_index_state",
        lambda video_id: next(state_sequence),
    )

    monkeypatch.setattr(
        indexing,
        "create_index",
        lambda video_id, languages=None: (
            calls.__setitem__(
                "create",
                calls["create"] + 1,
            )
            or vector_store
        ),
    )

    monkeypatch.setattr(
        indexing,
        "load_vector_store",
        lambda video_id: (
            calls.__setitem__(
                "load",
                calls["load"] + 1,
            )
            or vector_store
        ),
    )

    result = indexing.ensure_index(
        "Gfr50f6ZBvo"
    )

    assert result is vector_store
    assert calls["create"] == 1
    assert calls["load"] == 1
    assert disable_cache_clear == [
        "clear",
        "clear",
    ]


def test_ensure_index_propagates_load_failure(
    monkeypatch,
    disable_cache_clear,
):
    monkeypatch.setattr(
        indexing,
        "get_index_state",
        lambda video_id: indexing.IndexState.VALID,
    )

    def fake_load(video_id):
        raise RuntimeError(
            "FAISS load failed"
        )

    monkeypatch.setattr(
        indexing,
        "load_vector_store",
        fake_load,
    )

    with pytest.raises(
        RuntimeError,
        match="FAISS load failed",
    ):
        indexing.ensure_index(
            "Gfr50f6ZBvo"
        )

    assert disable_cache_clear == []


def test_ensure_index_passes_languages_to_create_index(
    monkeypatch,
    disable_cache_clear,
):
    vector_store = FakeVectorStore()

    state_sequence = iter(
        [
            indexing.IndexState.MISSING,
            indexing.IndexState.VALID,
        ]
    )

    received = {}

    monkeypatch.setattr(
        indexing,
        "get_index_state",
        lambda video_id: next(state_sequence),
    )

    def fake_create(
        video_id,
        languages=None,
    ):
        received["video_id"] = video_id
        received["languages"] = languages
        return vector_store

    monkeypatch.setattr(
        indexing,
        "create_index",
        fake_create,
    )

    monkeypatch.setattr(
        indexing,
        "load_vector_store",
        lambda video_id: vector_store,
    )

    result = indexing.ensure_index(
        "Gfr50f6ZBvo",
        languages=["en", "hi"],
    )

    assert result is vector_store
    assert received == {
        "video_id": "Gfr50f6ZBvo",
        "languages": ["en", "hi"],
    }


def test_ensure_index_rejects_failed_creation(
    monkeypatch,
    disable_cache_clear,
):
    monkeypatch.setattr(
        indexing,
        "get_index_state",
        lambda video_id: indexing.IndexState.MISSING,
    )

    def fake_create(
        video_id,
        languages=None,
    ):
        raise RuntimeError(
            "transcript unavailable"
        )

    monkeypatch.setattr(
        indexing,
        "create_index",
        fake_create,
    )

    with pytest.raises(
        RuntimeError,
        match="transcript unavailable",
    ):
        indexing.ensure_index(
            "Gfr50f6ZBvo"
        )

    assert disable_cache_clear == [
        "clear",
    ]