import pytest

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