import json

from Backend.rag_system.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    VECTOR_STORE_ROOT,
)

from Backend.rag_system.indexing import (
    INDEX_METADATA_VERSION,
    IndexState,
    get_index_state,
)


def create_index(
    video_id: str,
    root_path,
):
    video_store_path = root_path / video_id

    video_store_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    (video_store_path / "index.faiss").write_bytes(
        b"fake-faiss-index"
    )

    (video_store_path / "index.pkl").write_bytes(
        b"fake-pickle-index"
    )

    metadata = {
        "metadata_version": INDEX_METADATA_VERSION,
        "video_id": video_id,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunk_count": 10,
        "transcript_language": "English",
        "transcript_language_code": "en",
        "transcript_generated": True,
    }

    (video_store_path / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    return video_store_path


def test_missing_index_returns_missing(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "Backend.rag_system.indexing.VECTOR_STORE_ROOT",
        tmp_path,
    )

    assert (
        get_index_state("dQw4w9WgXcQ")
        is IndexState.MISSING
    )


def test_valid_index_returns_valid(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "Backend.rag_system.indexing.VECTOR_STORE_ROOT",
        tmp_path,
    )

    create_index(
        "dQw4w9WgXcQ",
        tmp_path,
    )

    assert (
        get_index_state("dQw4w9WgXcQ")
        is IndexState.VALID
    )


def test_missing_faiss_file_returns_invalid(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "Backend.rag_system.indexing.VECTOR_STORE_ROOT",
        tmp_path,
    )

    path = create_index(
        "dQw4w9WgXcQ",
        tmp_path,
    )

    (path / "index.faiss").unlink()

    assert (
        get_index_state("dQw4w9WgXcQ")
        is IndexState.INVALID
    )


def test_missing_pickle_file_returns_invalid(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "Backend.rag_system.indexing.VECTOR_STORE_ROOT",
        tmp_path,
    )

    path = create_index(
        "dQw4w9WgXcQ",
        tmp_path,
    )

    (path / "index.pkl").unlink()

    assert (
        get_index_state("dQw4w9WgXcQ")
        is IndexState.INVALID
    )


def test_missing_metadata_file_returns_invalid(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "Backend.rag_system.indexing.VECTOR_STORE_ROOT",
        tmp_path,
    )

    path = create_index(
        "dQw4w9WgXcQ",
        tmp_path,
    )

    (path / "metadata.json").unlink()

    assert (
        get_index_state("dQw4w9WgXcQ")
        is IndexState.INVALID
    )


def test_malformed_metadata_returns_invalid(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "Backend.rag_system.indexing.VECTOR_STORE_ROOT",
        tmp_path,
    )

    path = create_index(
        "dQw4w9WgXcQ",
        tmp_path,
    )

    (path / "metadata.json").write_text(
        "{invalid json",
        encoding="utf-8",
    )

    assert (
        get_index_state("dQw4w9WgXcQ")
        is IndexState.INVALID
    )


def test_wrong_video_id_returns_invalid(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "Backend.rag_system.indexing.VECTOR_STORE_ROOT",
        tmp_path,
    )

    path = create_index(
        "dQw4w9WgXcQ",
        tmp_path,
    )

    metadata = json.loads(
        (path / "metadata.json").read_text(
            encoding="utf-8"
        )
    )

    metadata["video_id"] = "abcdefghijk"

    (path / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    assert (
        get_index_state("dQw4w9WgXcQ")
        is IndexState.INVALID
    )


def test_wrong_metadata_version_returns_invalid(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "Backend.rag_system.indexing.VECTOR_STORE_ROOT",
        tmp_path,
    )

    path = create_index(
        "dQw4w9WgXcQ",
        tmp_path,
    )

    metadata = json.loads(
        (path / "metadata.json").read_text(
            encoding="utf-8"
        )
    )

    metadata["metadata_version"] = 999

    (path / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    assert (
        get_index_state("dQw4w9WgXcQ")
        is IndexState.INVALID
    )


def test_wrong_embedding_model_returns_stale(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "Backend.rag_system.indexing.VECTOR_STORE_ROOT",
        tmp_path,
    )

    path = create_index(
        "dQw4w9WgXcQ",
        tmp_path,
    )

    metadata = json.loads(
        (path / "metadata.json").read_text(
            encoding="utf-8"
        )
    )

    metadata["embedding_model"] = "older-embedding-model"

    (path / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    assert (
        get_index_state("dQw4w9WgXcQ")
        is IndexState.STALE
    )


def test_wrong_chunk_size_returns_stale(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "Backend.rag_system.indexing.VECTOR_STORE_ROOT",
        tmp_path,
    )

    path = create_index(
        "dQw4w9WgXcQ",
        tmp_path,
    )

    metadata = json.loads(
        (path / "metadata.json").read_text(
            encoding="utf-8"
        )
    )

    metadata["chunk_size"] = CHUNK_SIZE + 100

    (path / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    assert (
        get_index_state("dQw4w9WgXcQ")
        is IndexState.STALE
    )


def test_wrong_chunk_overlap_returns_stale(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "Backend.rag_system.indexing.VECTOR_STORE_ROOT",
        tmp_path,
    )

    path = create_index(
        "dQw4w9WgXcQ",
        tmp_path,
    )

    metadata = json.loads(
        (path / "metadata.json").read_text(
            encoding="utf-8"
        )
    )

    metadata["chunk_overlap"] = CHUNK_OVERLAP + 50

    (path / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8"
    )

    assert (
        get_index_state("dQw4w9WgXcQ")
        is IndexState.STALE
    )


def test_stale_configuration_is_not_invalid(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "Backend.rag_system.indexing.VECTOR_STORE_ROOT",
        tmp_path,
    )

    path = create_index(
        "dQw4w9WgXcQ",
        tmp_path,
    )

    metadata = json.loads(
        (path / "metadata.json").read_text(
            encoding="utf-8"
        )
    )

    metadata["embedding_model"] = "older-embedding-model"

    (path / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8"
    )

    state = get_index_state(
        "dQw4w9WgXcQ"
    )

    assert state is IndexState.STALE
    assert state is not IndexState.INVALID