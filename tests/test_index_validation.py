import json

from Backend.rag_system.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    VECTOR_STORE_ROOT,
)

from Backend.rag_system.indexing import (
    INDEX_METADATA_VERSION,
    validate_vector_store,
)


def create_valid_index(
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


def test_valid_index_returns_true(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "Backend.rag_system.indexing.VECTOR_STORE_ROOT",
        tmp_path,
    )

    create_valid_index(
        "dQw4w9WgXcQ",
        tmp_path,
    )

    assert validate_vector_store("dQw4w9WgXcQ") is True


def test_missing_vector_store_returns_false(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "Backend.rag_system.indexing.VECTOR_STORE_ROOT",
        tmp_path,
    )

    assert validate_vector_store("dQw4w9WgXcQ") is False


def test_missing_faiss_file_returns_false(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "Backend.rag_system.indexing.VECTOR_STORE_ROOT",
        tmp_path,
    )

    path = create_valid_index(
            "dQw4w9WgXcQ",
            tmp_path,
        )
    (path / "index.faiss").unlink()

    assert validate_vector_store("dQw4w9WgXcQ") is False


def test_missing_pickle_file_returns_false(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "Backend.rag_system.indexing.VECTOR_STORE_ROOT",
        tmp_path,
    )

    path = create_valid_index(
        "dQw4w9WgXcQ",
        tmp_path,
    )
    (path / "index.pkl").unlink()

    assert validate_vector_store("dQw4w9WgXcQ") is False


def test_missing_metadata_file_returns_false(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "Backend.rag_system.indexing.VECTOR_STORE_ROOT",
        tmp_path,
    )

    path = create_valid_index("dQw4w9WgXcQ", tmp_path)
    (path / "metadata.json").unlink()

    assert validate_vector_store("dQw4w9WgXcQ") is False


def test_malformed_metadata_returns_false(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "Backend.rag_system.indexing.VECTOR_STORE_ROOT",
        tmp_path,
    )

    path = create_valid_index("dQw4w9WgXcQ", tmp_path)

    (path / "metadata.json").write_text(
        "{invalid json",
        encoding="utf-8",
    )

    assert validate_vector_store("dQw4w9WgXcQ") is False


def test_missing_required_metadata_returns_false(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "Backend.rag_system.indexing.VECTOR_STORE_ROOT",
        tmp_path,
    )

    path = create_valid_index("dQw4w9WgXcQ", tmp_path)

    metadata = json.loads(
        (path / "metadata.json").read_text(
            encoding="utf-8"
        )
    )

    del metadata["embedding_model"]

    (path / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    assert validate_vector_store("dQw4w9WgXcQ") is False


def test_wrong_video_id_returns_false(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "Backend.rag_system.indexing.VECTOR_STORE_ROOT",
        tmp_path,
    )

    path = create_valid_index("dQw4w9WgXcQ", tmp_path)

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

    assert validate_vector_store("dQw4w9WgXcQ") is False


def test_wrong_embedding_model_returns_false(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "Backend.rag_system.indexing.VECTOR_STORE_ROOT",
        tmp_path,
    )

    path = create_valid_index("dQw4w9WgXcQ", tmp_path)

    metadata = json.loads(
        (path / "metadata.json").read_text(
            encoding="utf-8"
        )
    )

    metadata["embedding_model"] = "wrong-model"

    (path / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    assert validate_vector_store("dQw4w9WgXcQ") is False


def test_wrong_chunk_configuration_returns_false(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "Backend.rag_system.indexing.VECTOR_STORE_ROOT",
        tmp_path,
    )

    path = create_valid_index("dQw4w9WgXcQ", tmp_path)

    metadata = json.loads(
        (path / "metadata.json").read_text(
            encoding="utf-8"
        )
    )

    metadata["chunk_size"] = 500

    (path / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    assert validate_vector_store("dQw4w9WgXcQ") is False


def test_invalid_chunk_count_returns_false(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "Backend.rag_system.indexing.VECTOR_STORE_ROOT",
        tmp_path,
    )

    path = create_valid_index("dQw4w9WgXcQ", tmp_path)

    metadata = json.loads(
        (path / "metadata.json").read_text(
            encoding="utf-8"
        )
    )

    metadata["chunk_count"] = 0

    (path / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    assert validate_vector_store("dQw4w9WgXcQ") is False


def test_unsupported_metadata_version_returns_false(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "Backend.rag_system.indexing.VECTOR_STORE_ROOT",
        tmp_path,
    )

    path = create_valid_index("dQw4w9WgXcQ", tmp_path)

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

    assert validate_vector_store("dQw4w9WgXcQ") is False
    
    