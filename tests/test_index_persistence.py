import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from Backend.rag_system import indexing


class FakeVectorStore:
    def __init__(self, fail_on_save=False):
        self.fail_on_save = fail_on_save

    def save_local(self, path):
        if self.fail_on_save:
            raise RuntimeError("simulated save failure")

        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)

        (target / "index.faiss").write_bytes(b"fake-faiss")
        (target / "index.pkl").write_bytes(b"fake-pickle")


def _chunks(video_id="abc123"):
    return [
        Document(
            page_content="test content",
            metadata={
                "video_id": video_id,
                "language": "English",
                "language_code": "en",
                "is_generated": True,
            },
        )
    ]

def test_save_vector_store_preserves_existing_index_when_staged_save_fails(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        indexing,
        "VECTOR_STORE_ROOT",
        tmp_path,
    )

    existing_path = tmp_path / "abc123"
    existing_path.mkdir()

    (existing_path / "index.faiss").write_bytes(
        b"original-faiss"
    )
    (existing_path / "index.pkl").write_bytes(
        b"original-pickle"
    )

    existing_metadata = indexing.build_index_metadata(
        _chunks("abc123")
    )

    with (existing_path / "metadata.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            existing_metadata,
            file,
        )

    class PartiallyFailingVectorStore:
        def save_local(self, path):
            target = Path(path)
            target.mkdir(
                parents=True,
                exist_ok=True,
            )

            (target / "index.faiss").write_bytes(
                b"new-faiss"
            )

            raise RuntimeError(
                "simulated failure after partial write"
            )

    with pytest.raises(
        RuntimeError,
        match="simulated failure after partial write",
    ):
        indexing.save_vector_store(
            PartiallyFailingVectorStore(),
            "abc123",
            _chunks("abc123"),
        )

    assert existing_path.is_dir()

    assert (
        (existing_path / "index.faiss").read_bytes()
        == b"original-faiss"
    )

    assert (
        (existing_path / "index.pkl").read_bytes()
        == b"original-pickle"
    )

    with (existing_path / "metadata.json").open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    assert metadata == existing_metadata

def test_save_vector_store_creates_valid_index(tmp_path, monkeypatch):
    monkeypatch.setattr(
        indexing,
        "VECTOR_STORE_ROOT",
        tmp_path,
    )

    saved_path = indexing.save_vector_store(
        FakeVectorStore(),
        "abc123",
        _chunks(),
    )

    assert saved_path == tmp_path / "abc123"
    assert (saved_path / "index.faiss").is_file()
    assert (saved_path / "index.pkl").is_file()
    assert (saved_path / "metadata.json").is_file()

    with (saved_path / "metadata.json").open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    assert metadata["video_id"] == "abc123"


def test_save_vector_store_preserves_existing_index_on_save_failure(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        indexing,
        "VECTOR_STORE_ROOT",
        tmp_path,
    )

    existing_path = tmp_path / "abc123"
    existing_path.mkdir()

    (existing_path / "index.faiss").write_bytes(
        b"original-faiss"
    )
    (existing_path / "index.pkl").write_bytes(
        b"original-pickle"
    )

    existing_metadata = {
        "metadata_version": 1,
        "video_id": "abc123",
        "embedding_model": indexing.EMBEDDING_MODEL,
        "chunk_size": indexing.CHUNK_SIZE,
        "chunk_overlap": indexing.CHUNK_OVERLAP,
        "chunk_count": 1,
        "transcript_language": "English",
        "transcript_language_code": "en",
        "transcript_generated": True,
    }

    with (existing_path / "metadata.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(existing_metadata, file)

    with pytest.raises(
        RuntimeError,
        match="simulated save failure",
    ):
        indexing.save_vector_store(
            FakeVectorStore(fail_on_save=True),
            "abc123",
            _chunks(),
        )

    assert existing_path.is_dir()
    assert (
        (existing_path / "index.faiss").read_bytes()
        == b"original-faiss"
    )
    assert (
        (existing_path / "index.pkl").read_bytes()
        == b"original-pickle"
    )


def test_save_vector_store_preserves_backup_when_restore_fails(
    monkeypatch,
    tmp_path,
):
    from Backend.rag_system import indexing

    video_id = "Gfr50f6ZBvo"

    monkeypatch.setattr(
        indexing,
        "VECTOR_STORE_ROOT",
        tmp_path,
    )

    final_path = tmp_path / video_id
    final_path.mkdir()

    (final_path / "index.faiss").write_text(
        "old-faiss",
        encoding="utf-8",
    )
    (final_path / "index.pkl").write_text(
        "old-pickle",
        encoding="utf-8",
    )
    (final_path / "metadata.json").write_text(
        "{}",
        encoding="utf-8",
    )

    chunks = [
        Document(
            page_content="test content",
            metadata={
                "video_id": video_id,
                "language": "English",
                "language_code": "en",
                "is_generated": True,
                "start": 0.0,
                "duration": 1.0,
            },
        )
    ]

    class FakeVectorStore:
        def save_local(self, path):
            path = Path(path)

            (path / "index.faiss").write_text(
                "new-faiss",
                encoding="utf-8",
            )
            (path / "index.pkl").write_text(
                "new-pickle",
                encoding="utf-8",
            )

    real_rename = Path.rename
    rename_calls = 0

    def controlled_rename(self, target):
        nonlocal rename_calls

        rename_calls += 1

        # 1 = final -> backup
        # 2 = staging -> final  -> fail
        # 3 = backup -> final    -> fail
        if rename_calls >= 2:
            raise OSError("simulated rename failure")

        return real_rename(self, target)

    monkeypatch.setattr(
        Path,
        "rename",
        controlled_rename,
    )

    with pytest.raises(RuntimeError, match="failed to restore"):
        indexing.save_vector_store(
            FakeVectorStore(),
            video_id,
            chunks,
        )

    backup_paths = list(
        tmp_path.glob(f".{video_id}.backup-*")
    )

    assert backup_paths
    assert not final_path.exists()
    

def test_save_vector_store_rejects_empty_chunks(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        indexing,
        "VECTOR_STORE_ROOT",
        tmp_path,
    )

    with pytest.raises(
        ValueError,
        match="no chunks",
    ):
        indexing.save_vector_store(
            FakeVectorStore(),
            "abc123",
            [],
        )