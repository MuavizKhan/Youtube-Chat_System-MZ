import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from Backend.rag_system.indexing import (
    INDEX_METADATA_VERSION,
    build_index_metadata,
    save_vector_store,
)
from Backend.rag_system import indexing


def make_chunks():
    return [
        Document(
            page_content="First transcript chunk.",
            metadata={
                "video_id": "Gfr50f6ZBvo",
                "language": "English",
                "language_code": "en",
                "is_generated": True,
                "start": 0.0,
                "end": 10.0,
                "duration": 10.0,
                "chunk_id": 0,
                "chunk_length": 24,
                "source": "youtube_transcript",
            },
        ),
        Document(
            page_content="Second transcript chunk.",
            metadata={
                "video_id": "Gfr50f6ZBvo",
                "language": "English",
                "language_code": "en",
                "is_generated": True,
                "start": 10.0,
                "end": 20.0,
                "duration": 10.0,
                "chunk_id": 1,
                "chunk_length": 25,
                "source": "youtube_transcript",
            },
        ),
    ]


def test_build_index_metadata():
    chunks = make_chunks()

    metadata = build_index_metadata(chunks)

    assert metadata == {
        "metadata_version": INDEX_METADATA_VERSION,
        "video_id": "Gfr50f6ZBvo",
        "embedding_model": indexing.EMBEDDING_MODEL,
        "chunk_size": indexing.CHUNK_SIZE,
        "chunk_overlap": indexing.CHUNK_OVERLAP,
        "chunk_count": 2,
        "transcript_language": "English",
        "transcript_language_code": "en",
        "transcript_generated": True,
    }


def test_build_index_metadata_rejects_empty_chunks():
    with pytest.raises(
        ValueError,
        match="no chunks were produced",
    ):
        build_index_metadata([])


def test_build_index_metadata_rejects_missing_metadata():
    chunks = make_chunks()

    del chunks[0].metadata["language_code"]

    with pytest.raises(
        ValueError,
        match="language_code",
    ):
        build_index_metadata(chunks)


def test_build_index_metadata_rejects_multiple_video_ids():
    chunks = make_chunks()

    chunks[1].metadata["video_id"] = "AnotherVideo"

    with pytest.raises(
        ValueError,
        match="multiple video IDs",
    ):
        build_index_metadata(chunks)


def test_save_vector_store_writes_metadata(tmp_path, monkeypatch):
    chunks = make_chunks()

    class FakeVectorStore:
        def save_local(self, path):
            path = Path(path)
            path.mkdir(
                parents=True,
                exist_ok=True,
            )

            (path / "index.faiss").write_bytes(
                b"fake-faiss"
            )

            (path / "index.pkl").write_bytes(
                b"fake-pickle"
            )

    monkeypatch.setattr(
        indexing,
        "VECTOR_STORE_ROOT",
        tmp_path,
    )

    saved_path = save_vector_store(
        FakeVectorStore(),
        "Gfr50f6ZBvo",
        chunks,
    )

    metadata_path = (
        saved_path
        / "metadata.json"
    )

    assert metadata_path.exists()

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as metadata_file:

        metadata = json.load(
            metadata_file
        )

    assert metadata["video_id"] == "Gfr50f6ZBvo"
    assert metadata["chunk_count"] == 2
    assert metadata["embedding_model"] == indexing.EMBEDDING_MODEL