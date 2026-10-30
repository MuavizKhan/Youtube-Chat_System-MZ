import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from Backend.rag_system.indexing import (
    INDEX_METADATA_VERSION,
    TRANSCRIPT_SEGMENT_SCHEMA_VERSION,
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
                "transcript_segments": [
                    {
                        "segment_id": 0,
                        "start": 0.0,
                        "end": 10.0,
                        "text": "First transcript chunk.",
                    },
                ],
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
                "transcript_segments": [
                    {
                        "segment_id": 1,
                        "start": 10.0,
                        "end": 20.0,
                        "text": "Second transcript chunk.",
                    },
                ],
            },
        ),
    ]


def test_build_index_metadata():
    chunks = make_chunks()

    metadata = build_index_metadata(chunks)

    assert metadata == {
        "metadata_version": INDEX_METADATA_VERSION,
        "transcript_segment_schema_version": TRANSCRIPT_SEGMENT_SCHEMA_VERSION,
        "video_id": "Gfr50f6ZBvo",
        "embedding_model": indexing.EMBEDDING_MODEL,
        "chunk_size": indexing.CHUNK_SIZE,
        "chunk_overlap": indexing.CHUNK_OVERLAP,
        "chunk_count": 2,
        "transcript_language": "English",
        "transcript_language_code": "en",
        "transcript_generated": True,
        "transcript_segment_count": 2,
        "transcript_character_count": 47,
        "transcript_timeline_start_seconds": 0.0,
        "transcript_timeline_end_seconds": 20.0,
        "transcript_timeline_span_seconds": 20.0,
        "requested_languages": ["en", "hi", "ur", "ar"],
    }


def test_build_index_metadata_rejects_missing_transcript_segments():
    chunks = make_chunks()
    del chunks[0].metadata["transcript_segments"]

    with pytest.raises(
        ValueError,
        match="missing transcript segment metadata",
    ):
        build_index_metadata(chunks)


def test_build_index_metadata_rejects_malformed_transcript_segments():
    chunks = make_chunks()
    chunks[0].metadata["transcript_segments"][0]["start"] = float("nan")

    with pytest.raises(
        ValueError,
        match="valid times",
    ):
        build_index_metadata(chunks)


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
    assert (
        metadata["transcript_segment_schema_version"]
        == indexing.TRANSCRIPT_SEGMENT_SCHEMA_VERSION
    )
    assert metadata["embedding_model"] == indexing.EMBEDDING_MODEL
    assert metadata["requested_languages"] == ["en", "hi", "ur", "ar"]


def test_build_index_metadata_counts_unique_caption_segments_across_overlapping_chunks():
    chunks = make_chunks()
    # Chunk overlap may repeat the same source caption. It must be counted once.
    chunks[1].metadata["transcript_segments"].append(
        dict(chunks[0].metadata["transcript_segments"][0])
    )

    metadata = build_index_metadata(chunks)

    assert metadata["transcript_segment_count"] == 2
    assert metadata["transcript_character_count"] == 47
    assert metadata["transcript_timeline_start_seconds"] == 0.0
    assert metadata["transcript_timeline_end_seconds"] == 20.0
    assert metadata["transcript_timeline_span_seconds"] == 20.0


def test_build_index_metadata_rejects_conflicting_duplicate_segment_ids():
    chunks = make_chunks()
    conflicting_duplicate = dict(
        chunks[0].metadata["transcript_segments"][0]
    )
    conflicting_duplicate["text"] = "Conflicting text."
    chunks[1].metadata["transcript_segments"].append(conflicting_duplicate)

    with pytest.raises(ValueError, match="conflicting records"):
        build_index_metadata(chunks)


def test_build_index_metadata_preserves_explicit_language_order():
    metadata = build_index_metadata(
        make_chunks(),
        requested_languages=["ar", "en"],
    )

    assert metadata["requested_languages"] == ["ar", "en"]


def test_index_language_preferences_read_from_persisted_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(indexing, "VECTOR_STORE_ROOT", tmp_path)
    video_dir = tmp_path / "arabic-video"
    video_dir.mkdir()
    (video_dir / "metadata.json").write_text(
        json.dumps({"requested_languages": ["ar", "en"]}),
        encoding="utf-8",
    )

    assert indexing.get_index_language_preferences("arabic-video") == ["ar", "en"]
    assert indexing.index_matches_requested_languages(
        "arabic-video",
        ["ar", "en"],
    )
    assert not indexing.index_matches_requested_languages(
        "arabic-video",
        ["en", "ar"],
    )


def test_index_language_preferences_missing_from_legacy_metadata_returns_unknown(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(indexing, "VECTOR_STORE_ROOT", tmp_path)
    video_dir = tmp_path / "legacy-video"
    video_dir.mkdir()
    (video_dir / "metadata.json").write_text("{}", encoding="utf-8")

    assert indexing.get_index_language_preferences("legacy-video") is None
    assert not indexing.index_matches_requested_languages(
        "legacy-video",
        ["ar"],
    )
