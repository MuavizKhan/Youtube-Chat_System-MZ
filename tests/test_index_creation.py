from types import SimpleNamespace

import pytest
from unittest.mock import MagicMock

from Backend.rag_system import indexing


VIDEO_ID = "Gfr50f6ZBvo"


def sample_transcript():
    return SimpleNamespace(
        video_id=VIDEO_ID,
        language="English",
        language_code="en",
        is_generated=True,
        snippets=[
            SimpleNamespace(
                text="Interviewer asks a question.",
                start=0.0,
                duration=1.5,
            ),
            SimpleNamespace(
                text="A guest begins an answer long enough to cross chunks.",
                start=2.0,
                duration=3.0,
            ),
            SimpleNamespace(
                text="Interviewer follows up.",
                start=5.0,
                duration=1.5,
            ),
        ],
    )


def test_transcript_to_documents_preserves_caption_segment_ids():
    documents = indexing.transcript_to_documents(sample_transcript())

    assert [document.metadata["segment_id"] for document in documents] == [0, 1, 2]
    assert [document.page_content for document in documents] == [
        "Interviewer asks a question.",
        "A guest begins an answer long enough to cross chunks.",
        "Interviewer follows up.",
    ]


def test_timestamped_chunks_preserve_caption_segments_across_chunk_boundaries(
    monkeypatch,
):
    monkeypatch.setattr(indexing, "CHUNK_SIZE", 20)
    monkeypatch.setattr(indexing, "CHUNK_OVERLAP", 5)

    documents = indexing.transcript_to_documents(sample_transcript())
    chunks = indexing.split_into_timestamped_chunks(documents)

    assert len(chunks) > 1
    assert all(chunk.page_content.strip() for chunk in chunks)
    assert all(chunk.metadata["transcript_segments"] for chunk in chunks)

    segments_by_id = {}
    for chunk in chunks:
        for segment in chunk.metadata["transcript_segments"]:
            existing = segments_by_id.get(segment["segment_id"])
            if existing is not None:
                assert existing == segment
            segments_by_id[segment["segment_id"]] = segment

    assert set(segments_by_id) == {0, 1, 2}
    assert segments_by_id[0] == {
        "segment_id": 0,
        "start": 0.0,
        "end": 1.5,
        "text": "Interviewer asks a question.",
    }
    assert segments_by_id[1]["text"] == (
        "A guest begins an answer long enough to cross chunks."
    )
    assert segments_by_id[2]["text"] == "Interviewer follows up."

def test_get_transcript_uses_supported_language_priority(monkeypatch):
    api = MagicMock()
    fetched_transcript = MagicMock()
    api.fetch.return_value = fetched_transcript
    monkeypatch.setattr(indexing, "YouTubeTranscriptApi", lambda: api)

    assert indexing.PREFERRED_LANGUAGES == ["en", "hi", "ur", "ar"]
    result = indexing.get_transcript(VIDEO_ID)

    assert result is fetched_transcript
    api.fetch.assert_called_once_with(
        VIDEO_ID,
        languages=["en", "hi", "ur", "ar"],
    )


def test_get_transcript_normalizes_explicit_language_preferences(monkeypatch):
    api = MagicMock()
    fetched_transcript = MagicMock()
    api.fetch.return_value = fetched_transcript
    monkeypatch.setattr(indexing, "YouTubeTranscriptApi", lambda: api)

    result = indexing.get_transcript(
        VIDEO_ID,
        languages=[" AR ", "en", "ar"],
    )

    assert result is fetched_transcript
    api.fetch.assert_called_once_with(
        VIDEO_ID,
        languages=["ar", "en"],
    )


def test_get_transcript_rejects_unsupported_language(monkeypatch):
    api = MagicMock()
    monkeypatch.setattr(indexing, "YouTubeTranscriptApi", lambda: api)

    with pytest.raises(ValueError, match="Unsupported transcript language code"):
        indexing.get_transcript(VIDEO_ID, languages=["fr"])

    api.fetch.assert_not_called()




def test_create_index_rejects_empty_video_id():
    with pytest.raises(ValueError, match="video_id cannot be empty"):
        indexing.create_index("")


def test_create_index_builds_and_saves_index(monkeypatch):
    transcript = MagicMock()
    transcript.language = "English"
    transcript.language_code = "en"
    transcript.is_generated = True
    transcript.snippets = [MagicMock(), MagicMock()]

    documents = [
        MagicMock(),
        MagicMock(),
    ]

    chunks = [
        MagicMock(),
        MagicMock(),
    ]

    vector_store = MagicMock()
    saved_path = MagicMock()

    get_transcript = MagicMock(
        return_value=transcript
    )

    transcript_to_documents = MagicMock(
        return_value=documents
    )

    split_into_timestamped_chunks = MagicMock(
        return_value=chunks
    )

    create_vector_store = MagicMock(
        return_value=vector_store
    )

    save_vector_store = MagicMock(
        return_value=saved_path
    )

    monkeypatch.setattr(
        indexing,
        "get_transcript",
        get_transcript,
    )

    monkeypatch.setattr(
        indexing,
        "transcript_to_documents",
        transcript_to_documents,
    )

    monkeypatch.setattr(
        indexing,
        "split_into_timestamped_chunks",
        split_into_timestamped_chunks,
    )

    monkeypatch.setattr(
        indexing,
        "create_vector_store",
        create_vector_store,
    )

    monkeypatch.setattr(
        indexing,
        "save_vector_store",
        save_vector_store,
    )

    result = indexing.create_index(
        VIDEO_ID
    )

    assert result is vector_store

    get_transcript.assert_called_once_with(
        VIDEO_ID,
        languages=["en", "hi", "ur", "ar"],
    )

    transcript_to_documents.assert_called_once_with(
        transcript
    )

    split_into_timestamped_chunks.assert_called_once_with(
        documents
    )

    create_vector_store.assert_called_once_with(
        chunks
    )

    save_vector_store.assert_called_once_with(
        vector_store,
        VIDEO_ID,
        chunks,
        requested_languages=["en", "hi", "ur", "ar"],
    )