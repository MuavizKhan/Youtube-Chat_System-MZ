from types import SimpleNamespace

import pytest

from Backend.rag_system import browser_transcripts, indexing
from Backend.rag_system.browser_transcripts import (
    BrowserTranscriptSubmission,
    load_browser_transcript,
    stage_browser_transcript,
)
from Backend.rag_system.indexing import TranscriptAcquisitionError
from Backend.rag_system import index_service


VIDEO_ID = "C-KqNLIKrXM"


def make_submission(**overrides):
    values = {
        "video_id": VIDEO_ID,
        "language": "English",
        "language_code": "en",
        "is_generated": True,
        "snippets": [
            {"text": "First caption", "start": 0.0, "duration": 1.2},
            {"text": "Second caption", "start": 1.2, "duration": 2.0},
        ],
    }
    values.update(overrides)
    return BrowserTranscriptSubmission(**values)


@pytest.mark.unit
def test_browser_transcript_is_staged_and_retained_until_index_success(tmp_path, monkeypatch):
    monkeypatch.setattr(
        browser_transcripts,
        "BROWSER_TRANSCRIPT_DIRECTORY",
        tmp_path,
    )

    submission = make_submission()
    staged_path = stage_browser_transcript(submission)

    assert staged_path.is_file()

    transcript = load_browser_transcript(VIDEO_ID)

    assert transcript is not None
    assert transcript.video_id == VIDEO_ID
    assert transcript.language == "English"
    assert transcript.language_code == "en"
    assert transcript.is_generated is True
    assert transcript.source == "youtube_browser_captions"
    assert [item.text for item in transcript.snippets] == [
        "First caption",
        "Second caption",
    ]
    assert transcript.snippets[1].start == 1.2
    assert load_browser_transcript(VIDEO_ID) is not None
    assert staged_path.exists()


@pytest.mark.unit
def test_browser_transcript_rejects_another_video_id():
    with pytest.raises(ValueError, match="video ID"):
        make_submission(video_id="not-valid")


@pytest.mark.unit
def test_browser_transcript_rejects_excessive_total_text():
    with pytest.raises(ValueError, match="maximum supported text size"):
        make_submission(
            snippets=[
                {
                    "text": "x" * 5_000,
                    "start": index * 10,
                    "duration": 1.0,
                }
                for index in range(301)
            ]
        )


@pytest.mark.unit
def test_get_transcript_prefers_staged_browser_captions(monkeypatch):
    expected = SimpleNamespace(
        video_id=VIDEO_ID,
        language="English",
        language_code="en",
        is_generated=True,
        snippets=[
            SimpleNamespace(text="Browser caption", start=0.0, duration=2.0)
        ],
        source="youtube_browser_captions",
    )

    monkeypatch.setattr(
        indexing,
        "load_browser_transcript",
        lambda video_id: expected if video_id == VIDEO_ID else None,
    )
    monkeypatch.setattr(
        indexing,
        "YouTubeTranscriptApi",
        lambda: pytest.fail("server transcript API should not run when browser captions exist"),
    )

    assert indexing.get_transcript(VIDEO_ID) is expected


@pytest.mark.unit
def test_backend_ip_block_is_reported_as_a_specific_transcript_failure(monkeypatch):
    class BlockedApi:
        def fetch(self, video_id, languages):
            raise indexing.IpBlocked(video_id)

    monkeypatch.setattr(indexing, "load_browser_transcript", lambda video_id: None)
    monkeypatch.setattr(indexing, "YouTubeTranscriptApi", lambda: BlockedApi())

    with pytest.raises(TranscriptAcquisitionError) as error:
        indexing.get_transcript(VIDEO_ID)

    assert error.value.code == "YOUTUBE_IP_BLOCKED"
    assert "Keep the video open in YouTube" in error.value.public_message



@pytest.mark.unit
def test_submit_browser_transcript_stages_before_dispatching_index_job(monkeypatch):
    calls = []
    queued = index_service.IndexStatus(
        video_id=VIDEO_ID,
        state="queued",
        action="create",
        ready=False,
        job_id="browser-caption-job",
    )

    monkeypatch.setattr(
        index_service,
        "get_index_status",
        lambda video_id: index_service.IndexStatus(
            video_id=video_id,
            state="missing",
            action="create",
            ready=False,
        ),
    )
    monkeypatch.setattr(
        index_service,
        "stage_browser_transcript",
        lambda submission: calls.append(("stage", submission.video_id)),
    )
    monkeypatch.setattr(
        index_service,
        "submit_index",
        lambda video_id: calls.append(("submit", video_id)) or queued,
    )

    result = index_service.submit_browser_transcript(make_submission())

    assert result == queued
    assert calls == [
        ("stage", VIDEO_ID),
        ("submit", VIDEO_ID),
    ]


@pytest.mark.unit
def test_browser_transcript_upload_route_accepts_caption_payload(monkeypatch):
    from fastapi.testclient import TestClient
    from Backend import app as backend_app

    queued = index_service.IndexStatus(
        video_id=VIDEO_ID,
        state="queued",
        action="create",
        ready=False,
        job_id="browser-caption-api-job",
    )
    monkeypatch.setattr(
        backend_app,
        "submit_browser_transcript",
        lambda submission: queued,
    )

    response = TestClient(backend_app.app).post(
        "/index/browser-transcript",
        json=make_submission().model_dump(),
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["video_id"] == VIDEO_ID
    assert payload["state"] == "queued"
    assert payload["job_id"] == "browser-caption-api-job"



@pytest.mark.unit
def test_submit_browser_transcript_stages_for_an_existing_active_job(monkeypatch):
    active = index_service.IndexStatus(
        video_id=VIDEO_ID,
        state="building",
        action="create",
        ready=False,
        job_id="older-server-fetch-job",
    )
    staged = []
    monkeypatch.setattr(index_service, "get_index_status", lambda video_id: active)
    monkeypatch.setattr(
        index_service,
        "stage_browser_transcript",
        lambda submission: staged.append(submission.video_id),
    )
    monkeypatch.setattr(
        index_service,
        "submit_index",
        lambda video_id: pytest.fail("An active durable job must not be duplicated."),
    )

    result = index_service.submit_browser_transcript(make_submission())

    assert result == active
    assert staged == [VIDEO_ID]
