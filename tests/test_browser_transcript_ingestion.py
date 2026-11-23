from types import SimpleNamespace

import pytest

from Backend.rag_system import browser_transcripts, indexing
from Backend.rag_system.browser_transcripts import (
    BrowserTranscriptSubmission,
    fingerprint_browser_transcript,
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
    assert transcript.fingerprint == fingerprint_browser_transcript(submission)
    assert [item.text for item in transcript.snippets] == [
        "First caption",
        "Second caption",
    ]
    assert transcript.snippets[1].start == 1.2
    assert load_browser_transcript(VIDEO_ID) is not None
    assert staged_path.exists()


@pytest.mark.unit
def test_browser_transcript_rejects_another_video_id():
    with pytest.raises(ValueError):
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
    class FakeJobStore:
        def enqueue_browser_transcript(self, video_id, action, fingerprint):
            calls.append(("enqueue", video_id, action, fingerprint))
            return SimpleNamespace(
                job_id=queued.job_id,
                status=queued.state,
                action=queued.action,
            )

    monkeypatch.setattr(index_service, "_JOB_STORE", FakeJobStore())
    monkeypatch.setattr(
        index_service,
        "_dispatch_video",
        lambda video_id: calls.append(("dispatch", video_id)),
    )
    monkeypatch.setattr(index_service, "get_index_transcript_provenance", lambda video_id: None)
    monkeypatch.setattr(index_service, "get_index_state", lambda video_id: indexing.IndexState.MISSING)

    result = index_service.submit_browser_transcript(make_submission())

    assert result == queued
    assert calls[0] == ("stage", VIDEO_ID)
    assert calls[1][0:3] == ("enqueue", VIDEO_ID, "create")
    assert calls[1][3] == fingerprint_browser_transcript(make_submission())
    assert calls[2] == ("dispatch", VIDEO_ID)


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
    class ActiveJobStore:
        def enqueue_browser_transcript(self, video_id, action, fingerprint):
            staged.append(("registered", video_id, fingerprint))
            return SimpleNamespace(
                job_id=active.job_id,
                status=active.state,
                action=active.action,
            )

    monkeypatch.setattr(index_service, "_JOB_STORE", ActiveJobStore())
    monkeypatch.setattr(index_service, "_dispatch_video", lambda video_id: None)
    monkeypatch.setattr(index_service, "get_index_transcript_provenance", lambda video_id: None)
    monkeypatch.setattr(index_service, "get_index_state", lambda video_id: indexing.IndexState.MISSING)

    result = index_service.submit_browser_transcript(make_submission())

    assert result == active
    assert staged[0] == VIDEO_ID
    assert staged[1][0:2] == ("registered", VIDEO_ID)
    assert staged[1][2] == fingerprint_browser_transcript(make_submission())



@pytest.mark.unit
def test_browser_transcript_fingerprint_changes_when_caption_content_changes():
    original = make_submission()
    changed = make_submission(
        snippets=[
            {"text": "Changed caption", "start": 0.0, "duration": 1.2},
            {"text": "Second caption", "start": 1.2, "duration": 2.0},
        ]
    )

    assert fingerprint_browser_transcript(original) != fingerprint_browser_transcript(changed)


@pytest.mark.unit
def test_browser_upload_requeues_a_ready_api_index(monkeypatch):
    submission = make_submission()
    dispatched = []
    queued = SimpleNamespace(
        job_id="replacement-browser-job",
        status="queued",
        action="rebuild",
    )

    monkeypatch.setattr(
        index_service,
        "get_index_status",
        lambda video_id: index_service.IndexStatus(
            video_id=video_id,
            state="ready",
            action="reuse",
            ready=True,
        ),
    )
    monkeypatch.setattr(
        index_service,
        "get_index_transcript_provenance",
        lambda video_id: {
            "source": "youtube_transcript_api",
            "fingerprint": "a" * 64,
        },
    )
    monkeypatch.setattr(index_service, "stage_browser_transcript", lambda item: None)
    monkeypatch.setattr(index_service, "get_index_state", lambda video_id: indexing.IndexState.VALID)

    class ReadyJobStore:
        def enqueue_browser_transcript(self, video_id, action, fingerprint):
            assert action == "rebuild"
            assert fingerprint == fingerprint_browser_transcript(submission)
            return queued

    monkeypatch.setattr(index_service, "_JOB_STORE", ReadyJobStore())
    monkeypatch.setattr(index_service, "_dispatch_video", lambda video_id: dispatched.append(video_id))

    result = index_service.submit_browser_transcript(submission)

    assert result.state == "queued"
    assert result.job_id == "replacement-browser-job"
    assert dispatched == [VIDEO_ID]



@pytest.mark.unit
def test_browser_transcript_provenance_is_propagated_to_documents_and_chunks():
    transcript = SimpleNamespace(
        video_id=VIDEO_ID,
        language="English",
        language_code="en",
        is_generated=True,
        source="youtube_browser_captions",
        fingerprint="f" * 64,
        snippets=[
            SimpleNamespace(
                text="This is a browser-captured caption segment.",
                start=3.0,
                duration=2.0,
            )
        ],
    )

    documents = indexing.transcript_to_documents(transcript)
    assert documents[0].metadata["transcript_source"] == "youtube_browser_captions"
    assert documents[0].metadata["transcript_fingerprint"] == "f" * 64

    chunks = indexing.split_into_timestamped_chunks(documents)
    assert chunks[0].metadata["transcript_source"] == "youtube_browser_captions"
    assert chunks[0].metadata["transcript_fingerprint"] == "f" * 64
