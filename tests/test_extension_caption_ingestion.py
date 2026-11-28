import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "chrome-extension" / "manifest.json"
BACKGROUND = ROOT / "chrome-extension" / "background.js"
APP = ROOT / "Backend" / "app.py"


def test_extension_can_read_caption_tracks_in_the_youtube_page_world():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    background = BACKGROUND.read_text(encoding="utf-8")

    assert "scripting" in manifest["permissions"]
    assert "https://www.youtube.com/*" in manifest["host_permissions"]
    assert manifest["version"] == "1.4.2"
    assert "chrome.scripting.executeScript" in background
    assert 'world: "MAIN"' in background
    assert "playerCaptionsTracklistRenderer" in background
    assert "orderedTracks.slice(0, 6)" in background
    assert 'INDEX_BROWSER_TRANSCRIPT_URL' in background
    assert '/index/browser-transcript' in background


def test_backend_exposes_a_separate_browser_caption_ingestion_route():
    app = APP.read_text(encoding="utf-8")
    assert '"/index/browser-transcript"' in app
    assert "def index_browser_transcript(" in app
    assert "submit_browser_transcript(submission)" in app
    assert "MAX_BROWSER_TRANSCRIPT_BODY_BYTES" in app



def test_extension_reconnects_open_youtube_tabs_after_reload():
    background = BACKGROUND.read_text(encoding="utf-8")
    content = (ROOT / "chrome-extension" / "content.js").read_text(encoding="utf-8")

    assert "chrome.runtime.onInstalled.addListener" in background
    assert "reconnectOpenYouTubeTabs" in background
    assert 'type: "extension_ping"' in background
    assert "chrome.scripting.executeScript" in background
    assert 'files: ["content.js"]' in background
    assert 'world: "ISOLATED"' in background
    assert "data-youtube-ai-chat-content-version" in content
    assert 'message.type === "extension_ping"' in content
    assert 'version: CONTENT_SCRIPT_VERSION' in content


def test_extension_content_script_is_safe_to_inject_more_than_once():
    content = (ROOT / "chrome-extension" / "content.js").read_text(encoding="utf-8")

    assert "(() => {" in content
    assert 'if (existingVersion === CONTENT_SCRIPT_VERSION)' in content
    assert 'document.getElementById("youtube-ai-chat-launcher")?.remove()' in content
    assert content.rstrip().endswith("})();")


def test_browser_caption_failure_is_visible_when_server_fallback_fails():
    background = BACKGROUND.read_text(encoding="utf-8")

    assert 'error.errorCode = capture?.code || "BROWSER_CAPTURE_FAILED"' in background
    assert "browserFailure = {" in background
    assert "stage: browserFailureStage" in background
    assert "Browser transcript " in background
    assert "BROWSER_TRANSCRIPT_UPLOAD_FAILED" in background
    assert "browserErrorCode: browserFailure.code" in background

@pytest.mark.unit
def test_browser_capture_uses_youtubes_native_transcript_panel_for_gated_tracks():
    background = BACKGROUND.read_text(encoding="utf-8")

    assert 'captionUrl.searchParams.get("exp") === "xpe"' in background
    assert "captureTranscriptFromPanel" in background
    assert "openedPanel = !panel" in background
    assert "transcript-segment-view-model" in background
    assert "ytd-transcript-segment-renderer" in background
    assert "ytd-video-description-transcript-section-renderer" in background
    assert "TRANSCRIPT_PANEL_INCOMPLETE" in background
    assert 'source: "youtube_transcript_panel"' in background


@pytest.mark.unit
def test_browser_failure_diagnostic_survives_queued_server_fallback():
    background = BACKGROUND.read_text(encoding="utf-8")
    content = (ROOT / "chrome-extension" / "content.js").read_text(encoding="utf-8")

    assert "browser_diagnostic: browserFailure" in background
    assert "describeBrowserAcquisitionDiagnostic" in content
    assert "indexAcquisitionDiagnosticVideoId === videoId" in content
    assert "Browser transcript " in content


@pytest.mark.unit
def test_parallel_tabs_share_one_index_preparation_per_video():
    background = BACKGROUND.read_text(encoding="utf-8")

    assert "const indexPreparationFlights = new Map()" in background
    assert "function dispatchIndexPreparation(videoId, tabId, sendResponse)" in background
    assert "dispatchIndexPreparation(" in background
    assert "handleIndexPreparation(videoId, tabId, resolveFlight)" in background


def test_index_status_polling_uses_backoff_and_prevents_overlapping_requests():
    content = (ROOT / "chrome-extension" / "content.js").read_text(encoding="utf-8")

    assert "INDEX_POLL_INITIAL_INTERVAL_MS = 3000" in content
    assert "INDEX_POLL_MAX_INTERVAL_MS = 10000" in content
    assert "Math.ceil(delayMs * 1.5)" in content
    assert "indexPollInFlight" in content
    assert "indexPollGeneration" in content
    assert "INDEX_POLL_INTERVAL_MS" not in content



def test_extension_polls_durable_whole_video_chat_jobs():
    background = BACKGROUND.read_text(encoding="utf-8")
    content = (ROOT / "chrome-extension" / "content.js").read_text(encoding="utf-8")

    assert 'message.type === "chat_job_status"' in background
    assert "handleChatJobStatus(jobId, videoId, sendResponse)" in background
    assert 'type: "chat_job_status"' in content
    assert "const pollChatJob = (jobId, delayMs = 5000)" in content
    assert "statusResponse.pending" in content
    assert "handleChatResponse(statusResponse, runtimeError)" in content
