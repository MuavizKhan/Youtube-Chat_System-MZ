import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "chrome-extension" / "manifest.json"
BACKGROUND = ROOT / "chrome-extension" / "background.js"
APP = ROOT / "Backend" / "app.py"


def test_extension_can_read_caption_tracks_in_the_youtube_page_world():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    background = BACKGROUND.read_text(encoding="utf-8")

    assert "scripting" in manifest["permissions"]
    assert "https://www.youtube.com/*" in manifest["host_permissions"]
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

