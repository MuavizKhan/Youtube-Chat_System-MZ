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
    assert 'INDEX_BROWSER_TRANSCRIPT_URL' in background
    assert '/index/browser-transcript' in background


def test_backend_exposes_a_separate_browser_caption_ingestion_route():
    app = APP.read_text(encoding="utf-8")
    assert '"/index/browser-transcript"' in app
    assert "def index_browser_transcript(" in app
    assert "submit_browser_transcript(submission)" in app
    assert "MAX_BROWSER_TRANSCRIPT_BODY_BYTES" in app
