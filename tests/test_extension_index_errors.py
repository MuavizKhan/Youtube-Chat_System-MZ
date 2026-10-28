from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKGROUND_JS = ROOT / "chrome-extension" / "background.js"
CONTENT_JS = ROOT / "chrome-extension" / "content.js"


def test_index_status_forwards_structured_failure_to_content_script():
    background = BACKGROUND_JS.read_text(encoding="utf-8")
    content = CONTENT_JS.read_text(encoding="utf-8")

    assert "error_code: data?.error_code || null" in background
    assert "error_message: data?.error_message || null" in background
    assert "response.error_message ||" in content
    assert "Check that captions are available in a supported language" in content
