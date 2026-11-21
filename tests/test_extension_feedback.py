from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTENT_JS = ROOT / "chrome-extension" / "content.js"
BACKGROUND_JS = ROOT / "chrome-extension" / "background.js"
STYLE_CSS = ROOT / "chrome-extension" / "style.css"
APP_PY = ROOT / "Backend" / "app.py"


def test_each_successful_answer_gets_answer_scoped_feedback():
    content = CONTENT_JS.read_text(encoding="utf-8")
    assert "function appendFeedbackControls(message, context)" in content
    assert 'yesButton.textContent = "Yes";' in content
    assert 'noButton.textContent = "No";' in content
    assert "appendFeedbackControls(aiMessage, {" in content
    assert "requestId: response.request_id" in content
    assert "videoId: requestedVideoId" in content


def test_negative_feedback_distinguishes_answer_transcript_and_context_problems():
    content = CONTENT_JS.read_text(encoding="utf-8")
    for category in (
        "incorrect_or_unsupported",
        "missing_information",
        "wrong_attribution",
        "wrong_timestamp_or_source",
        "transcript_problem",
        "question_context_misunderstood",
        "uncertain_context",
    ):
        assert category in content
    assert "Not sure — more context may be needed" in content


def test_context_sharing_is_explicit_and_off_by_default():
    content = CONTENT_JS.read_text(encoding="utf-8")
    assert "consent.checked = false;" in content
    assert "share_context: consent" in content
    assert "if (consent) {" in content
    assert "payload.context = baseContext;" in content
    assert "conversation history are not shared" in content


def test_background_carries_chat_request_id_and_validates_feedback_video():
    background = BACKGROUND_JS.read_text(encoding="utf-8")
    assert 'const FEEDBACK_URL =' in background
    assert 'request_id:' in background
    assert 'response.headers.get("X-Request-ID")' in background
    assert 'message.type === "feedback"' in background
    assert 'pageVideoId !== videoId' in background
    assert "handleFeedbackSubmission(message, sendResponse);" in background


def test_feedback_controls_have_compact_styles():
    css = STYLE_CSS.read_text(encoding="utf-8")
    assert ".answer-feedback {" in css
    assert ".answer-feedback-form {" in css
    assert ".answer-feedback-consent {" in css


def test_backend_returns_chat_correlation_id_and_exposes_write_only_feedback():
    app = APP_PY.read_text(encoding="utf-8")
    assert "request_id: str | None = None" in app
    assert '"request_id": request.state.request_id' in app
    assert '@app.post("/feedback", status_code=201)' in app
    assert "def submit_answer_feedback(" in app
    assert "There is no public read route" not in app
