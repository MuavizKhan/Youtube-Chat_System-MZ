"""Static integration contracts for the Phase 11.0E extension feedback UI."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "Backend" / "app.py"
BACKGROUND = ROOT / "chrome-extension" / "background.js"
CONTENT = ROOT / "chrome-extension" / "content.js"
STYLE = ROOT / "chrome-extension" / "style.css"


def test_feedback_api_is_separate_rate_limited_and_does_not_call_rag():
    app = APP.read_text(encoding="utf-8")
    start = app.index('# USER ANSWER FEEDBACK')
    end = app.index('# CHAT', start)
    route = app[start:end]

    assert '"/feedback"' in route
    assert "@limiter.limit(RATE_LIMIT)" in route
    assert "feedback_store.save(feedback_submission)" in route
    assert "answer_question(" not in route


def test_extension_routes_feedback_only_from_matching_youtube_video():
    background = BACKGROUND.read_text(encoding="utf-8")

    assert 'const FEEDBACK_URL =' in background
    start = background.index('if (message.type === "feedback")')
    end = background.index('if (message.type !== "chat")', start)
    route = background[start:end]

    assert "isYouTubeUrl(senderUrl)" in route
    assert "pageVideoId !== videoId" in route
    assert "handleFeedbackSubmission(feedback, sendResponse)" in route
    assert 'fetch(FEEDBACK_URL' in background


def test_successful_answer_displays_yes_no_feedback_controls():
    content = CONTENT.read_text(encoding="utf-8")

    assert "function renderAnswerFeedback(aiMessage, context)" in content
    assert 'yesButton.textContent = "Yes"' in content
    assert 'noButton.textContent = "No"' in content
    assert "Your rating, video ID and technical details are recorded" in content
    assert "source?.chunk_ids" in content
    assert "chunk_ids:" in content
    assert "answer_incorrect_or_incomplete" in content
    assert "question_or_context_misunderstood" in content
    assert "transcript_missing_or_inaccurate" in content


def test_context_sharing_is_explicitly_opt_in_and_only_then_sent():
    content = CONTENT.read_text(encoding="utf-8")
    start = content.index("function renderAnswerFeedback(")
    end = content.index("// REQUEST STATE RESET", start)
    renderer = content[start:end]

    assert "consent.checked = false" in renderer
    assert "if (shareContext)" in renderer
    assert "payload.question = context.question" in renderer
    assert "payload.answer = context.answer" in renderer
    assert "payload.sources = context.sources" in renderer


def test_feedback_form_uses_required_reason_placeholder_and_retryable_failure():
    content = CONTENT.read_text(encoding="utf-8")

    assert 'placeholderOption.textContent = "Select a reason"' in content
    assert "placeholderOption.disabled = true" in content
    assert "reasonSelect.required = true" in content
    assert 'status.dataset.state = "error"' in content
    assert "Please try again." in content


def test_feedback_ui_has_accessible_status_and_compact_styling():
    content = CONTENT.read_text(encoding="utf-8")
    css = STYLE.read_text(encoding="utf-8")

    assert 'status.setAttribute("role", "status")' in content
    assert 'status.setAttribute("aria-live", "polite")' in content
    assert ".answer-feedback {" in css
    assert ".answer-feedback-consent {" in css
