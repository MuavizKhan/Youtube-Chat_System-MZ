"""Tests for privacy, validation, persistence, and idempotency of feedback intake."""

import pytest
from pydantic import ValidationError

from Backend.rag_system.feedback import (
    FeedbackReason,
    FeedbackSource,
    FeedbackStore,
    FeedbackSubmission,
)


VIDEO_ID = "abcdefghijk"
ANSWER_ID = "answer-123456789"


def make_submission(**overrides):
    values = {
        "answer_id": ANSWER_ID,
        "video_id": VIDEO_ID,
        "helpful": False,
        "reason": FeedbackReason.ANSWER_INCORRECT_OR_INCOMPLETE,
        "share_context": False,
    }
    values.update(overrides)
    return FeedbackSubmission(**values)


def test_not_helpful_feedback_requires_a_structured_reason():
    with pytest.raises(ValidationError, match="requires a feedback reason"):
        make_submission(reason=None)


def test_question_and_answer_cannot_be_persisted_without_explicit_opt_in():
    with pytest.raises(ValidationError, match="require share_context=true"):
        make_submission(question="What happened?", answer="The speaker explains it.")


def test_shared_context_requires_question_and_answer():
    with pytest.raises(ValidationError, match="requires both the question and answer"):
        make_submission(share_context=True, question="What happened?")


def test_feedback_sources_must_belong_to_the_same_video():
    source = FeedbackSource(
        source_id=1,
        video_id="zzzzzzzzzzz",
        start=12.0,
        end=20.0,
    )
    with pytest.raises(ValidationError, match="must match the feedback video_id"):
        make_submission(
            share_context=True,
            question="What happened?",
            answer="The speaker explains it.",
            sources=[source],
        )


def test_context_is_omitted_from_database_without_opt_in(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    store.save(make_submission(comment="The answer seems incomplete."))

    row = store.list_records()[0]
    assert row["comment"] == "The answer seems incomplete."
    assert row["context_shared"] is False
    assert row["question"] is None
    assert row["answer"] is None
    assert row["sources"] == []


def test_opted_in_context_is_saved_for_review(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    store.save(
        make_submission(
            share_context=True,
            question="Why did the company face financial pressure?",
            answer="The speaker describes several pressures.",
            sources=[
                FeedbackSource(
                    source_id=2,
                    video_id=VIDEO_ID,
                    start=44.5,
                    end=61.0,
                )
            ],
        )
    )

    row = store.list_records()[0]
    assert row["context_shared"] is True
    assert row["question"] == "Why did the company face financial pressure?"
    assert row["answer"] == "The speaker describes several pressures."
    assert row["sources"][0]["start"] == 44.5


def test_duplicate_answer_submission_is_idempotent(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    first = store.save(make_submission())
    duplicate = store.save(
        make_submission(
            helpful=True,
            reason=None,
            comment="Changing the rating is not a second record.",
        )
    )

    assert first.status == "received"
    assert duplicate.status == "already_received"
    assert duplicate.feedback_id == first.feedback_id
    assert len(store.list_records()) == 1


def test_summary_contains_aggregates_not_user_content(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    store.save(
        make_submission(
            comment="Private free-text detail",
            share_context=True,
            question="Private question",
            answer="Private answer",
        )
    )

    summary = store.summary()
    serialized = str(summary)
    assert summary["total_feedback"] == 1
    assert summary["not_helpful"] == 1
    assert summary["context_shared"] == 1
    assert "Private question" not in serialized
    assert "Private answer" not in serialized
    assert "Private free-text detail" not in serialized


def test_submission_rejects_unrecognized_fields():
    with pytest.raises(ValidationError, match="extra"):
        make_submission(email="person@example.com")



def test_feedback_endpoint_persists_and_returns_receipt(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from Backend import app as backend_app

    store = FeedbackStore(tmp_path / "api-feedback.sqlite3")
    monkeypatch.setattr(backend_app, "feedback_store", store)
    client = TestClient(backend_app.app)

    response = client.post(
        "/feedback",
        json={
            "answer_id": "answer-api-test-12345",
            "video_id": VIDEO_ID,
            "helpful": False,
            "reason": "question_or_context_misunderstood",
            "share_context": False,
        },
    )

    assert response.status_code == 202
    receipt = response.json()
    assert receipt["status"] == "received"
    assert len(receipt["feedback_id"]) == 32
    assert store.summary()["not_helpful"] == 1
