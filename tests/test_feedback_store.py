from __future__ import annotations

import json

import pytest

from Backend.rag_system import feedback_store


@pytest.fixture
def feedback_db(tmp_path, monkeypatch):
    path = tmp_path / "feedback.sqlite3"
    monkeypatch.setattr(feedback_store, "FEEDBACK_DB_PATH", path)
    return path


def _context():
    return {
        "question": "Why did the project fail?",
        "answer": "The speaker attributes the delay to funding.",
        "sources": [
            {
                "source_id": 1,
                "chunk_ids": [4, 5],
                "video_id": "abcdefghijk",
                "start": 30.0,
                "end": 42.0,
                "duration": 12.0,
                "distance": 0.2,
                "transcript_text": "This field must never be stored.",
            }
        ],
        "model": "test-model",
        "retrieval_method": "hybrid",
        "retrieval_config": {"top_k": 4, "lambda_mult": 0.7},
        "conversation_history": [{"role": "user", "content": "private history"}],
    }


def test_context_is_not_stored_without_explicit_consent(feedback_db):
    record = feedback_store.create_feedback_record(
        request_id="a" * 32,
        video_id="abcdefghijk",
        vote="not_helpful",
        reason="uncertain_context",
        context=_context(),
        share_context=False,
    )
    stored = feedback_store.list_feedback_records()[0]
    assert record["context_shared"] is False
    assert stored["context"] is None


def test_shared_context_uses_an_allowlist_and_omits_transcript_text(feedback_db):
    feedback_store.create_feedback_record(
        request_id="b" * 32,
        video_id="abcdefghijk",
        vote="not_helpful",
        reason="incorrect_or_unsupported",
        comment="The speaker said the opposite.",
        context=_context(),
        share_context=True,
    )
    stored = feedback_store.list_feedback_records()[0]
    assert stored["context_shared"] is True
    assert stored["context"]["question"] == "Why did the project fail?"
    assert stored["context"]["answer"].startswith("The speaker attributes")
    assert "transcript_text" not in stored["context"]["sources"][0]
    assert "conversation_history" not in stored["context"]


def test_feedback_is_idempotent_per_chat_request(feedback_db):
    arguments = {
        "request_id": "c" * 32,
        "video_id": "abcdefghijk",
        "vote": "helpful",
    }
    feedback_store.create_feedback_record(**arguments)
    with pytest.raises(feedback_store.DuplicateFeedbackError):
        feedback_store.create_feedback_record(**arguments)


def test_feedback_review_requires_pseudonymous_reviewer_and_records_status(feedback_db):
    created = feedback_store.create_feedback_record(
        request_id="d" * 32,
        video_id="abcdefghijk",
        vote="not_helpful",
    )
    with pytest.raises(feedback_store.FeedbackStoreError):
        feedback_store.update_feedback_review(
            feedback_id=created["feedback_id"],
            status="verified_issue",
            reviewer_id="Muaviz Khan",
        )
    reviewed = feedback_store.update_feedback_review(
        feedback_id=created["feedback_id"],
        status="verified_issue",
        reviewer_id="R01",
        notes="Checked the timestamp and the surrounding transcript.",
    )
    assert reviewed["review_status"] == "verified_issue"
    assert reviewed["reviewer_id"] == "R01"
    assert reviewed["review_notes"].startswith("Checked the timestamp")


def test_evaluation_export_includes_only_approved_consented_candidates(feedback_db, tmp_path, monkeypatch):
    monkeypatch.setattr(feedback_store, "PROJECT_ROOT", tmp_path)
    for suffix, status in (("e", "evaluation_candidate"), ("f", "new")):
        created = feedback_store.create_feedback_record(
            request_id=suffix * 32,
            video_id="abcdefghijk",
            vote="not_helpful",
            reason="missing_information",
            context=_context(),
            share_context=True,
        )
        if status != "new":
            feedback_store.update_feedback_review(
                feedback_id=created["feedback_id"],
                status=status,
                reviewer_id="R02",
            )
    output = tmp_path / "evaluation_data" / "private" / "feedback_candidates.jsonl"
    count = feedback_store.export_evaluation_candidates(output)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert count == 1
    assert len(rows) == 1
    assert rows[0]["is_gold_label"] is False
    assert rows[0]["requires_source_and_transcript_verification"] is True


def test_candidate_export_refuses_public_or_repository_paths(feedback_db, tmp_path):
    with pytest.raises(feedback_store.FeedbackStoreError):
        feedback_store.export_evaluation_candidates(tmp_path / "public.jsonl")


def test_invalid_reason_and_missing_shared_context_are_rejected(feedback_db):
    with pytest.raises(feedback_store.FeedbackStoreError):
        feedback_store.create_feedback_record(
            request_id="g" * 32,
            video_id="abcdefghijk",
            vote="not_helpful",
            reason="wrong",
        )
    with pytest.raises(feedback_store.FeedbackStoreError):
        feedback_store.create_feedback_record(
            request_id="h" * 32,
            video_id="abcdefghijk",
            vote="not_helpful",
            share_context=True,
        )
