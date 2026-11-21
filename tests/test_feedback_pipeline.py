"""Tests for privacy, validation, persistence, and idempotency of feedback intake."""

import pytest
from pydantic import ValidationError
from types import SimpleNamespace

from scripts.phase11_feedback_queue import (
    attach_ai_assessments,
    build_evidence_snapshot,
    split_answer_claims,
)

from Backend.rag_system.feedback import (
    FeedbackConversationTurn,
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
    assert row["conversation_history"] == []


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
    assert row["sources"][0]["chunk_ids"] == []
    assert row["conversation_history"] == []


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



def test_ai_assessment_permission_requires_context_sharing():
    with pytest.raises(ValidationError, match="AI assessment requires a not-helpful report with share_context=true"):
        make_submission(allow_ai_assessment=True)


def test_ai_assessment_permission_is_saved_only_with_context_opt_in(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    store.save(
        make_submission(
            share_context=True,
            allow_ai_assessment=True,
            question="What happened?",
            answer="The speaker explains what happened.",
        )
    )
    assert store.list_records()[0]["allow_ai_assessment"] is True


def test_feedback_source_rejects_invalid_or_duplicate_chunk_ids():
    with pytest.raises(ValidationError, match="non-negative integers"):
        FeedbackSource(
            source_id=1,
            video_id=VIDEO_ID,
            start=1,
            end=2,
            chunk_ids=[1, True],
        )

    with pytest.raises(ValidationError, match="must not contain duplicates"):
        FeedbackSource(
            source_id=1,
            video_id=VIDEO_ID,
            start=1,
            end=2,
            chunk_ids=[3, 3],
        )


def test_feedback_store_persists_selected_chunk_ids(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    store.save(
        make_submission(
            share_context=True,
            question="What did the speaker say?",
            answer="The speaker described the event.",
            sources=[
                FeedbackSource(
                    source_id=1,
                    video_id=VIDEO_ID,
                    start=20,
                    end=30,
                    chunk_ids=[4, 5],
                )
            ],
        )
    )

    assert store.list_records()[0]["sources"][0]["chunk_ids"] == [4, 5]



def test_evidence_snapshot_resolves_only_exact_opted_in_video_chunks():
    record = {
        "context_shared": True,
        "video_id": VIDEO_ID,
        "sources": [{"chunk_ids": [4, 5]}],
    }
    documents = [
        SimpleNamespace(
            page_content="Cited transcript passage four.",
            metadata={
                "video_id": VIDEO_ID,
                "chunk_id": 4,
                "start": 20.0,
                "end": 28.0,
                "language_code": "en",
                "is_generated": False,
                "transcript_segments": [
                    {"segment_id": 7, "start": 20.0, "end": 23.0, "text": "Cited segment."}
                ],
            },
        ),
        SimpleNamespace(
            page_content="Wrong video's chunk five.",
            metadata={"video_id": "zzzzzzzzzzz", "chunk_id": 5},
        ),
        SimpleNamespace(
            page_content="Not a cited chunk.",
            metadata={"video_id": VIDEO_ID, "chunk_id": 6},
        ),
    ]

    snapshot = build_evidence_snapshot(record, documents)

    assert snapshot["status"] == "partial"
    assert snapshot["requested_chunk_ids"] == [4, 5]
    assert snapshot["missing_chunk_ids"] == [5]
    assert [item["chunk_id"] for item in snapshot["passages"]] == [4]
    assert snapshot["passages"][0]["text"] == "Cited transcript passage four."
    assert snapshot["passages"][0]["transcript_segments"][0]["text"] == "Cited segment."


def test_evidence_snapshot_refuses_to_resolve_context_without_opt_in():
    record = {
        "context_shared": False,
        "video_id": VIDEO_ID,
        "sources": [{"chunk_ids": [4]}],
    }

    snapshot = build_evidence_snapshot(
        record,
        [SimpleNamespace(page_content="Private transcript text", metadata={
            "video_id": VIDEO_ID, "chunk_id": 4
        })],
    )

    assert snapshot["status"] == "not_shared"
    assert snapshot["passages"] == []


def test_evidence_snapshot_marks_index_errors_without_fabricating_passages():
    record = {
        "context_shared": True,
        "video_id": VIDEO_ID,
        "sources": [{"chunk_ids": [4]}],
    }

    snapshot = build_evidence_snapshot(record, [], index_error="video_index_not_found")

    assert snapshot["status"] == "index_unavailable"
    assert snapshot["missing_chunk_ids"] == [4]
    assert snapshot["passages"] == []



def test_answer_claim_splitter_handles_bullets_and_multilingual_punctuation():
    claims = split_answer_claims(
        "- Revenue increased by 12%.\n"
        "- The speaker denied that the loan was fully repaid।\n"
        "The transcript is incomplete"
    )

    assert claims == [
        "Revenue increased by 12%.",
        "The speaker denied that the loan was fully repaid।",
        "The transcript is incomplete",
    ]


def test_ai_assessment_queue_skips_provider_for_reports_without_consent():
    records = [
        {
            "helpful": False,
            "context_shared": True,
            "allow_ai_assessment": False,
            "evidence_snapshot": {
                "status": "complete",
                "passages": [{"passage_id": "p1", "text": "Transcript", "chunk_id": 1}],
            },
        }
    ]

    assessed = attach_ai_assessments(records)

    assert assessed[0]["ai_proposed_assessment"]["status"] == "not_consented"


def test_ai_assessment_queue_skips_reports_with_incomplete_evidence():
    records = [
        {
            "helpful": False,
            "context_shared": True,
            "allow_ai_assessment": True,
            "evidence_snapshot": {
                "status": "partial",
                "passages": [{"passage_id": "p1", "text": "Transcript", "chunk_id": 1}],
            },
        }
    ]

    assessed = attach_ai_assessments(records)

    assert assessed[0]["ai_proposed_assessment"]["status"] == "evidence_incomplete"


def test_ai_assessment_queue_bounds_number_of_cases():
    with pytest.raises(ValueError, match="between 1 and 20"):
        attach_ai_assessments([], max_cases=21)



def test_existing_feedback_database_is_migrated_with_ai_consent_default_off(tmp_path):
    import sqlite3

    database = tmp_path / "legacy-feedback.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute(
        """
        CREATE TABLE answer_feedback (
            feedback_id TEXT PRIMARY KEY,
            answer_id TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            helpful INTEGER NOT NULL,
            reason TEXT,
            comment TEXT,
            video_id TEXT NOT NULL,
            request_id TEXT,
            context_shared INTEGER NOT NULL,
            question TEXT,
            answer TEXT,
            sources_json TEXT NOT NULL,
            model TEXT,
            retrieval_method TEXT,
            retrieval_config_json TEXT NOT NULL,
            review_status TEXT NOT NULL DEFAULT 'new'
        )
        """
    )
    connection.commit()
    connection.close()

    store = FeedbackStore(database)
    store.save(make_submission())

    record = store.list_records()[0]
    assert record["allow_ai_assessment"] is False
    assert record["conversation_history"] == []



def test_shared_conversation_history_requires_context_opt_in():
    turn = FeedbackConversationTurn(role="user", content="What about the earlier figure?")
    with pytest.raises(ValidationError, match="require share_context=true"):
        make_submission(conversation_history=[turn])


def test_shared_conversation_history_is_bounded_and_persisted(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    store.save(
        make_submission(
            share_context=True,
            question="What about that figure?",
            answer="The earlier figure was not confirmed.",
            conversation_history=[
                FeedbackConversationTurn(role="user", content="What was the figure?"),
                FeedbackConversationTurn(role="assistant", content="The speaker gave two figures."),
            ],
        )
    )

    row = store.list_records()[0]
    assert row["conversation_history"] == [
        {"role": "user", "content": "What was the figure?"},
        {"role": "assistant", "content": "The speaker gave two figures."},
    ]


def test_shared_conversation_history_total_length_is_limited():
    turns = [
        FeedbackConversationTurn(role="user", content="x" * 1_000)
        for _ in range(5)
    ]
    with pytest.raises(ValidationError, match="4,000 characters"):
        make_submission(
            share_context=True,
            question="Why?",
            answer="Because.",
            conversation_history=turns,
        )



def test_ai_assessment_permission_is_rejected_for_positive_ratings():
    with pytest.raises(
        ValidationError,
        match="AI assessment requires a not-helpful report",
    ):
        make_submission(
            helpful=True,
            reason=None,
            share_context=True,
            allow_ai_assessment=True,
            question="What happened?",
            answer="The speaker explained it.",
        )
