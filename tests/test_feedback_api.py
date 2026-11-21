from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import Backend.app as app_module
from Backend.rag_system import feedback_store


@pytest.fixture
def feedback_client(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback_store, "FEEDBACK_DB_PATH", tmp_path / "feedback.sqlite3")
    with TestClient(app_module.app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_feedback_endpoint_stores_basic_vote_without_context_consent(feedback_client):
    response = feedback_client.post(
        "/feedback",
        json={
            "request_id": "1" * 32,
            "video_id": "Gfr50f6ZBvo",
            "vote": "helpful",
            "share_context": False,
        },
    )
    assert response.status_code == 201
    assert response.json()["status"] == "received"
    assert response.json()["context_saved"] is False
    stored = feedback_store.list_feedback_records()
    assert len(stored) == 1
    assert stored[0]["context"] is None


def test_feedback_endpoint_stores_context_only_after_opt_in(feedback_client):
    response = feedback_client.post(
        "/feedback",
        json={
            "request_id": "2" * 32,
            "video_id": "Gfr50f6ZBvo",
            "vote": "not_helpful",
            "reason": "transcript_problem",
            "share_context": True,
            "context": {
                "question": "What did the speaker say?",
                "answer": "The answer may have misread the sentence.",
                "sources": [
                    {
                        "source_id": 1,
                        "chunk_ids": [2],
                        "video_id": "Gfr50f6ZBvo",
                        "start": 10.0,
                        "end": 16.0,
                        "duration": 6.0,
                        "distance": 0.1,
                    }
                ],
                "model": "test-model",
                "retrieval_method": "hybrid",
                "retrieval_config": {"top_k": 4},
            },
        },
    )
    assert response.status_code == 201
    assert response.json()["context_saved"] is True
    stored = feedback_store.list_feedback_records()[0]
    assert stored["context"]["question"] == "What did the speaker say?"


def test_feedback_endpoint_is_idempotent_per_chat_request(feedback_client):
    payload = {
        "request_id": "3" * 32,
        "video_id": "Gfr50f6ZBvo",
        "vote": "helpful",
    }
    first = feedback_client.post("/feedback", json=payload)
    second = feedback_client.post("/feedback", json=payload)
    assert first.status_code == 201
    assert second.status_code == 409
    assert len(feedback_store.list_feedback_records()) == 1


def test_feedback_endpoint_rejects_mismatched_source_video(feedback_client):
    response = feedback_client.post(
        "/feedback",
        json={
            "request_id": "4" * 32,
            "video_id": "Gfr50f6ZBvo",
            "vote": "not_helpful",
            "share_context": True,
            "context": {
                "question": "Question?",
                "answer": "Answer.",
                "sources": [
                    {
                        "source_id": 1,
                        "chunk_ids": [2],
                        "video_id": "different123",
                        "start": 10.0,
                        "end": 16.0,
                        "duration": 6.0,
                        "distance": 0.1,
                    }
                ],
                "model": "test-model",
                "retrieval_method": "hybrid",
                "retrieval_config": {"top_k": 4},
            },
        },
    )
    assert response.status_code == 422
    assert len(feedback_store.list_feedback_records()) == 0



def test_chat_response_exposes_request_id_for_feedback(feedback_client, monkeypatch):
    def answer_question_stub(**kwargs):
        return {
            "answer": "The speaker says the project was delayed by funding.",
            "video_id": "Gfr50f6ZBvo",
            "sources": [],
            "retrieved_chunks": 0,
            "model": "test-model",
            "source_segments": 0,
            "retrieval_method": "question_aware_mmr_lexical",
            "retrieval_config": {
                "top_k": 4,
                "fetch_k": 10,
                "lambda_mult": 0.7,
                "max_distance": 1.3,
            },
            "observability": {},
        }

    monkeypatch.setattr(app_module, "answer_question", answer_question_stub)
    response = feedback_client.post(
        "/chat",
        json={
            "video_id": "Gfr50f6ZBvo",
            "question": "Why was the project delayed?",
        },
    )

    assert response.status_code == 200
    assert response.headers.get("x-request-id")
    assert response.json()["request_id"] == response.headers["x-request-id"]
    assert len(response.json()["request_id"]) == 32
