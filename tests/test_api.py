import pytest
from types import SimpleNamespace

from fastapi.testclient import TestClient

import Backend.app as app_module
from Backend.rag_system.index_service import IndexNotReadyError


VALID_VIDEO_ID = "Gfr50f6ZBvo"


def result():
    return {
        "answer": "Test answer.",
        "video_id": VALID_VIDEO_ID,
        "sources": [],
        "retrieved_chunks": 0,
        "source_segments": 0,
        "model": "test-model",
        "retrieval_method": "question_aware_mmr_lexical",
        "retrieval_config": {
            "top_k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.7,
            "max_distance": 1.3,
        },
    }


@pytest.fixture
def client():
    with TestClient(
        app_module.app,
        raise_server_exceptions=False,
    ) as test_client:
        yield test_client


@pytest.mark.api
def test_chat_returns_409_when_index_is_building(
    client,
    monkeypatch,
):
    def not_ready(**kwargs):
        raise IndexNotReadyError(
            VALID_VIDEO_ID,
            "building",
            "job-building",
        )

    monkeypatch.setattr(
        app_module,
        "answer_question",
        not_ready,
    )

    response = client.post(
        "/chat",
        json={
            "video_id": VALID_VIDEO_ID,
            "question": "What happened?",
        },
    )

    assert response.status_code == 409
    assert response.json()["error"] == "index_not_ready"
    assert response.json()["state"] == "building"
    assert response.json()["job_id"] == "job-building"
    assert response.headers["retry-after"] == "2"
    assert response.json()["request_id"]


@pytest.mark.api
def test_chat_returns_409_when_index_is_missing(
    client,
    monkeypatch,
):
    def not_ready(**kwargs):
        raise IndexNotReadyError(
            VALID_VIDEO_ID,
            "missing",
            None,
        )

    monkeypatch.setattr(
        app_module,
        "answer_question",
        not_ready,
    )

    response = client.post(
        "/chat",
        json={
            "video_id": VALID_VIDEO_ID,
            "question": "What happened?",
        },
    )

    assert response.status_code == 409
    assert response.json()["error"] == "index_not_ready"
    assert response.json()["state"] == "missing"
    assert response.json()["job_id"] is None


@pytest.mark.api
def test_chat_returns_404_for_unexpected_file_not_found(
    client,
    monkeypatch,
):
    monkeypatch.setattr(
        app_module,
        "answer_question",
        lambda **kwargs: (_ for _ in ()).throw(
            FileNotFoundError("secret vector-store path")
        ),
    )

    response = client.post(
        "/chat",
        json={
            "video_id": VALID_VIDEO_ID,
            "question": "What happened?",
        },
    )

    assert response.status_code == 404
    assert response.json()["error"] == "video_not_indexed"
    assert "secret vector-store path" not in response.text


@pytest.mark.api
def test_chat_returns_502_when_rag_service_fails(
    client,
    monkeypatch,
):
    monkeypatch.setattr(
        app_module,
        "answer_question",
        lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("secret upstream")
        ),
    )

    response = client.post(
        "/chat",
        json={
            "video_id": VALID_VIDEO_ID,
            "question": "What happened?",
        },
    )

    assert response.status_code == 502
    assert response.json()["error"] == "ai_service_error"
    assert "secret upstream" not in response.text


@pytest.mark.api
def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.api
def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.api
def test_ready(client):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


@pytest.mark.api
def test_ready_reports_missing_configuration(
    client,
    monkeypatch,
):
    monkeypatch.setattr(
        app_module,
        "HF_TOKEN",
        None,
    )

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["error"] == "service_not_ready"
    assert response.json()["request_id"]


@pytest.mark.api
def test_chat_success_and_request_id(
    client,
    monkeypatch,
):
    monkeypatch.setattr(
        app_module,
        "answer_question",
        lambda video_reference, question: result(),
    )

    response = client.post(
        "/chat",
        json={
            "video_id": VALID_VIDEO_ID,
            "question": "What happened?",
        },
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "Test answer."
    assert response.headers["X-Request-ID"]


@pytest.mark.api
def test_chat_invalid_video_reference(client):
    response = client.post(
        "/chat",
        json={
            "video_id": "not-valid",
            "question": "What happened?",
        },
    )

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_video_reference"
    assert response.json()["request_id"]


@pytest.mark.api
def test_chat_value_error_contract(
    client,
    monkeypatch,
):
    monkeypatch.setattr(
        app_module,
        "answer_question",
        lambda **kwargs: (_ for _ in ()).throw(
            ValueError("secret")
        ),
    )

    response = client.post(
        "/chat",
        json={
            "video_id": VALID_VIDEO_ID,
            "question": "What happened?",
        },
    )

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_request"


@pytest.mark.api
def test_chat_unexpected_error_contract(
    client,
    monkeypatch,
):
    monkeypatch.setattr(
        app_module,
        "answer_question",
        lambda **kwargs: (_ for _ in ()).throw(
            Exception("secret internal")
        ),
    )

    response = client.post(
        "/chat",
        json={
            "video_id": VALID_VIDEO_ID,
            "question": "What happened?",
        },
    )

    assert response.status_code == 500
    assert response.json()["error"] == "internal_server_error"
    assert "secret internal" not in response.text


@pytest.mark.api
def test_validation_contract(client):
    response = client.post(
        "/chat",
        json={
            "video_id": VALID_VIDEO_ID,
        },
    )

    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"
    assert response.json()["request_id"]


@pytest.mark.api
def test_question_length_limit(client):
    response = client.post(
        "/chat",
        json={
            "video_id": VALID_VIDEO_ID,
            "question": "x" * 1001,
        },
    )

    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


@pytest.mark.api
def test_cors_preflight(client):
    response = client.options(
        "/chat",
        headers={
            "Origin": "https://www.youtube.com",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"


@pytest.mark.api
def test_rate_limit_contract(
    client,
    monkeypatch,
):
    monkeypatch.setattr(
        app_module,
        "answer_question",
        lambda video_reference, question: result(),
    )

    app_module.limiter.enabled = True

    try:
        responses = [
            client.post(
                "/chat",
                json={
                    "video_id": VALID_VIDEO_ID,
                    "question": "What happened?",
                },
            )
            for _ in range(21)
        ]

        assert [
            response.status_code
            for response in responses[:20]
        ] == [200] * 20

        assert responses[20].status_code == 429
        assert (
            responses[20].json()["error"]
            == "rate_limit_exceeded"
        )
        assert responses[20].json()["request_id"]

    finally:
        app_module.limiter.enabled = False
        app_module.limiter.reset()


@pytest.mark.api
def test_prepare_index_endpoint_returns_202_for_queued_job(
    client,
    monkeypatch,
):
    monkeypatch.setattr(
        app_module,
        "prepare_index",
        lambda video_id: SimpleNamespace(
            video_id=VALID_VIDEO_ID,
            state="queued",
            action="create",
            ready=False,
            job_id="job-123",
        ),
    )

    response = client.post(
        "/index",
        json={
            "video_id": (
                f"https://www.youtube.com/watch?v={VALID_VIDEO_ID}"
            ),
        },
    )

    assert response.status_code == 202
    assert response.json() == {
        "video_id": VALID_VIDEO_ID,
        "state": "queued",
        "action": "create",
        "ready": False,
        "job_id": "job-123",
    }
    assert response.headers["X-Request-ID"]


@pytest.mark.api
def test_prepare_index_endpoint_returns_200_when_already_ready(
    client,
    monkeypatch,
):
    monkeypatch.setattr(
        app_module,
        "prepare_index",
        lambda video_id: SimpleNamespace(
            video_id=VALID_VIDEO_ID,
            state="ready",
            action="reuse",
            ready=True,
            job_id=None,
        ),
    )

    response = client.post(
        "/index",
        json={"video_id": VALID_VIDEO_ID},
    )

    assert response.status_code == 200
    assert response.json()["state"] == "ready"
    assert response.json()["action"] == "reuse"
    assert response.json()["ready"] is True
    assert response.json()["job_id"] is None


@pytest.mark.api
def test_prepare_index_endpoint_rejects_invalid_video_reference(
    client,
):
    response = client.post(
        "/index",
        json={"video_id": "not-valid"},
    )

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_video_reference"
    assert response.json()["request_id"]


@pytest.mark.api
def test_prepare_index_endpoint_sanitizes_failure(
    client,
    monkeypatch,
):
    monkeypatch.setattr(
        app_module,
        "prepare_index",
        lambda video_id: (_ for _ in ()).throw(
            RuntimeError(
                "private transcript provider details"
            )
        ),
    )

    response = client.post(
        "/index",
        json={"video_id": VALID_VIDEO_ID},
    )

    assert response.status_code == 502
    assert response.json()["error"] == "index_preparation_failed"
    assert (
        "private transcript provider details"
        not in response.text
    )


@pytest.mark.api
def test_index_status_endpoint_returns_queued_job(
    client,
    monkeypatch,
):
    monkeypatch.setattr(
        app_module,
        "get_index_status",
        lambda video_id: SimpleNamespace(
            video_id=video_id,
            state="queued",
            action="create",
            ready=False,
            job_id="job-queued",
        ),
    )

    response = client.get(
        f"/index/{VALID_VIDEO_ID}"
    )

    assert response.status_code == 200
    assert response.json() == {
        "video_id": VALID_VIDEO_ID,
        "state": "queued",
        "action": "create",
        "ready": False,
        "job_id": "job-queued",
    }


@pytest.mark.api
def test_index_status_endpoint_rejects_invalid_video_id(
    client,
):
    response = client.get("/index/not-valid")

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_video_reference"


@pytest.mark.api
def test_index_prepare_validation_error_uses_api_contract(
    client,
):
    response = client.post(
        "/index",
        json={},
    )

    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"
    assert response.json()["request_id"]
