import pytest
from fastapi.testclient import TestClient
import Backend.app as app_module

VALID_VIDEO_ID="Gfr50f6ZBvo"


def result():
    return {
        "answer":"Test answer.",
        "video_id":VALID_VIDEO_ID,
        "sources":[],
        "retrieved_chunks":0,
        "source_segments":0,
        "model":"test-model",
        "retrieval_method":"question_aware_mmr_lexical",
        "retrieval_config":{
            "top_k":4,"fetch_k":10,"lambda_mult":0.7,"max_distance":1.3
        },
    }


@pytest.fixture
def client():
    with TestClient(app_module.app,raise_server_exceptions=False) as c:
        yield c


@pytest.mark.api
def test_root(client):
    r=client.get("/")
    assert r.status_code==200
    assert r.json()["status"]=="ok"


@pytest.mark.api
def test_health(client):
    r=client.get("/health")
    assert r.status_code==200
    assert r.json()["status"]=="ok"


@pytest.mark.api
def test_ready(client):
    r=client.get("/ready")
    assert r.status_code==200
    assert r.json()["status"]=="ready"


@pytest.mark.api
def test_ready_reports_missing_configuration(client,monkeypatch):
    monkeypatch.setattr(app_module,"HF_TOKEN",None)
    r=client.get("/ready")
    assert r.status_code==503
    assert r.json()["error"]=="service_not_ready"
    assert r.json()["request_id"]


@pytest.mark.api
def test_chat_success_and_request_id(client,monkeypatch):
    monkeypatch.setattr(app_module,"answer_question",lambda video_reference,question:result())
    r=client.post("/chat",json={"video_id":VALID_VIDEO_ID,"question":"What happened?"})
    assert r.status_code==200
    assert r.json()["answer"]=="Test answer."
    assert r.headers["X-Request-ID"]


@pytest.mark.api
def test_chat_invalid_video_reference(client):
    r=client.post("/chat",json={"video_id":"not-valid","question":"What happened?"})
    assert r.status_code==400
    assert r.json()["error"]=="invalid_video_reference"
    assert r.json()["request_id"]


@pytest.mark.api
def test_chat_missing_vector_store_is_sanitized(client,monkeypatch):
    def missing(**kwargs):
        raise FileNotFoundError("secret vector-store path")
    monkeypatch.setattr(app_module,"answer_question",missing)
    r=client.post("/chat",json={"video_id":VALID_VIDEO_ID,"question":"What happened?"})
    assert r.status_code==404
    assert r.json()["error"]=="video_not_indexed"
    assert "secret vector-store path" not in r.text
    assert "vector-store" not in r.json()["message"].lower()


@pytest.mark.api
def test_chat_value_error_contract(client,monkeypatch):
    monkeypatch.setattr(app_module,"answer_question",lambda **kwargs: (_ for _ in ()).throw(ValueError("secret")))
    r=client.post("/chat",json={"video_id":VALID_VIDEO_ID,"question":"What happened?"})
    assert r.status_code==400
    assert r.json()["error"]=="invalid_request"


@pytest.mark.api
def test_chat_runtime_error_contract(client,monkeypatch):
    monkeypatch.setattr(app_module,"answer_question",lambda **kwargs: (_ for _ in ()).throw(RuntimeError("secret upstream")))
    r=client.post("/chat",json={"video_id":VALID_VIDEO_ID,"question":"What happened?"})
    assert r.status_code==502
    assert r.json()["error"]=="ai_service_error"
    assert "secret upstream" not in r.text


@pytest.mark.api
def test_chat_unexpected_error_contract(client,monkeypatch):
    monkeypatch.setattr(app_module,"answer_question",lambda **kwargs: (_ for _ in ()).throw(Exception("secret internal")))
    r=client.post("/chat",json={"video_id":VALID_VIDEO_ID,"question":"What happened?"})
    assert r.status_code==500
    assert r.json()["error"]=="internal_server_error"
    assert "secret internal" not in r.text


@pytest.mark.api
def test_validation_contract(client):
    r=client.post("/chat",json={"video_id":VALID_VIDEO_ID})
    assert r.status_code==422
    assert r.json()["error"]=="validation_error"
    assert r.json()["request_id"]


@pytest.mark.api
def test_question_length_limit(client):
    r=client.post("/chat",json={"video_id":VALID_VIDEO_ID,"question":"x"*1001})
    assert r.status_code==422
    assert r.json()["error"]=="validation_error"


@pytest.mark.api
def test_cors_preflight(client):
    r=client.options("/chat",headers={
        "Origin":"https://www.youtube.com",
        "Access-Control-Request-Method":"POST",
    })
    assert r.status_code==200
    assert r.headers["access-control-allow-origin"]=="*"


@pytest.mark.api
def test_rate_limit_contract(client,monkeypatch):
    monkeypatch.setattr(app_module,"answer_question",lambda video_reference,question:result())
    app_module.limiter.enabled=True
    try:
        responses=[
            client.post("/chat",json={"video_id":VALID_VIDEO_ID,"question":"What happened?"})
            for _ in range(21)
        ]
        assert [r.status_code for r in responses[:20]]==[200]*20
        assert responses[20].status_code==429
        assert responses[20].json()["error"]=="rate_limit_exceeded"
        assert responses[20].json()["request_id"]
        assert "x-ratelimit-limit" in {k.lower() for k in responses[0].headers}
    finally:
        app_module.limiter.enabled=False
        app_module.limiter.reset()
