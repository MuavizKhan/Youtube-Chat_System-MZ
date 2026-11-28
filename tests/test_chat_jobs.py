from Backend.rag_system.chat_jobs import (
    COMPLETED,
    FAILED,
    QUEUED,
    RUNNING,
    ChatJobStore,
)


def test_chat_job_lifecycle_and_active_request_deduplication(tmp_path):
    store = ChatJobStore(tmp_path / "chat.sqlite3", lease_seconds=60)
    history = [{"role": "user", "content": "Earlier question"}]

    first = store.enqueue("VKrtiTgBeng", "Summarize the whole video.", history)
    duplicate = store.enqueue("VKrtiTgBeng", "Summarize the whole video.", history)

    assert first.job_id == duplicate.job_id
    assert first.status == QUEUED

    claimed = store.claim(first.job_id, worker_id="worker-1")
    assert claimed is not None
    assert claimed.status == RUNNING
    assert claimed.attempt == 1

    active_duplicate = store.enqueue(
        "VKrtiTgBeng",
        "Summarize the whole video.",
        history,
    )
    assert active_duplicate.job_id == first.job_id
    assert active_duplicate.status == RUNNING

    result = {
        "answer": "1. Opening.\n2. Middle.\n3. Ending.",
        "video_id": "VKrtiTgBeng",
        "sources": [],
        "retrieved_chunks": 3,
        "source_segments": 3,
        "model": "test-model",
        "retrieval_method": "video_wide_overview",
        "retrieval_config": {},
        "observability": {},
    }
    assert store.complete(first.job_id, "worker-1", result) is True

    completed = store.get(first.job_id)
    assert completed is not None
    assert completed.status == COMPLETED
    assert completed.result == result
    assert completed.error_code is None


def test_failed_chat_job_can_be_retried_with_a_new_job_id(tmp_path):
    store = ChatJobStore(tmp_path / "chat.sqlite3", lease_seconds=60)
    job = store.enqueue("VKrtiTgBeng", "Summarize the entire video.", [])

    claimed = store.claim(job.job_id, worker_id="worker-1")
    assert claimed is not None
    assert store.fail(
        job.job_id,
        worker_id="worker-1",
        error_code="ai_provider_rate_limited",
        error_message="Provider quota window did not reset within the job budget.",
        retry_after_seconds=30,
    )

    failed = store.get(job.job_id)
    assert failed is not None
    assert failed.status == FAILED
    assert failed.error_code == "ai_provider_rate_limited"
    assert failed.retry_after_seconds == 30

    retry = store.enqueue("VKrtiTgBeng", "Summarize the entire video.", [])
    assert retry.job_id != job.job_id
    assert retry.status == QUEUED


def test_expired_chat_job_lease_is_requeued_and_attempt_is_preserved(
    tmp_path,
    monkeypatch,
):
    import Backend.rag_system.chat_jobs as chat_jobs

    now = [10_000.0]
    monkeypatch.setattr(chat_jobs.time, "time", lambda: now[0])
    store = ChatJobStore(tmp_path / "chat.sqlite3", lease_seconds=60)
    job = store.enqueue("VKrtiTgBeng", "Summarize the entire video.", [])

    claimed = store.claim(job.job_id, worker_id="worker-1")
    assert claimed is not None
    assert claimed.status == RUNNING
    assert claimed.attempt == 1

    now[0] += 61.0
    assert store.recover_expired() == 1

    recovered = store.get(job.job_id)
    assert recovered is not None
    assert recovered.status == QUEUED
    assert recovered.worker_id is None

    claimed_again = store.claim(job.job_id, worker_id="worker-2")
    assert claimed_again is not None
    assert claimed_again.status == RUNNING
    assert claimed_again.attempt == 2


def test_chat_job_rejects_short_lease_duration(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        ChatJobStore(tmp_path / "chat.sqlite3", lease_seconds=59)



def test_background_worker_reports_daily_provider_quota_exhaustion(
    tmp_path,
    monkeypatch,
):
    import Backend.app as app_module

    store = ChatJobStore(tmp_path / "worker.sqlite3", lease_seconds=60)
    job = store.enqueue("VKrtiTgBeng", "Summarize the entire video.", [])

    def exhausted_provider(**_kwargs):
        raise RuntimeError(
            "Groq daily token quota is exhausted for the configured model."
        )

    monkeypatch.setattr(app_module, "_CHAT_JOB_STORE", store)
    monkeypatch.setattr(app_module, "answer_question", exhausted_provider)
    monkeypatch.setattr(app_module, "_on_chat_job_finished", lambda _job_id: None)

    app_module._run_chat_job(job.job_id)

    failed = store.get(job.job_id)
    assert failed is not None
    assert failed.status == FAILED
    assert failed.error_code == "ai_provider_daily_quota_exhausted"
    assert "daily token quota" in failed.error_message.lower()
