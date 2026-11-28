import pytest

from Backend.rag_system.provider_rate_limiter import (
    GroqTokenPacer,
    ProviderQuotaWaitError,
    estimate_request_tokens,
    parse_provider_duration,
)


def test_provider_duration_parser_supports_groq_header_formats():
    assert parse_provider_duration("7.66s") == pytest.approx(7.66)
    assert parse_provider_duration("1m2.5s") == pytest.approx(62.5)
    assert parse_provider_duration("250ms") == pytest.approx(0.25)
    assert parse_provider_duration(3.5) == pytest.approx(3.5)
    assert parse_provider_duration("invalid") is None


def test_request_token_estimate_includes_prompt_and_completion_reserve():
    estimate = estimate_request_tokens(
        [
            {"role": "system", "content": "You must use transcript evidence."},
            {"role": "user", "content": "Summarize this video."},
        ],
        max_completion_tokens=500,
    )
    assert estimate > 500
    assert estimate < 800


def test_sqlite_pacer_reserves_tokens_and_waits_for_rolling_window(tmp_path):
    now = [100.0]

    def sleep(seconds):
        now[0] += seconds

    pacer = GroqTokenPacer(
        tmp_path / "quota.sqlite3",
        default_tpm=100,
        safety_factor=1.0,
        clock=lambda: now[0],
        sleeper=sleep,
    )

    pacer.acquire("test-model", 60)
    pacer.acquire("test-model", 30)

    before = pacer.snapshot("test-model")
    assert before["local_window_reservation_count"] == 2
    assert before["local_window_reserved_tokens"] == 90

    pacer.acquire("test-model", 20)

    after = pacer.snapshot("test-model")
    assert now[0] >= 160.0
    assert after["local_window_reserved_tokens"] == 20


def test_provider_remaining_and_reset_headers_override_local_estimates(tmp_path):
    now = [500.0]

    def sleep(seconds):
        now[0] += seconds

    pacer = GroqTokenPacer(
        tmp_path / "quota.sqlite3",
        default_tpm=8000,
        safety_factor=0.90,
        clock=lambda: now[0],
        sleeper=sleep,
    )
    pacer.record_headers(
        "openai/gpt-oss-20b",
        {
            "x-ratelimit-limit-tokens": "8000",
            "x-ratelimit-remaining-tokens": "1000",
            "x-ratelimit-reset-tokens": "7.5s",
        },
    )

    pacer.acquire("openai/gpt-oss-20b", 1500)

    assert now[0] >= 508.0
    snapshot = pacer.snapshot("openai/gpt-oss-20b")
    assert snapshot["local_window_reserved_tokens"] == 1500
    assert snapshot["provider_state"]["limit_tokens"] == 8000


def test_retry_after_header_is_used_when_token_reset_header_is_missing(tmp_path):
    now = [1000.0]

    def sleep(seconds):
        now[0] += seconds

    pacer = GroqTokenPacer(
        tmp_path / "quota.sqlite3",
        default_tpm=100,
        safety_factor=1.0,
        clock=lambda: now[0],
        sleeper=sleep,
    )
    pacer.record_headers(
        "test-model",
        {
            "x-ratelimit-limit-tokens": "100",
            "x-ratelimit-remaining-tokens": "0",
            "retry-after": "4s",
        },
    )
    pacer.acquire("test-model", 20)
    assert now[0] >= 1004.0


def test_pacer_respects_request_scoped_retry_wait_budget(tmp_path):
    pacer = GroqTokenPacer(
        tmp_path / "quota.sqlite3",
        default_tpm=100,
        safety_factor=1.0,
        clock=lambda: 100.0,
        sleeper=lambda _seconds: pytest.fail("must not sleep after wait budget is rejected"),
    )
    pacer.record_headers(
        "test-model",
        {
            "x-ratelimit-limit-tokens": "100",
            "x-ratelimit-remaining-tokens": "0",
            "x-ratelimit-reset-tokens": "20s",
        },
    )

    with pytest.raises(ProviderQuotaWaitError) as error:
        pacer.acquire(
            "test-model",
            20,
            reserve_wait=lambda _seconds: False,
        )

    assert error.value.retry_after_seconds == pytest.approx(20.0)


def test_pacer_rejects_a_request_larger_than_the_safe_model_window(tmp_path):
    pacer = GroqTokenPacer(
        tmp_path / "quota.sqlite3",
        default_tpm=100,
        safety_factor=0.9,
        clock=lambda: 10.0,
        sleeper=lambda _seconds: pytest.fail("oversized requests must fail immediately"),
    )

    with pytest.raises(ProviderQuotaWaitError):
        pacer.acquire("test-model", 91)
