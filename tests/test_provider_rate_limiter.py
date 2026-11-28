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

    first_reservation = pacer.acquire("test-model", 60)
    second_reservation = pacer.acquire("test-model", 30)
    pacer.settle_reservation(first_reservation)
    pacer.settle_reservation(second_reservation)

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



def test_pacer_uses_rolling_sixty_second_fallback_without_reset_headers(tmp_path):
    now = [2000.0]

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
        },
    )

    pacer.acquire("test-model", 20)

    assert now[0] >= 2060.0
    assert pacer.snapshot("test-model")["local_window_reserved_tokens"] == 20



def test_header_snapshot_does_not_discard_an_in_flight_reservation(tmp_path):
    now = [3000.0]

    def sleep(seconds):
        now[0] += seconds

    pacer = GroqTokenPacer(
        tmp_path / "quota.sqlite3",
        default_tpm=100,
        safety_factor=1.0,
        clock=lambda: now[0],
        sleeper=sleep,
    )

    in_flight = pacer.acquire("test-model", 60)
    now[0] += 0.1
    completed_request = pacer.acquire("test-model", 20)
    now[0] += 0.1

    # The second request's response reports 80 tokens remaining. The first
    # reservation is still in flight and is not represented by this snapshot.
    pacer.record_headers(
        "test-model",
        {
            "x-ratelimit-limit-tokens": "100",
            "x-ratelimit-remaining-tokens": "80",
            "x-ratelimit-reset-tokens": "60s",
        },
    )
    pacer.settle_reservation(completed_request)

    # A new 30-token request must not consume the 60 tokens reserved by the
    # first in-flight request merely because it started before observed_at.
    next_reservation = pacer.acquire("test-model", 30)
    assert now[0] >= 3060.0
    pacer.settle_reservation(next_reservation)
    pacer.settle_reservation(in_flight)



def test_abandoned_in_flight_reservation_expires_after_its_lease(tmp_path):
    now = [4000.0]

    def sleep(seconds):
        now[0] += seconds

    db_path = tmp_path / "quota.sqlite3"
    pacer = GroqTokenPacer(
        db_path,
        default_tpm=100,
        safety_factor=1.0,
        clock=lambda: now[0],
        sleeper=sleep,
    )
    reservation = pacer.acquire("test-model", 90)

    import sqlite3
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE groq_token_reservations SET lease_until = ? WHERE reservation_id = ?",
            (now[0] - 1.0, reservation),
        )

    now[0] += 61.0
    replacement = pacer.acquire("test-model", 90)

    assert replacement != reservation
    assert pacer.snapshot("test-model")["local_window_reserved_tokens"] == 90
    pacer.settle_reservation(replacement)


def test_existing_in_flight_rows_receive_a_lease_during_schema_migration(tmp_path):
    import sqlite3

    now = 5000.0
    db_path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE groq_token_reservations (
                reservation_id TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                created_at REAL NOT NULL,
                reserved_tokens INTEGER NOT NULL,
                completed_at REAL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO groq_token_reservations (
                reservation_id, model, created_at, reserved_tokens, completed_at
            )
            VALUES ('legacy-reservation', 'test-model', ?, 40, NULL)
            """,
            (now,),
        )

    GroqTokenPacer(
        db_path,
        default_tpm=100,
        safety_factor=1.0,
        clock=lambda: now,
        sleeper=lambda _seconds: None,
    )

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT completed_at, lease_until FROM groq_token_reservations "
            "WHERE reservation_id = 'legacy-reservation'"
        ).fetchone()

    assert row[0] is None
    assert row[1] == pytest.approx(now + 300.0)



def test_retry_after_only_response_headers_still_block_groq_requests(tmp_path):
    now = [6000.0]

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
        {"retry-after": "8s"},
    )

    pacer.acquire("test-model", 20)

    assert now[0] >= 6008.0
    state = pacer.snapshot("test-model")["provider_state"]
    assert state["limit_tokens"] == 100
    assert state["remaining_tokens"] == 0
