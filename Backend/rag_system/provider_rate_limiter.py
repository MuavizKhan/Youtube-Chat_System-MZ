"""Shared, token-aware pacing for Groq requests.

Groq quotas apply across an organization, so rotating API keys does not create
additional capacity. This module coordinates requests from this backend through
a SQLite ledger and reconciles its estimates with Groq's authoritative rate
limit response headers. The SQLite file must be shared by processes using the
same provider organization.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
import re
import sqlite3
import time
import uuid
from typing import Any, Callable

from .config import (
    APP_ENV,
    GROQ_TPM_LIMIT,
    GROQ_TPM_SAFETY_FACTOR,
    PROVIDER_QUOTA_DB_PATH,
)


RESERVATION_LEASE_SECONDS = 300.0


class ProviderQuotaWaitError(RuntimeError):
    """Raised when the caller's wait budget cannot cover the next quota reset."""

    def __init__(self, retry_after_seconds: float):
        self.retry_after_seconds = max(1.0, float(retry_after_seconds))
        super().__init__("The Groq token budget requires a later retry.")


def parse_provider_duration(value: Any) -> float | None:
    """Parse Groq duration headers such as '7.66s', '1m2.5s', or '250ms'."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0.0, float(value))
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip().lower()
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*(ms|s|m|h)", raw)
    if not matches:
        return None
    multipliers = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    return sum(float(number) * multipliers[unit] for number, unit in matches)


def _normalize_headers(headers: Any) -> dict[str, str]:
    if headers is None:
        return {}
    try:
        return {str(key).lower(): str(value) for key, value in headers.items()}
    except Exception:
        return {}


def estimate_request_tokens(
    messages: Sequence[Mapping[str, Any]],
    max_completion_tokens: int,
) -> int:
    """Conservatively estimate prompt + reserved completion tokens.

    Groq's token-rate limiter can account for the requested completion ceiling,
    not just the eventual short answer. The 3.5-character estimate intentionally
    errs high for typical English and multilingual transcript prompts. Server
    headers replace estimates as soon as the first response is received.
    """
    characters = 0
    message_count = 0
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        message_count += 1
        for key in ("role", "content", "name"):
            value = message.get(key)
            if isinstance(value, str):
                characters += len(value)
    prompt_estimate = math.ceil(characters / 3.5) + (12 * message_count) + 16
    return max(1, prompt_estimate + max(1, int(max_completion_tokens)))


class GroqTokenPacer:
    """SQLite-coordinated sliding-window reservations plus provider-header state."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        default_tpm: int = 8000,
        safety_factor: float = 0.90,
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if isinstance(default_tpm, bool) or int(default_tpm) <= 0:
            raise ValueError("default_tpm must be a positive integer.")
        if not math.isfinite(float(safety_factor)) or not 0.1 <= safety_factor <= 1.0:
            raise ValueError("safety_factor must be between 0.1 and 1.0.")
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.default_tpm = int(default_tpm)
        self.safety_factor = float(safety_factor)
        self.clock = clock
        self.sleeper = sleeper
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS groq_token_reservations (
                    reservation_id TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    reserved_tokens INTEGER NOT NULL,
                    completed_at REAL,
                    lease_until REAL NOT NULL DEFAULT 0
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_groq_token_reservations_window
                ON groq_token_reservations(model, created_at)
                """
            )
            reservation_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(groq_token_reservations)"
                ).fetchall()
            }
            if "completed_at" not in reservation_columns:
                connection.execute(
                    "ALTER TABLE groq_token_reservations ADD COLUMN completed_at REAL"
                )
                # Rows created by the earlier pacer implementation represent
                # completed calls; no live request survives an application
                # restart. Preserve their 60-second window, not an infinite
                # in-flight reservation.
                connection.execute(
                    """
                    UPDATE groq_token_reservations
                    SET completed_at = created_at
                    WHERE completed_at IS NULL
                    """
                )
            if "lease_until" not in reservation_columns:
                connection.execute(
                    """
                    ALTER TABLE groq_token_reservations
                    ADD COLUMN lease_until REAL NOT NULL DEFAULT 0
                    """
                )
                connection.execute(
                    """
                    UPDATE groq_token_reservations
                    SET lease_until = created_at + ?
                    WHERE completed_at IS NULL AND lease_until = 0
                    """,
                    (RESERVATION_LEASE_SECONDS,),
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS groq_token_quota_state (
                    model TEXT PRIMARY KEY,
                    limit_tokens INTEGER NOT NULL,
                    remaining_tokens INTEGER NOT NULL,
                    observed_at REAL NOT NULL,
                    reset_at REAL
                )
                """
            )

    def record_headers(self, model: str, headers: Any) -> None:
        """Store the latest authoritative token limit, remaining count and reset."""
        values = _normalize_headers(headers)
        limit_value = values.get("x-ratelimit-limit-tokens")
        remaining_value = values.get("x-ratelimit-remaining-tokens")
        reset_value = values.get("x-ratelimit-reset-tokens")
        try:
            limit_tokens = int(limit_value) if limit_value is not None else None
            remaining_tokens = int(remaining_value) if remaining_value is not None else None
        except (TypeError, ValueError, OverflowError):
            return
        if (
            limit_tokens is None
            or remaining_tokens is None
            or limit_tokens <= 0
            or remaining_tokens < 0
        ):
            return
        reset_seconds = parse_provider_duration(reset_value)
        if reset_seconds is None:
            # A 429 response may omit the token-reset header while still
            # providing the authoritative Retry-After duration.
            reset_seconds = parse_provider_duration(values.get("retry-after"))
        now = self.clock()
        reset_at = now + reset_seconds if reset_seconds is not None else None
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO groq_token_quota_state (
                    model, limit_tokens, remaining_tokens, observed_at, reset_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(model) DO UPDATE SET
                    limit_tokens = excluded.limit_tokens,
                    remaining_tokens = excluded.remaining_tokens,
                    observed_at = excluded.observed_at,
                    reset_at = excluded.reset_at
                """,
                (model, limit_tokens, remaining_tokens, now, reset_at),
            )

    def _try_reserve(self, model: str, estimated_tokens: int) -> tuple[str | None, float]:
        now = self.clock()
        reservation_id = uuid.uuid4().hex
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                DELETE FROM groq_token_reservations
                WHERE created_at <= ?
                  AND (
                    completed_at IS NOT NULL
                    OR lease_until <= ?
                  )
                """,
                (now - 60.0, now),
            )
            state = connection.execute(
                "SELECT * FROM groq_token_quota_state WHERE model = ?",
                (model,),
            ).fetchone()
            rows = connection.execute(
                """
                SELECT created_at, reserved_tokens, completed_at
                FROM groq_token_reservations
                WHERE model = ?
                  AND (
                    created_at > ?
                    OR (
                        completed_at IS NULL
                        AND lease_until > ?
                    )
                  )
                ORDER BY created_at ASC
                """,
                (
                    model,
                    (state["observed_at"] if state is not None else now - 60.0),
                    now,
                ),
            ).fetchall()

            provider_reset_at = (
                float(state["reset_at"])
                if state is not None and state["reset_at"] is not None
                else (
                    float(state["observed_at"]) + 60.0
                    if state is not None
                    else None
                )
            )
            if state is not None and provider_reset_at is not None and now >= provider_reset_at:
                # Prefer Groq's reset timestamp. If the header is absent, the
                # provider's rolling TPM window still expires 60 seconds after
                # the last authoritative remaining-token snapshot.
                connection.execute(
                    """
                    UPDATE groq_token_quota_state
                    SET remaining_tokens = limit_tokens, observed_at = ?, reset_at = NULL
                    WHERE model = ?
                    """,
                    (now, model),
                )
                state = connection.execute(
                    "SELECT * FROM groq_token_quota_state WHERE model = ?",
                    (model,),
                ).fetchone()
                rows = connection.execute(
                    """
                    SELECT created_at, reserved_tokens, completed_at
                    FROM groq_token_reservations
                    WHERE model = ?
                      AND (
                        created_at > ?
                        OR (
                            completed_at IS NULL
                            AND lease_until > ?
                        )
                      )
                    ORDER BY created_at ASC
                    """,
                    (model, now, now),
                ).fetchall()

            limit_tokens = (
                int(state["limit_tokens"])
                if state is not None
                else self.default_tpm
            )
            safe_limit = max(1, int(limit_tokens * self.safety_factor))
            used_since_snapshot = sum(int(row["reserved_tokens"]) for row in rows)
            if state is not None:
                provider_remaining = max(
                    0,
                    int(state["remaining_tokens"]) - used_since_snapshot,
                )
                available = min(
                    provider_remaining,
                    max(0, safe_limit - used_since_snapshot),
                )
            else:
                # Before the first response header, use a conservative local
                # rolling window. External organization usage is learned from
                # the first provider response and all subsequent header updates.
                all_rows = connection.execute(
                    """
                    SELECT created_at, reserved_tokens, completed_at
                    FROM groq_token_reservations
                    WHERE model = ?
                      AND (
                        created_at > ?
                        OR (
                            completed_at IS NULL
                            AND lease_until > ?
                        )
                      )
                    ORDER BY created_at ASC
                    """,
                    (model, now - 60.0, now),
                ).fetchall()
                used_since_snapshot = sum(int(row["reserved_tokens"]) for row in all_rows)
                available = max(0, safe_limit - used_since_snapshot)
                rows = all_rows

            if estimated_tokens > safe_limit:
                raise ProviderQuotaWaitError(60.0)

            if available < estimated_tokens:
                if state is not None and state["reset_at"] is not None:
                    wait_seconds = max(0.25, float(state["reset_at"]) - now)
                elif rows:
                    tokens_to_release = estimated_tokens - available
                    released = 0
                    wait_seconds = 60.0
                    for row in rows:
                        released += int(row["reserved_tokens"])
                        if row["completed_at"] is None:
                            # An in-flight request may outlive its nominal
                            # rolling-window expiry. Recheck periodically but
                            # keep its reservation until a response is seen.
                            row_wait = max(
                                5.0,
                                float(row["created_at"]) + 60.0 - now,
                            )
                        else:
                            row_wait = max(
                                0.25,
                                float(row["created_at"]) + 60.0 - now,
                            )
                        wait_seconds = max(0.25, row_wait)
                        if released >= tokens_to_release:
                            break
                elif state is not None and state["reset_at"] is None:
                    wait_seconds = max(
                        0.25,
                        float(state["observed_at"]) + 60.0 - now,
                    )
                else:
                    wait_seconds = 60.0
                connection.rollback()
                return None, wait_seconds

            connection.execute(
                """
                INSERT INTO groq_token_reservations (
                    reservation_id, model, created_at, reserved_tokens,
                    completed_at, lease_until
                )
                VALUES (?, ?, ?, ?, NULL, ?)
                """,
                (
                    reservation_id,
                    model,
                    now,
                    estimated_tokens,
                    now + RESERVATION_LEASE_SECONDS,
                ),
            )
            connection.commit()
        return reservation_id, 0.0

    def settle_reservation(self, reservation_id: str) -> None:
        """Mark a provider request complete while retaining quota until its window expires."""
        if not reservation_id or not isinstance(reservation_id, str):
            return
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE groq_token_reservations
                SET completed_at = ?, lease_until = 0
                WHERE reservation_id = ? AND completed_at IS NULL
                """,
                (self.clock(), reservation_id),
            )

    def acquire(
        self,
        model: str,
        estimated_tokens: int,
        *,
        reserve_wait: Callable[[float], bool] | None = None,
    ) -> str:
        """Wait until a request fits the current TPM window, then reserve it."""
        if not model or not isinstance(model, str):
            raise ValueError("model must be a non-empty string.")
        if isinstance(estimated_tokens, bool) or int(estimated_tokens) <= 0:
            raise ValueError("estimated_tokens must be a positive integer.")
        waited = 0.0
        while True:
            reservation_id, wait_seconds = self._try_reserve(
                model,
                int(estimated_tokens),
            )
            if reservation_id is not None:
                if waited:
                    import logging
                    logging.getLogger(__name__).info(
                        "Groq quota pacing completed model=%s waited_seconds=%.2f "
                        "reserved_tokens=%s",
                        model,
                        waited,
                        estimated_tokens,
                    )
                return reservation_id
            if reserve_wait is not None and not reserve_wait(wait_seconds):
                raise ProviderQuotaWaitError(wait_seconds)
            self.sleeper(wait_seconds + 0.25)
            waited += wait_seconds + 0.25

    def snapshot(self, model: str) -> dict[str, Any]:
        """Expose quota state for tests and safe diagnostics."""
        with self._connect() as connection:
            state = connection.execute(
                "SELECT * FROM groq_token_quota_state WHERE model = ?",
                (model,),
            ).fetchone()
            count = connection.execute(
                """
                SELECT COUNT(*) AS count, COALESCE(SUM(reserved_tokens), 0) AS tokens
                FROM groq_token_reservations
                WHERE model = ?
                  AND (
                    created_at > ?
                    OR (
                        completed_at IS NULL
                        AND lease_until > ?
                    )
                  )
                """,
                (model, self.clock() - 60.0, self.clock()),
            ).fetchone()
        return {
            "provider_state": dict(state) if state is not None else None,
            "local_window_reservation_count": int(count["count"]),
            "local_window_reserved_tokens": int(count["tokens"]),
        }


_PACER = GroqTokenPacer(
    PROVIDER_QUOTA_DB_PATH,
    default_tpm=GROQ_TPM_LIMIT,
    safety_factor=GROQ_TPM_SAFETY_FACTOR,
)


def groq_completion_create(
    client: Any,
    request_kwargs: dict[str, Any],
    *,
    reserve_wait: Callable[[float], bool] | None = None,
) -> Any:
    """Create a completion while pacing against TPM and recording response headers."""
    model = str(request_kwargs.get("model") or "")
    messages = request_kwargs.get("messages") or []
    completion_ceiling = int(request_kwargs.get("max_completion_tokens") or 1)

    reservation_id = None
    if APP_ENV != "test":
        estimated_tokens = estimate_request_tokens(messages, completion_ceiling)
        reservation_id = _PACER.acquire(
            model,
            estimated_tokens,
            reserve_wait=reserve_wait,
        )

    endpoint = client.chat.completions
    raw_endpoint = getattr(endpoint, "with_raw_response", None)
    raw_create = getattr(raw_endpoint, "create", None)

    try:
        if callable(raw_create):
            raw_response = raw_create(**request_kwargs)
            _PACER.record_headers(model, getattr(raw_response, "headers", None))
            parser = getattr(raw_response, "parse", None)
            return parser() if callable(parser) else raw_response

        response = endpoint.create(**request_kwargs)
        headers = getattr(response, "headers", None)
        if headers is None:
            headers = getattr(getattr(response, "_response", None), "headers", None)
        _PACER.record_headers(model, headers)
        return response
    except Exception as error:
        response = getattr(error, "response", None)
        _PACER.record_headers(
            model,
            getattr(response, "headers", None),
        )
        raise
    finally:
        if reservation_id is not None:
            _PACER.settle_reservation(reservation_id)
