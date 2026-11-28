import pytest

from Backend.rag_system.provider_retry_budget import (
    begin_groq_retry_budget,
    current_groq_retry_budget,
    end_groq_retry_budget,
    reserve_groq_retry_wait,
)


def test_request_retry_budget_accumulates_waits_and_rejects_over_budget():
    token = begin_groq_retry_budget(max_wait_seconds=25.0)
    try:
        assert reserve_groq_retry_wait(20.0) is True
        assert reserve_groq_retry_wait(5.0) is True
        assert reserve_groq_retry_wait(0.01) is False
        assert current_groq_retry_budget() == {
            "max_wait_seconds": 25.0,
            "waited_seconds": 25.0,
        }
    finally:
        end_groq_retry_budget(token)


def test_request_retry_budget_is_isolated_and_nested_contexts_restore_previous():
    outer = begin_groq_retry_budget(max_wait_seconds=30.0)
    try:
        assert reserve_groq_retry_wait(7.0) is True
        inner = begin_groq_retry_budget(max_wait_seconds=3.0)
        try:
            assert current_groq_retry_budget() == {
                "max_wait_seconds": 3.0,
                "waited_seconds": 0.0,
            }
            assert reserve_groq_retry_wait(4.0) is False
        finally:
            end_groq_retry_budget(inner)
        assert current_groq_retry_budget() == {
            "max_wait_seconds": 30.0,
            "waited_seconds": 7.0,
        }
    finally:
        end_groq_retry_budget(outer)


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan"), True])
def test_request_retry_budget_rejects_invalid_limits(value):
    with pytest.raises(ValueError):
        begin_groq_retry_budget(max_wait_seconds=value)


def test_retry_reservation_rejects_invalid_wait_values():
    for value in (-1, float("inf"), float("nan"), True):
        with pytest.raises(ValueError):
            reserve_groq_retry_wait(value)
