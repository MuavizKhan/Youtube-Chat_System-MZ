import os


_TEST_ENVIRONMENT = {
    "APP_ENV": "test",
    "HF_TOKEN": "test-token",
    "HF_MODEL_ID": "test-model",
    "HF_PROVIDER": "auto",
    "CORS_ALLOW_ORIGINS": "*",
    "RATELIMIT_LIMIT": "20/minute",
    "RATELIMIT_STORAGE_URI": "memory://",
}

for _name, _value in _TEST_ENVIRONMENT.items():
    os.environ[_name] = _value


import pytest


@pytest.fixture(autouse=True)
def disable_rate_limiting_by_default():
    """Keep tests isolated; the rate-limit test enables the limiter."""
    from Backend.app import limiter

    previous_state = limiter.enabled
    limiter.enabled = False

    try:
        yield
    finally:
        limiter.enabled = previous_state
        limiter.reset()
