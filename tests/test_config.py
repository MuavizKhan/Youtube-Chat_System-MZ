import os
import subprocess
import sys


def run_config_import(extra_env: dict[str, str | None]):
    env = os.environ.copy()

    for key, value in extra_env.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value

    return subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import Backend.rag_system.config as config; "
                "print(config.LLM_PROVIDER); "
                "print(config.GROQ_MODEL_ID); "
                "print(config.GROQ_REASONING_EFFORT)"
            ),
        ],
        capture_output=True,
        text=True,
        cwd=str(os.getcwd()),
        env=env,
        check=False,
    )


def test_groq_provider_configuration_loads():
    result = run_config_import(
        {
            "LLM_PROVIDER": "groq",
            "GROQ_API_KEY": "test-key",
            "GROQ_MODEL_ID": "openai/gpt-oss-20b",
            "GROQ_REASONING_EFFORT": "low",
        }
    )

    assert result.returncode == 0
    assert result.stdout.splitlines() == [
        "groq",
        "openai/gpt-oss-20b",
        "low",
    ]


def test_groq_provider_requires_api_key():
    result = run_config_import(
        {
            "LLM_PROVIDER": "groq",
            "GROQ_API_KEY": None,
        }
    )

    assert result.returncode != 0
    assert "GROQ_API_KEY must be configured when LLM_PROVIDER=groq." in result.stderr


def test_llm_provider_rejects_unknown_provider():
    result = run_config_import(
        {
            "LLM_PROVIDER": "not-a-provider",
            "GROQ_API_KEY": "test-key",
        }
    )

    assert result.returncode != 0
    assert "LLM_PROVIDER must be one of: huggingface, groq." in result.stderr
