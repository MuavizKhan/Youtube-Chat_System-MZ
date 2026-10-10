from pathlib import Path

import pytest
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

    env["PYTHON_DOTENV_DISABLED"] = "true"

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
        cwd=str(Path(__file__).resolve().parents[1]),
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


def run_retrieval_config_import(extra_env: dict[str, str | None]):
    env = os.environ.copy()

    for key, value in extra_env.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value

    if "RAG_DENSE_LEXICAL_RRF_WEIGHT" not in extra_env:
        env.pop("RAG_DENSE_LEXICAL_RRF_WEIGHT", None)

    env["PYTHON_DOTENV_DISABLED"] = "true"

    return subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import Backend.rag_system.config as config; "
                "print(config.RAG_RERANK_ENABLED); "
                "print(config.RAG_RERANK_MODEL); "
                "print(config.RAG_RERANK_CANDIDATE_K); "
                "print(config.RAG_RERANK_BATCH_SIZE); "
                "print(config.RAG_RERANK_MAX_LENGTH); "
                "print(config.DENSE_LEXICAL_RRF_WEIGHT)"
            ),
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        env=env,
        check=False,
    )


def test_cross_encoder_reranking_configuration_loads():
    result = run_retrieval_config_import(
        {
            "RAG_RERANK_ENABLED": "true",
            "RAG_RERANK_MODEL": "test/reranker",
            "RAG_RERANK_CANDIDATE_K": "12",
            "RAG_RERANK_BATCH_SIZE": "8",
            "RAG_RERANK_MAX_LENGTH": "256",
            "RAG_DENSE_LEXICAL_RRF_WEIGHT": "1.4",
        }
    )

    assert result.returncode == 0
    assert result.stdout.splitlines() == [
        "True",
        "test/reranker",
        "12",
        "8",
        "256",
        "1.4",
    ]


def test_dense_lexical_rrf_weight_defaults_to_documented_value():
    result = run_retrieval_config_import(
        {
            "RAG_RERANK_ENABLED": "false",
            "RAG_RERANK_MODEL": "test/reranker",
            "RAG_RERANK_CANDIDATE_K": "24",
            "RAG_RERANK_BATCH_SIZE": "16",
            "RAG_RERANK_MAX_LENGTH": "512",
        }
    )

    assert result.returncode == 0
    assert result.stdout.splitlines()[-1] == "1.25"


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_dense_lexical_rrf_weight_rejects_invalid_values(value):
    result = run_retrieval_config_import(
        {"RAG_DENSE_LEXICAL_RRF_WEIGHT": value}
    )

    assert result.returncode != 0
    assert "RAG_DENSE_LEXICAL_RRF_WEIGHT must be finite and greater than 0." in result.stderr


def test_cross_encoder_reranking_rejects_invalid_boolean():
    result = run_retrieval_config_import(
        {
            "RAG_RERANK_ENABLED": "maybe",
        }
    )

    assert result.returncode != 0
    assert "RAG_RERANK_ENABLED" in result.stderr


def test_cross_encoder_reranking_rejects_non_positive_candidate_k():
    result = run_retrieval_config_import(
        {
            "RAG_RERANK_CANDIDATE_K": "0",
        }
    )

    assert result.returncode != 0
    assert "RAG_RERANK_CANDIDATE_K must be greater than 0." in result.stderr
