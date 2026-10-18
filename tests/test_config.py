import os
import subprocess
import sys
from pathlib import Path

import pytest


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

def test_embedding_model_defaults_to_multilingual_model():
    env = os.environ.copy()
    env.pop("HF_EMBEDDING_MODEL", None)
    env["PYTHON_DOTENV_DISABLED"] = "true"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import Backend.rag_system.config as config; "
                "print(config.EMBEDDING_MODEL)"
            ),
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        env=env,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )



def run_index_namespace_import(extra_env: dict[str, str | None]):
    env = os.environ.copy()
    env.pop("RAG_INDEX_NAMESPACE", None)
    env.pop("INDEX_JOB_DB_PATH", None)

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
                "print(f'namespace={config.RAG_INDEX_NAMESPACE}'); "
                "print(f'vector_store_root={config.VECTOR_STORE_ROOT}'); "
                "print(f'index_job_db_path={config.INDEX_JOB_DB_PATH}'); "
                "print(f'project_root={config.PROJECT_ROOT}')"
            ),
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        env=env,
        check=False,
    )


def test_index_namespace_keeps_default_storage_paths_unchanged():
    result = run_index_namespace_import({})

    assert result.returncode == 0
    values = dict(line.split("=", 1) for line in result.stdout.splitlines())
    project_root = Path(values["project_root"])

    assert values["namespace"] == ""
    assert Path(values["vector_store_root"]) == (
        project_root / "Backend" / "rag_system" / "vector_stores"
    )
    assert Path(values["index_job_db_path"]) == (
        project_root / "Backend" / "rag_system" / "index_jobs.sqlite3"
    )


def test_index_namespace_isolates_vector_stores_and_job_database():
    result = run_index_namespace_import(
        {"RAG_INDEX_NAMESPACE": "experiment-bge-m3"}
    )

    assert result.returncode == 0
    values = dict(line.split("=", 1) for line in result.stdout.splitlines())
    project_root = Path(values["project_root"])

    assert values["namespace"] == "experiment-bge-m3"
    assert Path(values["vector_store_root"]) == (
        project_root
        / "Backend" / "rag_system" / "vector_stores"
        / "experiments" / "experiment-bge-m3"
    )
    assert Path(values["index_job_db_path"]) == (
        project_root / "Backend" / "rag_system" / "index_jobs_experiment-bge-m3.sqlite3"
    )


def test_index_namespace_honors_explicit_process_job_database_override():
    result = run_index_namespace_import(
        {
            "RAG_INDEX_NAMESPACE": "experiment-bge-m3",
            "INDEX_JOB_DB_PATH": "custom-experiment-jobs.sqlite3",
        }
    )

    assert result.returncode == 0
    values = dict(line.split("=", 1) for line in result.stdout.splitlines())
    project_root = Path(values["project_root"])

    assert Path(values["index_job_db_path"]) == (
        project_root / "custom-experiment-jobs.sqlite3"
    )


@pytest.mark.parametrize(
    "namespace",
    ["../baseline", "nested/path", "bad namespace", ".", "..", "x" * 65],
)
def test_index_namespace_rejects_unsafe_path_segments(namespace):
    result = run_index_namespace_import({"RAG_INDEX_NAMESPACE": namespace})

    assert result.returncode != 0
    assert "RAG_INDEX_NAMESPACE must be 1-64 characters" in result.stderr


def _run_citation_policy_import(value: str | None):
    env = os.environ.copy()
    env.pop("RAG_CITATION_INCLUDE_CONTEXT_NEIGHBORS", None)
    if value is not None:
        env["RAG_CITATION_INCLUDE_CONTEXT_NEIGHBORS"] = value
    env["PYTHON_DOTENV_DISABLED"] = "true"

    return subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import Backend.rag_system.config as config; "
                "print(config.RAG_CITATION_INCLUDE_CONTEXT_NEIGHBORS)"
            ),
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        env=env,
        check=False,
    )


def test_context_neighbor_citation_experiment_defaults_off():
    result = _run_citation_policy_import(None)
    assert result.returncode == 0
    assert result.stdout.strip() == "False"


def test_context_neighbor_citation_experiment_can_be_enabled():
    result = _run_citation_policy_import("true")
    assert result.returncode == 0
    assert result.stdout.strip() == "True"


def test_context_neighbor_citation_experiment_rejects_invalid_boolean():
    result = _run_citation_policy_import("sometimes")
    assert result.returncode != 0
    assert "RAG_CITATION_INCLUDE_CONTEXT_NEIGHBORS" in result.stderr
