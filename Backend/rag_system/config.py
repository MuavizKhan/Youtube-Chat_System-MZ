"""
config.py

Central configuration for the YouTube RAG system.

This file contains configuration only.
Application logic belongs in the other modules.
"""

from pathlib import Path
import os

from dotenv import load_dotenv


# ============================================================
# PROJECT PATHS
# ============================================================

# config.py
#     ↓
# rag_system/
#     ↓ parents[0] = rag_system
#     ↓ parents[1] = Backend
#     ↓ parents[2] = project root

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BACKEND_ROOT = PROJECT_ROOT / "Backend"

RAG_ROOT = Path(__file__).resolve().parent

VECTOR_STORE_ROOT = RAG_ROOT / "vector_stores"


# ============================================================
# ENVIRONMENT
# ============================================================

ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_FILE)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _get_int(
    name: str,
    default: int,
) -> int:

    value = os.getenv(name)

    if value is None:
        return default

    try:
        return int(value)

    except ValueError as error:

        raise RuntimeError(
            f"Environment variable '{name}' must be an integer."
        ) from error


def _get_float(
    name: str,
    default: float,
) -> float:

    value = os.getenv(name)

    if value is None:
        return default

    try:
        return float(value)

    except ValueError as error:

        raise RuntimeError(
            f"Environment variable '{name}' must be a number."
        ) from error


def _get_list(
    name: str,
    default: list[str],
) -> list[str]:
    """
    Read a comma-separated environment variable
    and return a cleaned list of values.
    """

    value = os.getenv(name)

    if value is None or not value.strip():
        return default.copy()

    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]

# ============================================================
# INDEX JOB COORDINATION
# ============================================================

_default_index_job_db_path = RAG_ROOT / "index_jobs.sqlite3"

INDEX_JOB_DB_PATH = Path(
    os.getenv(
        "INDEX_JOB_DB_PATH",
        str(_default_index_job_db_path),
    )
).expanduser()

if not INDEX_JOB_DB_PATH.is_absolute():
    INDEX_JOB_DB_PATH = PROJECT_ROOT / INDEX_JOB_DB_PATH

INDEX_JOB_WORKERS = _get_int(
    "INDEX_JOB_WORKERS",
    2,
)

INDEX_JOB_LEASE_SECONDS = _get_int(
    "INDEX_JOB_LEASE_SECONDS",
    1800,
)


# ============================================================
# APPLICATION
# ============================================================

APP_ENV = os.getenv(
    "APP_ENV",
    "development",
).strip().lower()


if APP_ENV not in {
    "development",
    "production",
    "test",
}:

    raise RuntimeError(
        "APP_ENV must be one of: "
        "development, production, test."
    )


CORS_ALLOW_ORIGINS = _get_list(
    "CORS_ALLOW_ORIGINS",
    ["*"] if APP_ENV == "development" else [],
)


if "*" in CORS_ALLOW_ORIGINS and len(
    CORS_ALLOW_ORIGINS
) > 1:

    raise RuntimeError(
        "CORS_ALLOW_ORIGINS cannot combine "
        "'*' with explicit origins."
    )


if (
    APP_ENV == "production"
    and not CORS_ALLOW_ORIGINS
):

    raise RuntimeError(
        "CORS_ALLOW_ORIGINS must be configured "
        "when APP_ENV=production."
    )

# ============================================================
# NETWORK / HOST HARDENING
# ============================================================

MAX_REQUEST_BODY_BYTES = _get_int(
    "MAX_REQUEST_BODY_BYTES",
    16384,
)

TRUSTED_HOSTS = _get_list(
    "TRUSTED_HOSTS",
    (
        ["testserver", "127.0.0.1", "localhost"]
        if APP_ENV == "test"
        else ["127.0.0.1", "localhost"]
    ),
)

if APP_ENV == "production" and not TRUSTED_HOSTS:
    raise RuntimeError(
        "TRUSTED_HOSTS must be configured when APP_ENV=production."
    )


# ============================================================
# RATE LIMITING
# ============================================================

RATE_LIMIT = os.getenv(
    "RATELIMIT_LIMIT",
    "20/minute",
).strip()


RATE_LIMIT_STORAGE_URI = os.getenv(
    "RATELIMIT_STORAGE_URI",
    "memory://",
).strip()


if not RATE_LIMIT:

    raise RuntimeError(
        "RATELIMIT_LIMIT cannot be empty."
    )


if not RATE_LIMIT_STORAGE_URI:

    raise RuntimeError(
        "RATELIMIT_STORAGE_URI cannot be empty."
    )


if (
    APP_ENV == "production"
    and RATE_LIMIT_STORAGE_URI == "memory://"
):

    raise RuntimeError(
        "Production rate limiting requires "
        "shared storage. Configure "
        "RATELIMIT_STORAGE_URI for Redis "
        "or another shared backend."
    )



# ============================================================
# YOUTUBE / INDEXING
# ============================================================

PREFERRED_LANGUAGES = ["en"]

CHUNK_SIZE = _get_int(
    "RAG_CHUNK_SIZE",
    1000,
)

CHUNK_OVERLAP = _get_int(
    "RAG_CHUNK_OVERLAP",
    200,
)


# ============================================================
# EMBEDDINGS
# ============================================================

EMBEDDING_MODEL = os.getenv(
    "HF_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)


# ============================================================
# RETRIEVAL
# ============================================================

TOP_K = _get_int(
    "RAG_TOP_K",
    4,
)

MMR_FETCH_K = _get_int(
    "RAG_MMR_FETCH_K",
    10,
)

MMR_LAMBDA = _get_float(
    "RAG_MMR_LAMBDA",
    0.7,
)

# Reciprocal Rank Fusion constant used to combine semantic and lexical
# rankings without comparing their incompatible score scales.
RRF_K = _get_int(
    "RAG_RRF_K",
    60,
)

MAX_DISTANCE = _get_float(
    "RAG_MAX_DISTANCE",
    1.30,
)


# Optional context expansion around strong retrieval hits.
# A window of 1 adds at most one chronological neighbor on each side.
CONTEXT_EXPANSION_CHUNKS = _get_int(
    "RAG_CONTEXT_EXPANSION_CHUNKS",
    1,
)

# Hard upper bound for the final retrieved context before generation.
CONTEXT_MAX_CHUNKS = _get_int(
    "RAG_CONTEXT_MAX_CHUNKS",
    12,
)


# Evidence-dense questions need more semantic and downstream capacity because
# one answer can require multiple transcript regions.
DENSE_SEMANTIC_K = _get_int(
    "RAG_DENSE_SEMANTIC_K",
    24,
)

DENSE_SEMANTIC_FETCH_K = _get_int(
    "RAG_DENSE_SEMANTIC_FETCH_K",
    32,
)

DENSE_ANCHOR_LIMIT = _get_int(
    "RAG_DENSE_ANCHOR_LIMIT",
    16,
)

DENSE_CONTEXT_MAX_CHUNKS = _get_int(
    "RAG_DENSE_CONTEXT_MAX_CHUNKS",
    16,
)

# Reserve small anchor capacity for each planned evidence facet.
DENSE_FACET_MIN_ANCHORS = _get_int(
    "RAG_DENSE_FACET_MIN_ANCHORS",
    1,
)

DENSE_FACET_MAX_ANCHORS = _get_int(
    "RAG_DENSE_FACET_MAX_ANCHORS",
    3,
)


# ============================================================
# SOURCE SEGMENTATION
# ============================================================

# Retrieval chunks can overlap because chunk_overlap is used
# during indexing. For UI/source display, overlapping chunks
# are merged into one chronological evidence segment.

SOURCE_MERGE_GAP_SECONDS = _get_float(
    "RAG_SOURCE_MERGE_GAP_SECONDS",
    0.0,
)

# Validation
if SOURCE_MERGE_GAP_SECONDS < 0:

    raise RuntimeError(
        "RAG_SOURCE_MERGE_GAP_SECONDS cannot be negative."
    )

# ============================================================
# HUGGING FACE GENERATION
# ============================================================

HF_TOKEN = os.getenv(
    "HF_TOKEN",
)

HF_MODEL_ID = os.getenv(
    "HF_MODEL_ID",
    "openai/gpt-oss-20b",
)

HF_PROVIDER = os.getenv(
    "HF_PROVIDER",
    "auto",
)

HF_MAX_TOKENS = _get_int(
    "HF_MAX_TOKENS",
    1200,
)

HF_TEMPERATURE = _get_float(
    "HF_TEMPERATURE",
    0.1,
)

HF_REASONING_EFFORT = os.getenv(
    "HF_REASONING_EFFORT",
    "low",
).strip().lower()

if HF_REASONING_EFFORT not in {
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
}:
    raise RuntimeError(
        "HF_REASONING_EFFORT must be one of: "
        "none, minimal, low, medium, high, xhigh."
    )

HF_MAX_RETRIES = _get_int(
    "HF_MAX_RETRIES",
    2,
)

HF_RETRY_DELAY_SECONDS = _get_float(
    "HF_RETRY_DELAY_SECONDS",
    1.0,
)

if HF_MAX_RETRIES < 0:

    raise RuntimeError(
        "HF_MAX_RETRIES cannot be negative."
    )


if HF_RETRY_DELAY_SECONDS < 0:

    raise RuntimeError(
        "HF_RETRY_DELAY_SECONDS cannot be negative."
    )


# ============================================================
# VALIDATION
# ============================================================\n\nif MAX_REQUEST_BODY_BYTES <= 0:
    raise RuntimeError(
        "MAX_REQUEST_BODY_BYTES must be greater than 0."
    )




if INDEX_JOB_WORKERS <= 0:

    raise RuntimeError(
        "INDEX_JOB_WORKERS must be greater than 0."
    )


if INDEX_JOB_LEASE_SECONDS < 30:

    raise RuntimeError(
        "INDEX_JOB_LEASE_SECONDS must be at least 30 seconds."
    )

if CHUNK_SIZE <= 0:

    raise RuntimeError(
        "RAG_CHUNK_SIZE must be greater than 0."
    )


if CHUNK_OVERLAP < 0:

    raise RuntimeError(
        "RAG_CHUNK_OVERLAP cannot be negative."
    )


if CHUNK_OVERLAP >= CHUNK_SIZE:

    raise RuntimeError(
        "RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE."
    )


if TOP_K <= 0:

    raise RuntimeError(
        "RAG_TOP_K must be greater than 0."
    )


if MMR_FETCH_K <= 0:

    raise RuntimeError(
        "RAG_MMR_FETCH_K must be greater than 0."
    )


if TOP_K > MMR_FETCH_K:

    raise RuntimeError(
        "RAG_TOP_K cannot be greater than RAG_MMR_FETCH_K."
    )


if not 0.0 <= MMR_LAMBDA <= 1.0:

    raise RuntimeError(
        "RAG_MMR_LAMBDA must be between 0.0 and 1.0."
    )


if RRF_K <= 0:
    raise RuntimeError(
        "RAG_RRF_K must be greater than 0."
    )


if MAX_DISTANCE < 0:

    raise RuntimeError(
        "RAG_MAX_DISTANCE cannot be negative."
    )


if DENSE_FACET_MIN_ANCHORS <= 0:

    raise RuntimeError(
        "RAG_DENSE_FACET_MIN_ANCHORS must be greater than 0."
    )


if DENSE_FACET_MAX_ANCHORS <= 0:

    raise RuntimeError(
        "RAG_DENSE_FACET_MAX_ANCHORS must be greater than 0."
    )


if DENSE_FACET_MIN_ANCHORS > DENSE_FACET_MAX_ANCHORS:

    raise RuntimeError(
        "RAG_DENSE_FACET_MIN_ANCHORS cannot be greater than "
        "RAG_DENSE_FACET_MAX_ANCHORS."
    )


if DENSE_FACET_MIN_ANCHORS * 3 > DENSE_ANCHOR_LIMIT:

    raise RuntimeError(
        "DENSE_FACET_MIN_ANCHORS reserves more slots than "
        "RAG_DENSE_ANCHOR_LIMIT can provide."
    )


if DENSE_FACET_MAX_ANCHORS > DENSE_ANCHOR_LIMIT:

    raise RuntimeError(
        "DENSE_FACET_MAX_ANCHORS cannot be greater than "
        "RAG_DENSE_ANCHOR_LIMIT."
    )


if CONTEXT_EXPANSION_CHUNKS < 0:

    raise RuntimeError(
        "RAG_CONTEXT_EXPANSION_CHUNKS cannot be negative."
    )


if CONTEXT_MAX_CHUNKS <= 0:

    raise RuntimeError(
        "RAG_CONTEXT_MAX_CHUNKS must be greater than 0."
    )


if HF_MAX_TOKENS <= 0:

    raise RuntimeError(
        "HF_MAX_TOKENS must be greater than 0."
    )


if HF_TEMPERATURE < 0:

    raise RuntimeError(
        "HF_TEMPERATURE cannot be negative."
    )