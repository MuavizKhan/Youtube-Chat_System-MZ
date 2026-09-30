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

MAX_DISTANCE = _get_float(
    "RAG_MAX_DISTANCE",
    1.30,
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

HF_MAX_TOKENS = int(
    os.getenv(
        "HF_MAX_TOKENS",
        "800"
    )
)

HF_TEMPERATURE = _get_float(
    "HF_TEMPERATURE",
    0.1,
)


# ============================================================
# VALIDATION
# ============================================================

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


if MAX_DISTANCE < 0:

    raise RuntimeError(
        "RAG_MAX_DISTANCE cannot be negative."
    )


if HF_MAX_TOKENS <= 0:

    raise RuntimeError(
        "HF_MAX_TOKENS must be greater than 0."
    )


if HF_TEMPERATURE < 0:

    raise RuntimeError(
        "HF_TEMPERATURE cannot be negative."
    )