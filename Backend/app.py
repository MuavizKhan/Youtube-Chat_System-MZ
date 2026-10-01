"""
app.py

FastAPI entry point for the YouTube Video ChatSystem.

Flow:

Chrome Extension
        ↓
POST /chat
        ↓
FastAPI
        ↓
RAG chain
        ↓
MMR retrieval
        ↓
Prompt
        ↓
Hugging Face
        ↓
Answer + Sources
"""

import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from Backend.rag_system.chain import answer_question
from Backend.rag_system.retrieval import extract_video_id

from Backend.rag_system.config import (
    CORS_ALLOW_ORIGINS,
    HF_MODEL_ID,
    HF_TOKEN,
)

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(
    __name__
)


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(

    title="YouTube Video AI Chat",

    description=(
        "RAG-powered backend for chatting "
        "with YouTube videos."
    ),

    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=False,
    allow_methods=[
        "GET",
        "POST",
        "OPTIONS",
    ],
    allow_headers=[
        "Content-Type",
        "Accept",
    ],
    expose_headers=[
        "X-Request-ID",
    ],
)


# ============================================================
# REQUEST CONTEXT / LOGGING MIDDLEWARE
# ============================================================

@app.middleware("http")
async def request_context_middleware(
    request: Request,
    call_next,
):
    request_id = uuid.uuid4().hex

    request.state.request_id = request_id

    start_time = time.perf_counter()

    try:

        response = await call_next(
            request
        )

    except Exception:

        duration_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        logger.exception(
            "request failed "
            "request_id=%s method=%s path=%s "
            "duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            duration_ms,
        )

        raise

    duration_ms = (
        time.perf_counter()
        - start_time
    ) * 1000

    response.headers[
        "X-Request-ID"
    ] = request_id

    logger.info(
        "request completed "
        "request_id=%s method=%s path=%s "
        "status_code=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )

    return response


# ============================================================
# REQUEST MODELS
# ============================================================

class ChatRequest(BaseModel):

    video_id: str = Field(
    ...,
    min_length=1,
    max_length=2048,
    description=(
        "YouTube video ID or YouTube URL."
    ),
)

    question: str = Field(
    ...,
    min_length=1,
    max_length=1000,
    description=(
        "Question about the YouTube video."
    ),
)


class SourceResponse(BaseModel):

    source_id: int

    chunk_ids: list[int]

    video_id: str | None

    start: float

    end: float

    duration: float

    distance: float


class ChatResponse(BaseModel):

    answer: str

    video_id: str

    sources: list[SourceResponse]

    retrieved_chunks: int

    model: str

    source_segments: int

    retrieval_method: str

    retrieval_config: dict[str, int | float]


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "ok",
        "service": "YouTube Video AI Chat",
        "message": "FastAPI backend is running.",
        "version": app.version,
    }


# ============================================================
# HEALTH CHECKS
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "YouTube Video AI Chat",
        "version": app.version,
    }


@app.get("/ready")
def readiness():

    missing_dependencies = []

    if not HF_TOKEN:
        missing_dependencies.append(
            "HF_TOKEN"
        )

    if not HF_MODEL_ID:
        missing_dependencies.append(
            "HF_MODEL_ID"
        )

    if missing_dependencies:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "missing": missing_dependencies,
            },
        )

    return {
        "status": "ready",
        "service": "YouTube Video AI Chat",
        "version": app.version,
    }


# ============================================================
# CHAT
# ============================================================

@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
    http_request: Request,
):

    video_id = request.video_id.strip()
    question = request.question.strip()

    if not video_id:

        raise HTTPException(
            status_code=400,
            detail="video_id cannot be empty.",
        )

    if not question:

        raise HTTPException(
            status_code=400,
            detail="question cannot be empty.",
        )

    try:

        canonical_video_id = extract_video_id(
            video_id
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    try:

        result = answer_question(
            video_reference=canonical_video_id,
            question=question,
        )

        return result


    except FileNotFoundError as error:

        logger.warning(
            "Vector store not found. "
            "request_id=%s video_id=%s",
            http_request.state.request_id,
            canonical_video_id,
        )

        raise HTTPException(
            status_code=404,
            detail=(
                "No indexed transcript is available "
                "for this video."
            ),
        ) from error

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except RuntimeError as error:

        logger.exception(
            "Upstream RAG/Hugging Face error. "
            "request_id=%s video_id=%s",
            http_request.state.request_id,
            canonical_video_id,
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "The AI service could not process "
                "the request."
            ),
        ) from error

    except Exception as error:

        logger.exception(
            "Unexpected /chat error. "
            "request_id=%s",
            http_request.state.request_id,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "An unexpected error occurred "
                "while processing the request."
            ),
        ) from error