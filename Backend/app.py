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
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
# API ERROR HANDLERS
# ============================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request,
    exception: HTTPException,
):
    request_id = getattr(
        request.state,
        "request_id",
        "unknown",
    )

    detail = exception.detail

    if isinstance(detail, dict):

        error_code = detail.get(
            "error",
            "request_error",
        )

        message = detail.get(
            "message",
            "The request could not be processed.",
        )

    else:

        error_code = {
            400: "invalid_request",
            404: "not_found",
            405: "method_not_allowed",
            409: "conflict",
            429: "rate_limited",
            502: "upstream_error",
            503: "service_not_ready",
        }.get(
            exception.status_code,
            "request_error",
        )

        message = str(
            detail
        )

    logger.warning(
        "API error "
        "request_id=%s status_code=%s error=%s",
        request_id,
        exception.status_code,
        error_code,
    )

    response = JSONResponse(
        status_code=exception.status_code,
        content={
            "error": error_code,
            "message": message,
            "request_id": request_id,
        },
    )

    if exception.headers:

        for header_name, header_value in (
            exception.headers.items()
        ):

            response.headers[
                header_name
            ] = header_value

    return response


# Validation error handler for request body validation errors
@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request,
    exception: RequestValidationError,
):
    request_id = getattr(
        request.state,
        "request_id",
        "unknown",
    )

    logger.warning(
        "Request validation failed "
        "request_id=%s",
        request_id,
    )

    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": (
                "The request contains invalid "
                "or missing fields."
            ),
            "request_id": request_id,
        },
    )

# Global unexpected-error handler
@app.exception_handler(Exception)
async def unexpected_exception_handler(
    request: Request,
    exception: Exception,
):
    request_id = getattr(
        request.state,
        "request_id",
        "unknown",
    )

    logger.exception(
        "Unhandled application exception "
        "request_id=%s",
        request_id,
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": (
                "An unexpected error occurred "
                "while processing the request."
            ),
            "request_id": request_id,
        },
    )


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
# ERROR RESPONSE MODEL
# ============================================================

class ErrorResponse(BaseModel):

    error: str

    message: str

    request_id: str


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

        logger.error(
            "Service is not ready. "
            "Missing configuration: %s",
            ", ".join(
                missing_dependencies
            ),
        )

        raise HTTPException(
            status_code=503,
            detail={
                "error": "service_not_ready",
                "message": (
                    "The service is not ready "
                    "to accept requests."
                ),
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
            detail={
                "error": "invalid_video_reference",
                "message": (
                    "The supplied value is not a "
                    "valid YouTube video ID or URL."
                ),
            },
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
            detail={
                "error": "video_not_indexed",
                "message": (
                    "No indexed transcript is available "
                    "for this video."
                ),
            },
        ) from error

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_request",
                "message": (
                    "The request contains an invalid value."
                ),
            },
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
            detail={
                "error": "ai_service_error",
                "message": (
                    "The AI service could not process "
                    "the request."
                ),
            },
        ) from error