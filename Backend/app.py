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

from contextlib import asynccontextmanager
import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from typing import Literal

from Backend.rag_system.chain import answer_question
from Backend.rag_system.retrieval import extract_video_id
from Backend.rag_system.index_service import (
    IndexNotReadyError,
    get_index_status,
    recover_index_jobs,
    shutdown_index_workers,
    submit_index,
)

from Backend.rag_system.config import (
    CORS_ALLOW_ORIGINS,
    HF_MODEL_ID,
    HF_TOKEN,
    RATE_LIMIT,
    RATE_LIMIT_STORAGE_URI,
    MAX_REQUEST_BODY_BYTES,
    TRUSTED_HOSTS,
)

from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

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
# APPLICATION LIFECYCLE
# ============================================================

@asynccontextmanager
async def lifespan(_app: FastAPI):
    recover_index_jobs()
    try:
        yield
    finally:
        shutdown_index_workers()


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
    lifespan=lifespan,
)

# ============================================================
# RATE LIMITING
# ============================================================

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=RATE_LIMIT_STORAGE_URI,
    headers_enabled=True,
)

app.state.limiter = limiter


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
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset",
        "Retry-After",
    ],
)


# ============================================================
# HOST HARDENING
# ============================================================

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=TRUSTED_HOSTS,
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

    content_length = request.headers.get("content-length")

    if content_length:
        try:
            request_size = int(content_length)
        except ValueError:
            request_size = None

        if (
            request_size is not None
            and request_size > MAX_REQUEST_BODY_BYTES
        ):
            logger.warning(
                "request rejected: body too large "
                "request_id=%s size=%s limit=%s method=%s path=%s",
                request_id,
                request_size,
                MAX_REQUEST_BODY_BYTES,
                request.method,
                request.url.path,
            )

            response = JSONResponse(
                status_code=413,
                content={
                    "error": "request_too_large",
                    "message": "The request body is too large.",
                    "request_id": request_id,
                },
            )
            response.headers["X-Request-ID"] = request_id
            return response

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

    error_content = {
        "error": error_code,
        "message": message,
        "request_id": request_id,
    }

    if isinstance(detail, dict):
        if "state" in detail:
            error_content["state"] = detail["state"]
        if "job_id" in detail:
            error_content["job_id"] = detail["job_id"]

    response = JSONResponse(
        status_code=exception.status_code,
        content=error_content,
    )

    if exception.headers:

        for header_name, header_value in (
            exception.headers.items()
        ):

            response.headers[
                header_name
            ] = header_value

    return response


# ============================================================
# RATE LIMIT ERROR HANDLER
# ============================================================

@app.exception_handler(RateLimitExceeded)
async def rate_limit_exception_handler(
    request: Request,
    exception: RateLimitExceeded,
):

    request_id = getattr(
        request.state,
        "request_id",
        "unknown",
    )

    logger.warning(
        "Rate limit exceeded "
        "request_id=%s path=%s",
        request_id,
        request.url.path,
    )

    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "message": (
                "Too many requests. "
                "Please try again later."
            ),
            "request_id": request_id,
        },
    )



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

class IndexPrepareRequest(BaseModel):
    video_id: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="YouTube video ID or URL to prepare.",
    )


class IndexStatusResponse(BaseModel):
    video_id: str
    state: str
    action: str
    ready: bool
    job_id: str | None = None


class ConversationTurn(BaseModel):

    role: Literal["user", "assistant"]

    content: str = Field(
        ...,
        min_length=1,
        max_length=1000,
    )


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

    conversation_history: list[ConversationTurn] = Field(
        default_factory=list,
        max_length=6,
        description=(
            "Recent user/assistant turns used only to resolve "
            "follow-up references."
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

    retrieval_config: dict[str, int | float | bool | str]

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
# INDEX PREPARATION AND STATUS
# ============================================================

@app.post(
    "/index",
    response_model=IndexStatusResponse,
)
@limiter.limit(RATE_LIMIT)
def index_prepare(
    index_request: IndexPrepareRequest,
    request: Request,
    response: Response,
):
    """Queue, reuse, or rebuild a video's index without blocking the request."""


    try:
        video_id = extract_video_id(
            index_request.video_id.strip()
        )
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_video_reference",
                "message": "The supplied value is not a valid YouTube video ID or URL.",
            },
        ) from error

    try:
        result = submit_index(video_id)
        logger.info(
            "Index request accepted request_id=%s video_id=%s state=%s action=%s",
            request.state.request_id,
            video_id,
            result.state,
            result.action,
        )

        if result.state == "failed":
            response.status_code = 503
        elif not result.ready:
            response.status_code = 202

        return {
            "video_id": result.video_id,
            "state": result.state,
            "action": result.action,
            "ready": result.ready,
            "job_id": result.job_id,
        }

    except Exception as error:
        logger.exception(
            "Index preparation failed request_id=%s video_id=%s",
            request.state.request_id,
            video_id,
        )
        raise HTTPException(
            status_code=502,
            detail={
                "error": "index_queue_unavailable",
                "message": "The index preparation service is temporarily unavailable. Please retry.",
            },
        ) from error


@app.get(
    "/index/{video_id}",
    response_model=IndexStatusResponse,
)
@limiter.limit(RATE_LIMIT)
def index_status(
    video_id: str,
    request: Request,
    response: Response,
):
    """Return the currently persisted index state for a video."""

    try:
        result = get_index_status(video_id)
        return {
            "video_id": result.video_id,
            "state": result.state,
            "action": result.action,
            "ready": result.ready,
            "job_id": result.job_id,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_video_reference",
                "message": "The supplied value is not a valid YouTube video ID.",
            },
        ) from error


# ============================================================
# CHAT
# ============================================================

@app.post(
    "/chat",
    response_model=ChatResponse,
)
@limiter.limit(RATE_LIMIT)
def chat(
    chat_request: ChatRequest,
    request: Request,
    response: Response,
):

    video_id = chat_request.video_id.strip()
    question = chat_request.question.strip()

    conversation_history = [
        {
            "role": turn.role,
            "content": turn.content.strip(),
        }
        for turn in chat_request.conversation_history
        if turn.content.strip()
    ][-6:]

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
            conversation_history=conversation_history,
        )

        return result

    except IndexNotReadyError as error:

        logger.info(
            "Index not ready request_id=%s video_id=%s state=%s",
            request.state.request_id,
            canonical_video_id,
            error.state,
        )

        if error.state in {"queued", "building"}:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "index_not_ready",
                    "message": "The video index is still being prepared. Retry chat after it becomes ready.",
                    "state": error.state,
                    "job_id": error.job_id,
                },
            ) from error

        if error.state == "failed":
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "index_preparation_failed",
                    "message": "The video index could not be prepared. Open the index status and retry preparation.",
                    "state": error.state,
                    "job_id": error.job_id,
                },
            ) from error

        raise HTTPException(
            status_code=404,
            detail={
                "error": "video_not_indexed",
                "message": "No indexed transcript is available for this video.",
            },
        ) from error

    except FileNotFoundError as error:

        logger.warning(
            "Vector store not found. "
            "request_id=%s video_id=%s",
            request.state.request_id,
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
            request.state.request_id,
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