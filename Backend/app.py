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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from Backend.rag_system.chain import answer_question


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
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

    allow_origins=["*"],

    allow_credentials=False,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# REQUEST MODELS
# ============================================================

class ChatRequest(BaseModel):

    video_id: str = Field(
        ...,
        min_length=1,
        description=(
            "YouTube video ID or YouTube URL."
        ),
    )

    question: str = Field(
        ...,
        min_length=1,
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
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {

        "status":
            "ok",

        "service":
            "YouTube Video AI Chat",

        "message":
            "FastAPI backend is running.",
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
):

    video_id = request.video_id.strip()

    question = request.question.strip()


    if not video_id:

        raise HTTPException(

            status_code=400,

            detail=(
                "video_id cannot be empty."
            ),
        )


    if not question:

        raise HTTPException(

            status_code=400,

            detail=(
                "question cannot be empty."
            ),
        )


    try:

        result = answer_question(

            video_reference=video_id,

            question=question,
        )


        return result


    except FileNotFoundError as error:

        raise HTTPException(

            status_code=404,

            detail=str(error),

        ) from error


    except ValueError as error:

        raise HTTPException(

            status_code=400,

            detail=str(error),

        ) from error


    except RuntimeError as error:

        raise HTTPException(

            status_code=502,

            detail=str(error),

        ) from error


    except Exception as error:

        logger.exception(
            "Unexpected /chat error."
        )


        raise HTTPException(

            status_code=500,

            detail=(
                "An unexpected error occurred "
                "while processing the request."
            ),

        ) from error