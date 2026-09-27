"""
app.py

FastAPI backend for the YouTube RAG chatbot.

Flow:

Chrome Extension
        ↓
POST /chat
        ↓
FastAPI
        ↓
generation.generate_answer()
        ↓
MMR Retrieval
        ↓
Augmentation
        ↓
Hugging Face LLM
        ↓
JSON Response
"""


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# MAKE RAG MODULES IMPORTABLE
# ============================================================

# Current project structure uses:
#
# Backend/
# ├── app.py
# └── rag-system/
#     ├── generation.py
#     ├── retriever.py
#     └── augmentation.py
#
# Because "rag-system" contains a hyphen, it cannot be imported
# as a normal Python package name.
#
# Therefore we add the directory to sys.path.

RAG_DIR = (
    Path(__file__).resolve().parent
    / "rag-system"
)


if not RAG_DIR.exists():

    raise RuntimeError(
        f"RAG directory was not found:\n{RAG_DIR}"
    )


if str(RAG_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(RAG_DIR)
    )


from importlib import import_module

generate_answer = import_module("generation").generate_answer


# ============================================================
# CREATE FASTAPI APPLICATION
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

# Useful during local Chrome-extension development.
#
# The Chrome extension already has host permissions for the
# local FastAPI server, but keeping CORS enabled makes the
# backend easier to test from browser-based clients as well.

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=False,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):

    video_id: str

    question: str


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "status": "ok",
        "service": "YouTube Video AI Chat",
        "message": "FastAPI backend is running.",
    }


# ============================================================
# CHAT ENDPOINT
# ============================================================

@app.post("/chat")
def chat(
    request: ChatRequest
):

    # --------------------------------------------------------
    # Validate video ID
    # --------------------------------------------------------

    video_id = request.video_id.strip()

    if not video_id:

        raise HTTPException(
            status_code=400,
            detail="video_id cannot be empty.",
        )


    # --------------------------------------------------------
    # Validate question
    # --------------------------------------------------------

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="question cannot be empty.",
        )


    # --------------------------------------------------------
    # Run complete RAG pipeline
    # --------------------------------------------------------

    try:

        result = generate_answer(

            video_reference=video_id,

            question=question,
        )

        return result


    # --------------------------------------------------------
    # Vector store / indexing problem
    # --------------------------------------------------------

    except FileNotFoundError as error:

        raise HTTPException(

            status_code=404,

            detail=str(error),
        ) from error


    # --------------------------------------------------------
    # Invalid input
    # --------------------------------------------------------

    except ValueError as error:

        raise HTTPException(

            status_code=400,

            detail=str(error),
        ) from error


    # --------------------------------------------------------
    # Hugging Face / external model error
    # --------------------------------------------------------

    except RuntimeError as error:

        raise HTTPException(

            status_code=502,

            detail=str(error),
        ) from error


    # --------------------------------------------------------
    # Unexpected backend error
    # --------------------------------------------------------

    except Exception as error:

        print(
            "\nUnexpected /chat error:"
        )

        print(
            repr(error)
        )

        raise HTTPException(

            status_code=500,

            detail=(
                "An unexpected error occurred "
                "while processing the request."
            ),
        ) from error