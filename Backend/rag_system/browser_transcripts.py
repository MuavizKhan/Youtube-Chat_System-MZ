"""Validated browser-acquired YouTube caption tracks for local indexing.

A Chrome extension can read the same caption track the YouTube player uses.
This module stages that bounded transcript on disk so the existing durable
index worker can consume it without changing the RAG or FAISS pipeline.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


VIDEO_ID_PATTERN = r"^[A-Za-z0-9_-]{11}$"
LANGUAGE_CODE_PATTERN = r"^[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*$"
MAX_BROWSER_TRANSCRIPT_SEGMENTS = 100_000
MAX_BROWSER_TRANSCRIPT_CHARACTERS = 1_500_000

BROWSER_TRANSCRIPT_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "data" / "browser_transcripts"
)


class BrowserTranscriptSnippet(BaseModel):
    """One timestamped caption segment captured by the active YouTube tab."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=5_000)
    start: float = Field(ge=0, le=86_400, allow_inf_nan=False)
    duration: float = Field(ge=0, le=86_400, allow_inf_nan=False)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("caption text must not be blank.")
        return cleaned


class BrowserTranscriptSubmission(BaseModel):
    """A bounded transcript captured from the current YouTube video player."""

    model_config = ConfigDict(extra="forbid")

    video_id: str = Field(min_length=11, max_length=11, pattern=VIDEO_ID_PATTERN)
    language: str = Field(min_length=1, max_length=100)
    language_code: str = Field(
        min_length=2,
        max_length=35,
        pattern=LANGUAGE_CODE_PATTERN,
    )
    is_generated: bool
    snippets: list[BrowserTranscriptSnippet] = Field(
        min_length=1,
        max_length=MAX_BROWSER_TRANSCRIPT_SEGMENTS,
    )

    @field_validator("language")
    @classmethod
    def normalize_language(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("caption language must not be blank.")
        return cleaned

    @field_validator("language_code")
    @classmethod
    def normalize_language_code(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def validate_total_transcript_size(self) -> "BrowserTranscriptSubmission":
        total_characters = sum(len(item.text) for item in self.snippets)
        if total_characters > MAX_BROWSER_TRANSCRIPT_CHARACTERS:
            raise ValueError(
                "the browser transcript exceeds the maximum supported text size."
            )
        return self


def _transcript_path(video_id: str) -> Path:
    if not re.fullmatch(VIDEO_ID_PATTERN, video_id or ""):
        raise ValueError("invalid YouTube video ID for browser transcript staging.")
    return BROWSER_TRANSCRIPT_DIRECTORY / f"{video_id}.json"


def stage_browser_transcript(submission: BrowserTranscriptSubmission) -> Path:
    """Atomically stage a validated transcript for the durable index worker."""
    destination = _transcript_path(submission.video_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{submission.video_id}.{uuid.uuid4().hex}.tmp"
    )

    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(
                submission.model_dump(),
                stream,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    return destination


def discard_browser_transcript(video_id: str) -> None:
    """Remove a staged browser transcript after success or explicit cleanup."""
    _transcript_path(video_id).unlink(missing_ok=True)


def load_browser_transcript(video_id: str) -> Any | None:
    """Read a staged transcript without deleting it before index success."""
    source = _transcript_path(video_id)
    try:
        with source.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except FileNotFoundError:
        return None

    if payload.get("video_id") != video_id:
        raise ValueError("staged browser transcript video ID did not match the index job.")

    snippets = [
        SimpleNamespace(
            text=item["text"],
            start=float(item["start"]),
            duration=float(item["duration"]),
        )
        for item in payload["snippets"]
    ]

    return SimpleNamespace(
        video_id=video_id,
        language=payload["language"],
        language_code=payload["language_code"],
        is_generated=bool(payload["is_generated"]),
        snippets=snippets,
        source="youtube_browser_captions",
    )
