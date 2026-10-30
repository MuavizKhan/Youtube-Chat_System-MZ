import pytest

from scripts.phase11_0_unseen_video_smoke import (
    _validate_case,
    classify_transcript_timeline,
)


@pytest.mark.parametrize(
    ("span_seconds", "expected"),
    [
        (0, "short"),
        (600, "short"),
        (601, "medium"),
        (1799, "medium"),
        (1800, "long"),
        (7200, "long"),
    ],
)
def test_classify_transcript_timeline_boundaries(span_seconds, expected):
    assert classify_transcript_timeline(span_seconds) == expected


@pytest.mark.parametrize(
    "span_seconds",
    [-1, float("nan"), float("inf")],
)
def test_classify_transcript_timeline_rejects_invalid_values(span_seconds):
    with pytest.raises((ValueError, OverflowError)):
        classify_transcript_timeline(span_seconds)


def test_validate_case_accepts_unseen_video_with_explicit_language_and_variant():
    _validate_case({
        "case_id": "UVE01",
        "video_id": "XGLx-Anjxvg",
        "languages": ["en"],
        "expected_outcome": "ready",
        "expected_language_code": "en",
        "expected_generated": True,
    })


@pytest.mark.parametrize(
    "case",
    [
        {
            "case_id": "BAD01",
            "video_id": "short",
            "languages": ["en"],
            "expected_outcome": "ready",
            "expected_language_code": "en",
        },
        {
            "case_id": "BAD02",
            "video_id": "XGLx-Anjxvg",
            "languages": ["en", "en"],
            "expected_outcome": "ready",
            "expected_language_code": "en",
        },
        {
            "case_id": "BAD03",
            "video_id": "XGLx-Anjxvg",
            "languages": ["fr"],
            "expected_outcome": "ready",
            "expected_language_code": "fr",
        },
        {
            "case_id": "BAD04",
            "video_id": "XGLx-Anjxvg",
            "languages": ["en"],
            "expected_outcome": "failed",
        },
    ],
)
def test_validate_case_rejects_malformed_smoke_manifest_entries(case):
    with pytest.raises(ValueError):
        _validate_case(case)
