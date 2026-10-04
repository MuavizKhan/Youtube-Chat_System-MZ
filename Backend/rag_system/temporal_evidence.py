"""
Timestamp-grounded gold annotations for the temporal retrieval benchmark.

The windows are based on published chapter markers for the benchmark video.
They are coarse section-level reference intervals, not word-level transcript
boundaries. Scoring therefore checks whether retrieved chunks overlap a
known relevant section rather than requiring an exact chunk ID.
"""

from dataclasses import dataclass


GOLD_TEMPORAL_VIDEO_ID = "MdeQMVBuGgY"
TEMPORAL_ANNOTATION_SOURCE = (
    "https://socialcounts.org/youtube-video-live-view-count/MdeQMVBuGgY"
)


@dataclass(frozen=True)
class TemporalEvidenceWindow:
    """A chapter-level time interval containing relevant evidence."""

    section_id: str
    start_seconds: float
    end_seconds: float
    label: str


@dataclass(frozen=True)
class TemporalEvidenceCase:
    """Timestamp windows expected to contain evidence for one question."""

    question_id: str
    windows: tuple[TemporalEvidenceWindow, ...]
    min_window_coverage: float


TEMPORAL_EVIDENCE_CASES = (
    TemporalEvidenceCase(
        question_id="Q11",
        windows=(
            TemporalEvidenceWindow(
                section_id="kingfisher_rise_and_fall",
                start_seconds=4040.0,  # 01:07:20
                end_seconds=5700.0,    # 01:35:00
                label="Rise & Fall of Kingfisher Airlines",
            ),
            TemporalEvidenceWindow(
                section_id="kingfisher_financial_turmoil",
                start_seconds=7171.0,  # 01:59:31
                end_seconds=8172.0,    # 02:16:12
                label="Turmoil at Kingfisher Airlines",
            ),
        ),
        # Q11 asks broadly where the topic is discussed. Either major
        # chapter is a valid answer; the benchmark does not require both.
        min_window_coverage=0.5,
    ),
)

TEMPORAL_BENCHMARK_THRESHOLDS = {
    "hit_rate_at_k": 1.0,
    "mean_window_coverage": 0.5,
    "timestamp_validity_rate": 1.0,
    "source_identity_validity_rate": 1.0,
    "mrr": 0.5,
}

_TEMPORAL_BY_ID = {
    case.question_id: case
    for case in TEMPORAL_EVIDENCE_CASES
}


def get_temporal_evidence_case(question_id: str) -> TemporalEvidenceCase:
    """Return the curated temporal annotation for a question."""

    try:
        return _TEMPORAL_BY_ID[question_id]
    except KeyError as error:
        raise ValueError(
            f"No temporal evidence annotation exists for {question_id}."
        ) from error
