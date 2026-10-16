"""Declarative, video-scoped registry for deterministic RAG benchmarks.

Question IDs are only unique within an individual VideoBenchmark. Keep source
metadata next to annotations so future corpus additions can be audited.
"""

from dataclasses import dataclass
from typing import Any

from .gold_evidence import (
    GOLD_EVIDENCE_CASES,
    GOLD_EVIDENCE_SOURCE_URL,
    GOLD_EVIDENCE_VIDEO_ID,
    GoldEvidenceCase,
)
from .temporal_evidence import (
    TEMPORAL_ANNOTATION_SOURCE,
    TEMPORAL_EVIDENCE_CASES,
    TemporalEvidenceCase,
)


TEST_QUESTIONS = [
    {
        "id": "Q01",
        "type": "direct_fact",
        "question": "What is Kingfisher Airlines?",
    },
    {
        "id": "Q02",
        "type": "direct_fact",
        "question": (
            "What connection does Vijay Mallya discuss "
            "with Formula One?"
        ),
    },
    {
        "id": "Q03",
        "type": "direct_fact",
        "question": (
            "What businesses or companies does Vijay Mallya "
            "discuss in the interview?"
        ),
    },
    {
        "id": "Q04",
        "type": "explanation",
        "question": (
            "Why does Vijay Mallya say Kingfisher Airlines "
            "struggled or failed?"
        ),
    },
    {
        "id": "Q05",
        "type": "explanation",
        "question": (
            "How does Vijay Mallya describe the challenges "
            "he faced while running Kingfisher Airlines?"
        ),
    },
    {
        "id": "Q06",
        "type": "explanation",
        "question": (
            "What does Vijay Mallya say about building "
            "the Kingfisher brand?"
        ),
    },
    {
        "id": "Q07",
        "type": "specific_detail",
        "question": (
            "What does Vijay Mallya say about the amount "
            "of debt or money involved in the Kingfisher "
            "situation?"
        ),
    },
    {
        "id": "Q08",
        "type": "specific_detail",
        "question": (
            "What does Vijay Mallya say about banks and "
            "the recovery of money?"
        ),
    },
    {
        "id": "Q09",
        "type": "specific_detail",
        "question": (
            "What does Vijay Mallya say about the role of "
            "Indian government policy in the problems "
            "faced by Kingfisher Airlines?"
        ),
    },
    {
        "id": "Q10",
        "type": "multi_part",
        "question": (
            "What does Vijay Mallya say about the failure "
            "of Kingfisher Airlines, and what reasons does "
            "he give for it?"
        ),
    },
    {
        "id": "Q11",
        "type": "temporal",
        "question": (
            "Where in the video does Vijay Mallya discuss "
            "Kingfisher Airlines, and what is discussed "
            "there?"
        ),
    },
    {
        "id": "Q12",
        "type": "unanswerable",
        "question": (
            "What programming language does Vijay Mallya "
            "say is his favorite?"
        ),
    },
    {
        "id": "Q13",
        "type": "unrelated",
        "question": "What is the capital of Australia?",
    },
]

EXPECTED_RETRIEVAL = {
    **{f"Q{i:02d}": "must_retrieve" for i in range(1, 12)},
    "Q12": "may_retrieve",
    "Q13": "must_not_retrieve",
}


@dataclass(frozen=True)
class VideoBenchmark:
    """Question suite and curated annotations for exactly one video."""

    video_id: str
    title: str
    transcript_language: str
    video_url: str
    transcript_source_kind: str
    transcript_source_url: str
    question_cases: tuple[dict[str, str], ...]
    expected_retrieval: dict[str, str]
    gold_evidence_cases: tuple[GoldEvidenceCase, ...]
    temporal_evidence_cases: tuple[TemporalEvidenceCase, ...]
    annotation_source_urls: tuple[str, ...]

    @property
    def gold_evidence_by_question(self) -> dict[str, GoldEvidenceCase]:
        return {case.question_id: case for case in self.gold_evidence_cases}

    @property
    def temporal_evidence_by_question(self) -> dict[str, TemporalEvidenceCase]:
        return {case.question_id: case for case in self.temporal_evidence_cases}

    def question_cases_for_evaluation(self) -> list[dict[str, str]]:
        """Return independent cases with their per-video expectation attached."""
        return [
            {
                **case,
                "expected": self.expected_retrieval[case["id"]],
            }
            for case in self.question_cases
        ]


KINGFISHER_BENCHMARK = VideoBenchmark(
    video_id=GOLD_EVIDENCE_VIDEO_ID,
    title="Kingfisher Airlines interview (benchmark label)",
    transcript_language="en",
    video_url=f"https://www.youtube.com/watch?v={GOLD_EVIDENCE_VIDEO_ID}",
    transcript_source_kind="published_transcript_copy",
    transcript_source_url=GOLD_EVIDENCE_SOURCE_URL,
    question_cases=tuple(TEST_QUESTIONS),
    expected_retrieval=dict(EXPECTED_RETRIEVAL),
    gold_evidence_cases=tuple(GOLD_EVIDENCE_CASES),
    temporal_evidence_cases=tuple(TEMPORAL_EVIDENCE_CASES),
    annotation_source_urls=(
        GOLD_EVIDENCE_SOURCE_URL,
        TEMPORAL_ANNOTATION_SOURCE,
    ),
)

# The key is the exact YouTube video ID. Unknown IDs are intentionally absent:
# no annotation or passing result is inferred for an unregistered video.
VIDEO_BENCHMARKS: dict[str, VideoBenchmark] = {
    KINGFISHER_BENCHMARK.video_id: KINGFISHER_BENCHMARK,
}


def get_video_benchmark(video_id: str) -> VideoBenchmark | None:
    """Return the configured benchmark for a video ID, if one exists."""
    return VIDEO_BENCHMARKS.get(video_id)


def describe_video_benchmark(video_id: str) -> dict[str, Any]:
    """Return JSON-serializable registry metadata or explicit missing status."""
    benchmark = get_video_benchmark(video_id)
    if benchmark is None:
        return {
            "status": "not_configured",
            "video_id": video_id,
            "message": (
                "No benchmark is configured for this video; "
                "no quality pass/fail is claimed."
            ),
        }

    return {
        "status": "configured",
        "video_id": benchmark.video_id,
        "title": benchmark.title,
        "transcript_language": benchmark.transcript_language,
        "video_url": benchmark.video_url,
        "transcript_source": {
            "kind": benchmark.transcript_source_kind,
            "url": benchmark.transcript_source_url,
        },
        "annotation_source_urls": list(benchmark.annotation_source_urls),
        "question_ids": [case["id"] for case in benchmark.question_cases],
        "gold_evidence_question_ids": [
            case.question_id for case in benchmark.gold_evidence_cases
        ],
        "temporal_question_ids": [
            case.question_id for case in benchmark.temporal_evidence_cases
        ],
        "question_count": len(benchmark.question_cases),
        "gold_evidence_case_count": len(benchmark.gold_evidence_cases),
        "temporal_evidence_case_count": len(benchmark.temporal_evidence_cases),
    }
