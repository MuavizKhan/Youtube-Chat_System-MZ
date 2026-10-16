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
    runtime_seconds: int | None = None
    transcript_word_count: int | None = None
    caption_track_kind: str = "not_audited"
    caption_validation_note: str = ""

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



# Phase 10.3 pilot corpus. Caption pages below expose the YouTube caption
# track (rather than a third-party re-transcription). The four answerable
# evidence phrases per entry were checked against the visible caption text.
# The unrelated fifth question is a negative retrieval control and has no gold
# transcript phrase annotation. New entries use video-local question IDs.

ENGLISH_WORKSHOP_VIDEO_ID = "pOnl-Zndcls"
ENGLISH_WORKSHOP_SOURCE_URL = (
    "https://prepublish.ai/youtube-transcript/pOnl-Zndcls"
)
ENGLISH_WORKSHOP_QUESTIONS = (
    {
        "id": "EW01",
        "type": "direct_fact",
        "question": "How many ways does the speaker promise to express himself in meetings?",
    },
    {
        "id": "EW02",
        "type": "specific_detail",
        "question": "What short structure does the speaker offer for explaining work?",
    },
    {
        "id": "EW03",
        "type": "explanation",
        "question": "What does the speaker say is difficult about making conversation with strangers?",
    },
    {
        "id": "EW04",
        "type": "multi_part",
        "question": "What email structure does the speaker promise, and what outcome is it meant to get?",
    },
    {
        "id": "EW05",
        "type": "unrelated",
        "question": "What is the capital of Australia?",
    },
)
ENGLISH_WORKSHOP_EXPECTED = {
    "EW01": "must_retrieve",
    "EW02": "must_retrieve",
    "EW03": "must_retrieve",
    "EW04": "must_retrieve",
    "EW05": "must_not_retrieve",
}
ENGLISH_WORKSHOP_GOLD = (
    GoldEvidenceCase(
        question_id="EW01",
        groups=(("three ways to express yourself",),),
        min_group_coverage=1.0,
    ),
    GoldEvidenceCase(
        question_id="EW02",
        groups=(("30 second structure for explaining your work",),),
        min_group_coverage=1.0,
    ),
    GoldEvidenceCase(
        question_id="EW03",
        groups=(("small talk is difficult",),),
        min_group_coverage=1.0,
    ),
    GoldEvidenceCase(
        question_id="EW04",
        groups=(("simple email format that gets replies",),),
        min_group_coverage=1.0,
    ),
)

ENGLISH_AI_VIDEO_ID = "A-JBaNvv3Tk"
ENGLISH_AI_SOURCE_URL = (
    "https://prepublish.ai/youtube-transcript/A-JBaNvv3Tk"
)
ENGLISH_AI_QUESTIONS = (
    {
        "id": "AI01",
        "type": "direct_fact",
        "question": "What does the guest predict will become easier for anyone?",
    },
    {
        "id": "AI02",
        "type": "explanation",
        "question": "What concern does the guest raise about some future open-source models?",
    },
    {
        "id": "AI03",
        "type": "specific_detail",
        "question": "What has the guest unified into one system?",
    },
    {
        "id": "AI04",
        "type": "multi_part",
        "question": "What does the guest say about his shipping pace and how quickly he is shipping?",
    },
    {
        "id": "AI05",
        "type": "unrelated",
        "question": "What is the capital of Australia?",
    },
)
ENGLISH_AI_EXPECTED = {
    "AI01": "must_retrieve",
    "AI02": "must_retrieve",
    "AI03": "must_retrieve",
    "AI04": "must_retrieve",
    "AI05": "must_not_retrieve",
}
ENGLISH_AI_GOLD = (
    GoldEvidenceCase(
        question_id="AI01",
        groups=(("easy to hack anyone",),),
        min_group_coverage=1.0,
    ),
    GoldEvidenceCase(
        question_id="AI02",
        groups=(("alignment whatsoever",),),
        min_group_coverage=1.0,
    ),
    GoldEvidenceCase(
        question_id="AI03",
        groups=(("central system",),),
        min_group_coverage=1.0,
    ),
    GoldEvidenceCase(
        question_id="AI04",
        groups=(("how fast I'm shipping",),),
        min_group_coverage=1.0,
    ),
)

HINDI_HISTORY_VIDEO_ID = "INpnuL7qnzY"
HINDI_HISTORY_SOURCE_URL = (
    "https://prepublish.ai/youtube-transcript/INpnuL7qnzY"
)
HINDI_HISTORY_QUESTIONS = (
    {
        "id": "HI01",
        "type": "direct_fact",
        "question": "Which peasant movement is named as the first major movement?",
    },
    {
        "id": "HI02",
        "type": "explanation",
        "question": "How does the caption locate the first major peasant movement in relation to Bijolia?",
    },
    {
        "id": "HI03",
        "type": "specific_detail",
        "question": "How many types of taxes does the speaker mention?",
    },
    {
        "id": "HI04",
        "type": "multi_part",
        "question": "Which tax is identified as new, and what name does the speaker give for it?",
    },
    {
        "id": "HI05",
        "type": "unrelated",
        "question": "What is the capital of Australia?",
    },
)
HINDI_HISTORY_EXPECTED = {
    "HI01": "must_retrieve",
    "HI02": "must_retrieve",
    "HI03": "must_retrieve",
    "HI04": "must_retrieve",
    "HI05": "must_not_retrieve",
}
HINDI_HISTORY_GOLD = (
    GoldEvidenceCase(
        question_id="HI01",
        groups=(("पहला बड़ा किसान आंदोलन",),),
        min_group_coverage=1.0,
    ),
    GoldEvidenceCase(
        question_id="HI02",
        groups=(("बिजोलिया में",),),
        min_group_coverage=1.0,
    ),
    GoldEvidenceCase(
        question_id="HI03",
        groups=(("84 प्रकार के कर",),),
        min_group_coverage=1.0,
    ),
    GoldEvidenceCase(
        question_id="HI04",
        groups=(("नया टैक्स लगाया गया था चवरी कर",),),
        min_group_coverage=1.0,
    ),
)

ARABIC_SILENCE_VIDEO_ID = "fAd2nKfrDB8"
ARABIC_SILENCE_SOURCE_URL = (
    "https://prepublish.ai/youtube-transcript/fAd2nKfrDB8"
)
ARABIC_SILENCE_QUESTIONS = (
    {
        "id": "AR01",
        "type": "explanation",
        "question": "Does complete quiet provide the rest people expect?",
    },
    {
        "id": "AR02",
        "type": "specific_detail",
        "question": "How much might one hour in the world's quietest room cost?",
    },
    {
        "id": "AR03",
        "type": "specific_detail",
        "question": "What bodily sounds might people begin to hear in the room?",
    },
    {
        "id": "AR04",
        "type": "multi_part",
        "question": "What does the episode say sound contributes to perception, and how does it describe sound's role?",
    },
    {
        "id": "AR05",
        "type": "unrelated",
        "question": "What is the capital of Australia?",
    },
)
ARABIC_SILENCE_EXPECTED = {
    "AR01": "must_retrieve",
    "AR02": "must_retrieve",
    "AR03": "must_retrieve",
    "AR04": "must_retrieve",
    "AR05": "must_not_retrieve",
}
ARABIC_SILENCE_GOLD = (
    GoldEvidenceCase(
        question_id="AR01",
        groups=(("الهدوء التام مبيديش الراحة",),),
        min_group_coverage=1.0,
    ),
    GoldEvidenceCase(
        question_id="AR02",
        groups=(("600 دولار",),),
        min_group_coverage=1.0,
    ),
    GoldEvidenceCase(
        question_id="AR03",
        groups=(("نبض قلبك حركة الدم",),),
        min_group_coverage=1.0,
    ),
    GoldEvidenceCase(
        question_id="AR04",
        groups=(("الصوت هو بوصلة مهمة",),),
        min_group_coverage=1.0,
    ),
)

BENCHMARK_CURATION_BLOCKERS = {
    "language": "ur",
    "status": "blocked_pending_source_verification",
    "reason": (
        "No candidate with an accessible, verified Urdu-script YouTube caption "
        "track was confirmed during this curation slice. YouTube's automatic "
        "caption language list does not include Urdu, so a valid candidate "
        "needs a publisher-uploaded Urdu caption track that can be verified "
        "against the video. Do not substitute an unverified translation."
    ),
    "verification_reference": (
        "https://support.google.com/youtube/answer/6373554?hl=en-GBAnd"
    ),
}

ENGLISH_WORKSHOP_BENCHMARK = VideoBenchmark(
    video_id=ENGLISH_WORKSHOP_VIDEO_ID,
    title="How to Improve Your English for Your Career (Live Workshop)",
    transcript_language="en",
    video_url=f"https://www.youtube.com/watch?v={ENGLISH_WORKSHOP_VIDEO_ID}",
    transcript_source_kind="youtube_auto_generated_captions",
    transcript_source_url=ENGLISH_WORKSHOP_SOURCE_URL,
    question_cases=ENGLISH_WORKSHOP_QUESTIONS,
    expected_retrieval=ENGLISH_WORKSHOP_EXPECTED,
    gold_evidence_cases=ENGLISH_WORKSHOP_GOLD,
    temporal_evidence_cases=(),
    annotation_source_urls=(ENGLISH_WORKSHOP_SOURCE_URL,),
    runtime_seconds=3280,
    transcript_word_count=7304,
    caption_track_kind="youtube_auto_generated",
    caption_validation_note=(
        "Caption page identifies English captions generated automatically by "
        "YouTube; visible transcript text reviewed for the four gold phrases."
    ),
)

ENGLISH_AI_BENCHMARK = VideoBenchmark(
    video_id=ENGLISH_AI_VIDEO_ID,
    title="Ex-Apple dev reveals his Agentic Engineering Workflow",
    transcript_language="en",
    video_url=f"https://www.youtube.com/watch?v={ENGLISH_AI_VIDEO_ID}",
    transcript_source_kind="youtube_auto_generated_captions",
    transcript_source_url=ENGLISH_AI_SOURCE_URL,
    question_cases=ENGLISH_AI_QUESTIONS,
    expected_retrieval=ENGLISH_AI_EXPECTED,
    gold_evidence_cases=ENGLISH_AI_GOLD,
    temporal_evidence_cases=(),
    annotation_source_urls=(ENGLISH_AI_SOURCE_URL,),
    runtime_seconds=4265,
    transcript_word_count=13185,
    caption_track_kind="youtube_auto_generated",
    caption_validation_note=(
        "Caption page identifies English captions generated automatically by "
        "YouTube; transcript text reviewed for the four gold phrases."
    ),
)

HINDI_HISTORY_BENCHMARK = VideoBenchmark(
    video_id=HINDI_HISTORY_VIDEO_ID,
    title=(
        "किसान आंदोलन | Rajasthan ke kisan andolan by Rajveer Sir "
        "revision class Rajasthan history all exams"
    ),
    transcript_language="hi",
    video_url=f"https://www.youtube.com/watch?v={HINDI_HISTORY_VIDEO_ID}",
    transcript_source_kind="youtube_auto_generated_captions",
    transcript_source_url=HINDI_HISTORY_SOURCE_URL,
    question_cases=HINDI_HISTORY_QUESTIONS,
    expected_retrieval=HINDI_HISTORY_EXPECTED,
    gold_evidence_cases=HINDI_HISTORY_GOLD,
    temporal_evidence_cases=(),
    annotation_source_urls=(HINDI_HISTORY_SOURCE_URL,),
    runtime_seconds=2420,
    transcript_word_count=7228,
    caption_track_kind="youtube_auto_generated",
    caption_validation_note=(
        "Searchable caption page identifies Hindi captions generated "
        "automatically by YouTube. The indexed caption excerpt exposes the "
        "four phrases used here; full-page fetch was unavailable in research."
    ),
)

ARABIC_SILENCE_BENCHMARK = VideoBenchmark(
    video_id=ARABIC_SILENCE_VIDEO_ID,
    title="الدحيح | هل الصمت التام راحة؟",
    transcript_language="ar",
    video_url=f"https://www.youtube.com/watch?v={ARABIC_SILENCE_VIDEO_ID}",
    transcript_source_kind="youtube_channel_published_captions",
    transcript_source_url=ARABIC_SILENCE_SOURCE_URL,
    question_cases=ARABIC_SILENCE_QUESTIONS,
    expected_retrieval=ARABIC_SILENCE_EXPECTED,
    gold_evidence_cases=ARABIC_SILENCE_GOLD,
    temporal_evidence_cases=(),
    annotation_source_urls=(ARABIC_SILENCE_SOURCE_URL,),
    runtime_seconds=1443,
    transcript_word_count=4008,
    caption_track_kind="youtube_channel_published",
    caption_validation_note=(
        "Caption page identifies Arabic captions published by the channel; "
        "visible transcript text reviewed for the four gold phrases."
    ),
)


# The key is the exact YouTube video ID. Unknown IDs are intentionally absent:
# no annotation or passing result is inferred for an unregistered video.
VIDEO_BENCHMARKS: dict[str, VideoBenchmark] = {
    KINGFISHER_BENCHMARK.video_id: KINGFISHER_BENCHMARK,
    ENGLISH_WORKSHOP_BENCHMARK.video_id: ENGLISH_WORKSHOP_BENCHMARK,
    ENGLISH_AI_BENCHMARK.video_id: ENGLISH_AI_BENCHMARK,
    HINDI_HISTORY_BENCHMARK.video_id: HINDI_HISTORY_BENCHMARK,
    ARABIC_SILENCE_BENCHMARK.video_id: ARABIC_SILENCE_BENCHMARK,
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
            "caption_track_kind": benchmark.caption_track_kind,
            "validation_note": benchmark.caption_validation_note,
        },
        "runtime_seconds": benchmark.runtime_seconds,
        "transcript_word_count": benchmark.transcript_word_count,
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
