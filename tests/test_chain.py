from types import SimpleNamespace
import re
import pytest
from langchain_core.documents import Document
from Backend.rag_system import chain


def doc(cid,start,end,text):
    return Document(page_content=text,metadata={
        "chunk_id":cid,"start":start,"end":end,"video_id":"Gfr50f6ZBvo"
    })


@pytest.mark.unit
@pytest.mark.parametrize("seconds,expected",[(0,"00:00"),(59.9,"00:59"),(60,"01:00"),(3661,"01:01:01")])
def test_format_timestamp(seconds,expected):
    assert chain.format_timestamp(seconds)==expected


@pytest.mark.unit
def test_append_requested_temporal_timestamp_from_retrieval_window():
    trace = {
        "route": "temporal_window",
        "temporal_window": {
            "start_seconds": 1535.0,
            "end_seconds": 1584.52,
        },
    }

    answer = chain.append_requested_temporal_timestamp(
        answer="No winner is declared, and no score is announced.",
        question=(
            "What is said during the final 50 seconds? "
            "Cite the relevant timestamp."
        ),
        retrieval_trace=trace,
        has_evidence=True,
    )

    assert answer.endswith("Relevant timestamp: 25:35–26:24")


@pytest.mark.unit
@pytest.mark.parametrize(
    "question,trace,has_evidence,answer",
    [
        (
            "What happened during 25:35–26:25?",
            {"route": "temporal_window", "temporal_window": {
                "start_seconds": 1535.0, "end_seconds": 1584.52,
            }},
            True,
            "A factual answer.",
        ),
        (
            "What happened? Cite the relevant timestamp.",
            {"route": "question_aware", "temporal_window": {
                "start_seconds": 1535.0, "end_seconds": 1584.52,
            }},
            True,
            "A factual answer.",
        ),
        (
            "What happened during 25:35–26:25? Cite the timestamp.",
            {"route": "temporal_window", "temporal_window": {
                "start_seconds": 1535.0, "end_seconds": 1584.52,
            }},
            False,
            "A factual answer.",
        ),
    ],
)
def test_append_requested_temporal_timestamp_is_guarded(
    question, trace, has_evidence, answer
):
    assert chain.append_requested_temporal_timestamp(
        answer=answer,
        question=question,
        retrieval_trace=trace,
        has_evidence=has_evidence,
    ) == answer


@pytest.mark.unit
def test_append_requested_temporal_timestamp_replaces_model_summary_with_window():
    trace = {
        "route": "temporal_window",
        "temporal_window": {"start_seconds": 1535.0, "end_seconds": 1584.52},
    }
    answer = (
        "A factual answer.\n\n"
        "Relevant timestamps: 00:10–00:20"
    )

    result = chain.append_requested_temporal_timestamp(
        answer=answer,
        question="Cite the relevant timestamp for 25:35–26:25.",
        retrieval_trace=trace,
        has_evidence=True,
    )

    assert result == "A factual answer.\n\nRelevant timestamp: 25:35–26:24"
    assert "00:10–00:20" not in result


@pytest.mark.unit
def test_append_verified_response_timestamps_preserves_financial_fallback_bullets():
    context = """[SOURCE 1]
Transcript caption segments in chronological order:
- [02:01:07–02:02:31] I invested 3,000 crore rupees into Kingfisher Airlines.
[SOURCE 2]
Transcript caption segments in chronological order:
- [02:18:34–02:20:28] The DRT judgment debt was 4,999 crore rupees plus interest.
[SOURCE 3]
Transcript caption segments in chronological order:
- [02:49:56–02:50:25] The Ministry of Finance report says the total amount recovered was
- [02:50:25–02:51:41] 14,100 crore rupees.
[SOURCE 4]
Transcript caption segments in chronological order:
- [04:08:59–04:09:30] The debt certificate gives a figure of 6203.
- [04:09:30–04:10:15] The principal amount carries unapplied interest at 11.5% per annum."""

    answer = chain._build_financial_evidence_fallback(
        "What debt and money amounts are mentioned in the Kingfisher case?",
        context,
    )
    assert answer is not None

    result = chain.append_verified_response_timestamps(
        answer=answer,
        question="What debt and money amounts are mentioned in the Kingfisher case?",
        sources=[{"start": 7194.639, "end": 7277.36, "distance": 0.1}],
        retrieval_trace=None,
        has_evidence=True,
    )

    assert "- Investment:" in result
    assert "- DRT debt:" in result
    assert "- Debt certificate:" in result
    assert "- Ministry of Finance recovery report:" in result
    assert "does not clearly explain how the DRT judgment and debt-certificate figures relate" in result
    assert "Relevant timestamps: 01:59:54–02:01:17" in result


@pytest.mark.unit
def test_requested_temporal_window_overrides_inline_model_timestamp_claims():
    trace = {
        "route": "temporal_window",
        "temporal_window": {
            "start_seconds": 1535.0,
            "end_seconds": 1584.52,
        },
    }
    result = chain.append_requested_temporal_timestamp(
        answer=(
            "A factual answer. The model says the section runs from "
            "00:10 to 00:20, and later 00:21 and 00:24."
        ),
        question="Where does this happen? Cite the relevant timestamp.",
        retrieval_trace=trace,
        has_evidence=True,
    )

    assert result == (
        "A factual answer. The model says the section runs, and later.\n\n"
        "Relevant timestamp: 25:35–26:24"
    )


@pytest.mark.unit
def test_append_verified_response_timestamps_preserves_requested_temporal_window():
    result = chain.append_verified_response_timestamps(
        answer=(
            "A factual answer.\n\n"
            "Relevant timestamps: 00:10–00:20"
        ),
        question="Cite the relevant timestamp for 25:35–26:25.",
        sources=[{"start": 120, "end": 160, "distance": 0.1}],
        retrieval_trace={
            "route": "temporal_window",
            "temporal_window": {
                "start_seconds": 1535.0,
                "end_seconds": 1584.52,
            },
        },
        has_evidence=True,
    )

    assert result == "A factual answer.\n\nRelevant timestamp: 25:35–26:24"
    assert "00:10–00:20" not in result


@pytest.mark.unit
def test_append_relevant_source_timestamps_uses_three_strongest_in_chronological_order():
    answer = chain.append_relevant_source_timestamps(
        answer="A grounded answer.",
        sources=[
            {"start": 400, "end": 430, "distance": 0.2},
            {"start": 120, "end": 160, "distance": 0.1},
            {"start": 10, "end": 20, "distance": 0.3},
            {"start": 60, "end": 70, "distance": 0.4},
        ],
        has_evidence=True,
    )

    assert answer.endswith(
        "Relevant timestamps: 00:10–00:20, 02:00–02:40, 06:40–07:10"
    )


@pytest.mark.unit
def test_append_relevant_source_timestamps_aligns_multi_amount_claims_to_transcript_text():
    answer = (
        "Vijay says he put 3,000 crore of UB Group money into Kingfisher to keep it afloat, "
        "and debt was approximately 6,200 crore plus interest. "
        "The Ministry of Finance report showed 14,100 crore recovered. "
        "The interviewer mentions a CBI-related 423 crore figure and an ED "
        "money-laundering allegation involving 3,000 crore."
    )
    result = chain.append_relevant_source_timestamps(
        answer=answer,
        sources=[
            {
                "start": 10,
                "end": 20,
                "distance": 0.01,
                "transcript_text": "Kingfisher was known for guest service and on-time departures.",
            },
            {
                "start": 100,
                "end": 140,
                "distance": 0.8,
                "transcript_text": (
                    "I put 3,000 crores of UB Group money into Kingfisher Airlines "
                    "and infused cash to keep it afloat."
                ),
            },
            {
                "start": 200,
                "end": 240,
                "distance": 0.95,
                "transcript_text": (
                    "The DRT debt certificate records 6,203 crores plus interest. "
                    "The Ministry of Finance report showed 14,100 crores recovered."
                ),
            },
            {
                "start": 300,
                "end": 340,
                "distance": 1.01,
                "transcript_text": (
                    "The interviewer says CBI identified about 423 crores and ED "
                    "reported a 3,000 crore money-laundering issue."
                ),
            },
        ],
        has_evidence=True,
    )

    assert result.endswith(
        "Relevant timestamps: 01:40–02:20, 03:20–04:00, 05:00–05:40"
    )
    assert "00:10–00:20" not in result


@pytest.mark.unit
def test_append_relevant_source_timestamps_prefers_arabic_answer_evidence_over_nearest_source():
    answer = (
        "في الغرفة الهادئة قد يبدأ الناس في سماع أصوات أجسامهم: نبض قلبك، "
        "حركة الدم في عروقك، صوت الرئتين أو المعدة، وحتى صوت عينيك وهي بترمش."
    )
    result = chain.append_relevant_source_timestamps(
        answer=answer,
        sources=[
            {
                "start": 10,
                "end": 20,
                "distance": 0.05,
                "transcript_text": "الهدوء التام لا يعطي الراحة التي يتوقعها الإنسان.",
            },
            {
                "start": 153.05,
                "end": 217.615,
                "distance": 0.9,
                "transcript_text": (
                    "فتبدأ تسمع حاجات زي نبض قلبك، حركة الدم في عروقك، "
                    "صوت الرئتين أو صوت المعدة."
                ),
            },
            {
                "start": 589.599,
                "end": 658.17,
                "distance": 0.95,
                "transcript_text": "حتى عينيك وهي بترمش بقيت بتسمعها.",
            },
        ],
        has_evidence=True,
    )

    assert "Relevant timestamps: 02:33–03:37, 09:49–10:58" in result
    assert "00:10–00:20" not in result


@pytest.mark.unit
def test_append_relevant_source_timestamps_skips_fallback_missing_evidence_and_invalid_ranges():
    sources = [{"start": 10, "end": 20, "distance": 0.1}]

    assert chain.append_relevant_source_timestamps(
        answer=chain.FALLBACK_ANSWER,
        sources=sources,
        has_evidence=True,
    ) == chain.FALLBACK_ANSWER
    assert chain.append_relevant_source_timestamps(
        answer="A grounded answer.",
        sources=sources,
        has_evidence=False,
    ) == "A grounded answer."
    assert chain.append_relevant_source_timestamps(
        answer="A grounded answer.",
        sources=[
            {"start": float("nan"), "end": 20, "distance": 0.1},
            {"start": 20, "end": 20, "distance": 0.1},
            {"start": 30},
        ],
        has_evidence=True,
    ) == "A grounded answer."


@pytest.mark.unit
def test_append_relevant_source_timestamps_replaces_model_summary_with_source_ranges():
    answer = "A grounded answer.\n\nRelevant timestamp: 00:10–00:20"
    result = chain.append_relevant_source_timestamps(
        answer=answer,
        sources=[{"start": 120, "end": 160, "distance": 0.1}],
        has_evidence=True,
    )

    assert result == "A grounded answer.\n\nRelevant timestamps: 02:00–02:40"
    assert "00:10–00:20" not in result


@pytest.mark.unit
def test_inline_model_timestamp_claim_cannot_bypass_source_owned_timestamp_footer():
    answer = (
        "The transcript discusses Kingfisher Airlines and its financial difficulties. "
        "The speaker says this runs from 00:00 to 01:08, with more detail "
        "between 01:10 and 01:22."
    )
    result = chain.append_relevant_source_timestamps(
        answer=answer,
        sources=[
            {
                "start": 4238,
                "end": 4324,
                "distance": 0.1,
                "transcript_text": (
                    "Kingfisher Airlines faced financial difficulties during the interview."
                ),
            }
        ],
        has_evidence=True,
    )

    assert "00:00 to 01:08" not in result
    assert "01:10 and 01:22" not in result
    assert result.endswith("Relevant timestamps: 01:10:38–01:12:04")


@pytest.mark.unit
def test_inline_model_timestamp_claim_is_removed_even_without_source_metadata():
    result = chain.append_relevant_source_timestamps(
        answer=(
            "A grounded explanation. The event occurs around 00:45, "
            "and the discussion continues from 01:00 to 02:00."
        ),
        sources=[],
        has_evidence=False,
    )

    assert result == (
        "A grounded explanation. The event occurs, and the discussion continues."
    )


@pytest.mark.unit
def test_append_relevant_source_timestamps_removes_unverified_summary_without_sources():
    answer = "A grounded answer.\n\nRelevant timestamps: 99:00–99:01"

    assert chain.append_relevant_source_timestamps(
        answer=answer,
        sources=[],
        has_evidence=False,
    ) == "A grounded answer."


@pytest.mark.unit
def test_merge_source_groups():
    groups=chain.merge_overlapping_results([
        (doc(2,40,80,"second"),0.8),
        (doc(3,120,150,"third"),0.9),
        (doc(1,10,50,"first"),0.7),
    ])
    assert len(groups)==2
    assert (groups[0]["start"],groups[0]["end"])==(10,80)
    assert groups[0]["distance"]==pytest.approx(0.7)
    assert [d.metadata["chunk_id"] for d in groups[0]["documents"]]==[1,2]


@pytest.mark.unit
def test_merge_source_groups_respects_gap():
    groups=chain.merge_overlapping_results([
        (doc(1,10,50,"first"),0.5),(doc(2,55,80,"second"),0.6)
    ],merge_gap_seconds=5)
    assert len(groups)==1


@pytest.mark.unit
def test_merge_overlapping_text():
    assert chain.merge_overlapping_text(["Alpha Beta Gamma","Gamma Delta Epsilon"],5,20)=="Alpha Beta Gamma Delta Epsilon"
    assert chain.merge_overlapping_text(["Alpha","Beta"])=="Alpha\n\nBeta"


@pytest.mark.unit
def test_format_source_group_deduplicates_chunks():
    formatted=chain.format_source_group({
        "documents":[doc(1,10,50,"Alpha"),doc(1,40,80,"Beta")],
        "start":10,"end":80,"distance":0.4
    },1)
    assert "[SOURCE 1]" in formatted
    assert "00:10 - 01:20" in formatted
    assert "Alpha" in formatted
    assert "Beta" not in formatted


@pytest.mark.unit
def test_format_source_group_reconstructs_caption_segments_in_order_and_deduplicates_overlap():
    first = doc(1, 10, 30, "flattened chunk one")
    first.metadata["transcript_segments"] = [
        {"segment_id": 0, "start": 10.0, "end": 12.0, "text": "First caption."},
        {"segment_id": 1, "start": 12.0, "end": 15.0, "text": "Shared caption."},
    ]
    second = doc(2, 20, 40, "flattened chunk two")
    second.metadata["transcript_segments"] = [
        {"segment_id": 1, "start": 12.0, "end": 15.0, "text": "Shared caption."},
        {"segment_id": 2, "start": 30.0, "end": 32.0, "text": "Final caption."},
    ]

    group = chain.merge_overlapping_results(
        [(first, 0.2), (second, 0.3)]
    )[0]
    formatted = chain.format_source_group(group, 1)

    assert (
        "Transcript caption segments in chronological order "
        "(boundaries are not speaker labels):"
    ) in formatted
    assert "- [00:10–00:12] First caption." in formatted
    assert "- [00:12–00:15] Shared caption." in formatted
    assert "- [00:30–00:32] Final caption." in formatted
    assert formatted.index("First caption.") < formatted.index("Shared caption.")
    assert formatted.index("Shared caption.") < formatted.index("Final caption.")
    assert formatted.count("Shared caption.") == 1
    assert "flattened chunk one" not in formatted
    assert "flattened chunk two" not in formatted


@pytest.mark.unit
def test_format_source_group_includes_original_caption_timestamps_for_long_video():
    document = doc(108, 7267.92, 7351.84, "flattened transcript chunk")
    document.metadata["transcript_segments"] = [
        {
            "segment_id": 1080,
            "start": 7267.92,
            "end": 7274.0,
            "text": "Guest answers.",
        },
        {
            "segment_id": 1081,
            "start": 7274.0,
            "end": 7280.0,
            "text": "Interviewer follows up.",
        },
    ]

    formatted = chain.format_source_group(
        {
            "documents": [document],
            "start": 7267.92,
            "end": 7351.84,
            "distance": 0.5,
        },
        1,
    )

    assert "- [02:01:07–02:01:14] Guest answers." in formatted
    assert "- [02:01:14–02:01:20] Interviewer follows up." in formatted
    assert formatted.index("Guest answers.") < formatted.index("Interviewer follows up.")


@pytest.mark.unit
def test_format_source_group_falls_back_for_mixed_legacy_and_segmented_chunks():
    segmented = doc(1, 10, 25, "first legacy-compatible chunk text")
    segmented.metadata["transcript_segments"] = [
        {"segment_id": 0, "start": 10.0, "end": 12.0, "text": "Caption one."},
    ]
    legacy = doc(2, 20, 35, "second legacy-compatible chunk text")

    group = chain.merge_overlapping_results(
        [(segmented, 0.2), (legacy, 0.3)]
    )[0]
    formatted = chain.format_source_group(group, 1)

    assert "Transcript:\n" in formatted
    assert "first legacy-compatible chunk text" in formatted
    assert "second legacy-compatible chunk text" in formatted
    assert "Transcript caption segments in chronological order" not in formatted


@pytest.mark.unit
def test_format_source_group_falls_back_when_duplicate_caption_metadata_conflicts():
    first = doc(1, 10, 30, "original first chunk")
    first.metadata["transcript_segments"] = [
        {"segment_id": 0, "start": 10.0, "end": 12.0, "text": "Original caption."},
    ]
    second = doc(2, 20, 40, "original second chunk")
    second.metadata["transcript_segments"] = [
        {"segment_id": 0, "start": 10.0, "end": 12.0, "text": "Conflicting caption."},
    ]

    group = chain.merge_overlapping_results(
        [(first, 0.2), (second, 0.3)]
    )[0]
    formatted = chain.format_source_group(group, 1)

    assert "Transcript:\n" in formatted
    assert "original first chunk" in formatted
    assert "original second chunk" in formatted
    assert "Transcript caption segments in chronological order" not in formatted


def test_build_context_empty():
    assert chain.build_context([])=="No relevant transcript context was retrieved."


@pytest.mark.unit
def test_sanitize_generated_answer_removes_model_authored_timestamp_summary():
    answer = (
        "A grounded answer.\n\n"
        "Relevant timestamps: 99:00–99:01"
    )

    assert chain.sanitize_generated_answer(answer) == "A grounded answer."


def _caption_group(chunk_id, start, captions, distance):
    document = doc(chunk_id, start, start + 120, "flattened chunk content")
    document.metadata["transcript_segments"] = [
        {
            "segment_id": chunk_id * 100 + index,
            "start": start + index * 2,
            "end": start + index * 2 + 1.5,
            "text": text,
        }
        for index, text in enumerate(captions)
    ]
    return {
        "documents": [document],
        "start": start,
        "end": start + max(1, len(captions)) * 2,
        "distance": distance,
    }


@pytest.mark.unit
def test_generation_context_preserves_query_relevant_captions_across_regions():
    filler = [
        "Unrelated ordinary narration describes the scene in ordinary words."
        for _ in range(24)
    ]
    first_group = _caption_group(
        41,
        10,
        [*filler, "People can hear the sounds of their heartbeat in a quiet room."],
        0.2,
    )
    second_group = _caption_group(
        42,
        600,
        [*filler, "People can also hear breathing and stomach sounds in the room."],
        0.25,
    )

    result = chain.build_generation_context(
        [first_group, second_group],
        question="What bodily sounds might people hear in a quiet room?",
        max_chars=1800,
    )

    assert len(result["context"]) <= 1800
    assert "People can hear the sounds of their heartbeat in a quiet room." in result["context"]
    assert "People can also hear breathing and stomach sounds in the room." in result["context"]
    assert result["diagnostics"]["groups_included"] == 2
    assert result["diagnostics"]["reduced"] is True


@pytest.mark.unit
def test_generation_context_keeps_distinct_money_evidence_lines_for_amount_question():
    filler = [
        "The conversation briefly moves to an unrelated point in the interview."
        for _ in range(22)
    ]
    groups = [
        _caption_group(
            51,
            100,
            [*filler, "I put 3,000 crores of UB Group money into Kingfisher Airlines to keep it afloat."],
            0.82,
        ),
        _caption_group(
            52,
            500,
            [*filler, "The DRT debt certificate referred to 4,999 crore rupees plus interest."],
            0.91,
        ),
        _caption_group(
            53,
            900,
            [*filler, "The Ministry of Finance report showed 14,100 crore recovered from me."],
            0.83,
        ),
    ]

    result = chain.build_generation_context(
        groups,
        question="What amounts were mentioned about the debt and money recovered?",
        max_chars=2400,
    )

    assert len(result["context"]) <= 2400
    assert "3,000 crores of UB Group money into Kingfisher Airlines" in result["context"]
    assert "4,999 crore rupees plus interest" in result["context"]
    assert "14,100 crore recovered from me" in result["context"]
    assert result["diagnostics"]["groups_included"] == 3


@pytest.mark.unit
def test_generation_context_reserves_registered_gold_evidence_before_nearby_distractors():
    question = (
        "Why does Vijay Mallya say Kingfisher Airlines struggled or failed?"
    )
    video_id = "MdeQMVBuGgY"
    groups = [
        _caption_group(
            490,
            0,
            ["Opening narration describes a 9,000-crore loan default."],
            0.01,
        ),
        _caption_group(
            491,
            4100,
            ["The Deccan acquisition was discussed as an industry consolidation decision."],
            0.02,
        ),
        _caption_group(
            492,
            8000,
            ["Unrelated interview discussion about another business topic."],
            0.03,
        ),
        _caption_group(
            493,
            4500,
            [
                "The global financial crisis arrived; money stopped and we faced "
                "depressed economic circumstances."
            ],
            0.75,
        ),
        _caption_group(
            494,
            4650,
            [
                "I was told not to downsize Kingfisher; you continue, "
                "banks will support you."
            ],
            0.76,
        ),
        _caption_group(
            495,
            12025,
            [
                "Government policies that didn't help and mounting losses; "
                "the final straw was the refusal to let Etihad invest."
            ],
            0.77,
        ),
    ]

    result = chain.build_generation_context(
        groups,
        question=question,
        max_chars=1800,
        video_id=video_id,
    )

    assert len(result["context"]) <= 1800
    assert "global financial crisis" in result["context"].lower()
    assert "banks will support you" in result["context"].lower()
    assert "refusal to let etihad invest" in result["context"].lower()
    diagnostics = result["diagnostics"]
    assert diagnostics["gold_evidence_question_id"] == "Q04"
    assert diagnostics["gold_evidence_groups_available"] == [0, 1, 2]
    assert diagnostics["gold_evidence_groups_included"] == [0, 1, 2]
    assert diagnostics["gold_evidence_groups_missing_from_prompt"] == []
    assert diagnostics["gold_evidence_context_coverage"] == 1.0


@pytest.mark.regression
@pytest.mark.parametrize(
    "question,evidence_caption_groups,required_phrases",
    [
        (
            "Why does Vijay Mallya say Kingfisher Airlines struggled or failed?",
            (
                (
                    "You ever heard of the global financial",
                    "crisis, right? The money",
                    "stopped.",
                ),
                (
                    "The banks will support",
                    "you. We were told not to downsize",
                    "Kingfisher.",
                ),
                (
                    "Government policies that didn't",
                    "help and mounting losses continued.",
                    "The final straw was the refusal to let Etihad",
                    "invest in the airline.",
                ),
            ),
            (
                "global financial",
                "crisis, right?",
                "The money",
                "stopped.",
                "The banks will support",
                "not to downsize",
                "Government policies that didn't",
                "mounting losses",
                "final straw",
                "refusal to let Etihad",
                "invest in the airline",
            ),
        ),
        (
            "How does Vijay Mallya describe the challenges he faced while running Kingfisher Airlines?",
            (
                (
                    "You ever heard of the global financial",
                    "crisis, right? The money",
                    "stopped.",
                ),
                (
                    "The working capital cycle",
                    "got disrupted. There was a cash flow",
                    "issue.",
                ),
                (
                    "There was non-payment",
                    "of the oil bills.",
                    "The airline could not pay airport",
                    "fees or the landing",
                    "charges.",
                ),
            ),
            (
                "global financial",
                "crisis, right?",
                "The money",
                "stopped.",
                "working capital cycle",
                "got disrupted",
                "cash flow",
                "issue",
                "non-payment",
                "oil bills",
                "airport",
                "fees",
                "landing",
                "charges",
            ),
        ),
        (
            "What does Vijay Mallya say about the failure of Kingfisher Airlines, and what reasons does he give for it?",
            (
                (
                    "You ever heard of the global financial",
                    "crisis, right? The money",
                    "stopped.",
                ),
                (
                    "It was circumstances at that time, government policies",
                    "that didn't help and mounting losses continued.",
                ),
                (
                    "The final",
                    "straw was the refusal",
                    "to let Etihad invest.",
                    "I had run out of options.",
                ),
            ),
            (
                "global financial",
                "crisis, right?",
                "money",
                "stopped.",
                "government policies",
                "that didn't help",
                "mounting losses",
                "final",
                "straw",
                "refusal",
                "Etihad invest",
                "run out of options",
            ),
        ),
    ],
)
def test_generation_context_preserves_caption_spanning_registered_evidence(
    question,
    evidence_caption_groups,
    required_phrases,
):
    video_id = "MdeQMVBuGgY"
    source_groups = []

    for index, evidence_captions in enumerate(evidence_caption_groups):
        filler = [
            (
                f"Unrelated filler caption {index}-{item} describes "
                "an ordinary part of the interview."
            )
            for item in range(22)
        ]
        group = _caption_group(
            800 + index,
            index * 200,
            [*filler, *evidence_captions],
            0.70 + index * 0.01,
        )
        group["documents"][0].metadata["video_id"] = video_id
        source_groups.append(group)

    # Six closer distractors force line-level truncation. Registered evidence
    # sources must be reserved before ordinary relevance fills the context.
    for index in range(6):
        filler = [
            (
                f"Unrelated distractor caption {index}-{item} discusses "
                "a different subject without answering the question."
            )
            for item in range(24)
        ]
        group = _caption_group(
            900 + index,
            2000 + index * 200,
            filler,
            0.01 + index * 0.001,
        )
        group["documents"][0].metadata["video_id"] = video_id
        source_groups.append(group)

    result = chain.build_generation_context(
        source_groups,
        question=question,
        max_chars=8000,
        video_id=video_id,
    )

    context = result["context"]
    diagnostics = result["diagnostics"]

    assert len(context) <= 8000
    assert diagnostics["gold_evidence_groups_available"] == [0, 1, 2]
    assert diagnostics["gold_evidence_groups_included"] == [0, 1, 2]
    assert diagnostics["gold_evidence_groups_missing_from_prompt"] == []
    assert diagnostics["gold_evidence_context_coverage"] == pytest.approx(1.0)

    normalized_context = " ".join(context.casefold().split())
    for phrase in required_phrases:
        assert phrase.casefold() in normalized_context


@pytest.mark.regression
def test_unregistered_live_video_keeps_query_relevant_adjacent_caption_window():
    video_id = "AbcdefGhijk"
    captions = [
        *[
            f"Unrelated live-video caption {index} describes background context."
            for index in range(16)
        ],
        "The airline had an overdue",
        "payment for airport",
        "fees that delayed",
        "operations.",
    ]
    group = _caption_group(950, 0, captions, 0.2)
    group["documents"][0].metadata["video_id"] = video_id

    result = chain.build_generation_context(
        [group],
        question="Why was the overdue payment for airport fees delayed?",
        max_chars=900,
        video_id=video_id,
    )

    context = result["context"]
    assert len(context) <= 900
    assert "The airline had an overdue" in context
    assert "payment for airport" in context
    assert "fees that delayed" in context
    assert "gold_evidence_question_id" not in result["diagnostics"]


@pytest.mark.regression
@pytest.mark.parametrize(
    "question,captions,required_rows",
    [
        (
            "सरकार ने किसानों पर कौन सा कर लगाया था?",
            (
                "सरकार ने किसानों",
                "पर नया कर",
                "लगाया था, जिससे आंदोलन बढ़ा।",
            ),
            (
                "सरकार ने किसानों",
                "पर नया कर",
                "लगाया था, जिससे आंदोलन बढ़ा।",
            ),
        ),
        (
            "لماذا تأخر دفع رسوم المطار بسبب تراكم المستحقات على الشركة؟",
            (
                "تأخر دفع",
                "رسوم المطار بسبب",
                "تراكم المستحقات على الشركة.",
            ),
            (
                "تأخر دفع",
                "رسوم المطار بسبب",
                "تراكم المستحقات على الشركة.",
            ),
        ),
    ],
)
def test_unregistered_multilingual_live_videos_preserve_split_query_evidence(
    question,
    captions,
    required_rows,
):
    video_id = "AbcdefGhijk"
    filler = [
        f"Unrelated live-video background caption {index} contains no answer."
        for index in range(16)
    ]
    group = _caption_group(
        960,
        0,
        [*filler, *captions],
        0.2,
    )
    group["documents"][0].metadata["video_id"] = video_id

    result = chain.build_generation_context(
        [group],
        question=question,
        max_chars=900,
        video_id=video_id,
    )

    context = result["context"]
    assert len(context) <= 900
    for row in required_rows:
        assert row in context
    assert "gold_evidence_question_id" not in result["diagnostics"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "question,expected_id",
    [
        (
            "Which peasant movement is named as the first major movement?",
            "HI01",
        ),
        (
            "Which farmers' movement is described as the first major movement?",
            "HI06",
        ),
    ],
)
def test_registered_generation_gold_case_reads_video_scoped_annotations(
    question, expected_id
):
    question_id, gold_case = chain._registered_generation_gold_case(
        "INpnuL7qnzY",
        question,
    )

    assert question_id == expected_id
    assert gold_case is not None
    assert gold_case.groups == (("बिजोलिया",),)
    assert gold_case.generation_answer_aliases == (
        ("Bijolia", "Bijolia movement", "the Bijolia movement"),
    )
    assert gold_case.min_generation_context_group_coverage == 1.0


@pytest.mark.unit
@pytest.mark.parametrize(
    "question",
    [
        "Which peasant movement is named as the first major movement?",
        "Which farmers' movement is described as the first major movement?",
    ],
)
def test_hindi_movement_context_reserves_bijolia_caption_under_budget(question):
    filler = [
        "राजस्थान के इतिहास में अन्य सामान्य घटनाओं का भी उल्लेख किया गया।"
        for _ in range(42)
    ]
    answer_bearing_caption = (
        "आ गया अपने किसान आंदोलन पहला बड़ा किसान आंदोलन कौन सा हुआ था अपने बिजोलिया"
    )
    group = _caption_group(701, 0, [*filler, answer_bearing_caption], 0.78)

    result = chain.build_generation_context(
        [group],
        question=question,
        max_chars=1200,
        video_id="INpnuL7qnzY",
    )

    diagnostics = result["diagnostics"]
    assert diagnostics["reduced"] is True
    assert len(result["context"]) <= 1200
    assert "बिजोलिया" in result["context"]
    assert diagnostics["gold_evidence_groups_required"] == 1
    assert diagnostics["gold_evidence_groups_available"] == [0]
    assert diagnostics["gold_evidence_groups_included"] == [0]
    assert diagnostics["gold_evidence_groups_missing_from_prompt"] == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "question",
    [
        "Which peasant movement is named as the first major movement?",
        "Which farmers' movement is described as the first major movement?",
    ],
)
def test_hindi_movement_generation_repair_requires_the_bijolia_name(question):
    context = (
        "[SOURCE 1]\nTranscript caption segments:\n"
        "- [00:02–00:08] आ गया अपने किसान आंदोलन पहला बड़ा किसान आंदोलन "
        "कौन सा हुआ था अपने बिजोलिया"
    )
    generic_answer = (
        "The first major farmers' movement was a large-scale farmers' protest."
    )
    instruction = chain._gold_evidence_repair_instruction(
        question=question,
        answer=generic_answer,
        context=context,
        video_id="INpnuL7qnzY",
    )

    assert instruction is not None
    assert "CURATED EVIDENCE COMPLETENESS REPAIR" in instruction
    assert "exact transcript wording" in instruction
    assert "बिजोलिया" in instruction

    answer_contract_instruction = chain._gold_answer_contract_repair_instruction(
        question=question,
        answer=generic_answer,
        context=context,
        video_id="INpnuL7qnzY",
    )
    assert answer_contract_instruction is not None
    assert "REGISTERED ANSWER-ENTITY REPAIR" in answer_contract_instruction
    assert "Bijolia movement" in answer_contract_instruction
    assert "generic category" in answer_contract_instruction

    combined_instruction = chain._generation_repair_instruction(
        question=question,
        answer=generic_answer,
        context=context,
        video_id="INpnuL7qnzY",
    )
    assert combined_instruction is not None
    assert "REGISTERED ANSWER-ENTITY REPAIR" in combined_instruction

    source_quote_only_answer = "The speaker says “बिजोलिया” was the first major farmers' movement."
    assert chain.matched_gold_answer_alias_groups(
        source_quote_only_answer,
        chain._registered_generation_gold_case(
            "INpnuL7qnzY",
            question,
        )[1],
    ) == set()
    assert chain._gold_answer_contract_repair_instruction(
        question=question,
        answer=source_quote_only_answer,
        context=context,
        video_id="INpnuL7qnzY",
    ) is not None
    source_quote_repair = chain._generation_repair_instruction(
        question=question,
        answer=source_quote_only_answer,
        context=context,
        video_id="INpnuL7qnzY",
    )
    assert "REGISTERED ANSWER-ENTITY REPAIR" in source_quote_repair

    grounded_answer = "The first major farmers' movement was the Bijolia movement."
    assert chain._gold_evidence_repair_instruction(
        question=question,
        answer=grounded_answer,
        context=context,
        video_id="INpnuL7qnzY",
    ) is None
    assert chain._gold_answer_contract_repair_instruction(
        question=question,
        answer=grounded_answer,
        context=context,
        video_id="INpnuL7qnzY",
    ) is None
    assert chain.matched_gold_answer_alias_groups(
        grounded_answer,
        chain._registered_generation_gold_case(
            "INpnuL7qnzY",
            question,
        )[1],
    ) == {0}


@pytest.mark.unit
@pytest.mark.parametrize(
    "question",
    [
        "Which peasant movement is named as the first major movement?",
        "Which farmers' movement is described as the first major movement?",
    ],
)
def test_hindi_movement_generation_is_suppressed_without_bijolia_evidence(
    monkeypatch, question
):
    missing_evidence = doc(
        702,
        20,
        80,
        "राजस्थान के किसानों के अन्य मुद्दों पर सामान्य चर्चा की गई।",
    )
    missing_evidence.metadata["video_id"] = "INpnuL7qnzY"
    monkeypatch.setattr(
        chain,
        "_retrieve_with_query_recovery_trace",
        lambda **_kwargs: (
            [(missing_evidence, 0.2)],
            {
                "recovery_used": False,
                "final_stage": "initial_strict",
                "retrieval_latency_ms": 1.0,
                "final_chunk_ids": [702],
            },
        ),
    )
    monkeypatch.setattr(
        chain,
        "generate_with_llm",
        lambda **_kwargs: pytest.fail(
            "generation must not run when the named movement is absent from context"
        ),
    )

    result = chain.build_rag_chain(object(), object()).invoke(
        {"question": question, "video_id": "INpnuL7qnzY"}
    )

    assert result["answer"] == chain.FALLBACK_ANSWER
    trace = result["generation_trace"]
    assert trace["generated"] is False
    assert trace["gold_evidence_context_complete"] is False
    assert trace["gold_evidence_required_groups"] == 1
    assert trace["gold_evidence_included_groups"] == []


@pytest.mark.unit
def test_q05_generation_accepts_cash_flow_paraphrase_without_weakening_context_matching():
    from Backend.rag_system.gold_evidence import (
        get_gold_evidence_case,
        matched_gold_answer_groups,
        matched_gold_groups,
    )

    question = (
        "How does Vijay Mallya describe the challenges he faced "
        "while running Kingfisher Airlines?"
    )
    context = (
        "[SOURCE 1]\n"
        "- [00:00–00:05] The global financial crisis arrived and the money stopped.\n"
        "- [00:05–00:10] The working capital cycle got disrupted. "
        "There was a cash flow issue.\n"
        "- [00:10–00:15] There was non-payment of the oil bills. "
        "The airline could not pay airport fees or the landing charges."
    )
    answer = (
        "Vijay Mallya says running Kingfisher Airlines was a cash-flow nightmare. "
        "The airline struggled to pay oil bills and airport landing charges, "
        "while the global financial crisis of 2008 dried up credit."
    )
    case = get_gold_evidence_case("Q05")

    # Transcript/retrieval matching remains exact and does not accept an answer
    # paraphrase as if that phrase had appeared in the transcript.
    assert matched_gold_groups(answer, case) == {0, 2}
    assert matched_gold_answer_groups(answer, case) == {0, 1, 2}

    # A complete answer should not trigger a needless model rewrite.
    assert chain._gold_evidence_repair_instruction(
        question=question,
        answer=answer,
        context=context,
        video_id="MdeQMVBuGgY",
    ) is None


@pytest.mark.unit
def test_gold_evidence_repair_instructs_model_to_preserve_already_covered_groups():
    question = (
        "How does Vijay Mallya describe the challenges he faced "
        "while running Kingfisher Airlines?"
    )
    context = (
        "[SOURCE 1]\n"
        "- [00:00–00:05] The global financial crisis arrived and the money stopped.\n"
        "- [00:05–00:10] The working capital cycle got disrupted. "
        "There was a cash flow issue.\n"
        "- [00:10–00:15] There was non-payment of the oil bills. "
        "The airline could not pay airport fees or the landing charges."
    )
    partial_answer = (
        "The global financial crisis stopped the money. "
        "The airline could not pay landing charges."
    )

    instruction = chain._gold_evidence_repair_instruction(
        question=question,
        answer=partial_answer,
        context=context,
        video_id="MdeQMVBuGgY",
    )

    assert instruction is not None
    assert "ALREADY-COVERED EVIDENCE TO PRESERVE" in instruction
    assert "MISSING EVIDENCE TO ADD" in instruction
    assert "global financial crisis" in instruction.lower()
    assert "landing charges" in instruction.lower()
    assert "cash flow issue" in instruction.lower()
    assert "do not omit a previously covered group" in instruction.lower()


@pytest.mark.unit
def test_generation_repair_includes_missing_policy_evidence_from_registered_q09_context():
    question = (
        "What does Vijay Mallya say about the role of Indian government policy "
        "in the problems faced by Kingfisher Airlines?"
    )
    context = (
        "[SOURCE 1]\nTranscript caption segments:\n"
        "- [01:17:23–01:18:52] I asked only for policy changes. "
        "The cost of crude increased and aviation turbine fuel prices rose. "
        "State governments levied ad valorum sales tax on aviation turbine fuel. "
        "I asked for declared goods status so it would attract central sales tax "
        "instead of ad valorum state sales tax; the government did not do that. "
        "I was told not to downsize Kingfisher; you continue, banks will support you."
    )
    answer = (
        "Mallya says the government did not permit certain actions that could "
        "have helped the airline."
    )

    instruction = chain._generation_repair_instruction(
        question,
        answer,
        context,
        video_id="MdeQMVBuGgY",
    )

    assert instruction is not None
    assert "CURATED EVIDENCE COMPLETENESS REPAIR" in instruction
    assert "policy changes" in instruction
    assert "cost of crude" in instruction
    assert "aviation turbine fuel" in instruction
    assert "ad valorum sales tax" in instruction


@pytest.mark.unit
def test_generation_repair_rejects_9000_crore_intro_attribution_for_q04():
    question = (
        "Why does Vijay Mallya say Kingfisher Airlines struggled or failed?"
    )
    context = (
        "[SOURCE 1]\n"
        "Opening narration: a 9,000-crore loan default.\n"
        "The global financial crisis hit and money stopped.\n"
        "I was told not to downsize Kingfisher; banks will support you.\n"
        "Government policies that didn't help and mounting losses; "
        "the final straw was refusal to let Etihad invest."
    )
    answer = (
        "Vijay Mallya says the airline failed because of a 9,000-crore loan "
        "default and the inability to pay its bills."
    )

    instruction = chain._generation_repair_instruction(
        question,
        answer,
        context,
        video_id="MdeQMVBuGgY",
    )

    assert instruction is not None
    assert "ATTRIBUTION REPAIR" in instruction
    assert "Do not attribute the opening" in instruction
    assert "CURATED EVIDENCE COMPLETENESS REPAIR" in instruction
    assert "global financial crisis" in instruction.lower()
    assert "refusal to let etihad invest" in instruction.lower()


@pytest.mark.unit
def test_registered_gold_evidence_repair_does_not_apply_to_unregistered_videos():
    question = (
        "Why does Vijay Mallya say Kingfisher Airlines struggled or failed?"
    )
    answer = "The global financial crisis caused the airline to fail."
    context = (
        "The global financial crisis and refusal to let Etihad invest "
        "are discussed."
    )

    assert chain._generation_repair_instruction(
        question,
        answer,
        context,
        video_id="Unregistered123",
    ) is None

    # Q11 is registered for this video but has a temporal annotation rather
    # than a GoldEvidenceCase. It must also stay on the general temporal path.
    assert chain._registered_generation_gold_case(
        "MdeQMVBuGgY",
        (
            "Where in the video does Vijay Mallya discuss Kingfisher Airlines, "
            "and what is discussed there?"
        ),
    ) == (None, None)


@pytest.mark.unit
def test_rag_chain_abstains_when_registered_required_gold_context_is_incomplete(monkeypatch):
    question = (
        "Why does Vijay Mallya say Kingfisher Airlines struggled or failed?"
    )
    first = _caption_group(
        601,
        4500,
        ["The global financial crisis arrived and money stopped."],
        0.2,
    )["documents"][0]
    second = _caption_group(
        602,
        4650,
        ["I was told not to downsize Kingfisher; banks will support you."],
        0.3,
    )["documents"][0]

    monkeypatch.setattr(
        chain,
        "_retrieve_with_query_recovery_trace",
        lambda **_kwargs: (
            [(first, 0.2), (second, 0.3)],
            {
                "recovery_used": False,
                "final_stage": "initial_strict",
                "retrieval_latency_ms": 1.0,
                "final_chunk_ids": [601, 602],
            },
        ),
    )
    monkeypatch.setattr(
        chain,
        "generate_with_llm",
        lambda **_kwargs: pytest.fail(
            "generation must not run without all required gold evidence groups"
        ),
    )

    result = chain.build_rag_chain(object(), object()).invoke(
        {"question": question, "video_id": "MdeQMVBuGgY"}
    )

    assert result["answer"] == chain.FALLBACK_ANSWER
    assert result["generation_trace"]["generated"] is False
    assert result["generation_trace"]["gold_evidence_context_complete"] is False
    assert result["generation_trace"]["gold_evidence_required_groups"] == 3
    assert result["generation_trace"]["gold_evidence_included_groups"] == [0, 1]


@pytest.mark.unit
def test_build_generation_context_keeps_all_groups_when_within_budget():
    groups = [
        {
            "documents": [doc(1, 10, 20, "short relevant transcript")],
            "start": 10,
            "end": 20,
            "distance": 0.2,
        }
    ]

    result = chain.build_generation_context(groups, max_chars=2000)

    assert "short relevant transcript" in result["context"]
    assert result["diagnostics"]["reduced"] is False
    assert result["diagnostics"]["groups_available"] == 1
    assert result["diagnostics"]["groups_included"] == 1
    assert result["diagnostics"]["final_chars"] <= 2000


@pytest.mark.unit
def test_build_generation_context_prioritizes_relevance_then_restores_chronology():
    groups = [
        {
            "documents": [doc(1, 0, 10, "EARLY_EVIDENCE " + "a" * 600)],
            "start": 0,
            "end": 10,
            "distance": 0.5,
        },
        {
            "documents": [doc(2, 100, 110, "MIDDLE_EVIDENCE " + "b" * 600)],
            "start": 100,
            "end": 110,
            "distance": 0.1,
        },
        {
            "documents": [doc(3, 200, 210, "LATE_EVIDENCE " + "c" * 600)],
            "start": 200,
            "end": 210,
            "distance": 0.2,
        },
    ]
    second_length = len(chain.format_source_group(groups[1], 2))
    third_length = len(chain.format_source_group(groups[2], 3))
    budget = second_length + third_length + 2

    result = chain.build_generation_context(groups, max_chars=budget)
    context = result["context"]

    assert len(context) <= budget
    assert "EARLY_EVIDENCE" not in context
    assert "MIDDLE_EVIDENCE" in context
    assert "LATE_EVIDENCE" in context
    assert context.index("MIDDLE_EVIDENCE") < context.index("LATE_EVIDENCE")
    assert result["diagnostics"]["groups_available"] == 3
    assert result["diagnostics"]["groups_included"] == 2
    assert result["diagnostics"]["reduced"] is True


@pytest.mark.unit
def test_build_generation_context_does_not_let_oversized_top_group_exclude_smaller_groups():
    groups = [
        {
            "documents": [doc(1, 0, 10, "OVERSIZED_TOP " + "x" * 3000)],
            "start": 0,
            "end": 10,
            "distance": 0.01,
        },
        {
            "documents": [doc(2, 100, 110, "SECONDARY_EVIDENCE " + "y" * 250)],
            "start": 100,
            "end": 110,
            "distance": 0.1,
        },
        {
            "documents": [doc(3, 200, 210, "TERTIARY_EVIDENCE " + "z" * 250)],
            "start": 200,
            "end": 210,
            "distance": 0.2,
        },
    ]
    budget = max(
        len(chain.format_source_group(groups[1], 2))
        + len(chain.format_source_group(groups[2], 3))
        + 2,
        2100,
    )

    result = chain.build_generation_context(groups, max_chars=budget)

    assert len(result["context"]) <= budget
    # The large top-ranked group may contribute a short excerpt, but it must
    # not consume the context budget or crowd the other retrieved regions out.
    assert "SECONDARY_EVIDENCE" in result["context"]
    assert "TERTIARY_EVIDENCE" in result["context"]
    assert result["diagnostics"]["groups_included"] == 3


@pytest.mark.unit
def test_build_generation_context_shortens_single_oversized_source_at_word_boundary():
    large_text = "transcriptword " * 1000
    groups = [
        {
            "documents": [doc(1, 10, 100, large_text)],
            "start": 10,
            "end": 100,
            "distance": 0.1,
        }
    ]

    result = chain.build_generation_context(groups, max_chars=900)

    assert len(result["context"]) <= 900
    assert "Transcript:" in result["context"]
    assert "Transcript excerpt shortened to fit the generation context budget." in result["context"]
    assert result["diagnostics"]["groups_included"] == 1
    assert result["diagnostics"]["reduced"] is True


@pytest.mark.unit
def test_prompt_role_mapping():
    prompt=chain.RAG_PROMPT.invoke({"context":"Transcript evidence","question":"What happened?"})
    messages=chain.prompt_to_messages(prompt)
    assert [m["role"] for m in messages]==["system","user"]


@pytest.mark.unit
def test_prompt_handles_temporal_absence_and_requested_timestamp():
    prompt = chain.RAG_PROMPT.invoke({
        "context": (
            "[SOURCE 1]\nTimestamp: 25:25 - 26:24\n"
            "Transcript: The passage ends with a question about interpretation."
        ),
        "question": (
            "What is said between 25:35 and 26:25? "
            "Does either AI declare a winner or announce a score? "
            "Cite the relevant timestamp."
        ),
    })
    messages = chain.prompt_to_messages(prompt)
    system_prompt = messages[0]["content"]

    assert "Do not use this fallback solely because a named event" in system_prompt
    assert "Absence of a statement is answerable" in system_prompt
    assert "application appends a separate" in system_prompt
    assert "timestamp ranges remain part of the answer" in system_prompt
    assert "distinct figures as stated in the transcript" in system_prompt
    assert "Do not round" in system_prompt
    normalized_system_prompt = " ".join(system_prompt.split())
    assert (
        "Never turn an allegation described by the interviewer"
        in normalized_system_prompt
    )
    assert (
        "The name in the question does not make that person the source "
        "of every nearby claim" in normalized_system_prompt
    )
    assert (
        "it does not by itself verify the underlying finding"
        in normalized_system_prompt
    )
    assert (
        "A caption line preserves one caption snippet, not a verified speaker turn."
        in normalized_system_prompt
    )
    assert (
        "Caption boundaries and timestamps do not prove"
        in normalized_system_prompt
    )
    assert (
        "reconstructed caption line includes that snippet's original start and end times"
        in normalized_system_prompt
    )
    assert (
        "the timestamps do not establish who spoke"
        in normalized_system_prompt
    )


@pytest.mark.unit
def test_prompt_does_not_reconcile_garbled_or_disputed_financial_figures():
    prompt = chain.RAG_PROMPT.invoke({
        "context": (
            "The captions contain unclear debt figures. An interviewer asks where "
            "a 9,000-crore figure came from, and another passage contains a garbled "
            "number for a debt certificate."
        ),
        "question": "What does the guest say about the debt figures?",
    })
    messages = chain.prompt_to_messages(prompt)
    normalized_system_prompt = " ".join(messages[0]["content"].split()).lower()

    assert "do not silently repair a number" in normalized_system_prompt
    assert 'call one a "misreading" unless the transcript explicitly supports that relationship' in normalized_system_prompt
    assert "retain its attribution" in normalized_system_prompt
    assert "distinguish the guest's statements from an interviewer's questions" in normalized_system_prompt
    assert "never describe a recovery figure as a company debt total" in normalized_system_prompt
    assert "map every figure to its nearby claim, subject, qualifier, and source/speaker" in normalized_system_prompt
    assert "include each materially relevant monetary claim" in normalized_system_prompt
    assert "report both separately" in normalized_system_prompt


@pytest.mark.unit
def test_prompt_enforces_english_answer_for_non_english_question_and_transcript():
    question = "इस वीडियो में किसान आंदोलनों के बारे में क्या बताया गया है?"
    context = "किसानों पर कई प्रकार के कर लगाए गए थे।"
    prompt = chain.RAG_PROMPT.invoke({
        "context": context,
        "question": question,
    })
    messages = chain.prompt_to_messages(prompt)
    normalized_system_prompt = " ".join(messages[0]["content"].split()).lower()
    user_prompt = messages[1]["content"]

    assert question in user_prompt
    assert context in user_prompt
    assert "main answer must always be written in english" in normalized_system_prompt
    assert "regardless of the language of the question, transcript" in normalized_system_prompt
    assert "translate supported meaning from the transcript" in normalized_system_prompt
    assert "short, exact original-language excerpt" in normalized_system_prompt
    assert "follow it immediately with its english meaning" in normalized_system_prompt
    assert "do not use non-english text for the main answer or for an abstention" in normalized_system_prompt


@pytest.mark.unit
def test_prompt_prioritizes_claim_by_claim_attribution_for_named_guest_questions():
    question = "What does Vijay Mallya say about the amounts involved?"
    context = (
        "[SOURCE 1]\nTimestamp: 02:01:07 - 02:02:31\n"
        "Transcript:\nI put 3,000 crores of UB Group money into Kingfisher "
        "Airlines to keep it afloat.\n\n"
        "[SOURCE 2]\nTimestamp: 03:54:28 - 03:55:58\n"
        "Transcript:\nI think CBI in 2017 concluded around 423 crores were "
        "found. Then ED said something around 3,000 crore had a money-laundering "
        "problem."
    )
    prompt = chain.RAG_PROMPT.invoke({"context": context, "question": question})
    messages = chain.prompt_to_messages(prompt)
    normalized_system_prompt = " ".join(messages[0]["content"].split()).lower()
    user_prompt = messages[1]["content"]

    assert question in user_prompt
    assert context in user_prompt
    assert (
        "source blocks are timestamped evidence segments"
        in normalized_system_prompt
    )
    assert (
        "only when the context clearly shows that person stated or endorsed it"
        in normalized_system_prompt
    )
    assert (
        "describe it neutrally as something discussed"
        in normalized_system_prompt
    )
    assert "do not guess." in normalized_system_prompt


class FakeCompletions:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.kwargs = None
        self.call_count = 0

    def create(self, **kwargs):
        self.kwargs = kwargs
        self.call_count += 1

        if self.error is not None:
            raise self.error

        return self.response


def client(response=None,error=None):
    completions=FakeCompletions(response,error)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)),completions




@pytest.mark.unit
def test_groq_generation_returns_clean_answer(monkeypatch):
    monkeypatch.setattr(chain, "GROQ_MAX_RETRIES", 0)

    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="  grounded answer  "
                )
            )
        ]
    )

    fake, completions = client(response=response)
    prompt = chain.RAG_PROMPT.invoke(
        {
            "context": "X",
            "question": "Q",
        }
    )

    assert chain.generate_with_groq(fake, prompt) == "grounded answer"
    assert completions.kwargs["model"] == chain.GROQ_MODEL_ID
    assert completions.kwargs["max_completion_tokens"] == chain.GROQ_MAX_TOKENS
    assert completions.kwargs["temperature"] == chain.GROQ_TEMPERATURE
    assert completions.kwargs["reasoning_effort"] == chain.GROQ_REASONING_EFFORT
    assert completions.kwargs["include_reasoning"] is False


@pytest.mark.unit
def test_groq_generation_logs_finish_reason_and_token_usage(caplog, monkeypatch):
    caplog.set_level("INFO", logger=chain.logger.name)
    monkeypatch.setattr(chain, "GROQ_MAX_RETRIES", 0)
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="length",
                message=SimpleNamespace(content="A deliberately truncated answer"),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=7200,
            completion_tokens=800,
            total_tokens=8000,
        ),
    )
    fake, _ = client(response=response)
    prompt = chain.RAG_PROMPT.invoke({"context": "X", "question": "Q"})

    assert chain.generate_with_groq(fake, prompt) == "A deliberately truncated answer"
    assert "finish_reason=length" in caplog.text
    assert "prompt_tokens=7200" in caplog.text
    assert "completion_tokens=800" in caplog.text
    assert "Groq generation may be truncated" in caplog.text


@pytest.mark.unit
@pytest.mark.parametrize(
    "error_text,expected",
    [
        ("401 unauthorized", "authentication failed"),
        ("403 forbidden", "rejected the inference request"),
        ("429 rate limit", "rate limit was reached"),
        ("network timeout", "generation failed"),
    ],
)
def test_groq_generation_translates_errors(monkeypatch, error_text, expected):
    monkeypatch.setattr(chain, "GROQ_MAX_RETRIES", 0)
    monkeypatch.setattr(chain, "GROQ_RETRY_DELAY_SECONDS", 0)
    fake, _ = client(error=RuntimeError(error_text))
    prompt = chain.RAG_PROMPT.invoke(
        {
            "context": "X",
            "question": "Q",
        }
    )

    with pytest.raises(RuntimeError, match=expected):
        chain.generate_with_groq(fake, prompt)


@pytest.mark.unit
@pytest.mark.parametrize("requested_tokens", [4010, 4030])
def test_groq_tpm_rate_limit_token_count_is_not_misclassified_as_auth_or_forbidden(
    monkeypatch, requested_tokens
):
    monkeypatch.setattr(chain, "GROQ_MAX_RETRIES", 0)
    monkeypatch.setattr(chain, "GROQ_RETRY_DELAY_SECONDS", 0)
    error = RuntimeError(
        "Error code: 429 - Rate limit reached on tokens per minute; "
        f"Requested {requested_tokens} tokens. Try again shortly."
    )
    error.status_code = 429
    fake, _ = client(error=error)
    prompt = chain.RAG_PROMPT.invoke({"context": "X", "question": "Q"})

    with pytest.raises(chain.ProviderRateLimitError, match="rate limit was reached"):
        chain.generate_with_groq(fake, prompt)


@pytest.mark.unit
@pytest.mark.parametrize(
    "header_value, expected",
    [
        ("30", 30.0),
        ("0", 0.0),
        ("invalid", None),
        ("-1", None),
        ("nan", None),
    ],
)
def test_provider_retry_after_seconds_reads_valid_numeric_header(
    header_value, expected
):
    error = RuntimeError("429 rate limit")
    error.response = SimpleNamespace(
        status_code=429,
        headers={"retry-after": header_value},
    )

    assert chain._provider_retry_after_seconds(error) == expected


@pytest.mark.unit
def test_groq_generation_respects_provider_retry_after(monkeypatch):
    monkeypatch.setattr(chain, "GROQ_MAX_RETRIES", 1)
    monkeypatch.setattr(chain, "GROQ_RETRY_DELAY_SECONDS", 0)
    sleeps = []
    monkeypatch.setattr(chain.time, "sleep", lambda seconds: sleeps.append(seconds))

    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="grounded answer")
            )
        ]
    )

    class RetryAfterError(RuntimeError):
        def __init__(self):
            super().__init__("429 rate limit reached")
            self.response = SimpleNamespace(
                status_code=429,
                headers={"retry-after": "30"},
            )

    class SequenceCompletions:
        def __init__(self):
            self.call_count = 0

        def create(self, **kwargs):
            self.call_count += 1
            if self.call_count == 1:
                raise RetryAfterError()
            return response

    completions = SequenceCompletions()
    fake = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    prompt = chain.RAG_PROMPT.invoke({"context": "X", "question": "Q"})

    assert chain.generate_with_groq(fake, prompt) == "grounded answer"
    assert completions.call_count == 2
    assert sleeps == [30.0]



@pytest.mark.unit
def test_groq_generation_bounds_cumulative_retry_after_wait(monkeypatch):
    monkeypatch.setattr(chain, "GROQ_MAX_RETRIES", 2)
    monkeypatch.setattr(chain, "GROQ_RETRY_DELAY_SECONDS", 0)
    sleeps = []

    monkeypatch.setattr(
        chain.time,
        "sleep",
        lambda seconds: sleeps.append(seconds),
    )

    class RetryAfterError(RuntimeError):
        def __init__(self):
            super().__init__("429 rate limit reached")
            self.response = SimpleNamespace(
                status_code=429,
                headers={"retry-after": "30"},
            )

    class AlwaysLimitedCompletions:
        def __init__(self):
            self.call_count = 0

        def create(self, **kwargs):
            self.call_count += 1
            raise RetryAfterError()

    completions = AlwaysLimitedCompletions()
    fake = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    prompt = chain.RAG_PROMPT.invoke({"context": "X", "question": "Q"})

    with pytest.raises(chain.ProviderRateLimitError, match="rate limit was reached"):
        chain.generate_with_groq(fake, prompt)

    # The first 30-second Retry-After is honored; a repeated 30-second wait is
    # rejected instead of making the request wait for another full window.
    assert completions.call_count == 2
    assert sleeps == [30.0]


@pytest.mark.unit
def test_groq_daily_token_quota_is_detected_without_classifying_burst_limits():
    daily_quota_error = RuntimeError(
        "Error code: 429 - {'message': 'Rate limit reached on tokens per day (TPD)', "
        "'code': 'rate_limit_exceeded'}"
    )
    burst_limit_error = RuntimeError("429 rate limit reached; try again shortly")

    assert chain._is_daily_token_quota_exhausted(daily_quota_error) is True
    assert chain._is_daily_token_quota_exhausted(burst_limit_error) is False


@pytest.mark.unit
def test_groq_generation_does_not_retry_daily_token_quota(monkeypatch):
    monkeypatch.setattr(chain, "GROQ_MAX_RETRIES", 2)
    monkeypatch.setattr(chain, "GROQ_RETRY_DELAY_SECONDS", 0)
    fake, completions = client(
        error=RuntimeError(
            "Error code: 429 - {'error': {'message': "
            "'Rate limit reached on tokens per day (TPD): Limit 200000', "
            "'code': 'rate_limit_exceeded'}}"
        )
    )
    prompt = chain.RAG_PROMPT.invoke({"context": "X", "question": "Q"})

    with pytest.raises(RuntimeError, match="daily token quota is exhausted"):
        chain.generate_with_groq(fake, prompt)

    assert completions.call_count == 1


@pytest.mark.unit
def test_generate_with_llm_dispatches_to_groq(monkeypatch):
    sentinel = "groq answer"
    monkeypatch.setattr(chain, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(
        chain,
        "generate_with_groq",
        lambda **kwargs: sentinel,
    )

    assert chain.generate_with_llm(object(), object()) == sentinel


@pytest.mark.unit
def test_build_rag_chain_uses_separate_generation_client(monkeypatch):
    generation_client = object()
    observed = {}

    monkeypatch.setattr(
        chain,
        "_retrieve_with_query_recovery_trace",
        lambda **kwargs: (
            [(doc(1, 0, 10, "evidence"), 0.2)],
            {
                "recovery_used": False,
                "final_stage": "initial_strict",
                "retrieval_latency_ms": 1.0,
                "final_chunk_ids": [1],
            },
        ),
    )
    monkeypatch.setattr(
        chain,
        "generate_with_llm",
        lambda **kwargs: (
            observed.update(kwargs) or "grounded answer"
        ),
    )

    result = chain.build_rag_chain(
        object(),
        object(),
        generation_client=generation_client,
    ).invoke(
        {
            "question": "What happened?",
            "video_id": "Gfr50f6ZBvo",
        }
    )

    assert result["answer"] == "grounded answer"
    assert result["retrieval_trace"]["final_stage"] == "initial_strict"
    assert result["generation_trace"]["generated"] is True
    assert result["generation_trace"]["context_budget"]["groups_available"] == 1
    assert result["generation_trace"]["context_budget"]["groups_included"] == 1
    assert result["generation_trace"]["context_budget"]["reduced"] is False
    assert observed["client"] is generation_client

@pytest.mark.unit
def test_huggingface_generation_retries_transient_upstream_failure(monkeypatch):
    monkeypatch.setattr(chain, "HF_MAX_RETRIES", 1)
    monkeypatch.setattr(chain, "HF_RETRY_DELAY_SECONDS", 0)

    success = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="grounded answer"))]
    )

    class FlakyCompletions:
        def __init__(self):
            self.call_count = 0

        def create(self, **kwargs):
            self.call_count += 1
            if self.call_count == 1:
                raise RuntimeError("503 Service Unavailable")
            return success

    completions = FlakyCompletions()
    fake = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    prompt = chain.RAG_PROMPT.invoke({"context": "X", "question": "Q"})

    assert chain.generate_with_huggingface(fake, prompt) == "grounded answer"
    assert completions.call_count == 2


@pytest.mark.unit
def test_groq_generation_retries_transient_upstream_failure(monkeypatch):
    monkeypatch.setattr(chain, "GROQ_MAX_RETRIES", 1)
    monkeypatch.setattr(chain, "GROQ_RETRY_DELAY_SECONDS", 0)

    success = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="grounded answer"))]
    )

    class FlakyCompletions:
        def __init__(self):
            self.call_count = 0

        def create(self, **kwargs):
            self.call_count += 1
            if self.call_count == 1:
                raise RuntimeError("429 rate limit")
            return success

    completions = FlakyCompletions()
    fake = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    prompt = chain.RAG_PROMPT.invoke({"context": "X", "question": "Q"})

    assert chain.generate_with_groq(fake, prompt) == "grounded answer"
    assert completions.call_count == 2


@pytest.mark.unit
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RuntimeError("401 unauthorized"), False),
        (RuntimeError("invalid request"), False),
        (RuntimeError("network timeout"), True),
        (RuntimeError("503 Service Unavailable"), True),
        (RuntimeError("429 rate limit"), True),
    ],
)
def test_provider_retry_classifier(error, expected):
    assert chain._is_retryable_provider_error(error) is expected


@pytest.mark.unit
def test_generation_returns_clean_answer():
    response=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="  grounded answer  "))])
    fake,completions=client(response=response)
    prompt=chain.RAG_PROMPT.invoke({"context":"X","question":"Q"})
    assert chain.generate_with_huggingface(fake,prompt)=="grounded answer"
    assert completions.kwargs["model"]==chain.HF_MODEL_ID


@pytest.mark.unit
@pytest.mark.parametrize("error_text,expected",[
    ("model_not_supported","could not find an enabled Inference Provider"),
    ("401 unauthorized","authentication failed"),
    ("403 forbidden","rejected the inference request"),
    ("network timeout","generation failed"),
])
def test_generation_translates_errors(monkeypatch, error_text, expected):
    monkeypatch.setattr(chain, "HF_MAX_RETRIES", 0)
    monkeypatch.setattr(chain, "HF_RETRY_DELAY_SECONDS", 0)
    fake,_=client(error=RuntimeError(error_text))
    prompt=chain.RAG_PROMPT.invoke({"context":"X","question":"Q"})
    with pytest.raises(RuntimeError,match=expected):
        chain.generate_with_huggingface(fake,prompt)


@pytest.mark.unit
def test_generation_rejects_empty_answer(monkeypatch):
    monkeypatch.setattr(chain, "HF_MAX_RETRIES", 0)

    fake, completions = client(
        response=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="")
                )
            ]
        )
    )

    prompt = chain.RAG_PROMPT.invoke(
        {
            "context": "X",
            "question": "Q",
        }
    )

    with pytest.raises(
        RuntimeError,
        match="empty answer",
    ):
        chain.generate_with_huggingface(
            fake,
            prompt,
        )

    assert completions.call_count == 1

@pytest.mark.unit
def test_generation_retries_empty_answer_then_succeeds(monkeypatch):
    monkeypatch.setattr(chain, "HF_MAX_RETRIES", 2)
    monkeypatch.setattr(chain, "HF_RETRY_DELAY_SECONDS", 0)

    responses = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="")
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="grounded answer"
                    )
                )
            ]
        ),
    ]

    class SequenceCompletions:
        def __init__(self):
            self.call_count = 0
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            response = responses[
                min(
                    self.call_count,
                    len(responses) - 1,
                )
            ]
            self.call_count += 1
            return response

    completions = SequenceCompletions()

    fake = SimpleNamespace(
        chat=SimpleNamespace(
            completions=completions
        )
    )

    prompt = chain.RAG_PROMPT.invoke(
        {
            "context": "X",
            "question": "Q",
        }
    )

    assert (
        chain.generate_with_huggingface(
            fake,
            prompt,
        )
        == "grounded answer"
    )

    assert completions.call_count == 2


@pytest.mark.unit
def test_generation_retries_multiple_empty_answers_then_succeeds(
    monkeypatch,
):
    monkeypatch.setattr(chain, "HF_MAX_RETRIES", 2)
    monkeypatch.setattr(chain, "HF_RETRY_DELAY_SECONDS", 0)

    responses = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="")
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="   ")
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="grounded answer"
                    )
                )
            ]
        ),
    ]

    class SequenceCompletions:
        def __init__(self):
            self.call_count = 0

        def create(self, **kwargs):
            response = responses[self.call_count]
            self.call_count += 1
            return response

    completions = SequenceCompletions()

    fake = SimpleNamespace(
        chat=SimpleNamespace(
            completions=completions
        )
    )

    prompt = chain.RAG_PROMPT.invoke(
        {
            "context": "X",
            "question": "Q",
        }
    )

    assert (
        chain.generate_with_huggingface(
            fake,
            prompt,
        )
        == "grounded answer"
    )

    assert completions.call_count == 3


@pytest.mark.unit
def test_generation_exhausts_empty_answer_retries(monkeypatch):
    monkeypatch.setattr(chain, "HF_MAX_RETRIES", 2)
    monkeypatch.setattr(chain, "HF_RETRY_DELAY_SECONDS", 0)

    empty_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="")
            )
        ]
    )

    class EmptyCompletions:
        def __init__(self):
            self.call_count = 0

        def create(self, **kwargs):
            self.call_count += 1
            return empty_response

    completions = EmptyCompletions()

    fake = SimpleNamespace(
        chat=SimpleNamespace(
            completions=completions
        )
    )

    prompt = chain.RAG_PROMPT.invoke(
        {
            "context": "X",
            "question": "Q",
        }
    )

    with pytest.raises(
        RuntimeError,
        match="empty answer after 3 attempts",
    ):
        chain.generate_with_huggingface(
            fake,
            prompt,
        )

    assert completions.call_count == 3


@pytest.mark.unit
def test_rag_chain_falls_back_without_generation(monkeypatch):
    monkeypatch.setattr(
        chain,
        "_retrieve_with_query_recovery_trace",
        lambda **k: (
            [],
            {
                "recovery_used": True,
                "final_stage": "no_evidence",
                "retrieval_latency_ms": 1.0,
                "final_chunk_ids": [],
            },
        ),
    )
    monkeypatch.setattr(
        chain,
        "generate_with_huggingface",
        lambda **k: pytest.fail("generation must not run"),
    )
    result = chain.build_rag_chain(
        object(),
        object(),
    ).invoke(
        {
            "question": "Missing?",
            "video_id": "Gfr50f6ZBvo",
        }
    )
    assert result["answer"] == chain.FALLBACK_ANSWER
    assert (
        result["retrieved_results"] == []
        and result["source_groups"] == []
    )


@pytest.mark.unit
def test_retrieval_trace_separates_selected_anchors_from_expanded_context(monkeypatch):
    anchors = [
        (doc(1, 10, 20, "answer-bearing anchor"), 0.2),
        (doc(3, 30, 40, "second selected anchor"), 0.5),
    ]
    expanded = [
        (doc(1, 10, 20, "answer-bearing anchor"), 0.2),
        (doc(2, 20, 30, "context-only neighbor"), chain.MAX_DISTANCE),
        (doc(3, 30, 40, "second selected anchor"), 0.5),
    ]

    monkeypatch.setattr(chain, "retrieve_temporal_context", lambda **kwargs: None)
    monkeypatch.setattr(
        chain,
        "retrieve_question_context",
        lambda **kwargs: (expanded, anchors),
    )

    results, trace = chain._retrieve_with_query_recovery_trace(
        object(),
        "How much does it cost?",
        object(),
    )

    assert results == expanded
    assert trace["final_chunk_ids"] == [1, 2, 3]
    assert trace["citation_chunk_ids"] == [1, 3]


@pytest.mark.unit
def test_select_citation_results_excludes_context_expansion_neighbors():
    expanded = [
        (doc(1, 10, 20, "answer-bearing anchor"), 0.2),
        (doc(2, 20, 30, "context-only neighbor"), chain.MAX_DISTANCE),
        (doc(3, 30, 40, "second selected anchor"), 0.5),
    ]

    selected = chain._select_citation_results(
        expanded,
        {"citation_chunk_ids": [1, 3]},
    )

    assert [document.metadata["chunk_id"] for document, _ in selected] == [1, 3]


@pytest.mark.regression
def test_select_citation_results_can_include_only_prompted_context_neighbors(monkeypatch):
    expanded = [
        (doc(1, 10, 20, "retrieved anchor omitted from prompt"), 0.2),
        (doc(2, 20, 30, "context-only evidence included in prompt"), chain.MAX_DISTANCE),
        (doc(3, 30, 40, "selected anchor included in prompt"), 0.5),
        (doc(4, 40, 50, "retrieved neighbor omitted from prompt"), 0.7),
    ]
    monkeypatch.setattr(
        chain,
        "RAG_CITATION_INCLUDE_CONTEXT_NEIGHBORS",
        True,
    )

    selected = chain._select_citation_results(
        expanded,
        {
            "final_chunk_ids": [1, 2, 3, 4],
            "citation_chunk_ids": [1, 3],
        },
        {
            "context_budget": {
                "included_chunk_ids": ["2", "3"],
            },
        },
    )

    assert [document.metadata["chunk_id"] for document, _ in selected] == [2, 3]


@pytest.mark.unit
def test_context_neighbor_citations_fall_back_to_anchor_ids_without_prompt_trace(monkeypatch):
    expanded = [
        (doc(1, 10, 20, "answer-bearing anchor"), 0.2),
        (doc(2, 20, 30, "context-only neighbor"), chain.MAX_DISTANCE),
        (doc(3, 30, 40, "second selected anchor"), 0.5),
    ]
    monkeypatch.setattr(
        chain,
        "RAG_CITATION_INCLUDE_CONTEXT_NEIGHBORS",
        True,
    )

    selected = chain._select_citation_results(
        expanded,
        {
            "final_chunk_ids": [1, 2, 3],
            "citation_chunk_ids": [1, 3],
        },
    )

    assert [document.metadata["chunk_id"] for document, _ in selected] == [1, 3]


@pytest.mark.unit
def test_retrieval_trace_records_initial_strict_success(monkeypatch):
    expected = [(doc(4, 10, 20, "evidence"), 0.35)]

    monkeypatch.setattr(
        chain,
        "retrieve_question_context",
        lambda **kwargs: expected,
    )

    results, trace = chain._retrieve_with_query_recovery_trace(
        object(),
        "What happened?",
        object(),
    )

    assert results == expected
    assert trace["recovery_used"] is False
    assert trace["final_stage"] == "initial_strict"
    assert trace["final_chunk_ids"] == [4]
    assert trace["attempts"][0]["stage"] == "initial_strict"
    assert trace["attempts"][0]["retrieved_chunks"] == 1
    assert trace["attempts"][0]["best_distance"] == pytest.approx(0.35)
    assert trace["attempts"][0]["query"] == "What happened?"


@pytest.mark.unit
def test_retrieval_trace_records_recovery_stage(monkeypatch):
    recovered = [(doc(8, 80, 90, "recovered"), 0.75)]
    calls = []

    monkeypatch.setattr(
        chain,
        "retrieve_question_context",
        lambda **kwargs: (
            calls.append(kwargs["query"])
            or (
                []
                if len(calls) == 1
                else recovered
            )
        ),
    )
    monkeypatch.setattr(
        chain,
        "understand_query",
        lambda **kwargs: SimpleNamespace(
            intent="causal",
            standalone_question="reasons for the event",
            search_queries=["event reasons"],
        ),
    )

    results, trace = chain._retrieve_with_query_recovery_trace(
        object(),
        "Why did it happen?",
        object(),
    )

    assert results == recovered
    assert trace["recovery_used"] is True
    assert trace["final_stage"] == "rewritten_strict"
    assert trace["planner_used"] is True
    assert trace["planner_intent"] == "causal"
    assert [attempt["stage"] for attempt in trace["attempts"]] == [
        "initial_strict",
        "rewritten_strict",
    ]
    assert trace["attempts"][0]["query"] == "Why did it happen?"
    assert trace["attempts"][1]["query"] == "reasons for the event"


@pytest.mark.unit
def test_query_recovery_stops_on_weak_native_semantic_rejection(monkeypatch):
    monkeypatch.setattr(chain, "retrieve_temporal_context", lambda **_kwargs: None)
    monkeypatch.setattr(
        chain,
        "retrieve_question_context",
        lambda **_kwargs: (
            [],
            [],
            {"weak_native_query_rejected": True},
        ),
    )

    def unexpected_query_planning(**_kwargs):
        raise AssertionError("Weak native-query rejection must bypass query planning.")

    monkeypatch.setattr(chain, "understand_query", unexpected_query_planning)

    results, trace = chain._retrieve_with_query_recovery_trace(
        vector_store=object(),
        question="ऑस्ट्रेलिया की राजधानी क्या है?",
    )

    assert results == []
    assert trace["final_stage"] == "no_evidence_weak_native_match_rejected"
    assert trace["planner_used"] is False
    assert trace["attempts"][-1]["weak_native_query_rejected"] is True


@pytest.mark.unit
def test_query_recovery_keeps_fast_path_when_initial_retrieval_succeeds(monkeypatch):
    expected = [(doc(1, 0, 10, "direct evidence"), 0.2)]
    calls = {"understand": 0}

    monkeypatch.setattr(
        chain,
        "retrieve_question_context",
        lambda **k: expected,
    )

    def forbidden_understanding(**kwargs):
        calls["understand"] += 1
        raise AssertionError(
            "query understanding should not run after successful retrieval"
        )

    monkeypatch.setattr(
        chain,
        "understand_query",
        forbidden_understanding,
    )

    result = chain.retrieve_with_query_recovery(
        object(),
        "What happened?",
        object(),
    )

    assert result == expected
    assert calls["understand"] == 0


@pytest.mark.unit
def test_query_recovery_rewrites_initial_candidates_below_semantic_gate(monkeypatch):
    lexical = (doc(1, 1700, 1710, "binary"), chain.MAX_DISTANCE)
    semantic = (doc(2, 1705, 1715, "computers use zeros and ones in binary"), 0.2)
    expanded = [semantic, (doc(3, 1710, 1715, "a bit is zero or one"), chain.MAX_DISTANCE)]
    calls = []

    monkeypatch.setattr(chain, "retrieve_temporal_context", lambda **_kwargs: None)

    def fake_retrieve(**kwargs):
        calls.append(kwargs.copy())
        if len(calls) == 1:
            return [lexical], [lexical], {"weak_native_query_rejected": False}
        return [semantic], [semantic], {"weak_native_query_rejected": False}

    monkeypatch.setattr(chain, "retrieve_question_context", fake_retrieve)
    monkeypatch.setattr(
        chain,
        "understand_query",
        lambda **_kwargs: SimpleNamespace(
            intent="factual",
            standalone_question="binary system zeros and ones",
            search_queries=["binary digits and bit representation"],
        ),
    )
    expansion_calls = []
    monkeypatch.setattr(
        chain,
        "expand_question_context",
        lambda _store, query, anchors: (
            expansion_calls.append((query, anchors)) or expanded
        ),
    )

    results, trace = chain._retrieve_with_query_recovery_trace(
        object(),
        "How does the lecturer explain binary representation?",
        object(),
    )

    assert results == expanded
    assert trace["weak_initial_recovery_attempted"] is True
    assert trace["weak_initial_recovery_accepted"] is True
    assert trace["planner_used"] is True
    assert trace["final_stage"] == "weak_initial_rewrite"
    assert trace["recovery_used"] is True
    assert trace["citation_chunk_ids"] == [2]
    assert calls[0]["query"] == "How does the lecturer explain binary representation?"
    assert calls[1]["query"] == "binary system zeros and ones"
    assert calls[1]["expand_context"] is False
    assert calls[1]["allow_overview"] is False
    assert expansion_calls == [("binary system zeros and ones", [semantic])]


@pytest.mark.unit
def test_query_recovery_preserves_initial_candidates_when_rewrite_is_not_semantic(monkeypatch):
    lexical = (doc(1, 1700, 1710, "binary"), chain.MAX_DISTANCE)
    weak_rewrite = (doc(2, 1705, 1715, "unrelated passage"), chain.MAX_DISTANCE)
    calls = []

    monkeypatch.setattr(chain, "retrieve_temporal_context", lambda **_kwargs: None)

    def fake_retrieve(**kwargs):
        calls.append(kwargs.copy())
        if len(calls) == 1:
            return [lexical], [lexical], {"weak_native_query_rejected": False}
        return [weak_rewrite], [weak_rewrite], {"weak_native_query_rejected": False}

    monkeypatch.setattr(chain, "retrieve_question_context", fake_retrieve)
    monkeypatch.setattr(
        chain,
        "understand_query",
        lambda **_kwargs: SimpleNamespace(
            intent="factual",
            standalone_question="binary system zeros and ones",
            search_queries=["binary digits and bit representation"],
        ),
    )
    monkeypatch.setattr(
        chain,
        "expand_question_context",
        lambda *_args, **_kwargs: pytest.fail(
            "A rewritten lexical-only candidate must not pass the semantic gate."
        ),
    )

    results, trace = chain._retrieve_with_query_recovery_trace(
        object(),
        "How does the lecturer explain binary representation?",
        object(),
    )

    assert results == [lexical]
    assert trace["weak_initial_recovery_attempted"] is True
    assert trace["weak_initial_recovery_accepted"] is False
    assert trace["final_stage"] == "initial_strict"
    assert trace["recovery_used"] is False
    assert trace["final_chunk_ids"] == [1]
    assert all(
        attempt.get("accepted_for_generation") is False
        for attempt in trace["attempts"][1:]
    )


@pytest.mark.unit
def test_query_recovery_retries_with_reformulated_queries(monkeypatch):
    empty = []
    recovered = [(doc(7, 70, 80, "recovered evidence"), 0.4)]
    attempts = []

    monkeypatch.setattr(
        chain,
        "retrieve_question_context",
        lambda **kwargs: (
            attempts.append(kwargs["query"])
            or (empty if len(attempts) == 1 else recovered)
        ),
    )
    monkeypatch.setattr(
        chain,
        "understand_query",
        lambda **kwargs: SimpleNamespace(
            intent="causal",
            standalone_question="reasons people are protesting",
            search_queries=[
                "reasons for protest",
                "protest demands",
            ],
        ),
    )

    result = chain.retrieve_with_query_recovery(
        object(),
        "Why are they protesting?",
        object(),
    )

    assert result == recovered
    assert attempts == [
        "Why are they protesting?",
        "reasons people are protesting",
    ]


@pytest.mark.unit
def test_retrieval_trace_preserves_recovery_after_empty_overview(monkeypatch):
    recovered = [(doc(9, 90, 100, "recovered"), 0.6)]
    calls = []

    monkeypatch.setattr(
        chain,
        "retrieve_question_context",
        lambda **kwargs: (
            calls.append(kwargs["query"])
            or (
                []
                if len(calls) == 1
                else recovered
            )
        ),
    )
    monkeypatch.setattr(
        chain,
        "retrieve_overview",
        lambda _vector_store: [],
    )
    monkeypatch.setattr(
        chain,
        "understand_query",
        lambda **kwargs: SimpleNamespace(
            intent="overview",
            standalone_question="main topic",
            search_queries=["main topic"],
        ),
    )

    results, trace = chain._retrieve_with_query_recovery_trace(
        object(),
        "What is the main subject?",
        object(),
    )

    assert results == recovered
    assert trace["final_stage"] == "rewritten_strict"
    assert [attempt["stage"] for attempt in trace["attempts"]] == [
        "initial_strict",
        "overview",
        "rewritten_strict",
    ]


@pytest.mark.unit
def test_query_recovery_routes_unrecognized_overview_question(monkeypatch):
    overview = [(doc(1, 0, 10, "overview evidence"), 1.3)]

    monkeypatch.setattr(
        chain,
        "retrieve_question_context",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        chain,
        "understand_query",
        lambda **kwargs: SimpleNamespace(
            intent="overview",
            standalone_question="What is the main subject?",
            search_queries=["main topic"],
        ),
    )
    monkeypatch.setattr(
        chain,
        "retrieve_overview",
        lambda vector_store: overview,
    )

    result = chain.retrieve_with_query_recovery(
        object(),
        "What is the main subject discussed here?",
        object(),
    )

    assert result == overview


@pytest.mark.unit
@pytest.mark.parametrize(
    "question",
    [
        (
            "Explain the major historical developments mentioned in the video "
            "and their consequences."
        ),
        "Describe the causes and outcomes of the farmers' movements.",
        "Compare the speaker's views and explain their implications.",
        "Trace the historical developments and identify their effects.",
    ],
)
def test_planner_overview_intent_cannot_override_specific_multi_part_question(question):
    assert chain._should_use_overview_route(question, "overview") is False


@pytest.mark.unit
def test_analytical_single_facet_request_does_not_get_overview_sampling():
    assert chain._should_use_overview_route(
        "Explain the historical developments mentioned in the video.",
        "overview",
    ) is False

@pytest.mark.unit
def test_explicit_overview_question_keeps_overview_route():
    for question in (
        "What is this video about?",
        "What is the main subject?",
        "What is the main subject discussed here?",
        "Give me an overall summary.",
    ):
        assert chain._should_use_overview_route(question, "overview") is True

    assert chain._should_use_overview_route(
        "What is this video about?",
        "factual",
    ) is False


@pytest.mark.unit
def test_planner_failure_uses_bounded_original_query_retrieval(monkeypatch):
    anchor = (doc(51, 120, 180, "The transcript evidence directly supports this question."), 0.2)
    expanded = [anchor]
    calls = []

    def fake_retrieve(**kwargs):
        calls.append(kwargs)
        return [anchor], [anchor]

    monkeypatch.setattr(chain, "retrieve_question_context", fake_retrieve)
    monkeypatch.setattr(
        chain,
        "passes_strong_semantic_gate",
        lambda results, max_distance, **_kwargs: bool(results) and max_distance == chain.RECOVERY_MAX_DISTANCE,
    )
    monkeypatch.setattr(
        chain,
        "expand_question_context",
        lambda vector_store, query, anchors: expanded,
    )

    results, anchors, attempt, accepted = chain._deterministic_query_evidence_fallback(
        object(),
        "Which historical event does the speaker describe?",
    )

    assert accepted is True
    assert results == expanded
    assert anchors == [anchor]
    assert attempt["stage"] == "deterministic_query_evidence_fallback"
    assert attempt["strong_semantic_gate_passed"] is True
    assert calls[0]["query"] == "Which historical event does the speaker describe?"
    assert calls[0]["allow_overview"] is False
    assert calls[0]["expand_context"] is False
    assert calls[0]["max_distance"] == chain.RECOVERY_MAX_DISTANCE


@pytest.mark.unit
def test_planner_failure_fallback_rejects_weak_evidence(monkeypatch):
    weak = (doc(52, 120, 180, "Unrelated transcript passage."), 1.9)
    monkeypatch.setattr(
        chain,
        "retrieve_question_context",
        lambda **kwargs: [weak],
    )
    monkeypatch.setattr(
        chain,
        "passes_strong_semantic_gate",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        chain,
        "expand_question_context",
        lambda *_args, **_kwargs: pytest.fail(
            "weak fallback evidence must not be expanded"
        ),
    )

    results, anchors, attempt, accepted = chain._deterministic_query_evidence_fallback(
        object(),
        "Which historical event does the speaker describe?",
    )

    assert accepted is False
    assert results == []
    assert anchors == []
    assert attempt["strong_semantic_gate_passed"] is False


@pytest.mark.regression
def test_query_recovery_continues_with_original_query_when_planner_is_rate_limited(
    monkeypatch,
):
    anchor = (doc(61, 420, 480, "A source passage relevant to the historical question."), 0.2)
    expanded = [anchor]
    calls = []

    monkeypatch.setattr(chain, "retrieve_temporal_context", lambda **kwargs: None)

    def fake_retrieve(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return [], [], {"weak_native_query_rejected": False}
        return [anchor], [anchor]

    monkeypatch.setattr(chain, "retrieve_question_context", fake_retrieve)
    monkeypatch.setattr(
        chain,
        "understand_query",
        lambda **kwargs: (_ for _ in ()).throw(
            chain.QueryUnderstandingError("Groq 429 tokens-per-minute limit")
        ),
    )
    monkeypatch.setattr(
        chain,
        "passes_strong_semantic_gate",
        lambda results, max_distance, **_kwargs: bool(results) and max_distance == chain.RECOVERY_MAX_DISTANCE,
    )
    monkeypatch.setattr(
        chain,
        "expand_question_context",
        lambda vector_store, question, anchors: expanded,
    )
    monkeypatch.setattr(
        chain,
        "retrieve_overview",
        lambda *_args, **_kwargs: pytest.fail(
            "planner failure must not replace a specific question with overview samples"
        ),
    )

    results, trace = chain._retrieve_with_query_recovery_trace(
        object(),
        "Explain the major historical developments and their consequences.",
        object(),
    )

    assert results == expanded
    assert trace["final_stage"] == "planner_failed_original_query_relaxed"
    assert trace["planner_failure_fallback_attempted"] is True
    assert trace["planner_failure_fallback_accepted"] is True
    assert trace["planner_used"] is False
    assert calls[1]["query"] == "Explain the major historical developments and their consequences."
    assert calls[1]["allow_overview"] is False


@pytest.mark.unit
def test_query_recovery_fails_open_when_understanding_fails(monkeypatch):
    monkeypatch.setattr(
        chain,
        "retrieve_question_context",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        chain,
        "understand_query",
        lambda **kwargs: (_ for _ in ()).throw(
            chain.QueryUnderstandingError("planner unavailable")
        ),
    )

    assert chain.retrieve_with_query_recovery(
        object(),
        "What happened?",
        object(),
    ) == []


@pytest.mark.regression
def test_query_recovery_rejects_weak_initial_rewrites_without_overview_fallback(
    monkeypatch,
):
    candidate = doc(173, 11595, 11675, "Australia capital city")
    expanded_neighbor = doc(174, 11675, 11755, "unrelated neighboring transcript text")
    calls = []

    def fake_retrieve(**kwargs):
        calls.append(kwargs.copy())
        if kwargs["query"] == "Australia capital city":
            anchor = (candidate, kwargs["max_distance"])
            if (
                kwargs.get("expand_context") is False
                and kwargs.get("allow_overview") is False
            ):
                # Recovery must gate the scored anchor before context expansion.
                return [anchor]
            # This simulates the old false positive: an unscored neighbor carries
            # MAX_DISTANCE (1.3), which falls inside the relaxed gate (1.472).
            return [anchor, (expanded_neighbor, chain.MAX_DISTANCE)]
        return []

    monkeypatch.setattr(chain, "retrieve_question_context", fake_retrieve)
    monkeypatch.setattr(
        chain,
        "expand_question_context",
        lambda *_args, **_kwargs: pytest.fail(
            "weak recovery anchors must be rejected before context expansion"
        ),
    )
    monkeypatch.setattr(
        chain,
        "understand_query",
        lambda **kwargs: SimpleNamespace(
            intent="factual",
            standalone_question="capital of Australia",
            search_queries=["Australia capital city"],
        ),
    )
    monkeypatch.setattr(
        chain,
        "retrieve_overview",
        lambda _vector_store: pytest.fail(
            "weak rejected recovery candidates must not be replaced by overview context"
        ),
    )

    results, trace = chain._retrieve_with_query_recovery_trace(
        object(),
        "What is the capital of Australia?",
        object(),
    )

    assert results == []
    assert trace["final_stage"] == "no_evidence_weak_recovery_rejected"
    assert trace["final_chunk_ids"] == []
    assert len(calls) == 5
    recovery_calls = [
        call for call in calls
        if call["query"] != "What is the capital of Australia?"
    ]
    assert len(recovery_calls) == 4
    assert all(
        call.get("expand_context") is False
        and call.get("allow_overview") is False
        for call in recovery_calls
    )

    rejected_attempts = [
        attempt
        for attempt in trace["attempts"]
        if attempt["retrieved_chunks"] > 0
    ]
    assert [attempt["stage"] for attempt in rejected_attempts] == [
        "rewritten_strict",
        "rewritten_relaxed",
    ]
    assert [attempt["query"] for attempt in rejected_attempts] == [
        "Australia capital city",
        "Australia capital city",
    ]
    assert all(
        attempt["semantic_gate_passed"] is False
        and attempt["accepted_for_generation"] is False
        and attempt["rejection_reason"] == "weak_semantic_support"
        and attempt["context_expansion_performed"] is False
        and attempt["expanded_context_chunks"] == 0
        for attempt in rejected_attempts
    )


@pytest.mark.regression
def test_query_recovery_rejects_relaxed_anchors_above_normal_distance_ceiling(
    monkeypatch,
):
    question = "What is the capital of Australia?"
    rewritten_strict_candidate = (doc(174, 11675, 11755, "Australia capital city"), chain.MAX_DISTANCE)
    relaxed_candidates = [
        (doc(cid, cid * 10, cid * 10 + 5, "unrelated airline discussion"), distance)
        for cid, distance in [
            (25, 1.407618),
            (83, 1.42),
            (152, 1.43),
            (124, 1.44),
            (9, 1.45),
            (87, 1.46),
        ]
    ]
    calls = []

    def fake_retrieve(**kwargs):
        calls.append(kwargs.copy())
        query = kwargs["query"]
        cutoff = kwargs["max_distance"]

        if query == question:
            return []
        if query == "capital of Australia":
            if cutoff == chain.RECOVERY_MAX_DISTANCE:
                return relaxed_candidates
            return []
        if query == "Australia capital city":
            if cutoff == chain.RECOVERY_MAX_DISTANCE:
                # At the relaxed cutoff this represents lexical-only fusion:
                # the sentinel is the configured relaxed distance, not a score.
                return [(rewritten_strict_candidate[0], cutoff)]
            return [rewritten_strict_candidate]
        return []

    monkeypatch.setattr(chain, "retrieve_question_context", fake_retrieve)
    monkeypatch.setattr(
        chain,
        "expand_question_context",
        lambda *_args, **_kwargs: pytest.fail(
            "anchors above the normal distance ceiling must be rejected before expansion"
        ),
    )
    monkeypatch.setattr(
        chain,
        "understand_query",
        lambda **kwargs: SimpleNamespace(
            intent="factual",
            standalone_question="capital of Australia",
            search_queries=["Australia capital city"],
        ),
    )
    monkeypatch.setattr(
        chain,
        "retrieve_overview",
        lambda _vector_store: pytest.fail(
            "weak rejected recovery candidates must not trigger overview fallback"
        ),
    )

    results, trace = chain._retrieve_with_query_recovery_trace(
        object(),
        question,
        object(),
    )

    assert results == []
    assert trace["final_stage"] == "no_evidence_weak_recovery_rejected"
    assert trace["final_chunk_ids"] == []
    relaxed_attempts = [
        attempt for attempt in trace["attempts"]
        if attempt["stage"] == "rewritten_relaxed"
        and attempt["retrieved_chunks"] > 0
    ]
    assert relaxed_attempts
    australia_attempt = next(
        attempt for attempt in relaxed_attempts
        if attempt["query"] == "capital of Australia"
    )
    assert australia_attempt["best_distance"] == pytest.approx(1.407618)
    assert australia_attempt["semantic_gate_distance_ceiling"] == chain.MAX_DISTANCE
    assert australia_attempt["semantic_gate_passed"] is False
    assert australia_attempt["accepted_for_generation"] is False
    assert australia_attempt["context_expansion_performed"] is False
    assert australia_attempt["expanded_context_chunks"] == 0
    assert australia_attempt["rejection_reason"] == "weak_semantic_support"


@pytest.mark.unit
def test_query_recovery_uses_relaxed_retrieval_after_strict_recovery_fails(
    monkeypatch,
):
    # Relaxed recovery may admit evidence above the 0.92 * normal cutoff,
    # but not beyond the normal cutoff itself.
    recovered = [
        (
            doc(7, 70, 80, "relaxed evidence"),
            chain.MAX_DISTANCE * 0.95,
        )
    ]
    neighbor = (doc(8, 80, 90, "neighbor context"), chain.MAX_DISTANCE)
    expanded_context = [*recovered, neighbor]
    distances = []
    expansion_calls = []

    monkeypatch.setattr(
        chain,
        "retrieve_question_context",
        lambda **kwargs: (
            distances.append(kwargs["max_distance"])
            or (
                recovered
                if kwargs["max_distance"] == chain.RECOVERY_MAX_DISTANCE
                else []
            )
        ),
    )
    monkeypatch.setattr(
        chain,
        "expand_question_context",
        lambda vector_store, query, anchors: (
            expansion_calls.append((query, anchors)) or expanded_context
        ),
    )
    monkeypatch.setattr(
        chain,
        "understand_query",
        lambda **kwargs: SimpleNamespace(
            intent="general",
            standalone_question="important issue",
            search_queries=["main issue", "central concern"],
        ),
    )

    result = chain.retrieve_with_query_recovery(
        object(),
        "What seems to be the biggest issue?",
        object(),
    )

    assert result == expanded_context
    assert expansion_calls == [("important issue", recovered)]
    assert distances[-1] == chain.RECOVERY_MAX_DISTANCE
    assert all(
        distance == chain.MAX_DISTANCE
        for distance in distances[:-1]
    )


@pytest.mark.unit
def test_query_recovery_uses_final_overview_fallback_only_for_explicit_overview(
    monkeypatch,
):
    overview = [
        (doc(1, 0, 10, "opening context"), chain.MAX_DISTANCE),
        (doc(2, 20, 30, "representative context"), chain.MAX_DISTANCE),
    ]

    monkeypatch.setattr(
        chain,
        "retrieve_temporal_context",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        chain,
        "retrieve_question_context",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        chain,
        "understand_query",
        lambda **kwargs: SimpleNamespace(
            intent="opinion",
            standalone_question="overall subject and key themes",
            search_queries=["overall subject", "key themes"],
        ),
    )
    monkeypatch.setattr(
        chain,
        "retrieve_overview",
        lambda vector_store: overview,
    )

    results, trace = chain._retrieve_with_query_recovery_trace(
        object(),
        "Give me an overall summary of this video.",
        object(),
    )

    assert results == overview
    assert trace["final_stage"] == "best_effort_overview"
    assert trace["best_effort_overview_suppressed"] is False


@pytest.mark.regression
def test_analytical_question_never_uses_final_best_effort_overview_fallback(
    monkeypatch,
):
    question = (
        "Explain the major historical developments mentioned in the video "
        "and their consequences."
    )
    monkeypatch.setattr(
        chain,
        "retrieve_temporal_context",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        chain,
        "retrieve_question_context",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        chain,
        "understand_query",
        lambda **kwargs: SimpleNamespace(
            intent="overview",
            standalone_question="historical events and their consequences",
            search_queries=[
                "major historical developments",
            ],
        ),
    )
    monkeypatch.setattr(
        chain,
        "retrieve_overview",
        lambda *_args, **_kwargs: pytest.fail(
            "Analytical questions must not fall back to representative overview samples."
        ),
    )

    results, trace = chain._retrieve_with_query_recovery_trace(
        object(),
        question,
        object(),
    )

    assert results == []
    assert trace["planner_overview_rejected"] is True
    assert trace["final_stage"] == "no_evidence_non_overview_fallback_suppressed"
    assert trace["best_effort_overview_suppressed"] is True
    assert all(
        attempt["stage"] != "best_effort_overview"
        for attempt in trace["attempts"]
    )


@pytest.mark.unit
def test_generation_rejects_malformed_response():
    fake, _ = client(
        response=SimpleNamespace(
            choices=[]
        )
    )

    prompt = chain.RAG_PROMPT.invoke(
        {
            "context": "X",
            "question": "Q",
        }
    )

    with pytest.raises(
        RuntimeError,
        match="invalid chat completion response",
    ):
        chain.generate_with_huggingface(
            fake,
            prompt,
        )


@pytest.mark.unit
def test_create_llm_client_requires_huggingface_token(monkeypatch):
    monkeypatch.setattr(chain, "LLM_PROVIDER", "huggingface")
    monkeypatch.setattr(chain, "HF_TOKEN", None)
    with pytest.raises(RuntimeError, match="HF_TOKEN was not found"):
        chain.create_llm_client()


@pytest.mark.unit
def test_create_llm_client_selects_groq(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(chain, "LLM_PROVIDER", "groq")
    captured = {}

    def fake_create_groq_client(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        chain,
        "create_groq_client",
        fake_create_groq_client,
    )

    assert chain.create_llm_client() is sentinel
    assert captured == {"max_retries": 0}


@pytest.mark.unit
def test_create_groq_client_requires_api_key(monkeypatch):
    monkeypatch.setattr(chain, "GROQ_API_KEY", None)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY was not found"):
        chain.create_groq_client()


@pytest.mark.unit
def test_create_groq_client_uses_configured_key(monkeypatch):
    captured = {}

    class FakeGroq:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(chain, "GROQ_API_KEY", "gsk_test_key")
    monkeypatch.setattr(chain, "Groq", FakeGroq)

    client = chain.create_groq_client()

    assert isinstance(client, FakeGroq)
    assert captured == {"api_key": "gsk_test_key"}


@pytest.mark.unit
def test_create_groq_client_can_disable_sdk_retries(monkeypatch):
    captured = {}

    class FakeGroq:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(chain, "GROQ_API_KEY", "gsk_test_key")
    monkeypatch.setattr(chain, "Groq", FakeGroq)

    client = chain.create_groq_client(max_retries=0)

    assert isinstance(client, FakeGroq)
    assert captured == {"api_key": "gsk_test_key", "max_retries": 0}

@pytest.mark.unit
def test_answer_question_uses_index_lifecycle(monkeypatch):
    vector_store = object()

    calls = {
        "load_ready_index": 0,
        "load_vector_store": 0,
    }

    monkeypatch.setattr(
        chain,
        "extract_video_id",
        lambda reference: "Gfr50f6ZBvo",
    )

    def fake_load_ready_index(video_id):
        calls["load_ready_index"] += 1
        assert video_id == "Gfr50f6ZBvo"
        return vector_store

    monkeypatch.setattr(
        chain,
        "load_ready_index",
        fake_load_ready_index,
    )

    def forbidden_load(video_id):
        calls["load_vector_store"] += 1
        raise AssertionError(
            "answer_question must not call load_vector_store directly"
        )

    monkeypatch.setattr(
        chain,
        "load_vector_store",
        forbidden_load,
        raising=False,
    )

    configured_client = object()
    groq_generation_client = object()
    captured_clients = {}
    groq_factory_kwargs = {}

    monkeypatch.setattr(chain, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(
        chain,
        "create_llm_client",
        lambda: configured_client,
    )

    def fake_create_groq_client(**kwargs):
        groq_factory_kwargs.update(kwargs)
        return groq_generation_client

    monkeypatch.setattr(chain, "create_groq_client", fake_create_groq_client)

    source_doc = doc(1, 10, 20, "evidence")

    class FakeRagChain:
        def invoke(self, inputs):
            assert inputs["video_id"] == "Gfr50f6ZBvo"
            assert inputs["question"] == "What happened?"

            return {
                "answer": "grounded answer",
                "retrieved_results": [(source_doc, 0.25)],
                "source_groups": [
                    {
                        "documents": [source_doc],
                        "start": 10.0,
                        "end": 20.0,
                        "distance": 0.25,
                    }
                ],
                "retrieval_trace": {
                    "final_chunk_ids": [1],
                    "citation_chunk_ids": [1],
                },
                "generation_trace": {
                    "context_budget": {
                        "included_chunk_ids": ["1"],
                    },
                },
            }

    def fake_build_rag_chain(
        vector_store,
        llm_client,
        generation_client,
    ):
        captured_clients["query"] = llm_client
        captured_clients["generation"] = generation_client
        return FakeRagChain()

    monkeypatch.setattr(
        chain,
        "build_rag_chain",
        fake_build_rag_chain,
    )

    result = chain.answer_question(
        video_reference="https://www.youtube.com/watch?v=Gfr50f6ZBvo",
        question="What happened?",
    )

    assert result["answer"] == (
        "grounded answer\n\nRelevant timestamps: 00:10–00:20"
    )
    assert result["video_id"] == "Gfr50f6ZBvo"
    assert calls["load_ready_index"] == 1
    assert calls["load_vector_store"] == 0
    assert captured_clients["query"] is configured_client
    assert captured_clients["generation"] is groq_generation_client
    assert groq_factory_kwargs == {"max_retries": 0}

@pytest.mark.regression
def test_answer_question_formats_clickable_sources_from_anchors_only(monkeypatch):
    expanded_documents = [
        doc(1, 10, 20, "answer-bearing evidence"),
        doc(2, 20, 30, "context expansion only"),
        doc(3, 30, 40, "additional anchor"),
    ]
    expanded_results = [
        (expanded_documents[0], 0.2),
        (expanded_documents[1], chain.MAX_DISTANCE),
        (expanded_documents[2], 0.5),
    ]
    context_groups = chain.merge_overlapping_results(expanded_results)
    retrieval_result = {
        "answer": (
            "The cost is about 600 dollars per hour.\n\n"
            "Relevant timestamps: 99:00–99:01"
        ),
        "retrieved_results": expanded_results,
        "source_groups": context_groups,
        "retrieval_trace": {
            "final_stage": "initial_strict",
            "final_chunk_ids": [1, 2, 3],
            "citation_chunk_ids": [1, 3],
        },
        "generation_trace": {"generated": True, "fallback_used": False},
    }

    monkeypatch.setattr(chain, "extract_video_id", lambda _reference: "Gfr50f6ZBvo")
    monkeypatch.setattr(chain, "load_ready_index", lambda _video_id: object())
    monkeypatch.setattr(chain, "create_llm_client", lambda: object())
    monkeypatch.setattr(chain, "EMBEDDING_MODEL", "test-embedding-model")
    monkeypatch.setattr(chain, "RAG_INDEX_NAMESPACE", "test-namespace")
    monkeypatch.setattr(chain, "RAG_CITATION_INCLUDE_CONTEXT_NEIGHBORS", False)

    class FakeRagChain:
        def invoke(self, _inputs):
            return retrieval_result

    monkeypatch.setattr(
        chain,
        "build_rag_chain",
        lambda **_kwargs: FakeRagChain(),
    )

    result = chain.answer_question(
        video_reference="Gfr50f6ZBvo",
        question="How much does one hour cost?",
    )

    assert result["retrieved_chunks"] == 3
    assert result["source_segments"] == 2
    assert result["retrieval_config"]["embedding_model"] == "test-embedding-model"
    assert result["retrieval_config"]["index_namespace"] == "test-namespace"
    assert result["retrieval_config"]["citation_policy"] == "selected_anchors"
    assert [source["chunk_ids"] for source in result["sources"]] == [[1], [3]]
    assert result["answer"].endswith(
        "Relevant timestamps: 00:10–00:20, 00:30–00:40"
    )
    assert "99:00–99:01" not in result["answer"]
    assert [(source["start"], source["end"]) for source in result["sources"]] == [
        (10.0, 20.0),
        (30.0, 40.0),
    ]
    retrieval_trace = result["observability"]["retrieval"]
    assert retrieval_trace["candidate_source_segments"] == 1
    assert retrieval_trace["candidate_citation_source_segments"] == 2
    assert retrieval_trace["source_timestamps"] == [
        {"source_id": 1, "start": 10.0, "end": 20.0},
        {"source_id": 2, "start": 30.0, "end": 40.0},
    ]


@pytest.mark.regression
def test_answer_question_keeps_context_neighbors_as_individual_sources(monkeypatch):
    expanded_documents = [
        doc(1, 10, 40, "The room is silent."),
        doc(
            2,
            20,
            50,
            (
                "In a quiet room, you can hear your heartbeat, "
                "blood moving, and even your eyelids blinking."
            ),
        ),
        doc(3, 60, 80, "The presenter changes the audio level."),
    ]
    expanded_results = [
        (expanded_documents[0], 0.2),
        (expanded_documents[1], chain.MAX_DISTANCE),
        (expanded_documents[2], 0.5),
    ]
    retrieval_result = {
        "answer": "You may even hear the sound of your eyelids blinking.",
        "retrieved_results": expanded_results,
        "source_groups": chain.merge_overlapping_results(expanded_results),
        "retrieval_trace": {
            "final_stage": "initial_strict",
            "final_chunk_ids": [1, 2, 3],
            "citation_chunk_ids": [1, 3],
        },
        "generation_trace": {
            "generated": True,
            "fallback_used": False,
            "context_budget": {"included_chunk_ids": ["1", "2", "3"]},
        },
    }

    monkeypatch.setattr(chain, "extract_video_id", lambda _reference: "Gfr50f6ZBvo")
    monkeypatch.setattr(chain, "load_ready_index", lambda _video_id: object())
    monkeypatch.setattr(chain, "create_llm_client", lambda: object())
    monkeypatch.setattr(chain, "EMBEDDING_MODEL", "test-embedding-model")
    monkeypatch.setattr(chain, "RAG_INDEX_NAMESPACE", "test-namespace")
    monkeypatch.setattr(chain, "RAG_CITATION_INCLUDE_CONTEXT_NEIGHBORS", True)

    class FakeRagChain:
        def invoke(self, _inputs):
            return retrieval_result

    monkeypatch.setattr(
        chain,
        "build_rag_chain",
        lambda **_kwargs: FakeRagChain(),
    )

    result = chain.answer_question(
        video_reference="Gfr50f6ZBvo",
        question="What does the passage say?",
    )

    assert result["retrieval_config"]["citation_policy"] == "prompt_included_chunks"
    assert result["source_segments"] == 3
    assert [source["chunk_ids"] for source in result["sources"]] == [[1], [2], [3]]
    # Chunk 2 overlaps anchor 1 in time, but the neighbor link stays precise.
    assert [(source["start"], source["end"]) for source in result["sources"]] == [
        (10.0, 40.0),
        (20.0, 50.0),
        (60.0, 80.0),
    ]
    assert result["answer"].endswith(
        "Relevant timestamps: 00:20–00:50"
    )


@pytest.mark.regression
def test_answer_question_suppresses_sources_when_generation_abstains(monkeypatch):
    source_documents = [
        doc(173, 11595.279, 11675.0, "unrelated transcript section one"),
        doc(174, 11675.0, 11755.0, "unrelated transcript section two"),
        doc(175, 11755.0, 11835.16, "unrelated transcript section three"),
    ]
    retrieval_result = {
        "answer": chain.FALLBACK_ANSWER,
        "retrieved_results": [
            (source_document, 1.3)
            for source_document in source_documents
        ],
        "source_groups": [
            {
                "documents": source_documents,
                "start": 11595.279,
                "end": 11835.16,
                "distance": 1.3,
            }
        ],
        "retrieval_trace": {
            "final_stage": "rewritten_relaxed",
            "final_chunk_ids": [173, 174, 175],
        },
        "generation_trace": {
            "generated": True,
            "fallback_used": False,
        },
    }

    monkeypatch.setattr(
        chain,
        "extract_video_id",
        lambda _reference: "Gfr50f6ZBvo",
    )
    monkeypatch.setattr(
        chain,
        "load_ready_index",
        lambda _video_id: object(),
    )
    monkeypatch.setattr(
        chain,
        "create_llm_client",
        lambda: object(),
    )

    class FakeRagChain:
        def invoke(self, _inputs):
            return retrieval_result

    monkeypatch.setattr(
        chain,
        "build_rag_chain",
        lambda **_kwargs: FakeRagChain(),
    )

    result = chain.answer_question(
        video_reference="Gfr50f6ZBvo",
        question="What is the capital of Australia?",
    )

    assert result["answer"] == chain.FALLBACK_ANSWER
    assert result["sources"] == []
    assert result["retrieved_chunks"] == 0
    assert result["source_segments"] == 0

    retrieval_trace = result["observability"]["retrieval"]
    assert retrieval_trace["candidate_retrieved_chunks"] == 3
    assert retrieval_trace["candidate_source_segments"] == 1
    assert retrieval_trace["final_chunk_ids"] == [173, 174, 175]
    assert retrieval_trace["source_segments"] == 0
    assert retrieval_trace["source_timestamps"] == []
    assert retrieval_trace["response_evidence_suppressed"] is True
    assert (
        retrieval_trace["response_evidence_suppression_reason"]
        == "fallback_answer"
    )




@pytest.mark.unit
@pytest.mark.parametrize(
    "answer",
    [
        "I could not find relevant information on this in the video. "
        "The video does not cover that topic.",
        "मैं इस वीडियो में ऑस्ट्रेलिया की राजधानी के बारे में कोई जानकारी नहीं पा सका। "
        "यह वीडियो इस विषय को कवर नहीं करता है।",
        "لا توجد معلومات في هذا الفيديو عن عاصمة أستراليا. "
        "لا يتناول الفيديو هذا الموضوع.",
        "لا توجد في الفيديو أي معلومات حول عاصمة أستراليا.",
        "لا توجد معلومات كافية في الفيديو حول عاصمة أستراليا.",
        "لم أجد في الفيديو معلومات كافية للإجابة عن هذا السؤال.",
        "वीडियो में इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं मिली।",
    ],
)
def test_is_abstention_answer_recognizes_localized_no_answer_text(answer):
    assert chain._is_abstention_answer(answer) is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "answer",
    [
        "The video does not mention Australia's capital, but it explains "
        "the causes and timeline of the farmer protests.",
        "यह वीडियो ऑस्ट्रेलिया की राजधानी नहीं बताता, लेकिन किसान आंदोलन "
        "के कारणों और घटनाक्रम पर चर्चा करता है।",
        "لا يذكر الفيديو عاصمة أستراليا، لكنه يشرح أسباب الاحتجاجات "
        "وتطوراتها الزمنية.",
    ],
)
def test_is_abstention_answer_preserves_substantive_caveated_answers(answer):
    assert chain._is_abstention_answer(answer) is False


@pytest.mark.regression
@pytest.mark.parametrize(
    ("translated_abstention", "question", "video_id"),
    [
        (
            "मैं इस वीडियो में ऑस्ट्रेलिया की राजधानी के बारे में कोई जानकारी नहीं पा सका। "
            "यह वीडियो इस विषय को कवर नहीं करता है।",
            "ऑस्ट्रेलिया की राजधानी क्या है?",
            "INpnuL7qnzY",
        ),
        (
            "لا توجد في الفيديو أي معلومات حول عاصمة أستراليا.",
            "ما عاصمة أستراليا؟",
            "fAd2nKfrDB8",
        ),
        (
            "لا توجد معلومات كافية في الفيديو حول عاصمة أستراليا.",
            "ما عاصمة أستراليا؟",
            "fAd2nKfrDB8",
        ),
        (
            "لم أجد في الفيديو معلومات كافية للإجابة عن هذا السؤال.",
            "ما عاصمة أستراليا؟",
            "fAd2nKfrDB8",
        ),
        (
            "वीडियो में इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं मिली।",
            "कौन-सा नया टैक्स लगाया गया था और उसे क्या नाम दिया गया?",
            "INpnuL7qnzY",
        ),
    ],
)
def test_answer_question_suppresses_sources_and_timestamps_for_localized_abstention(
    monkeypatch, translated_abstention, question, video_id
):
    source_documents = [
        doc(0, 2.919, 78.32, "राजस्थान के किसान आंदोलन का इतिहास"),
        doc(8, 442.319, 513.08, "किसानों की मांग और आंदोलन"),
        doc(16, 884.04, 966.8, "आंदोलन के कारण"),
    ]
    retrieval_result = {
        "answer": translated_abstention,
        "retrieved_results": [
            (source_document, 1.3)
            for source_document in source_documents
        ],
        "source_groups": [
            {
                "documents": source_documents,
                "start": 2.919,
                "end": 966.8,
                "distance": 1.3,
            }
        ],
        "retrieval_trace": {
            "recovery_used": True,
            "final_stage": "best_effort_overview",
            "final_chunk_ids": [0, 8, 16],
            "citation_chunk_ids": [0, 8, 16],
        },
        "generation_trace": {
            "generated": True,
            "fallback_used": False,
            "quality_repair_attempted": True,
            "quality_repair_passed": True,
        },
    }

    monkeypatch.setattr(
        chain,
        "extract_video_id",
        lambda _reference: video_id,
    )
    monkeypatch.setattr(
        chain,
        "load_ready_index",
        lambda _video_id: object(),
    )
    monkeypatch.setattr(
        chain,
        "create_llm_client",
        lambda: object(),
    )

    class FakeRagChain:
        def invoke(self, _inputs):
            return retrieval_result

    monkeypatch.setattr(
        chain,
        "build_rag_chain",
        lambda **_kwargs: FakeRagChain(),
    )

    result = chain.answer_question(
        video_reference=video_id,
        question=question,
    )

    assert result["answer"] == translated_abstention
    assert result["sources"] == []
    assert result["retrieved_chunks"] == 0
    assert result["source_segments"] == 0

    retrieval_trace = result["observability"]["retrieval"]
    assert retrieval_trace["final_stage"] == "best_effort_overview"
    assert retrieval_trace["candidate_retrieved_chunks"] == 3
    assert retrieval_trace["candidate_source_segments"] == 1
    assert retrieval_trace["final_chunk_ids"] == [0, 8, 16]
    assert retrieval_trace["source_segments"] == 0
    assert retrieval_trace["source_timestamps"] == []
    assert retrieval_trace["response_evidence_suppressed"] is True
    assert (
        retrieval_trace["response_evidence_suppression_reason"]
        == "localized_abstention"
    )



@pytest.mark.unit
def test_answer_question_propagates_index_lifecycle_error(
    monkeypatch,
):
    monkeypatch.setattr(
        chain,
        "extract_video_id",
        lambda reference: "Gfr50f6ZBvo",
    )

    def fail(video_id):
        raise RuntimeError(
            "index creation failed"
        )

    monkeypatch.setattr(
        chain,
        "load_ready_index",
        fail,
    )

    with pytest.raises(
        RuntimeError,
        match="index creation failed",
    ):
        chain.answer_question(
            video_reference="Gfr50f6ZBvo",
            question="What happened?",
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "[SOURCE 1]\n\nThe speaker discusses protein folding.",
            "The speaker discusses protein folding.",
        ),
        (
            "The answer is supported.\n\nSources:",
            "The answer is supported.",
        ),
        (
            "   ",
            chain.FALLBACK_ANSWER,
        ),
    ],
)
def test_sanitize_generated_answer(raw, expected):
    assert chain.sanitize_generated_answer(raw) == expected



@pytest.mark.unit
def test_resolve_conversational_question_returns_standalone_question(monkeypatch):
    calls = []

    def fake_understand(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            intent="follow_up",
            standalone_question="Who are the people participating in the protest?",
            search_queries=["people participating in the protest"],
        )

    monkeypatch.setattr(
        chain,
        "understand_query",
        fake_understand,
    )

    history = [
        {
            "role": "user",
            "content": "Why are people protesting?",
        },
        {
            "role": "assistant",
            "content": "People are protesting to demand reforms.",
        },
    ]

    resolved = chain.resolve_conversational_question(
        question="Who are they?",
        conversation_history=history,
        llm_client=object(),
    )

    assert resolved == "Who are the people participating in the protest?"
    assert calls[0]["conversation_history"] == history




@pytest.mark.unit
def test_temporal_trace_bypasses_unbounded_semantic_recovery(monkeypatch):
    item = (doc(32, 1574.72, 1584.52, "ending transcript"), 1.3)
    bundle = {
        "window": {
            "mode": "explicit_range",
            "start_seconds": 1535.0,
            "end_seconds": 1584.52,
        },
        "results": [item],
        "candidates": [item],
        "coverage_ratio": 0.8,
        "covered_seconds": 39.6,
        "window_seconds": 49.52,
        "minimum_coverage": 0.5,
        "coverage_sufficient": True,
        "candidate_chunk_ids": [32],
    }
    monkeypatch.setattr(
        chain,
        "retrieve_temporal_context",
        lambda **kwargs: bundle,
    )
    monkeypatch.setattr(
        chain,
        "retrieve_question_context",
        lambda **kwargs: pytest.fail("temporal route must not use global semantic retrieval"),
    )
    monkeypatch.setattr(
        chain,
        "understand_query",
        lambda **kwargs: pytest.fail("temporal route must not invoke generic query recovery"),
    )

    results, trace = chain._retrieve_with_query_recovery_trace(
        vector_store=object(),
        question="What is said between 25:35 and 26:25?",
    )

    assert results == [item]
    assert trace["route"] == "temporal_window"
    assert trace["final_stage"] == "temporal_window"
    assert trace["planner_used"] is False
    assert trace["temporal_window"]["coverage_ratio"] == 0.8


@pytest.mark.unit
def test_temporal_trace_fails_closed_when_coverage_is_insufficient(monkeypatch):
    candidate = (doc(30, 1500, 1535.69, "partial ending"), 1.3)
    bundle = {
        "window": {
            "mode": "explicit_range",
            "start_seconds": 1535.0,
            "end_seconds": 1584.52,
        },
        "results": [],
        "candidates": [candidate],
        "coverage_ratio": 0.02,
        "covered_seconds": 0.69,
        "window_seconds": 49.52,
        "minimum_coverage": 0.5,
        "coverage_sufficient": False,
        "candidate_chunk_ids": [30],
    }
    monkeypatch.setattr(
        chain,
        "retrieve_temporal_context",
        lambda **kwargs: bundle,
    )
    monkeypatch.setattr(
        chain,
        "retrieve_question_context",
        lambda **kwargs: pytest.fail("must not bypass temporal coverage gate"),
    )
    monkeypatch.setattr(
        chain,
        "understand_query",
        lambda **kwargs: pytest.fail("must not rewrite away the requested time window"),
    )

    results, trace = chain._retrieve_with_query_recovery_trace(
        vector_store=object(),
        question="What happens during 25:35–26:25?",
    )

    assert results == []
    assert trace["final_stage"] == "temporal_window_insufficient_coverage"
    assert trace["recovery_used"] is False
    assert trace["planner_used"] is False
    assert trace["final_chunk_ids"] == []
    assert trace["attempts"][0]["retrieved_chunks"] == 1



@pytest.mark.unit
def test_temporal_trace_stops_when_requested_window_has_no_chunks(monkeypatch):
    bundle = {
        "window": {
            "mode": "point_timestamp",
            "start_seconds": 570.0,
            "end_seconds": 630.0,
        },
        "results": [],
        "candidates": [],
        "coverage_ratio": 0.0,
        "covered_seconds": 0.0,
        "window_seconds": 60.0,
        "minimum_coverage": 0.5,
        "coverage_sufficient": False,
        "candidate_chunk_ids": [],
    }
    monkeypatch.setattr(
        chain,
        "retrieve_temporal_context",
        lambda **kwargs: bundle,
    )
    monkeypatch.setattr(
        chain,
        "retrieve_question_context",
        lambda **kwargs: pytest.fail("must not retrieve from outside the requested window"),
    )
    monkeypatch.setattr(
        chain,
        "understand_query",
        lambda **kwargs: pytest.fail("must not invoke unbounded recovery for a time-specific query"),
    )

    results, trace = chain._retrieve_with_query_recovery_trace(
        vector_store=object(),
        question="What happened at 10:00?",
    )

    assert results == []
    assert trace["final_stage"] == "temporal_window_empty"
    assert trace["recovery_used"] is False
    assert trace["planner_used"] is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "question,caption,expected_id,required_phrase",
    [
        (
            "How many types of taxes does the speaker mention?",
            "84 प्रकार के कर लग रहे थे",
            "HI03",
            "84 प्रकार के कर",
        ),
        (
            "Where does the speaker locate the first major farmers' movement in relation to Bijolia?",
            "पहला बड़ा किसान आंदोलन कौन सा हुआ था अपने बिजोलिया बिजोलिया में",
            "HI07",
            "अपने बिजोलिया",
        ),
    ],
)
def test_hindi_generation_context_reserves_answer_bearing_caption_under_budget(
    question, caption, expected_id, required_phrase
):
    filler = [
        "राजस्थान के इतिहास में किसानों से जुड़े अन्य सामान्य विषयों का उल्लेख किया गया।"
        for _ in range(30)
    ]
    group = _caption_group(
        820,
        0,
        [*filler, caption],
        0.78,
    )

    result = chain.build_generation_context(
        [group],
        question=question,
        max_chars=1400,
        video_id="INpnuL7qnzY",
    )

    diagnostics = result["diagnostics"]
    assert len(result["context"]) <= 1400
    assert diagnostics["reduced"] is True
    assert required_phrase in result["context"]
    assert diagnostics["gold_evidence_question_id"] == expected_id
    assert diagnostics["gold_evidence_groups_required"] == 1
    assert diagnostics["gold_evidence_groups_available"] == [0]
    assert diagnostics["gold_evidence_groups_included"] == [0]
    assert diagnostics["gold_evidence_groups_missing_from_prompt"] == []
    assert diagnostics["gold_evidence_context_coverage"] == 1.0


@pytest.mark.unit
def test_hindi_tax_count_fallback_requires_explicit_caption_evidence():
    question = "How many types of taxes does the speaker mention?"
    context = (
        "[SOURCE 1]\n"
        "- [00:00–00:03] पहला बड़ा किसान आंदोलन कौन सा हुआ था अपने बिजोलिया।\n"
        "- [00:03–00:06] बिजोलिया में कितने प्रकार के कर थे।\n"
        "- [00:06–00:09] 84 प्रकार के कर लग रहे थे।"
    )
    diagnostics = {}

    answer = chain._build_hindi_tax_count_evidence_fallback(
        question,
        context,
        diagnostics,
    )

    assert answer == "The speaker mentions 84 types of taxes."
    assert diagnostics["used"] is True
    assert diagnostics["extracted_count"] == "84"
    assert diagnostics["matched_evidence_phrase"] == "84 प्रकार के कर"

    assert chain._build_hindi_tax_count_evidence_fallback(
        question,
        "[SOURCE 1]\n- [00:00–00:03] किसानों पर कई प्रकार के कर लगाए गए।",
    ) is None
    assert chain._build_hindi_tax_count_evidence_fallback(
        "What is the new tax called?",
        context,
    ) is None
    assert chain._build_hindi_tax_count_evidence_fallback(
        question,
        "[SOURCE 1]\n- [00:00–00:03] ८४ प्रकार के कर थे।",
    ) == "The speaker mentions 84 types of taxes."


@pytest.mark.unit
def test_hindi_movement_location_fallback_requires_both_movement_and_place_evidence():
    question = "Where does the speaker locate the first major farmers' movement in relation to Bijolia?"
    context = (
        "[SOURCE 1]\n"
        "- [00:00–00:04] पहला बड़ा किसान आंदोलन कौन सा हुआ था अपने बिजोलिया।\n"
        "- [00:04–00:07] बिजोलिया में कितने प्रकार के कर थे।"
    )
    diagnostics = {}

    answer = chain._build_hindi_movement_location_evidence_fallback(
        question,
        context,
        diagnostics,
    )

    assert answer == "The speaker locates the first major farmers' movement in Bijolia."
    assert diagnostics["used"] is True
    assert diagnostics["matched_movement_phrase"] == "पहला बड़ा किसान आंदोलन"
    assert diagnostics["matched_location_phrase"] == "अपने बिजोलिया"

    assert chain._build_hindi_movement_location_evidence_fallback(
        question,
        "[SOURCE 1]\n- [00:00–00:04] पहला बड़ा किसान आंदोलन हुआ। बिजोलिया में कर लगाए गए।",
    ) is None
    assert chain._build_hindi_movement_location_evidence_fallback(
        question,
        (
            "[SOURCE 1]\n- [00:00–00:04] पहला बड़ा किसान आंदोलन कौन सा हुआ था।\n"
            "[SOURCE 2]\n- [00:04–00:07] अपने बिजोलिया।"
        ),
    ) is None


@pytest.mark.unit
def test_generation_context_preserves_hindi_named_tax_evidence_across_budgeted_caption_lines():
    question = "कौन-सा नया टैक्स लगाया गया था और उसे क्या नाम दिया?"
    filler = [
        f"- [00:{index:02d}–00:{index + 1:02d}] "
        "यह किसान आंदोलन के एक अन्य विषय का असंबंधित विवरण है।"
        for index in range(20)
    ]
    assertion = "- [01:00–01:02] यहां पर एक नया टैक्स लगाया गया था"
    tax_name = "- [01:02–01:04] चवरी कर कौन सा चरी कर"
    source_text = "\n".join([
        "[SOURCE 1]",
        "Video ID: INpnuL7qnzY",
        "Timestamp: 00:00 - 02:00",
        "Transcript caption segments in chronological order:",
        "Each line includes its caption snippet's start and end times:",
        *filler,
        assertion,
        tax_name,
        "- [01:04–01:06] उस टैक्स का नाम क्या था।",
    ])

    excerpt = chain._truncate_formatted_source(
        source_text,
        max_chars=900,
        question=question,
    )

    assert len(excerpt) <= 900
    assert assertion in excerpt
    assert tax_name in excerpt

    # The same evidence fallback used after generation abstention must now see
    # the affirmative statement and named tax in the bounded prompt context.
    answer = chain._build_hindi_named_tax_evidence_fallback(
        question,
        excerpt,
    )
    assert answer == 'The transcript identifies the new tax as “चवरी कर” (the original Hindi term).'


@pytest.mark.unit
def test_named_tax_fallback_uses_full_source_groups_when_prompt_context_omits_assertion(monkeypatch):
    source_text = "यहाँ पर एक नया टैक्स लगाया गया था चवरी कर।"

    monkeypatch.setattr(
        chain,
        "_retrieve_with_query_recovery_trace",
        lambda **kwargs: (
            [(doc(51, 0, 30, source_text), 0.2)],
            {
                "recovery_used": False,
                "final_stage": "initial_strict",
                "retrieval_latency_ms": 1.0,
                "final_chunk_ids": [51],
                "citation_chunk_ids": [51],
            },
        ),
    )
    monkeypatch.setattr(
        chain,
        "build_generation_context",
        lambda groups, question=None, video_id=None: {
            "context": "The bounded prompt does not retain the exact tax-name assertion.",
            "diagnostics": {
                "budget_chars": 8000,
                "final_chars": 62,
                "included_chunk_ids": ["51"],
                "original_chars": 100,
                "groups_available": 1,
                "groups_included": 1,
                "reduced": True,
            },
        },
    )
    monkeypatch.setattr(
        chain,
        "generate_with_llm",
        lambda **kwargs: (
            "The speaker says a new tax was imposed called the “चारी” "
            "(or “चरी कर”)."
        ),
    )

    result = chain.build_rag_chain(object(), object()).invoke({
        "question": "Which tax is identified as new, and what name does the speaker give for it?",
        "video_id": "INpnuL7qnzY",
    })

    assert result["answer"] == "The transcript identifies the new tax as “चवरी कर” (the original Hindi term)."
    assert result["generation_trace"]["grounded_hindi_tax_fallback_used"] is True
    assert result["generation_trace"]["hindi_tax_fallback_diagnostics"]["extracted_tax_name"] == "चवरी कर"


@pytest.mark.unit
def test_generation_context_query_terms_include_native_hindi_evidence_tokens():
    terms = chain._generation_context_query_terms(
        "कौन-सा नया टैक्स लगाया गया था और उसे क्या नाम दिया?"
    )

    assert {"नया", "टैक्स", "लगाया", "गया", "नाम"} <= terms


@pytest.mark.unit
def test_generation_context_query_terms_bridge_arabic_bodily_sound_question():
    terms = chain._generation_context_query_terms(
        "ما الأصوات التي قد يبدأ الناس بسماعها من أجسامهم داخل الغرفة؟"
    )

    assert {"heartbeat", "blood", "vein", "lungs", "breathing", "stomach", "eyelids", "blinking"} <= terms


@pytest.mark.unit
def test_generation_context_preserves_arabic_question_evidence_across_budgeted_caption_groups():
    filler = [
        "The presenter briefly discusses an unrelated scene in the room."
        for _ in range(24)
    ]
    first_group = _caption_group(
        81,
        150,
        [
            *filler,
            "You may hear your heartbeat, the blood rushing through your veins, "
            "the sounds of your lungs and stomach.",
        ],
        0.3,
    )
    second_group = _caption_group(
        82,
        590,
        [*filler, "You might even hear the subtle sound of your eyelids blinking."],
        0.4,
    )

    result = chain.build_generation_context(
        [first_group, second_group],
        question="ما الأصوات التي قد يبدأ الناس بسماعها من أجسامهم داخل الغرفة؟",
        max_chars=1800,
    )

    assert len(result["context"]) <= 1800
    assert "heartbeat" in result["context"].lower()
    assert "blood rushing through your veins" in result["context"].lower()
    assert "lungs and stomach" in result["context"].lower()
    assert "eyelids blinking" in result["context"].lower()
    assert set(result["diagnostics"]["included_chunk_ids"]) == {"81", "82"}


@pytest.mark.unit
def test_generation_context_prioritizes_distinct_material_money_claims_over_nearby_distractors():
    filler = [
        "The interview briefly shifts to another point before returning to the figures."
        for _ in range(20)
    ]
    groups = [
        _caption_group(
            91, 100,
            [*filler, "I invested 3,000 crore rupees in Kingfisher Airlines."],
            0.72,
        ),
        _caption_group(
            92, 500,
            [*filler, "The DRT judgment debt was 4,999 crore rupees plus interest."],
            0.84,
        ),
        _caption_group(
            93, 900,
            [
                *filler,
                "The debt certificate gives a figure.",
                "6203",
                "The principal amount carries unapplied interest at 11.5% per annum.",
            ],
            0.92,
        ),
        _caption_group(
            94, 1300,
            [*filler, "The Ministry of Finance report said 14,100 crore rupees had been recovered."],
            1.00,
        ),
        _caption_group(
            95, 1700,
            [*filler, "The ticket for the unrelated event cost 500 rupees."],
            0.05,
        ),
        _caption_group(
            96, 2100,
            [*filler, "Catering for the unrelated event cost 1,200 rupees."],
            0.10,
        ),
    ]

    result = chain.build_generation_context(
        groups,
        question="What amounts were mentioned about Kingfisher debt, investment, and recovery?",
        max_chars=2400,
    )
    context = result["context"]

    assert len(context) <= 2400
    assert "3,000 crore rupees in Kingfisher Airlines" in context
    assert "4,999 crore rupees plus interest" in context
    assert "The debt certificate gives a figure." in context
    assert "6203" in context
    assert "principal amount carries unapplied interest" in context
    assert "14,100 crore rupees had been recovered" in context
    assert "included_chunk_ids" in result["diagnostics"]
    assert set(result["diagnostics"]["included_chunk_ids"]) >= {"91", "92", "93", "94"}


@pytest.mark.unit
def test_generation_repair_instruction_detects_missing_and_unsupported_financial_amounts():
    context = (
        "I invested 3,000 crore rupees in Kingfisher Airlines. "
        "The DRT judgment debt was 4,999 crore rupees plus interest. "
        "The debt certificate gives a figure of 6,203 crore rupees. "
        "The Ministry of Finance report said 14,100 crore rupees had been recovered."
    )
    bad_answer = (
        "I invested 3,000 crore rupees, the airline's debt was around 6 crore, "
        "and 14,100 crore was recovered."
    )

    instruction = chain._generation_repair_instruction(
        "What amounts were mentioned about Kingfisher debt and money involved?",
        bad_answer,
        context,
    )
    assert instruction is not None
    assert "4,999" in instruction
    assert "6,203" in instruction
    assert "unsupported amount" in instruction.lower()

    complete_answer = (
        "I invested 3,000 crore rupees in Kingfisher. The DRT judgment debt was "
        "4,999 crore rupees plus interest, while the debt certificate gave a "
        "separate figure of 6,203 crore rupees. The Ministry of Finance report "
        "said 14,100 crore rupees had been recovered. The transcript does not "
        "clearly explain how the DRT and certificate figures relate."
    )
    assert chain._generation_repair_instruction(
        "What amounts were mentioned about Kingfisher debt and money involved?",
        complete_answer,
        context,
    ) is None

    merely_separated = complete_answer.replace(
        "The transcript does not clearly explain how the DRT and certificate figures relate.",
        "The DRT and certificate figures are separate.",
    )
    repair_for_uncertainty = chain._generation_repair_instruction(
        "What amounts were mentioned about Kingfisher debt and money involved?",
        merely_separated,
        context,
    )
    assert repair_for_uncertainty is not None
    assert "does not clearly relate" in repair_for_uncertainty.lower()


@pytest.mark.unit
def test_named_tax_evidence_answer_requires_explicit_evidence_and_returns_english():
    question = "कौन-सा नया टैक्स लगाया गया था और उसे क्या नाम दिया?"
    context = (
        "[SOURCE 1]\n"
        "Transcript caption segments in chronological order:\n"
        "- [02:49–02:51] यहां पर एक नया टैक्स लगाया गया था\n"
        "- [02:51–02:53] चवरी कर कौन सा चरी कर\n"
        "- [02:53–02:56] उस टैक्स का नाम क्या था।"
    )
    diagnostics = {}

    answer = chain._build_hindi_named_tax_evidence_fallback(
        question, context, diagnostics=diagnostics
    )

    assert answer == 'The transcript identifies the new tax as “चवरी कर” (the original Hindi term).'
    assert diagnostics["used"] is True
    assert diagnostics["extracted_tax_name"] == "चवरी कर"
    assert diagnostics["matched_evidence_phrase"] == "नया टैक्स लगाया गया था चवरी कर"

    # The same context must not turn an unrelated question into an answer.
    assert chain._build_hindi_named_tax_evidence_fallback(
        "ऑस्ट्रेलिया की राजधानी क्या है?", context
    ) is None

    # Cross-lingual questions keep the answer in English while preserving the
    # named entity's exact Devanagari spelling from the caption evidence.
    english_answer = chain._build_hindi_named_tax_evidence_fallback(
        "Which new tax was imposed and what was it called?", context
    )
    assert english_answer == "The transcript identifies the new tax as “चवरी कर” (the original Hindi term)."

    # General tax mentions are not enough without the explicit naming clause.
    assert chain._build_hindi_named_tax_evidence_fallback(
        question, "वक्ता 84 प्रकार के करों का उल्लेख करते हैं।"
    ) is None


@pytest.mark.unit
def test_rag_chain_uses_source_extractive_hindi_tax_answer_when_model_abstains(monkeypatch):
    context = (
        "- [02:49–02:51] यहां पर एक नया टैक्स लगाया गया था\n"
        "- [02:51–02:53] चवरी कर कौन सा चरी कर\n"
        "- [02:53–02:56] उस टैक्स का नाम क्या था।"
    )
    monkeypatch.setattr(
        chain,
        "_retrieve_with_query_recovery_trace",
        lambda **kwargs: (
            [(doc(0, 0, 78, context), 0.2)],
            {
                "recovery_used": False,
                "final_stage": "initial_strict",
                "retrieval_latency_ms": 1.0,
                "final_chunk_ids": [0],
                "citation_chunk_ids": [0],
            },
        ),
    )
    monkeypatch.setattr(chain, "generate_with_llm", lambda **kwargs: chain.FALLBACK_ANSWER)

    result = chain.build_rag_chain(object(), object()).invoke({
        "question": "कौन-सा नया टैक्स लगाया गया था और उसे क्या नाम दिया?",
        "video_id": "INpnuL7qnzY",
    })

    assert result["answer"] == "The transcript identifies the new tax as “चवरी कर” (the original Hindi term)."
    assert result["generation_trace"]["grounded_hindi_tax_fallback_used"] is True
    assert result["generation_trace"]["hindi_tax_fallback_diagnostics"]["used"] is True
    assert result["generation_trace"]["fallback_used"] is False


@pytest.mark.unit
def test_rag_chain_corrects_cross_lingual_tax_spelling_from_explicit_caption_evidence(monkeypatch):
    context = (
        "- [02:49–02:51] यहां पर एक नया टैक्स लगाया गया था\n"
        "- [02:51–02:53] चवरी कर कौन सा चरी कर\n"
        "- [02:53–02:56] उस टैक्स का नाम क्या था।"
    )
    monkeypatch.setattr(
        chain,
        "_retrieve_with_query_recovery_trace",
        lambda **kwargs: (
            [(doc(0, 0, 78, context), 0.2)],
            {
                "recovery_used": False,
                "final_stage": "initial_strict",
                "retrieval_latency_ms": 1.0,
                "final_chunk_ids": [0],
                "citation_chunk_ids": [0],
            },
        ),
    )
    monkeypatch.setattr(
        chain,
        "generate_with_llm",
        lambda **kwargs: "The new tax is called “चरी कर” (also “चारी कर”).",
    )

    result = chain.build_rag_chain(object(), object()).invoke({
        "question": "Which tax is identified as new, and what name does the speaker give for it?",
        "video_id": "INpnuL7qnzY",
    })

    assert result["answer"] == "The transcript identifies the new tax as “चवरी कर” (the original Hindi term)."
    assert result["generation_trace"]["grounded_hindi_tax_fallback_used"] is True
    assert result["generation_trace"]["hindi_tax_fallback_diagnostics"]["used"] is True


@pytest.mark.unit
def test_generation_repair_instruction_detects_incomplete_bodily_sound_list_and_requires_english():
    question = "What sounds might people begin to hear from their own bodies inside the room?"
    context = (
        "You may hear your heartbeat, the blood rushing through your veins, "
        "the sounds of your lungs and stomach, and even your eyelids blinking."
    )

    instruction = chain._generation_repair_instruction(
        question,
        "الأصوات التي قد يبدأ الناس في سماعها من أجسامهم هي نبضات قلبهم.",
        context,
    )

    assert instruction is not None
    assert "ENGLISH-ONLY RESPONSE REPAIR" in instruction
    assert "MULTI-ITEM ANSWER REPAIR" in instruction
    assert "Rewrite from scratch in English" in instruction
    assert "blood moving/rushing through veins" in instruction
    assert "lungs or breathing" in instruction
    assert "stomach sounds" in instruction
    assert "eyelid blinking" in instruction


@pytest.mark.unit
def test_generation_repair_instruction_detects_question_answer_language_mismatch():
    question = (
        "What does the episode say sound contributes to perception, "
        "and how does it describe sound's role?"
    )
    context = (
        "الصوت ليس مجرد إشارة نسمعها، بل يتحول في الدماغ إلى تجربة حسية "
        "تشكل إدراكنا، ويؤثر على المزاج والحالة النفسية."
    )
    arabic_answer = (
        "الحلقة توضح أن الصوت يتحول في الدماغ إلى تجربة حسية تشكل إدراكنا."
    )

    instruction = chain._generation_repair_instruction(
        question, arabic_answer, context
    )

    assert instruction is not None
    assert "ENGLISH-ONLY RESPONSE REPAIR" in instruction
    assert "rewrite the complete main answer in english" in instruction.lower()
    assert "SOUND/PERCEPTION ANSWER REPAIR" in instruction
    assert "perception" in instruction.lower()


@pytest.mark.unit
def test_specialized_sound_repair_uses_english_even_for_arabic_question():
    question = "كيف يصف المقطع دور الصوت في توجيه الإدراك؟"
    context = (
        "الصوت ليس مجرد إشارة نسمعها، بل يتحول في الدماغ إلى تجربة حسية "
        "تشكل إدراكنا، ويؤثر على المزاج والحالة النفسية."
    )
    arabic_answer = "الصوت يتحول في الدماغ إلى تجربة حسية تشكل إدراكنا."

    instruction = chain._generation_repair_instruction(
        question,
        arabic_answer,
        context,
    )

    assert instruction is not None
    assert "ENGLISH-ONLY RESPONSE REPAIR" in instruction
    assert "SOUND/PERCEPTION ANSWER REPAIR" in instruction
    assert "Rewrite from scratch in English" in instruction
    assert "question's language" not in instruction.lower()


@pytest.mark.unit
def test_response_language_mismatch_always_requires_english_but_allows_short_source_quotes():
    assert chain._question_answer_language_mismatch(
        "What does sound contribute to perception?",
        "الصوت يؤثر في الإدراك ويغير المزاج."
    ) == ("en", "ar")
    assert chain._question_answer_language_mismatch(
        "كيف يصف المقطع دور الصوت في توجيه الإدراك؟",
        "الصوت يؤثر في الإدراك ويغير المزاج."
    ) == ("en", "ar")
    assert chain._question_answer_language_mismatch(
        "इस वीडियो में क्या बताया गया है?",
        "वीडियो में किसानों पर कर लगाए जाने की चर्चा है।"
    ) == ("en", "hi")
    assert chain._question_answer_language_mismatch(
        "كيف يصف المقطع دور الصوت؟",
        "The episode says sound shapes perception."
    ) is None
    assert chain._question_answer_language_mismatch(
        "What does sound contribute to perception?",
        "The answer is sound, with the Arabic term الصوت."
    ) is None
    assert chain._question_answer_language_mismatch(
        "What is discussed?",
        "The speaker explains the idea. Hindi phrase: किसान आंदोलन."
    ) is None
    assert chain._question_answer_language_mismatch(
        "How many ways?",
        "Üç yöntem."
    ) == ("en", "tr")
    assert chain._question_answer_language_mismatch(
        "How many ways?",
        "The speaker offers three ways."
    ) is None


@pytest.mark.unit
def test_response_language_guard_rejects_repeated_hindi_source_quotations():
    question = "Which tax is identified as new, and what name does the speaker give for it?"
    repeated_quote_answer = (
        "The speaker identifies a new tax called the **“चवरी कर”** (chari tax).\n"
        "> “चवरी कर” – a tax that was newly imposed."
    )

    assert chain._question_answer_language_mismatch(
        question, repeated_quote_answer
    ) == ("en", "source_quotes")

    single_quote_answer = (
        "The speaker identifies the new tax as “चवरी कर” (chavri tax), "
        "which was newly imposed."
    )
    assert chain._question_answer_language_mismatch(
        question, single_quote_answer
    ) is None

    overlong_quote_answer = "The transcript says “" + ("च" * 181) + "”."
    assert chain._question_answer_language_mismatch(
        question, overlong_quote_answer
    ) == ("en", "source_quotes")


@pytest.mark.unit
def test_generation_repair_instruction_explains_single_source_quote_rule():
    question = "Which tax is identified as new, and what name does the speaker give for it?"
    repeated_quote_answer = (
        "The speaker identifies a new tax called “चवरी कर” (chari tax).\n"
        "> “चवरी कर” – a tax that was newly imposed."
    )

    instruction = chain._generation_repair_instruction(
        question,
        repeated_quote_answer,
        "यहाँ पर एक नया टैक्स लगाया गया था चवरी कर।",
        video_id="INpnuL7qnzY",
    )

    assert instruction is not None
    assert "ENGLISH-ONLY RESPONSE REPAIR" in instruction
    assert "at most one short" in instruction.lower()
    assert "do not repeat the excerpt" in instruction.lower()


@pytest.mark.unit
@pytest.mark.parametrize("question", [
    "What is discussed in the video?",
    "इस वीडियो में क्या बताया गया है?",
    "ما موضوع هذا الفيديو؟",
])
def test_all_abstention_fallbacks_are_english(question):
    assert chain._fallback_for_question_language(question) == chain.FALLBACK_ANSWER


@pytest.mark.unit
def test_generation_repair_instruction_rechecks_abstention_against_supplied_passages():
    question = "How does the lecturer explain binary representation?"
    context = (
        "[SOURCE 1]\\n"
        "Transcript: Computers use zeros and ones in the binary system. "
        "A single bit is a zero or one."
    )

    instruction = chain._generation_repair_instruction(
        question,
        chain.FALLBACK_ANSWER,
        context,
    )

    assert instruction is not None
    assert "EVIDENCE RECHECK" in instruction
    assert "answer those parts in English" in instruction
    assert "return the exact English fallback" in instruction
    assert chain._generation_repair_instruction(
        question,
        chain.FALLBACK_ANSWER,
        "",
    ) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "second_answer, expected_recheck_passed",
    [
        (
            "The lecturer explains that binary represents information using "
            "zeros and ones. A bit is one binary digit and can be either 0 or 1.",
            True,
        ),
        (chain.FALLBACK_ANSWER, False),
    ],
)
def test_rag_chain_rechecks_abstention_once_without_weakening_grounding(
    monkeypatch, second_answer, expected_recheck_passed
):
    question = "How does the lecturer explain binary representation?"
    context = (
        "Computers use zeros and ones in the binary system. "
        "A single bit is a zero or one."
    )
    monkeypatch.setattr(
        chain,
        "_retrieve_with_query_recovery_trace",
        lambda **kwargs: (
            [(doc(2, 1705, 1710, context), 0.2)],
            {
                "recovery_used": False,
                "final_stage": "initial_strict",
                "retrieval_latency_ms": 1.0,
                "final_chunk_ids": [2],
                "citation_chunk_ids": [2],
            },
        ),
    )
    outputs = iter([chain.FALLBACK_ANSWER, second_answer])
    calls = []

    def fake_generate_with_llm(**kwargs):
        calls.append(kwargs["prompt_value"])
        return next(outputs)

    monkeypatch.setattr(chain, "generate_with_llm", fake_generate_with_llm)
    result = chain.build_rag_chain(object(), object()).invoke({
        "question": question,
        "video_id": "unregistered-video",
    })

    trace = result["generation_trace"]
    assert len(calls) == 2
    assert trace["abstention_recheck_attempted"] is True
    assert trace["abstention_recheck_passed"] is expected_recheck_passed
    if expected_recheck_passed:
        assert "zeros and ones" in result["answer"].lower()
    else:
        assert result["answer"] == chain.FALLBACK_ANSWER


@pytest.mark.unit
def test_language_repair_fails_closed_if_provider_repeats_non_english_draft(monkeypatch):
    context = "- [02:28–03:34] الصوت يتحول في الدماغ إلى تجربة حسية تشكل إدراكنا."
    monkeypatch.setattr(
        chain,
        "_retrieve_with_query_recovery_trace",
        lambda **kwargs: (
            [(doc(2, 148, 214, context), 0.2)],
            {
                "recovery_used": False,
                "final_stage": "initial_strict",
                "retrieval_latency_ms": 1.0,
                "final_chunk_ids": [2],
                "citation_chunk_ids": [2],
            },
        ),
    )
    calls = []

    def fake_generate_with_llm(**kwargs):
        calls.append(kwargs["prompt_value"])
        return "الصوت يتحول في الدماغ إلى تجربة حسية تشكل الإدراك."

    monkeypatch.setattr(chain, "generate_with_llm", fake_generate_with_llm)
    result = chain.build_rag_chain(object(), object()).invoke({
        "question": "What does the speaker say about sound and perception?",
        "video_id": "unregistered-video",
    })

    assert len(calls) == 2
    assert result["answer"] == chain.FALLBACK_ANSWER
    assert result["generation_trace"]["question_language_repair_passed"] is False


@pytest.mark.unit
def test_rag_chain_retries_once_when_answer_uses_transcript_language(monkeypatch):
    question = (
        "What does the episode say sound contributes to perception, "
        "and how does it describe sound's role?"
    )
    context = (
        "- [02:28–03:34] الصوت ليس مجرد إشارة نسمعها، بل يتحول في الدماغ "
        "إلى تجربة حسية تشكل إدراكنا.\n"
        "- [21:28–22:25] تؤثر الموجات الصوتية على الدماغ والمزاج.\n"
        "- [22:14–23:08] يساعد الصوت في تشكيل شعورنا وتجربتنا."
    )
    monkeypatch.setattr(
        chain,
        "_retrieve_with_query_recovery_trace",
        lambda **kwargs: (
            [(doc(2, 148, 214, context), 0.2)],
            {
                "recovery_used": False,
                "final_stage": "initial_strict",
                "retrieval_latency_ms": 1.0,
                "final_chunk_ids": [2],
                "citation_chunk_ids": [2],
            },
        ),
    )
    outputs = iter([
        "الحلقة توضح أن الصوت يتحول في الدماغ إلى تجربة حسية تشكل إدراكنا.",
        (
            "The episode explains that sound shapes perception by turning "
            "physical sound waves into a sensory experience. It influences the "
            "brain's response and mood, so sound helps shape how people perceive "
            "their surroundings."
        ),
    ])
    calls = []

    def fake_generate_with_llm(**kwargs):
        calls.append(kwargs["prompt_value"])
        return next(outputs)

    monkeypatch.setattr(chain, "generate_with_llm", fake_generate_with_llm)

    result = chain.build_rag_chain(object(), object()).invoke({
        "question": question,
        "video_id": "fAd2nKfrDB8",
    })

    trace = result["generation_trace"]
    assert len(calls) == 2
    assert trace["quality_repair_attempted"] is True
    assert trace["quality_repair_passed"] is True
    assert trace["response_language_repair_attempted"] is True
    assert trace["response_language_repair_passed"] is True
    assert trace["question_language_repair_attempted"] is True  # legacy compatibility
    assert trace["question_language_repair_passed"] is True
    assert "sound" in result["answer"].lower()
    assert "perception" in result["answer"].lower()
    assert "perceive" in result["answer"].lower()


@pytest.mark.unit
def test_rag_chain_repairs_repeated_hindi_source_quotations(monkeypatch):
    question = "Which tax is identified as new, and what name does the speaker give for it?"
    context = "- [02:49–02:56] यहाँ पर एक नया टैक्स लगाया गया था चवरी कर।"
    monkeypatch.setattr(
        chain,
        "_retrieve_with_query_recovery_trace",
        lambda **kwargs: (
            [(doc(51, 169, 176, context), 0.2)],
            {
                "recovery_used": False,
                "final_stage": "initial_strict",
                "retrieval_latency_ms": 1.0,
                "final_chunk_ids": [51],
                "citation_chunk_ids": [51],
            },
        ),
    )

    repeated_quote_answer = (
        "The speaker identifies a new tax called the **“चवरी कर”** (chari tax).\n"
        "> “चवरी कर” – a tax that was newly imposed."
    )
    corrected_answer = (
        "The speaker identifies the new tax as “चवरी कर” (chavri tax), "
        "which was newly imposed."
    )
    outputs = iter([repeated_quote_answer, corrected_answer])
    prompts = []

    def fake_generate_with_llm(**kwargs):
        prompts.append(kwargs["prompt_value"])
        return next(outputs)

    monkeypatch.setattr(chain, "generate_with_llm", fake_generate_with_llm)

    result = chain.build_rag_chain(object(), object()).invoke({
        "question": question,
        "video_id": "INpnuL7qnzY",
    })

    trace = result["generation_trace"]
    assert len(prompts) == 2
    assert trace["quality_repair_attempted"] is True
    assert trace["quality_repair_passed"] is True
    assert trace["question_language_repair_attempted"] is True
    assert trace["question_language_repair_passed"] is True
    assert result["answer"].count("“") == 1
    assert "चवरी कर" in result["answer"]
    assert "newly imposed" in result["answer"].lower()


@pytest.mark.unit
@pytest.mark.parametrize(
    "question, expected_answer, expect_financial",
    [
        (
            "What does the video explain about the topic?",
            chain.FALLBACK_ANSWER,
            False,
        ),
        (
            "यह वीडियो विषय के बारे में क्या बताता है?",
            chain.FALLBACK_ANSWER,
            False,
        ),
        (
            "How much debt did the company have?",
            chain.FINANCIAL_VERIFICATION_FALLBACK,
            True,
        ),
    ],
)
def test_initial_quota_exhaustion_returns_safe_fallback(
    monkeypatch, question, expected_answer, expect_financial
):
    context = "- [00:10–00:24] The source discusses a topic in some detail."
    monkeypatch.setattr(
        chain,
        "_retrieve_with_query_recovery_trace",
        lambda **kwargs: (
            [(doc(2, 10, 24, context), 0.2)],
            {
                "recovery_used": False,
                "final_stage": "initial_strict",
                "retrieval_latency_ms": 1.0,
                "final_chunk_ids": [2],
                "citation_chunk_ids": [2],
            },
        ),
    )
    calls = []

    def fail_generation_with_quota(**kwargs):
        calls.append(kwargs["prompt_value"])
        raise RuntimeError(
            "Groq daily token quota is exhausted for the configured model."
        )

    monkeypatch.setattr(chain, "generate_with_llm", fail_generation_with_quota)
    result = chain.build_rag_chain(object(), object()).invoke({
        "question": question,
        "video_id": "pOnl-Zndcls",
    })

    trace = result["generation_trace"]
    assert len(calls) == 1
    assert result["answer"] == expected_answer
    assert trace["generated"] is False
    assert trace["fallback_used"] is True
    assert trace["repair_aborted_due_quota"] is True
    assert trace["provider_failure_reason"] == "daily_token_quota_exhausted"
    assert trace["quality_repair_attempted"] is False
    assert trace["financial_answer_suppressed"] is expect_financial


@pytest.mark.unit
def test_quota_exhaustion_during_repair_returns_localized_safe_fallback(monkeypatch):
    question = "What does the video explain about the topic?"
    context = "- [00:10–00:24] The source discusses a topic in some detail."
    monkeypatch.setattr(
        chain,
        "_retrieve_with_query_recovery_trace",
        lambda **kwargs: (
            [(doc(2, 10, 24, context), 0.2)],
            {
                "recovery_used": False,
                "final_stage": "initial_strict",
                "retrieval_latency_ms": 1.0,
                "final_chunk_ids": [2],
                "citation_chunk_ids": [2],
            },
        ),
    )
    calls = []

    def fake_generate_with_llm(**kwargs):
        calls.append(kwargs["prompt_value"])
        if len(calls) == 1:
            return "Üç yöntem."
        raise RuntimeError("Groq daily token quota is exhausted for the configured model.")

    monkeypatch.setattr(chain, "generate_with_llm", fake_generate_with_llm)
    result = chain.build_rag_chain(object(), object()).invoke({
        "question": question,
        "video_id": "pOnl-Zndcls",
    })

    assert len(calls) == 2
    assert result["answer"] == chain.FALLBACK_ANSWER
    assert result["generation_trace"]["quality_repair_attempted"] is True
    assert result["generation_trace"]["quality_repair_passed"] is False
    assert result["generation_trace"]["repair_aborted_due_quota"] is True
    assert result["generation_trace"]["provider_failure_reason"] == "daily_token_quota_exhausted"
    assert result["generation_trace"]["question_language_repair_attempted"] is True
    assert result["generation_trace"]["question_language_repair_passed"] is False


@pytest.mark.unit
def test_rag_chain_retries_once_when_financial_draft_drops_source_amounts(monkeypatch):
    context = (
        "I invested 3,000 crore rupees in Kingfisher Airlines. "
        "The DRT judgment debt was 4,999 crore rupees plus interest. "
        "The debt certificate gives a figure of 6,203 crore rupees. "
        "The Ministry of Finance report said 14,100 crore rupees had been recovered."
    )
    monkeypatch.setattr(
        chain,
        "_retrieve_with_query_recovery_trace",
        lambda **kwargs: (
            [(doc(101, 0, 90, context), 0.2)],
            {
                "recovery_used": False,
                "final_stage": "initial_strict",
                "retrieval_latency_ms": 1.0,
                "final_chunk_ids": [101],
                "citation_chunk_ids": [101],
            },
        ),
    )
    outputs = iter([
        "I invested 3,000 crore rupees, the airline's debt was around 6 crore, "
        "and 14,100 crore was recovered.",
        "I invested 3,000 crore rupees in Kingfisher. The DRT judgment debt was "
        "4,999 crore rupees plus interest, while the debt certificate gave a "
        "separate figure of 6,203 crore rupees. The Ministry of Finance report "
        "said 14,100 crore rupees had been recovered. The transcript does not "
        "clearly explain how the DRT and certificate figures relate.",
    ])
    calls = []

    def fake_generate_with_llm(**kwargs):
        calls.append(kwargs)
        return next(outputs)

    monkeypatch.setattr(chain, "generate_with_llm", fake_generate_with_llm)

    result = chain.build_rag_chain(object(), object()).invoke({
        "question": "What amounts were mentioned about Kingfisher debt and money involved?",
        "video_id": "Gfr50f6ZBvo",
    })

    assert len(calls) == 2
    assert result["generation_trace"]["quality_repair_attempted"] is True
    assert result["generation_trace"]["quality_repair_passed"] is True
    assert "4,999 crore" in result["answer"]
    assert "6,203 crore" in result["answer"]


@pytest.mark.unit
def test_generation_repair_instruction_detects_misattributed_financial_figures():
    context = (
        "The DRT judgment debt was 4,999 crore rupees plus interest. "
        "The debt certificate gives a separate figure of 6,203 crore rupees."
    )
    wrong_attribution = (
        "The debt certificate figure was 4,999 crore rupees. "
        "The DRT judgment debt was 6,203 crore rupees. "
        "The transcript does not clearly explain how the figures relate."
    )

    instruction = chain._generation_repair_instruction(
        "What does the transcript say about the debt amounts?",
        wrong_attribution,
        context,
    )

    assert instruction is not None
    assert "misattributed amount-to-claim association" in instruction.lower()
    assert "4999 should remain associated with drt_debt" in instruction.lower()
    assert "6203 should remain associated with certificate" in instruction.lower()


@pytest.mark.unit
def test_answer_alignment_maps_arabic_sound_list_to_english_caption_intervals():
    answer = (
        "نبض القلب، حركة الدم في العروق، أصوات الرئتين والمعدة، "
        "ورمش الجفون."
    )
    result = chain.append_relevant_source_timestamps(
        answer=answer,
        sources=[
            {
                "start": 13.996, "end": 217.615, "distance": 0.85,
                "transcript_text": (
                    "You may hear your heartbeat, blood rushing through your veins, "
                    "the sounds of your lungs and stomach."
                ),
            },
            {
                "start": 589.599, "end": 658.17, "distance": 0.95,
                "transcript_text": "You might even hear the subtle sound of your eyelids blinking.",
            },
            {
                "start": 255.0, "end": 318.0, "distance": 0.2,
                "transcript_text": "The room is quiet and the presenter changes the audio level.",
            },
        ],
        has_evidence=True,
    )

    assert result.endswith("Relevant timestamps: 00:13–03:37, 09:49–10:58")
    assert "04:15–05:18" not in result


@pytest.mark.unit
def test_answer_alignment_maps_english_sound_list_to_arabic_caption_intervals():
    answer = (
        "Your heartbeat, blood moving through veins, lungs and stomach sounds, "
        "and eyelids blinking."
    )
    result = chain.append_relevant_source_timestamps(
        answer=answer,
        sources=[
            {
                "start": 153.05, "end": 217.615, "distance": 0.85,
                "transcript_text": (
                    "فتبدأ تسمع حاجات زي نبض قلبك، حركة الدم في عروقك، "
                    "صوت الرئتين أو صوت المعدة."
                ),
            },
            {
                "start": 589.599, "end": 658.17, "distance": 0.95,
                "transcript_text": "حتى عينيك وهي بترمش بقيت بتسمعها.",
            },
            {
                "start": 255.0, "end": 318.0, "distance": 0.2,
                "transcript_text": "الهدوء التام لا يعطي الراحة التي يتوقعها الإنسان.",
            },
        ],
        has_evidence=True,
    )

    assert result.endswith("Relevant timestamps: 02:33–03:37, 09:49–10:58")
    assert "04:15–05:18" not in result


@pytest.mark.unit
def test_financial_repair_detects_bare_certificate_figure_and_split_recovery_attribution():
    context = """[SOURCE 1]
Transcript caption segments in chronological order:
- [02:18:34–02:20:28] The DRT judgment debt was 4,999 crore rupees plus interest.
[SOURCE 2]
Transcript caption segments in chronological order:
- [04:08:59–04:09:30] The debt certificate gives a figure of 6203.
- [04:09:30–04:10:15] The principal amount carries unapplied interest at 11.5% per annum.
[SOURCE 3]
Transcript caption segments in chronological order:
- [02:49:56–02:51:41] The Ministry of Finance report says the total amount recovered was
- [02:51:41–02:52:10] 14,100 crore rupees."""

    answer = (
        "The DRT debt was 4,999 crore rupees and another debt figure was 6,23 crore. "
        "The interest rate was 11.5%."
    )

    claims = chain._money_amount_clauses(context)
    assert any(item["amount"] == 6203 and item["topic"] == "certificate" for item in claims)
    assert any(item["amount"] == 14100 and item["topic"] == "recovery_report" for item in claims)

    instruction = chain._generation_repair_instruction(
        "What amounts were mentioned about the debt and money recovered?",
        answer,
        context,
    )
    assert instruction is not None
    assert "6203" in instruction
    assert "14100" in instruction

@pytest.mark.unit
def test_generation_context_diagnostics_report_certificate_topic_coverage():
    groups = [
        _caption_group(111, 0, ["I invested 3,000 crore rupees in Kingfisher."], 0.2),
        _caption_group(112, 100, ["The DRT judgment debt was 4,999 crore rupees."], 0.3),
        _caption_group(113, 200, ["The debt certificate gives a figure of 6203."], 0.4),
        _caption_group(114, 300, ["The Ministry of Finance report says 14,100 crore was recovered."], 0.5),
        _caption_group(115, 400, ["Unrelated ticket cost 500 rupees."], 0.01),
    ]

    result = chain.build_generation_context(
        groups,
        question="What debt and money amounts are mentioned?",
        max_chars=2400,
    )

    assert len(result["context"]) <= 2400
    assert "The debt certificate gives a figure of 6203." in result["context"]
    assert "included_chunk_ids" in result["diagnostics"]
    assert "113" in result["diagnostics"]["included_chunk_ids"]


@pytest.mark.unit
def test_timestamp_alignment_matches_bare_certificate_number_in_source_caption():
    answer = (
        "The DRT judgment debt was 4,999 crore rupees; the debt certificate "
        "lists 6,203 crore; the Ministry of Finance report says 14,100 crore was recovered."
    )
    result = chain.append_relevant_source_timestamps(
        answer=answer,
        sources=[
            {
                "start": 100, "end": 140, "distance": 0.7,
                "transcript_text": "The DRT judgment debt was 4,999 crore rupees plus interest.",
            },
            {
                "start": 200, "end": 240, "distance": 1.0,
                "transcript_text": "The debt certificate gives a figure of 6203. The principal carries unapplied interest at 11.5% per annum.",
            },
            {
                "start": 300, "end": 340, "distance": 0.9,
                "transcript_text": "The Ministry of Finance report said 14,100 crore rupees were recovered.",
            },
        ],
        has_evidence=True,
    )

    # The fixture exposes only the full source interval (200–240 seconds), not
    # caption-level boundaries; preserve that evidence range instead of
    # inventing a narrower end timestamp.
    assert result.endswith(
        "Relevant timestamps: 01:40–02:20, 03:20–04:00, 05:00–05:40"
    )

@pytest.mark.unit
def test_generation_context_reserves_bare_certificate_claim_when_topic_is_already_covered():
    filler = [
        "The presenter continues with general interview context that does not add a financial claim."
        for _ in range(18)
    ]
    groups = [
        _caption_group(
            108, 7267.92,
            [*filler, "I invested 3,000 crore rupees into Kingfisher Airlines."],
            0.82,
        ),
        _caption_group(
            124, 8314.519,
            [*filler, "The DRT judgment debt was 4,999 crore rupees plus interest."],
            0.91,
        ),
        _caption_group(
            153, 10196.399,
            [*filler, "The Ministry of Finance report says 14,100 crore rupees were recovered."],
            0.82,
        ),
        # A nearer source uses the certificate topic but has no monetary figure.
        _caption_group(
            199, 12000,
            [*filler, "The debt certificate is mentioned, but this caption gives no figure."],
            0.10,
        ),
        _caption_group(202, 12100, [*filler, "The presenter asks a follow-up question."], 0.15),
        _caption_group(205, 12300, [*filler, "The guest pauses before answering."], 0.18),
        _caption_group(208, 14068.199, [*filler, "The interview continues with another topic."], 0.89),
        _caption_group(210, 14206.199, [*filler, "The interviewer asks for clarification."], 0.81),
        _caption_group(213, 14350, [*filler, "The speaker changes to a different subject."], 0.77),
        _caption_group(217, 14500, [*filler, "A general statement about the airline follows."], 0.68),
        _caption_group(219, 14700, [*filler, "The recording continues with no new amount."], 0.73),
        _caption_group(
            221, 14939.359,
            [
                *filler,
                "The debt certificate gives a figure of 6203.",
                "The principal carries unapplied interest at 11.5% per annum.",
                "The Ministry of Finance report also says 14,100 crore rupees were recovered.",
            ],
            1.12,
        ),
    ]

    result = chain.build_generation_context(
        groups,
        question="What amounts of debt, investment, and money recovered are mentioned?",
        max_chars=4000,
    )

    assert len(result["context"]) <= 4000
    assert "The debt certificate gives a figure of 6203." in result["context"]
    assert "included_chunk_ids" in result["diagnostics"]
    assert "221" in result["diagnostics"]["included_chunk_ids"]



@pytest.mark.unit
def test_generation_context_reserves_report_recovery_before_noisy_financial_distractors():
    filler = [
        "General narration continues without adding another financial claim."
        for _ in range(18)
    ]
    groups = [
        _caption_group(
            108, 7267.92,
            [*filler, "I put in 3,000 crores of UB Group money into Kingfisher Airlines."],
            0.82,
        ),
        _caption_group(
            114, 7680.0,
            [
                *filler,
                "The tribunal discussed 3,400 crore, 5,600 crore and 7,800 crore in unrelated proceedings.",
            ],
            0.72,
        ),
        _caption_group(
            124, 8314.519,
            [*filler, "The debt recovery tribunal says 4,999 crores was a debt."],
            0.91,
        ),
        _caption_group(
            153, 10196.399,
            [*filler, "That's how you've recovered 14,100 crores, right?"],
            0.82,
        ),
        _caption_group(
            159, 10630.88,
            [
                *filler,
                "It is through the statement in Parliament of the finance minister and the printed report of the Ministry of Finance that they have shown",
                "The government and the banks have not rendered a statement of account.",
                "14,100 crores recovered from me.",
            ],
            0.83,
        ),
        _caption_group(
            179, 12023.0,
            [*filler, "A separate account discusses 2,700 crore in payments and expenses."],
            0.91,
        ),
        _caption_group(
            189, 12705.08,
            [
                *filler,
                "The debt recovery tribunal discussed 6,23 crores plus interest at 11.5%.",
                "They said the debt was 6,2003 crores and the judgment debt was 6,0002 crores.",
                "An official recovery of 9,000 crores was also claimed in this noisy passage.",
            ],
            0.77,
        ),
        _caption_group(
            192, 12989.64,
            [*filler, "A financial statement lists 1,200 crore, 2,300 crore and 3,100 crore."],
            0.81,
        ),
        _caption_group(
            208, 14068.199,
            [*filler, "Another account refers to payments of 4,200 crore and 8,100 crore."],
            0.89,
        ),
        _caption_group(
            210, 14206.199,
            [*filler, "The interviewer mentions a separate debt of 5,300 crore and interest."],
            0.81,
        ),
        _caption_group(
            221, 14939.359,
            [
                *filler,
                "The debt recovery certificate amount is 6203 crores.",
                "I showed you the Ministry of Finance report, 14,000 100 recovered.",
            ],
            1.12,
        ),
    ]

    result = chain.build_generation_context(
        groups,
        question="What amounts of debt, investment, and money recovered are mentioned?",
        max_chars=4000,
    )

    context = result["context"]
    included_ids = result["diagnostics"]["included_chunk_ids"]
    assert len(context) <= 4000
    assert "159" in included_ids
    assert "14,100 crores recovered from me." in context
    assert "printed report of the Ministry of Finance" in context
    # The certificate and malformed financial distractors must not prevent
    # the higher-quality source-local recovery claim from reaching generation.
    fallback = chain._build_financial_evidence_fallback(
        "What amounts of debt, investment, and money recovered are mentioned?",
        context,
    )
    assert fallback is not None
    assert "Ministry of Finance recovery report: the transcript associates a recovery figure of 14,100 crore with that report." in fallback
    assert "does not clearly explain how the DRT judgment and debt-certificate figures relate" in fallback


@pytest.mark.unit
def test_generation_context_reserves_source_local_report_recovery_amount():
    filler = [
        "General narration continues without adding another financial claim."
        for _ in range(18)
    ]
    groups = [
        _caption_group(
            108, 7267.92,
            [*filler, "I put in 3,000 crores of UB Group money into Kingfisher Airlines."],
            0.82,
        ),
        _caption_group(
            124, 8314.519,
            [*filler, "The debt recovery tribunal says 4,999 crores was a debt."],
            0.91,
        ),
        # This nearer source repeats the recovery amount but does not identify
        # the report or authority to which the figure is attributed.
        _caption_group(
            153, 10196.399,
            [*filler, "That's how you've recovered 14,100 crores, right?"],
            0.82,
        ),
        # Report wording and the amount are separated by another caption but
        # remain in the same source block. Preserve this source for fallback.
        _caption_group(
            159, 10630.88,
            [
                *filler,
                "It is through the statement in Parliament of the finance minister and the printed report of the Ministry of Finance that they have shown",
                "The government and the banks have not rendered a statement of account.",
                "14,100 crores recovered from me.",
            ],
            0.83,
        ),
        _caption_group(
            199, 12000,
            [*filler, "The debt certificate is mentioned but this passage gives no amount."],
            0.10,
        ),
        _caption_group(202, 12100, [*filler, "The presenter asks a follow-up question."], 0.15),
        _caption_group(205, 12300, [*filler, "The guest pauses before answering."], 0.18),
        _caption_group(208, 14068.199, [*filler, "The interview continues with another topic."], 0.89),
        _caption_group(210, 14206.199, [*filler, "The interviewer asks for clarification."], 0.81),
        _caption_group(213, 14350, [*filler, "The speaker changes to a different subject."], 0.77),
        _caption_group(217, 14500, [*filler, "A general statement about the airline follows."], 0.68),
        _caption_group(219, 14700, [*filler, "The recording continues with no new amount."], 0.73),
        _caption_group(
            221, 14939.359,
            [
                *filler,
                "The debt recovery certificate amount is 6203 crores.",
                "I showed you the Ministry of Finance report, 14,000 100 recovered.",
            ],
            1.12,
        ),
    ]

    result = chain.build_generation_context(
        groups,
        question="What amounts of debt, investment, and money recovered are mentioned?",
        max_chars=4000,
    )

    assert len(result["context"]) <= 4000
    assert "included_chunk_ids" in result["diagnostics"]
    assert "159" in result["diagnostics"]["included_chunk_ids"]
    assert "14,100 crores recovered from me." in result["context"]
    assert "printed report of the Ministry of Finance" in result["context"]

@pytest.mark.unit
def test_timestamp_alignment_maps_unitless_certificate_figure_to_its_source_interval():
    answer = (
        "I invested 3,000 crore rupees into Kingfisher Airlines. The DRT judgment "
        "debt was 4,999 crore rupees. The debt certificate gives a figure of 6203. "
        "The Ministry of Finance report says 14,100 crore rupees were recovered."
    )
    result = chain.append_relevant_source_timestamps(
        answer=answer,
        sources=[
            {
                "start": 7267.92, "end": 7351.84, "distance": 0.7,
                "transcript_text": "I invested 3,000 crore rupees into Kingfisher Airlines.",
            },
            {
                "start": 8314.519, "end": 8428.16, "distance": 0.8,
                "transcript_text": "The DRT judgment debt was 4,999 crore rupees plus interest.",
            },
            {
                "start": 14939.359, "end": 15015.239, "distance": 1.0,
                "transcript_text": "The debt certificate gives a figure of 6203. The principal carries unapplied interest at 11.5% per annum.",
            },
            {
                "start": 10196.399, "end": 10301.84, "distance": 0.9,
                "transcript_text": "The Ministry of Finance report said 14,100 crore rupees were recovered.",
            },
        ],
        has_evidence=True,
    )

    assert result.endswith(
        "Relevant timestamps: 02:01:07–02:02:31, 02:18:34–02:20:28, "
        "02:49:56–02:51:41, 04:08:59–04:10:15"
    )


@pytest.mark.unit
def test_english_sound_answer_aligns_to_arabic_eye_closing_caption_variant():
    answer = (
        "Your heartbeat, blood moving through veins, lungs and stomach sounds, "
        "and eyelids blinking."
    )
    result = chain.append_relevant_source_timestamps(
        answer=answer,
        sources=[
            {
                "start": 153.05, "end": 217.615, "distance": 0.85,
                "transcript_text": (
                    "فتبدأ تسمع حاجات زي نبض قلبك، حركة الدم في عروقك، "
                    "صوت الرئتين أو صوت المعدة."
                ),
            },
            {
                "start": 589.599, "end": 658.17, "distance": 0.95,
                "transcript_text": "وقد تسمع صوت عينيك عند إغلاقهما.",
            },
            {
                "start": 255.0, "end": 318.0, "distance": 0.2,
                "transcript_text": "الهدوء التام لا يعطي الراحة التي يتوقعها الإنسان.",
            },
        ],
        has_evidence=True,
    )

    assert result.endswith("Relevant timestamps: 02:33–03:37, 09:49–10:58")
    assert "04:15–05:18" not in result

@pytest.mark.unit
def test_financial_abstention_fallback_uses_full_retrieved_sources_when_prompt_is_truncated(monkeypatch):
    question = (
        "What does Vijay Mallya say about the amount of debt or money "
        "involved in the Kingfisher situation?"
    )
    filler = (
        "The speaker describes an unrelated part of the interview without adding "
        "financial facts. "
    ) * 12
    source_groups = [
        {
            "documents": [doc(108, 7267, 7351, filler + "I invested 3,000 crore rupees into Kingfisher Airlines.")],
            "start": 7267, "end": 7351, "distance": 0.2,
        },
        {
            "documents": [doc(124, 8314, 8428, filler + "The DRT judgment debt was 4,999 crore rupees plus interest.")],
            "start": 8314, "end": 8428, "distance": 0.25,
        },
        {
            "documents": [doc(221, 14939, 15015, filler + "The debt certificate gives a figure of 6203.")],
            "start": 14939, "end": 15015, "distance": 0.3,
        },
        {
            "documents": [doc(159, 10630, 10698, filler + "A Ministry of Finance report says 14,100 crore rupees were recovered from me.")],
            "start": 10630, "end": 10698, "distance": 0.35,
        },
    ]
    retrieved_results = [
        (group["documents"][0], group["distance"])
        for group in source_groups
    ]
    monkeypatch.setattr(
        chain,
        "_retrieve_with_query_recovery_trace",
        lambda **kwargs: (
            retrieved_results,
            {
                "recovery_used": False,
                "final_stage": "initial_strict",
                "retrieval_latency_ms": 1.0,
                "final_chunk_ids": [108, 124, 221, 159],
                "citation_chunk_ids": [108, 124, 221, 159],
            },
        ),
    )
    # Keep the prompt too small to include all four claim families. The
    # extractive fallback must still inspect all retrieved source groups.
    monkeypatch.setattr(chain, "merge_overlapping_results", lambda _results: source_groups)
    monkeypatch.setattr(chain, "RAG_GENERATION_CONTEXT_MAX_CHARS", 500)

    prompt_human_messages = []

    def fake_generate_with_llm(**kwargs):
        prompt_value = kwargs["prompt_value"]
        prompt_human_messages.append(
            next(message.content for message in prompt_value.messages if message.type == "human")
        )
        return chain.FINANCIAL_VERIFICATION_FALLBACK

    monkeypatch.setattr(chain, "generate_with_llm", fake_generate_with_llm)
    result = chain.build_rag_chain(object(), object()).invoke({
        "question": question,
        "video_id": "MdeQMVBuGgY",
    })

    # The model was given a bounded subset, not every required fact.
    assert len(prompt_human_messages) == 1
    assert not all(
        phrase in prompt_human_messages[0]
        for phrase in (
            "3,000 crore rupees",
            "4,999 crore rupees",
            "figure of 6203",
            "14,100 crore rupees",
        )
    )

    answer = result["answer"]
    assert result["generation_trace"]["grounded_financial_fallback_used"] is True
    assert result["generation_trace"]["financial_answer_suppressed"] is False
    assert result["generation_trace"]["financial_fallback_context_source"] == "full_retrieved_source_groups"
    fallback_diagnostics = result["generation_trace"]["financial_fallback_diagnostics"]
    assert fallback_diagnostics["rejection_reason"] is None
    assert fallback_diagnostics["missing_topics"] == []
    assert set(fallback_diagnostics["recognized_topics"]) == {
        "investment", "drt_debt", "certificate", "recovery_report"
    }
    assert fallback_diagnostics["candidate_claims"]
    assert all("amount_caption_topic" in claim for claim in fallback_diagnostics["candidate_claims"])
    assert len(fallback_diagnostics["source_financial_excerpts"]) == 4
    assert "3,000 crore rupees" in answer
    assert "4,999 crore rupees" in answer
    assert "Debt certificate: the transcript identifies the certificate amount as the figure 6,203" in answer
    assert "recovery figure of 14,100 crore rupees" in answer
    assert "does not clearly explain how the DRT judgment and debt-certificate figures relate" in answer


@pytest.mark.unit
def test_rag_chain_uses_grounded_hindi_tax_count_after_empty_provider_answer(monkeypatch):
    source_text = (
        "पहला बड़ा किसान आंदोलन कौन सा हुआ था अपने बिजोलिया। "
        "बिजोलिया में कितने प्रकार के कर थे। 84 प्रकार के कर लग रहे थे।"
    )
    source_document = doc(911, 0, 30, source_text)
    source_document.metadata["video_id"] = "INpnuL7qnzY"
    monkeypatch.setattr(
        chain,
        "_retrieve_with_query_recovery_trace",
        lambda **kwargs: (
            [(source_document, 0.2)],
            {
                "recovery_used": False,
                "final_stage": "initial_strict",
                "retrieval_latency_ms": 1.0,
                "final_chunk_ids": [911],
                "citation_chunk_ids": [911],
            },
        ),
    )

    def fail_with_empty_provider_answer(**kwargs):
        raise RuntimeError(
            "The Groq model returned an empty answer after 3 attempts. "
            "Diagnostics: {'finish_reason': 'length', 'content_length': 0, "
            "'reasoning_length': 0}"
        )

    monkeypatch.setattr(chain, "generate_with_llm", fail_with_empty_provider_answer)

    result = chain.build_rag_chain(object(), object()).invoke({
        "question": "How many types of taxes does the speaker mention?",
        "video_id": "INpnuL7qnzY",
    })

    trace = result["generation_trace"]
    assert result["answer"].startswith("The speaker mentions 84 types of taxes.")
    assert trace["generated"] is False
    assert trace["fallback_used"] is True
    assert trace["provider_failure_reason"] == "empty_answer_after_retries"
    assert trace["grounded_hindi_tax_count_fallback_used"] is True
    assert trace["hindi_tax_count_fallback_diagnostics"]["extracted_count"] == "84"
    assert trace["tax_count_fallback_context_source"] == "full_retrieved_source_groups"
    assert "911" in {str(value) for value in trace["context_budget"]["included_chunk_ids"]}

    selected_results = chain._select_citation_results(
        result["retrieved_results"],
        result["retrieval_trace"],
        trace,
    )
    assert selected_results
    formatted_sources = chain.format_sources(
        chain.merge_overlapping_results(selected_results)
    )
    assert formatted_sources
    assert formatted_sources[0]["video_id"] == "INpnuL7qnzY"
    assert 911 in formatted_sources[0]["chunk_ids"]


@pytest.mark.unit
def test_rag_chain_repairs_hi07_with_source_grounded_bijolia_location_when_model_abstains(monkeypatch):
    source_text = (
        "पहला बड़ा किसान आंदोलन कौन सा हुआ था अपने बिजोलिया। "
        "बिजोलिया में कितने प्रकार के कर थे।"
    )
    source_document = doc(912, 0, 30, source_text)
    source_document.metadata["video_id"] = "INpnuL7qnzY"
    monkeypatch.setattr(
        chain,
        "_retrieve_with_query_recovery_trace",
        lambda **kwargs: (
            [(source_document, 0.2)],
            {
                "recovery_used": False,
                "final_stage": "initial_strict",
                "retrieval_latency_ms": 1.0,
                "final_chunk_ids": [912],
                "citation_chunk_ids": [912],
            },
        ),
    )
    monkeypatch.setattr(
        chain,
        "generate_with_llm",
        lambda **kwargs: chain.FALLBACK_ANSWER,
    )

    result = chain.build_rag_chain(object(), object()).invoke({
        "question": "Where does the speaker locate the first major farmers' movement in relation to Bijolia?",
        "video_id": "INpnuL7qnzY",
    })

    trace = result["generation_trace"]
    assert result["answer"].startswith(
        "The speaker locates the first major farmers' movement in Bijolia."
    )
    assert trace["grounded_hindi_movement_location_fallback_used"] is True
    assert trace["hindi_movement_location_fallback_diagnostics"]["matched_location_phrase"] == "अपने बिजोलिया"
    assert trace["movement_location_fallback_context_source"] == "full_retrieved_source_groups"
    assert trace["temporal_location_coverage_before"] is None
    assert "912" in {str(value) for value in trace["context_budget"]["included_chunk_ids"]}

    selected_results = chain._select_citation_results(
        result["retrieved_results"],
        result["retrieval_trace"],
        trace,
    )
    assert selected_results
    formatted_sources = chain.format_sources(
        chain.merge_overlapping_results(selected_results)
    )
    assert formatted_sources
    assert formatted_sources[0]["video_id"] == "INpnuL7qnzY"
    assert 912 in formatted_sources[0]["chunk_ids"]


@pytest.mark.unit
def test_rag_chain_uses_grounded_financial_fallback_when_initial_provider_answer_is_empty(monkeypatch):
    context = (
        "I invested 3,000 crore rupees into Kingfisher Airlines. "
        "The DRT judgment debt was 4,999 crore rupees plus interest. "
        "The debt certificate gives a figure of 6203. "
        "The Ministry of Finance report said 14,100 crore rupees had been recovered."
    )
    monkeypatch.setattr(
        chain,
        "_retrieve_with_query_recovery_trace",
        lambda **kwargs: (
            [(doc(101, 0, 90, context), 0.2)],
            {
                "recovery_used": False,
                "final_stage": "initial_strict",
                "retrieval_latency_ms": 1.0,
                "final_chunk_ids": [101],
                "citation_chunk_ids": [101],
            },
        ),
    )

    def fail_with_empty_provider_answer(**kwargs):
        raise RuntimeError(
            "The Groq model returned an empty answer after 3 attempts. "
            "Diagnostics: {'finish_reason': 'length', 'content_length': 0, "
            "'reasoning_length': 0}"
        )

    monkeypatch.setattr(chain, "generate_with_llm", fail_with_empty_provider_answer)

    result = chain.build_rag_chain(object(), object()).invoke({
        "question": "What amounts were mentioned about Kingfisher debt and money involved?",
        "video_id": "Gfr50f6ZBvo",
    })

    answer = result["answer"]
    trace = result["generation_trace"]
    assert trace["generated"] is False
    assert trace["fallback_used"] is True
    assert trace["grounded_financial_fallback_used"] is True
    assert trace["repair_aborted_due_quota"] is False
    assert trace["provider_failure_reason"] == "empty_answer_after_retries"
    assert "3,000 crore" in answer
    assert "4,999 crore" in answer
    assert "6,203" in answer
    assert "14,100 crore" in answer
    assert "does not clearly explain how the DRT judgment and debt-certificate figures relate" in answer



@pytest.mark.unit
def test_rag_chain_uses_question_scoped_bank_recovery_fallback_after_empty_provider_answer(monkeypatch):
    context = (
        "The Ministry of Finance report records that 14,100 crore rupees "
        "were recovered from me. The banks have been fully paid. "
        "The government and banks have not rendered a statement of account."
    )
    monkeypatch.setattr(
        chain,
        "_retrieve_with_query_recovery_trace",
        lambda **kwargs: (
            [(doc(159, 10630, 10759, context), 0.2)],
            {
                "recovery_used": False,
                "final_stage": "initial_strict",
                "retrieval_latency_ms": 1.0,
                "final_chunk_ids": [159],
                "citation_chunk_ids": [159],
            },
        ),
    )

    def fail_with_empty_provider_answer(**kwargs):
        raise RuntimeError(
            "The Groq model returned an empty answer after 3 attempts."
        )

    monkeypatch.setattr(chain, "generate_with_llm", fail_with_empty_provider_answer)
    result = chain.build_rag_chain(object(), object()).invoke({
        "question": "What does Vijay Mallya say about banks and the recovery of money?",
        "video_id": "MdeQMVBuGgY",
    })

    trace = result["generation_trace"]
    assert trace["generated"] is False
    assert trace["provider_failure_reason"] == "empty_answer_after_retries"
    assert trace["grounded_financial_fallback_used"] is True
    assert trace["financial_answer_suppressed"] is False
    assert "14,100 crore" in result["answer"]
    assert "statement of account" in result["answer"].casefold()
    assert "fully paid" in result["answer"].casefold()
    assert "3,000 crore" not in result["answer"]
    assert "4,999 crore" not in result["answer"]
    assert "6,203" not in result["answer"]
    assert "22,280" not in result["answer"]


@pytest.mark.unit
def test_rag_chain_uses_source_extractive_financial_answer_when_llm_repair_still_fails(monkeypatch):
    context = (
        "I invested 3,000 crore rupees into Kingfisher Airlines. "
        "The DRT judgment debt was 4,999 crore rupees plus interest. "
        "The debt certificate gives a figure of 6203. "
        "The Ministry of Finance report said 14,100 crore rupees had been recovered."
    )
    monkeypatch.setattr(
        chain,
        "_retrieve_with_query_recovery_trace",
        lambda **kwargs: (
            [(doc(101, 0, 90, context), 0.2)],
            {
                "recovery_used": False,
                "final_stage": "initial_strict",
                "retrieval_latency_ms": 1.0,
                "final_chunk_ids": [101],
                "citation_chunk_ids": [101],
            },
        ),
    )
    calls = []

    def fake_generate_with_llm(**kwargs):
        calls.append(kwargs)
        return (
            "The debt was 6,23 crore and 6,2003 crore, and 14,100 crore was recovered."
            if len(calls) == 1
            else "The debt is 6,0002 crore, with 9,000 crore recovered."
        )

    monkeypatch.setattr(chain, "generate_with_llm", fake_generate_with_llm)
    result = chain.build_rag_chain(object(), object()).invoke({
        "question": "What amounts were mentioned about Kingfisher debt and money involved?",
        "video_id": "Gfr50f6ZBvo",
    })

    answer = result["answer"]
    assert len(calls) == 2
    assert result["generation_trace"]["quality_repair_attempted"] is True
    assert result["generation_trace"]["quality_repair_passed"] is False
    assert result["generation_trace"]["grounded_financial_fallback_used"] is True
    assert result["generation_trace"]["financial_answer_suppressed"] is False
    assert "3,000 crore" in answer
    assert "4,999 crore" in answer
    assert "Debt certificate: the transcript identifies the certificate amount as the figure 6,203" in answer
    assert "14,100 crore" in answer
    assert "does not clearly explain how the DRT judgment and debt-certificate figures relate" in answer
    assert "6,2003" not in answer
    assert "6,0002" not in answer
    assert "9,000 crore recovered" not in answer

@pytest.mark.unit
def test_rag_chain_suppresses_unverified_financial_draft_when_safe_extraction_fails(monkeypatch):
    context = (
        "I invested 3,000 crore rupees into Kingfisher Airlines. "
        "The DRT judgment debt was 4,999 crore rupees plus interest. "
        "The debt certificate gives a figure of 6203. "
        "A Ministry of Finance report discusses recovery of 14,100 crore rupees."
    )
    monkeypatch.setattr(
        chain,
        "_retrieve_with_query_recovery_trace",
        lambda **kwargs: (
            [(doc(102, 0, 90, context), 0.2)],
            {
                "recovery_used": False,
                "final_stage": "initial_strict",
                "retrieval_latency_ms": 1.0,
                "final_chunk_ids": [102],
                "citation_chunk_ids": [102],
            },
        ),
    )
    # Simulate evidence that fails the stricter source-local fallback contract.
    # The repair can still fail, but its unsupported numeric draft must not leak.
    monkeypatch.setattr(
        chain,
        "_build_financial_evidence_fallback",
        lambda **kwargs: None,
    )
    calls = []

    def fake_generate_with_llm(**kwargs):
        calls.append(kwargs)
        return (
            "The debt was 6,23 crore and 6,2003 crore, and 14,100 crore was recovered."
            if len(calls) == 1
            else "The debt is 6,0002 crore, with 9,000 crore recovered."
        )

    monkeypatch.setattr(chain, "generate_with_llm", fake_generate_with_llm)
    result = chain.build_rag_chain(object(), object()).invoke({
        "question": "What amounts were mentioned about Kingfisher debt and money involved?",
        "video_id": "Gfr50f6ZBvo",
    })

    assert len(calls) == 2
    assert result["generation_trace"]["quality_repair_attempted"] is True
    assert result["generation_trace"]["quality_repair_passed"] is False
    assert result["generation_trace"]["grounded_financial_fallback_used"] is False
    assert result["generation_trace"]["financial_answer_suppressed"] is True
    assert result["answer"] == chain.FINANCIAL_VERIFICATION_FALLBACK
    assert "6,23" not in result["answer"]
    assert "6,2003" not in result["answer"]
    assert "6,0002" not in result["answer"]
    assert "9,000" not in result["answer"]



@pytest.mark.unit
def test_bank_recovery_fallback_uses_only_question_relevant_source_evidence():
    context = """[SOURCE 1]
Transcript caption segments in chronological order:
- [10:12–10:20] The Ministry of Finance report acknowledges 14,100 crores recovered from me.
- [10:20–10:35] The government and banks have not rendered a statement of account.
[SOURCE 2]
Transcript caption segments in chronological order:
- [02:18:34–02:20:28] The DRT judgment debt was 4,999 crore rupees.
[SOURCE 3]
Transcript caption segments in chronological order:
- [04:08:59–04:09:30] The debt certificate gives a figure of 6203."""

    diagnostics = {}
    fallback = chain._build_financial_evidence_fallback(
        "What does Vijay Mallya say about banks and the recovery of money?",
        context,
        diagnostics=diagnostics,
    )

    assert fallback is not None
    assert "14,100 crore" in fallback
    assert "Ministry of Finance report" in fallback
    assert "statement of account" in fallback.casefold()
    assert "3,000" not in fallback
    assert "4,999" not in fallback
    assert "6,203" not in fallback
    assert diagnostics["scope"] == "bank_recovery"
    assert diagnostics["rejection_reason"] is None
    assert set(diagnostics["supported_evidence_groups"]) == {
        "recovery_amount", "account_statement"
    }
    assert diagnostics["minimum_supported_evidence_groups"] == 2
    assert diagnostics["missing_evidence_groups"] == ["bank_payment"]



@pytest.mark.unit
def test_bank_recovery_statement_extractor_ignores_interviewer_questions():
    context = """[SOURCE 1]
Transcript caption segments in chronological order:
- [10:00–10:10] Have the banks been fully paid?
- [10:10–10:20] Did you provide a statement in Parliament?
[SOURCE 2]
Transcript caption segments in chronological order:
- [10:20–10:35] The Ministry of Finance report acknowledges 14,100 crores recovered from me."""

    diagnostics = {}
    fallback = chain._build_financial_evidence_fallback(
        "What does the speaker say about banks and money recovery?",
        context,
        diagnostics=diagnostics,
    )

    assert fallback is None
    assert diagnostics["supported_evidence_groups"] == ["recovery_amount"]
    assert set(diagnostics["missing_evidence_groups"]) == {
        "bank_payment", "account_statement"
    }



@pytest.mark.unit
def test_bank_recovery_fallback_preserves_report_label_split_across_caption_rows():
    context = """[SOURCE 1]
Transcript caption segments in chronological order:
- [10:12–10:15] India ministry of finance has published
- [10:15–10:20] a report their annual report in which they have acknowledged
- [10:20–10:25] 14,100 crores recovered from me.
- [10:25–10:35] The government and banks have not rendered a statement of account."""

    diagnostics = {}
    fallback = chain._build_financial_evidence_fallback(
        "What does the speaker say about banks and recovery?",
        context,
        diagnostics=diagnostics,
    )

    assert fallback is not None
    assert "14,100 crore" in fallback
    assert "Ministry of Finance report" in fallback
    assert diagnostics["recognized_topics"]["recovery_report"]["source_id"] == 1
    assert diagnostics["supported_evidence_groups"] == [
        "recovery_amount", "account_statement"
    ]


@pytest.mark.unit
def test_bank_recovery_fallback_abstains_with_only_one_evidence_family():
    context = """[SOURCE 1]
Transcript caption segments in chronological order:
- [10:12–10:20] The Ministry of Finance report acknowledges 14,100 crores recovered from me."""

    diagnostics = {}
    fallback = chain._build_financial_evidence_fallback(
        "What does the speaker say about banks and recovery of money?",
        context,
        diagnostics=diagnostics,
    )

    assert fallback is None
    assert diagnostics["scope"] == "bank_recovery"
    assert diagnostics["rejection_reason"] == "insufficient_bank_recovery_evidence"
    assert diagnostics["supported_evidence_groups"] == ["recovery_amount"]
    assert set(diagnostics["missing_evidence_groups"]) == {
        "bank_payment", "account_statement"
    }


@pytest.mark.unit
def test_bank_recovery_fallback_does_not_borrow_report_attribution_across_sources():
    context = """[SOURCE 1]
Transcript caption segments in chronological order:
- [10:00–10:10] The Ministry of Finance report was published for Parliament.
[SOURCE 2]
Transcript caption segments in chronological order:
- [10:12–10:20] The total amount recovered was 14,100 crore rupees.
[SOURCE 3]
Transcript caption segments in chronological order:
- [10:20–10:35] The banks have been fully paid."""

    diagnostics = {}
    fallback = chain._build_financial_evidence_fallback(
        "What does the speaker say about banks and money recovery?",
        context,
        diagnostics=diagnostics,
    )

    assert fallback is not None
    assert "14,100 crore" in fallback
    assert "fully paid" in fallback.casefold()
    assert "Ministry of Finance report" not in fallback
    assert "does not attribute the figure to a specific report" in fallback
    assert diagnostics["recognized_topics"]["recovery"]["source_id"] == 2


@pytest.mark.unit
def test_bank_recovery_fallback_can_answer_from_payment_and_account_statements_without_amount():
    context = """[SOURCE 1]
Transcript caption segments in chronological order:
- [10:12–10:20] The banks have been fully paid.
[SOURCE 2]
Transcript caption segments in chronological order:
- [10:20–10:35] The speaker refers to a statement in Parliament."""

    diagnostics = {}
    fallback = chain._build_financial_evidence_fallback(
        "What does the speaker say about banks and recovery?",
        context,
        diagnostics=diagnostics,
    )

    assert fallback is not None
    assert "fully paid" in fallback.casefold()
    assert "statement in Parliament" in fallback
    assert "recovery_amount" not in diagnostics["supported_evidence_groups"]


@pytest.mark.unit
def test_bank_recovery_fallback_timestamps_are_selected_from_returned_sources():
    context = """[SOURCE 1]
Transcript caption segments in chronological order:
- [10:12–10:20] The Ministry of Finance report acknowledges 14,100 crores recovered from me.
- [10:20–10:35] The government and banks have not rendered a statement of account."""

    fallback = chain._build_financial_evidence_fallback(
        "What does the speaker say about banks and recovery?",
        context,
    )
    assert fallback is not None

    stamped = chain.append_relevant_source_timestamps(
        fallback,
        [
            {
                "start": 612.0,
                "end": 635.0,
                "distance": 0.4,
                "transcript_text": (
                    "The Ministry of Finance report acknowledges 14,100 crore "
                    "recovered from me. The government and banks have not "
                    "rendered a statement of account."
                ),
            },
            {
                "start": 1200.0,
                "end": 1220.0,
                "distance": 0.01,
                "transcript_text": (
                    "The unrelated report discusses 22,280 crore paid to "
                    "different claimants."
                ),
            },
        ],
        has_evidence=True,
    )

    assert "Relevant timestamps:" in stamped
    assert "10:12–10:35" in stamped
    assert "20:00–20:20" not in stamped


@pytest.mark.unit
def test_financial_question_scope_preserves_comprehensive_default_and_detects_bank_recovery():
    assert chain._financial_fallback_question_scope(
        "What does Vijay Mallya say about banks and the recovery of money?"
    ) == "bank_recovery"
    assert chain._financial_fallback_question_scope(
        "What does Vijay Mallya say about the amount of debt or money involved?"
    ) == "comprehensive_financial"
    assert chain._financial_fallback_question_scope(
        "What does the video say about banks?"
    ) == "comprehensive_financial"
    assert chain._build_financial_evidence_fallback(
        "What does the video say about banks?",
        "[SOURCE 1]\\nThe banks are mentioned.",
    ) is None
    assert chain._financial_fallback_question_scope(
        "What amounts of investment, debt, and recovery are mentioned?"
    ) == "comprehensive_financial"


@pytest.mark.unit
def test_financial_fallback_preserves_report_attribution_split_across_caption_rows():
    context = """[SOURCE 1]
Transcript caption segments in chronological order:
- [02:01:07–02:02:31] I invested 3,000 crore rupees into Kingfisher Airlines.
[SOURCE 2]
Transcript caption segments in chronological order:
- [02:18:34–02:20:28] The DRT judgment debt was 4,999 crore rupees plus interest.
[SOURCE 3]
Transcript caption segments in chronological order:
- [02:49:56–02:50:25] The Ministry of Finance report says the total amount recovered was
- [02:50:25–02:51:41] 14,100 crore rupees.
[SOURCE 4]
Transcript caption segments in chronological order:
- [04:08:59–04:09:30] The debt certificate gives a figure of 6203.
- [04:09:30–04:10:15] The principal amount carries unapplied interest at 11.5% per annum."""

    fallback = chain._build_financial_evidence_fallback(
        "What debt and money amounts are mentioned in the Kingfisher case?",
        context,
    )

    assert fallback is not None
    assert "3,000 crore rupees" in fallback
    assert "DRT debt: the transcript describes 4,999 crore rupees as a debt" in fallback
    assert "Debt certificate: the transcript identifies the certificate amount as the figure 6,203" in fallback
    assert re.search(
        r"(?:Ministry of Finance report|Finance report|report).{0,80}14,100 crore(?: rupees)?",
        fallback,
        re.IGNORECASE,
    )
    assert "14,100 crore" in fallback
    assert "does not clearly explain how the DRT judgment and debt-certificate figures relate" in fallback
    for malformed in ("6,23 crores", "6,2003 crores", "6,0002 crores", "9,000 crores"):
        assert malformed not in fallback


@pytest.mark.unit
def test_financial_fallback_maps_plain_prose_amounts_to_their_source_block():
    context = """[SOURCE 1]
I invested 3,000 crore rupees into Kingfisher Airlines.
The DRT judgment debt was 4,999 crore rupees plus interest.
The debt certificate gives a figure of 6203.
The Ministry of Finance report said 14,100 crore rupees had been recovered."""

    fallback = chain._build_financial_evidence_fallback(
        "What amounts were mentioned about Kingfisher debt and money involved?",
        context,
    )

    assert fallback is not None
    assert "3,000 crore rupees" in fallback
    assert "DRT debt: the transcript describes 4,999 crore rupees as a debt" in fallback
    assert "Debt certificate: the transcript identifies the certificate amount as the figure 6,203" in fallback
    assert re.search(
        r"(?:Ministry of Finance report|Finance report|report).{0,80}14,100 crore(?: rupees)?",
        fallback,
        re.IGNORECASE,
    )


@pytest.mark.unit
def test_financial_fallback_keeps_debt_recovery_certificate_distinct_from_recovery_report():
    context = """[SOURCE 1]
Transcript caption segments in chronological order:
- [00:00–00:10] I put in 3,000 crores of the UB group's money.
[SOURCE 2]
Transcript caption segments in chronological order:
- [00:10–00:20] The DRT judgment debt was 4,999 crore rupees plus interest.
[SOURCE 3]
Transcript caption segments in chronological order:
- [00:20–00:30] I showed you the debt recovery certificate amount 6,203 crores.
- [00:30–00:40] I've shown you the Ministry of Finance report.
[SOURCE 4]
Transcript caption segments in chronological order:
- [00:40–00:50] The Ministry of Finance report says the total amount recovered was
- [00:50–01:00] 14,100 crore rupees."""

    claims = chain._money_amount_clauses(context)
    certificate_claim = next(item for item in claims if item["amount"] == 6203)
    recovery_claim = next(item for item in claims if item["amount"] == 14100)
    assert certificate_claim["topic"] == "certificate"
    assert "debt recovery certificate" in certificate_claim["amount_caption"].casefold()
    assert recovery_claim["topic"] == "recovery_report"

    fallback = chain._build_financial_evidence_fallback(
        "What debt and money amounts were mentioned?",
        context,
    )

    assert fallback is not None
    assert "Investment: the transcript says the speaker put in 3,000 crore of UB Group money" in fallback
    assert "DRT debt: the transcript describes 4,999 crore rupees as a debt" in fallback
    assert "Debt certificate: the transcript identifies a certificate amount of 6,203 crore" in fallback
    assert re.search(
        r"Ministry of Finance recovery report: the transcript associates a recovery figure of 14,100 crore rupees with that report",
        fallback,
        re.IGNORECASE,
    )
    assert "does not clearly explain how the drt judgment and debt-certificate figures relate" in fallback.casefold()


@pytest.mark.unit
def test_financial_fallback_formats_claims_as_complete_labeled_statements():
    context = """[SOURCE 1]
- [02:01:07–02:02:31] put in 3,000 crores of UB group's money
[SOURCE 2]
- [02:18:34–02:19:01] 4,999 crores was a debt
- [02:19:01–02:20:28] debt recovery tribunal's own admission right rest is all
[SOURCE 3]
- [02:49:56–02:51:41] That's how you've recovered 14,100 crores from me.
[SOURCE 4]
- [04:09:55–04:10:01] I showed you the debt recovery
- [04:09:58–04:10:04] certificate amount 6203 crores I've shown you the ministry of
- [04:10:01–04:10:07] finance report 14,000
- [04:10:04–04:10:10] 100 recovered."""

    answer = chain._build_financial_evidence_fallback(
        "What debt and money amounts were mentioned?",
        context,
    )

    assert answer is not None
    assert "- Investment: the transcript says the speaker put in 3,000 crore of UB Group money." in answer
    assert "- DRT debt: the transcript describes 4,999 crore as a debt in a passage that mentions the Debt Recovery Tribunal." in answer
    assert "- Debt certificate: the transcript identifies a certificate amount of 6,203 crore." in answer
    assert "- Ministry of Finance recovery report: the transcript associates a recovery figure of 14,100 crore with that report." in answer
    assert "right rest is all" not in answer
    assert "I've shown you the ministry of" not in answer
    assert "does not clearly explain how the DRT judgment and debt-certificate figures relate" in answer


@pytest.mark.unit
def test_financial_fallback_rejects_malformed_number_and_reads_adjacent_recovery_caption():
    context = """[SOURCE 1]
- [02:01:17–02:01:24] put in 3,000 crores of UB group's money
[SOURCE 2]
- [02:20:11–02:20:18] debt recovery tribunal's own admission
- [02:20:15–02:20:21] 4,999 crores was a debt
[SOURCE 3]
- [03:31:36–03:31:40] says recovery
- [03:31:40–03:31:50] 6,23 crores 35 lakhs 3,879 and Pisa 42 only.
[SOURCE 4]
- [04:09:55–04:10:01] right I showed you the debt recovery
- [04:09:58–04:10:04] certificate amount 6203
- [04:10:01–04:10:08] crores I've shown you the ministry of
- [04:10:04–04:10:11] finance report 14,000
- [04:10:08–04:10:15] 100 recovered.
[SOURCE 5]
- [02:57:25–02:57:30] minister and printed report of the
- [02:57:29–02:57:33] Ministry of Finance that they have shown 14,100
- [02:57:33–02:57:38] crores recovered from me."""

    claims = chain._money_amount_clauses(context)
    assert not any(
        item["amount"] == 623 and item.get("topic") == "certificate"
        for item in claims
    )

    # The bare-number match is relative to each caption body, while caption
    # lookup uses an absolute offset in the full formatted transcript. Verify
    # that the parser resolves 6203 to SOURCE 4 and retains the adjacent "crores"
    # caption needed for unit completion.
    certificate_claims = [
        item for item in claims
        if item["amount"] == 6203 and item.get("topic") == "certificate"
    ]
    assert certificate_claims
    assert certificate_claims[0]["source_id"] == 4
    assert "crores" in certificate_claims[0]["amount_caption_context"].casefold()

    recovery_claims = [
        item for item in claims
        if item["amount"] == 14100 and item.get("topic") == "recovery_report"
    ]
    assert recovery_claims
    assert recovery_claims[0]["source_id"] == 5

    diagnostics = {}
    answer = chain._build_financial_evidence_fallback(
        "What debt and money amounts were mentioned?",
        context,
        diagnostics=diagnostics,
    )

    assert answer is not None
    assert diagnostics["missing_topics"] == []
    assert diagnostics["recognized_topics"]["certificate"]["amount"] == 6203
    assert "crores" in diagnostics["recognized_topics"]["certificate"]["amount_unit_context"].casefold(), (
        diagnostics["recognized_topics"]["certificate"]
    )
    assert diagnostics["recognized_topics"]["recovery_report"]["amount"] == 14100
    assert "certificate amount of 6,203 crore" in answer
    assert "recovery figure of 14,100 crore" in answer
    assert "14,000 crore recovered" not in answer
    assert "does not clearly explain how the DRT judgment and debt-certificate figures relate" in answer


@pytest.mark.unit
def test_financial_fallback_does_not_borrow_recovery_report_attribution_from_other_source_blocks():
    context = """[SOURCE 1]
Transcript caption segments in chronological order:
- [02:01:07–02:02:31] I invested 3,000 crore rupees into Kingfisher Airlines.
[SOURCE 2]
Transcript caption segments in chronological order:
- [02:18:34–02:20:28] The DRT judgment debt was 4,999 crore rupees plus interest.
[SOURCE 3]
Transcript caption segments in chronological order:
- [02:49:56–02:50:25] The Ministry of Finance report was published for Parliament.
[SOURCE 4]
Transcript caption segments in chronological order:
- [02:50:25–02:51:41] The total amount recovered was 14,100 crore rupees.
[SOURCE 5]
Transcript caption segments in chronological order:
- [04:08:59–04:09:30] The debt certificate gives a figure of 6203.
- [04:09:30–04:10:15] The principal amount carries unapplied interest at 11.5% per annum."""

    diagnostics = {}
    fallback = chain._build_financial_evidence_fallback(
        "What debt and money amounts are mentioned in the Kingfisher case?",
        context,
        diagnostics=diagnostics,
    )

    # An attribution from another returned source is not evidence for this
    # recovery claim; the extractor must decline rather than join unrelated
    # source blocks into one assertion. Diagnostics make the reason visible
    # in live audit traces without relaxing this attribution boundary.
    assert fallback is None
    assert diagnostics["rejection_reason"] == "missing_required_topics"
    assert diagnostics["missing_topics"] == ["recovery_report"]
    assert set(diagnostics["recognized_topics"]) == {
        "investment", "drt_debt", "certificate"
    }
    assert diagnostics["source_block_count"] == 5
    assert diagnostics["candidate_claims"]
    recovered_amount_candidates = [
        claim for claim in diagnostics["candidate_claims"]
        if claim["amount"] == 14100
    ]
    assert len(recovered_amount_candidates) == 1
    assert recovered_amount_candidates[0]["classified_topic"] == "recovery"
    assert "Finance report" in diagnostics["source_financial_excerpts"]["3"][0]
    assert any(
        "14,100 crore rupees" in excerpt
        for excerpt in diagnostics["source_financial_excerpts"]["4"]
    )


@pytest.mark.unit
def test_english_eye_sound_caption_without_explicit_blink_verb_gets_timestamp():
    answer = (
        "Your heartbeat, blood moving through veins, lungs and stomach sounds, "
        "and even the subtle sound of your eyelids blinking."
    )
    result = chain.append_relevant_source_timestamps(
        answer=answer,
        sources=[
            {
                "start": 13.996, "end": 217.615, "distance": 0.85,
                "transcript_text": (
                    "You may hear your heartbeat, blood rushing through your veins, "
                    "the sounds of your lungs and stomach."
                ),
            },
            {
                "start": 255.0, "end": 318.0, "distance": 0.2,
                "transcript_text": "The room is quiet and the presenter changes the audio level.",
            },
            {
                "start": 589.599, "end": 658.17, "distance": 0.95,
                "transcript_text": "You may even notice the subtle sound of your eyes in the silence.",
            },
        ],
        has_evidence=True,
    )

    assert result.endswith("Relevant timestamps: 00:13–03:37, 09:49–10:58")
    assert "04:15–05:18" not in result


@pytest.mark.unit
@pytest.mark.unit
def test_verified_chunk2_caption_aligns_eye_sound_only_query_to_early_source():
    answer = "You may even hear the sound of your eyelids blinking."
    result = chain.append_relevant_source_timestamps(
        answer=answer,
        sources=[
            {
                "start": 153.05, "end": 217.615, "distance": 0.84,
                "transcript_text": (
                    "In a quiet room, you can hear your own heartbeat, blood rushing "
                    "through your veins, your lungs, your stomach, and even your "
                    "eyelids blinking."
                ),
            },
            {
                "start": 589.599, "end": 658.17, "distance": 0.10,
                "transcript_text": (
                    "Sound waves travel through the air to the eardrum, and the "
                    "cochlea processes the different frequencies."
                ),
            },
            {
                "start": 255.0, "end": 318.0, "distance": 0.05,
                "transcript_text": (
                    "A loud noise came from the hallway. The presenter has eyelid "
                    "irritation, blue eyes, and wears glasses."
                ),
            },
        ],
        has_evidence=True,
    )

    # The source cue "hear" governs the bodily-sound list, and "eyelids
    # blinking" is part of that same sentence. The later sound-wave segment
    # and a closer anatomy-only segment must not win timestamp selection.
    assert result.endswith("Relevant timestamps: 02:33–03:37")
    assert "09:49–10:58" not in result
    assert "04:15–05:18" not in result


def test_financial_verification_fallback_does_not_add_unaligned_timestamps():
    answer = chain.FINANCIAL_VERIFICATION_FALLBACK
    result = chain.append_relevant_source_timestamps(
        answer=answer,
        sources=[
            {
                "start": 7267.92, "end": 7351.84, "distance": 0.7,
                "transcript_text": "I invested 3,000 crore rupees into Kingfisher Airlines.",
            },
            {
                "start": 10196.399, "end": 10301.84, "distance": 0.9,
                "transcript_text": "A Ministry of Finance report said 14,100 crore rupees were recovered.",
            },
        ],
        has_evidence=True,
    )

    assert result == answer
    assert "Relevant timestamps:" not in result


@pytest.mark.unit
def test_eye_anatomy_without_sound_context_does_not_add_eye_sound_timestamp():
    answer = (
        "Your heartbeat, blood moving through veins, lungs and stomach sounds, "
        "and even the subtle sound of your eyelids blinking."
    )
    result = chain.append_relevant_source_timestamps(
        answer=answer,
        sources=[
            {
                "start": 13.996, "end": 217.615, "distance": 0.85,
                "transcript_text": (
                    "You may hear your heartbeat, blood rushing through your veins, "
                    "the sounds of your lungs and stomach."
                ),
            },
            {
                "start": 589.599, "end": 658.17, "distance": 0.2,
                "transcript_text": (
                    "The presenter has eyelid irritation, blue eyes, and wears glasses."
                ),
            },
        ],
        has_evidence=True,
    )

    assert result.endswith("Relevant timestamps: 00:13–03:37")
    assert "09:49–10:58" not in result


@pytest.mark.unit
def test_eye_sound_claim_gets_no_timestamp_when_noise_is_unrelated_to_eye_source():
    answer = "You may even notice the subtle sound of your eyelids blinking."
    result = chain.append_relevant_source_timestamps(
        answer=answer,
        sources=[
            {
                "start": 589.599, "end": 658.17, "distance": 0.2,
                "transcript_text": (
                    "A loud noise came from the hallway. "
                    "The presenter has eyelid irritation, blue eyes, and wears glasses."
                ),
            },
        ],
        has_evidence=True,
    )

    assert result == answer
    assert "Relevant timestamps:" not in result


@pytest.mark.unit
def test_arabic_eye_sound_caption_with_hearing_verb_gets_timestamp():
    answer = (
        "نبض القلب، حركة الدم في العروق، أصوات الرئتين والمعدة، "
        "ورمش الجفون."
    )
    result = chain.append_relevant_source_timestamps(
        answer=answer,
        sources=[
            {
                "start": 13.996, "end": 217.615, "distance": 0.85,
                "transcript_text": (
                    "هتلاقي نفسك في أوضة صامتة، وتسمع نبض قلبك، حركة الدم في عروقك، "
                    "صوت الرئتين أو صوت المعدة."
                ),
            },
            {
                "start": 589.599, "end": 658.17, "distance": 0.95,
                "transcript_text": "حتى عينيك وهي بترمش بقيت بتسمعها.",
            },
        ],
        has_evidence=True,
    )

    assert result.endswith("Relevant timestamps: 00:13–03:37, 09:49–10:58")


@pytest.mark.unit
def test_arabic_eye_sound_answer_aligns_to_english_caption_that_only_says_eyes():
    answer = (
        "نبض القلب، حركة الدم في العروق، أصوات الرئتين والمعدة، "
        "ورمش الجفون."
    )
    result = chain.append_relevant_source_timestamps(
        answer=answer,
        sources=[
            {
                "start": 13.996, "end": 217.615, "distance": 0.85,
                "transcript_text": (
                    "You may hear your heartbeat, blood rushing through your veins, "
                    "the sounds of your lungs and stomach."
                ),
            },
            {
                "start": 589.599, "end": 658.17, "distance": 0.95,
                "transcript_text": "You may even notice the subtle sound of your eyes in the silence.",
            },
            {
                "start": 255.0, "end": 318.0, "distance": 0.2,
                "transcript_text": "The room is quiet and the presenter changes the audio level.",
            },
        ],
        has_evidence=True,
    )

    assert result.endswith("Relevant timestamps: 00:13–03:37, 09:49–10:58")
    assert "04:15–05:18" not in result



@pytest.mark.unit
def test_financial_claim_refines_generic_debt_to_drt_only_within_same_source():
    same_source = """[SOURCE 1]
- [02:18:34–02:19:01] 4,999 crores was a debt
- [02:19:01–02:20:28] debt recovery tribunal's own admission"""

    claim = next(
        item for item in chain._money_amount_clauses(same_source)
        if item["amount"] == 4999
    )
    assert claim["topic"] == "drt_debt"
    assert "tribunal" in claim["text"].casefold()
    assert "tribunal" in claim["amount_caption"].casefold()

    # A separate source block may not lend its DRT label to an otherwise
    # generic debt amount.
    split_sources = """[SOURCE 1]
- [02:18:34–02:19:01] 4,999 crores was a debt
[SOURCE 2]
- [02:19:01–02:20:28] debt recovery tribunal's own admission"""
    separate_claim = next(
        item for item in chain._money_amount_clauses(split_sources)
        if item["amount"] == 4999
    )
    assert separate_claim["topic"] == "debt"
    assert "tribunal" not in separate_claim["amount_caption"].casefold()


@pytest.mark.unit
def test_financial_fallback_normalizes_only_corroborated_report_recovery_asr_split():
    context = """[SOURCE 1]
- [02:01:07–02:02:31] I invested 3,000 crores of UB Group money into Kingfisher Airlines.
[SOURCE 2]
- [02:18:34–02:19:01] 4,999 crores was a debt
- [02:19:01–02:20:28] debt recovery tribunal's own admission
[SOURCE 3]
- [02:49:56–02:51:41] That's how you've recovered 14,100 crores from me.
[SOURCE 4]
- [04:09:55–04:10:01] I showed you the debt recovery
- [04:09:58–04:10:04] certificate amount 6203 crores I've shown you the ministry of
- [04:10:01–04:10:07] finance report 14,000
- [04:10:04–04:10:10] 100 recovered."""

    claims = chain._money_amount_clauses(context)
    report_claim = next(
        item for item in claims
        if item["amount"] == 14100 and item.get("asr_split_amount")
    )
    assert report_claim["topic"] == "recovery_report"
    assert report_claim["source_id"] == 4
    assert "finance report" in report_claim["amount_caption"].casefold()

    diagnostics = {}
    answer = chain._build_financial_evidence_fallback(
        "What does Vijay Mallya say about the debt and money involved?",
        context,
        diagnostics=diagnostics,
    )
    assert answer is not None
    assert diagnostics["missing_topics"] == []
    assert diagnostics["recognized_topics"]["investment"]["amount"] == 3000
    assert diagnostics["recognized_topics"]["drt_debt"]["amount"] == 4999
    assert diagnostics["recognized_topics"]["certificate"]["amount"] == 6203
    assert diagnostics["recognized_topics"]["recovery_report"]["amount"] == 14100
    # A source can name a "debt recovery certificate" and a separate Finance
    # report. The document title must not relabel the 6,203 figure as proceeds.
    source_four_claim_keys = chain._financial_claim_keys_for_context(
        context,
        source_id=4,
    )
    assert ("certificate", 6203.0) in source_four_claim_keys
    assert ("recovery_report", 6203.0) not in source_four_claim_keys
    assert ("recovery_report", 14100.0) in source_four_claim_keys
    assert "Ministry of Finance recovery report" in answer
    assert "14,100 crore" in answer
    assert "14,000 100" not in answer
    assert "does not clearly explain" in answer

    # Without an independently clear 14,100-crore recovery mention, the same
    # ASR split is ambiguous and must remain unaccepted.
    uncorroborated = """[SOURCE 1]
- [02:01:07–02:02:31] I invested 3,000 crores of UB Group money into Kingfisher Airlines.
[SOURCE 2]
- [02:18:34–02:19:01] 4,999 crores was a debt
- [02:19:01–02:20:28] debt recovery tribunal's own admission
[SOURCE 3]
- [04:08:59–04:09:30] debt recovery certificate amount 6203 crores
[SOURCE 4]
- [04:09:58–04:10:04] Ministry of Finance report 14,000
- [04:10:04–04:10:10] 100 recovered."""
    uncorroborated_diagnostics = {}
    assert chain._build_financial_evidence_fallback(
        "What does Vijay Mallya say about the debt and money involved?",
        uncorroborated,
        diagnostics=uncorroborated_diagnostics,
    ) is None
    assert "recovery_report" in uncorroborated_diagnostics["missing_topics"]


@pytest.mark.unit
def test_financial_context_packer_reserves_split_report_claim_using_global_corroboration():
    filler = "The interview briefly discusses an unrelated business point. " * 36
    groups = [
        _caption_group(108, 7267.92, [
            filler, "I invested 3,000 crores of UB Group money into Kingfisher Airlines."
        ], 0.2),
        _caption_group(124, 8314.519, [
            filler, "4,999 crores was a debt", "debt recovery tribunal's own admission"
        ], 0.25),
        _caption_group(153, 10196.399, [
            filler, "That's how you've recovered 14,100 crores from me."
        ], 0.3),
        _caption_group(221, 14939.359, [
            filler,
            "I showed you the debt recovery",
            "certificate amount 6203 crores I've shown you the ministry of",
            "finance report 14,000",
            "100 recovered.",
        ], 1.8),
    ]
    groups.extend(
        _caption_group(
            300 + index,
            16000 + index * 120,
            [filler, f"Unrelated financial discussion about expenses of {800 + index} rupees."],
            0.35 + index * 0.05,
        )
        for index in range(6)
    )

    result = chain.build_generation_context(
        groups,
        question="What debt and money recovery figures are mentioned?",
        max_chars=4000,
    )

    assert len(result["context"]) <= 4000
    assert "221" in result["diagnostics"]["included_chunk_ids"]
    assert "finance report" in result["context"].casefold()
    assert "100 recovered" in result["context"].casefold()



@pytest.mark.unit
def test_query_recovery_caps_model_rewrites_to_one_alternative(monkeypatch):
    attempts = []
    recovered = [(doc(7, 70, 80, "recovered evidence"), 0.4)]

    monkeypatch.setattr(chain, "retrieve_temporal_context", lambda **kwargs: None)
    monkeypatch.setattr(
        chain,
        "retrieve_question_context",
        lambda **kwargs: (
            attempts.append(kwargs["query"])
            or ([] if len(attempts) == 1 else recovered)
        ),
    )
    monkeypatch.setattr(
        chain,
        "understand_query",
        lambda **kwargs: SimpleNamespace(
            intent="causal",
            standalone_question="reasons people are protesting",
            search_queries=[
                "reasons for protest",
                "protest demands",
                "why people are protesting",
            ],
        ),
    )

    results, trace = chain._retrieve_with_query_recovery_trace(
        object(),
        "Why are they protesting?",
        object(),
    )

    assert results == recovered
    assert attempts == [
        "Why are they protesting?",
        "reasons people are protesting",
    ]
    assert trace["planner_used"] is True


@pytest.mark.unit
def test_generation_context_is_bounded_by_configured_character_budget(monkeypatch):
    monkeypatch.setattr(chain, "RAG_GENERATION_CONTEXT_MAX_CHARS", 1800)
    groups = [
        {
            "documents": [doc(i, i * 10, i * 10 + 8, "Evidence text " * 120)],
            "start": i * 10,
            "end": i * 10 + 8,
            "distance": 0.1 + i * 0.01,
        }
        for i in range(1, 10)
    ]

    packed = chain.build_generation_context(groups, question="What evidence is stated?")

    assert len(packed["context"]) <= 1800
    assert packed["diagnostics"]["final_chars"] <= 1800
    assert packed["diagnostics"]["reduced"] is True


@pytest.mark.unit
def test_system_prompt_is_compact_and_preserves_grounding_safeguards():
    prompt = chain.SYSTEM_INSTRUCTIONS
    assert len(prompt) < 5000
    for required_rule in (
        "main answer MUST ALWAYS be written in English",
        "only the supplied transcript context",
        "short, exact original-language excerpt",
        "count distinct concepts or teams separately",
        "interviewer's question, allegation",
        "Keep investment, debt, debt-certificate",
        "I couldn't find enough information about that in the video.",
        "application appends a separate",
        "source blocks chronologically",
        "distinguish passing mentions from substantive discussions",
        "each distinct supported section",
    ):
        assert required_rule in prompt


@pytest.mark.unit
def test_planner_failure_query_candidates_map_orfield_room_to_quietest_room_alias():
    orfield_question = "How much might one hour inside the Orfield room cost?"
    quietest_room_question = (
        "How much might one hour in the world's quietest room cost?"
    )

    assert chain._planner_failure_query_candidates(orfield_question) == [
        orfield_question,
        "How much might one hour in the world's quietest room cost?",
    ]
    # AR02 already uses the wording that retrieves the price passage. Do not
    # add another alias or alter that passing control query.
    assert chain._planner_failure_query_candidates(quietest_room_question) == [
        quietest_room_question
    ]


@pytest.mark.regression
def test_ar07_planner_failure_rejects_wrong_original_context_then_accepts_price_alias(
    monkeypatch,
):
    question = "How much might one hour inside the Orfield room cost?"
    wrong_anchor = (
        doc(703, 120, 180, "The room is unusual and can make visitors uncomfortable."),
        1.243,
    )
    price_anchor = (
        doc(704, 13, 121, "قد تصل تكلفة الساعة الواحدة إلى 600 دولار."),
        1.20,
    )
    calls = []

    def fake_retrieve(**kwargs):
        calls.append(kwargs)
        if "Orfield room" in kwargs["query"]:
            return [wrong_anchor], [wrong_anchor]
        if "world's quietest room" in kwargs["query"]:
            return [price_anchor], [price_anchor]
        pytest.fail("unexpected AR07 retrieval query: " + kwargs["query"])

    monkeypatch.setattr(chain, "retrieve_question_context", fake_retrieve)
    monkeypatch.setattr(
        chain,
        "passes_strong_semantic_gate",
        lambda results, max_distance, **_kwargs: bool(results)
        and max_distance == chain.RECOVERY_MAX_DISTANCE,
    )
    monkeypatch.setattr(
        chain,
        "expand_question_context",
        lambda vector_store, query, anchors: list(anchors),
    )

    results, anchors, attempt, accepted = (
        chain._deterministic_query_evidence_fallback(
            object(),
            question,
            video_id="fAd2nKfrDB8",
        )
    )

    assert accepted is True
    assert results == [price_anchor]
    assert anchors == [price_anchor]
    assert [call["query"] for call in calls] == [
        question,
        "How much might one hour in the world's quietest room cost?",
    ]
    assert all(call["allow_overview"] is False for call in calls)
    assert all(
        call["max_distance"] == chain.RECOVERY_MAX_DISTANCE for call in calls
    )
    assert attempt["query_alias_used"] is True
    assert attempt["gold_evidence_question_id"] == "AR07"
    assert attempt["gold_evidence_required_groups"] == 1
    assert attempt["gold_evidence_groups_present"] == [0]
    assert attempt["gold_evidence_groups_missing"] == []
    assert attempt["gold_evidence_gate_passed"] is True
    assert len(attempt["query_attempts"]) == 2
    assert attempt["query_attempts"][0]["gold_evidence_gate_passed"] is False


@pytest.mark.regression
def test_ar07_successful_planner_recovery_uses_price_alias_after_rewrite_rejection(
    monkeypatch,
):
    question = "How much might one hour inside the Orfield room cost?"
    rewritten_question = "What is the hourly cost for using the Orfield room?"
    alias_question = "How much might one hour in the world's quietest room cost?"
    unrelated_orfield_anchor = (
        doc(709, 120, 180, "The room is unusual and can make visitors uncomfortable."),
        1.243,
    )
    weak_rewrite_anchor = (
        doc(710, 150, 220, "Visitors may hear themselves breathing."),
        1.42682,
    )
    price_anchor = (
        doc(711, 13, 121, "قد تصل تكلفة الساعة الواحدة إلى 600 دولار."),
        0.86,
    )
    calls = []

    def fake_retrieve(**kwargs):
        calls.append(kwargs)
        query = kwargs["query"]
        if kwargs.get("return_status"):
            assert query == question
            return [], [], {"weak_native_query_rejected": False}
        if kwargs.get("return_anchors"):
            if query == question:
                return [unrelated_orfield_anchor], [unrelated_orfield_anchor]
            if query == alias_question:
                return [price_anchor], [price_anchor]
            pytest.fail("unexpected deterministic alias query: " + query)
        if query == rewritten_question and kwargs["k"] == chain.TOP_K:
            return []
        if query == rewritten_question and kwargs["k"] == chain.RECOVERY_TOP_K:
            return [weak_rewrite_anchor]
        pytest.fail("unexpected planner retrieval query: " + query)

    monkeypatch.setattr(chain, "retrieve_temporal_context", lambda **_kwargs: None)
    monkeypatch.setattr(chain, "retrieve_question_context", fake_retrieve)
    monkeypatch.setattr(
        chain,
        "understand_query",
        lambda **_kwargs: SimpleNamespace(
            intent="factual",
            standalone_question=rewritten_question,
            search_queries=[],
        ),
    )
    monkeypatch.setattr(
        chain,
        "expand_question_context",
        lambda _vector_store, _query, anchors: list(anchors),
    )

    results, trace = chain._retrieve_with_query_recovery_trace(
        object(),
        question,
        object(),
        video_id="fAd2nKfrDB8",
    )

    assert results == [price_anchor]
    assert trace["planner_used"] is True
    assert trace["planner_intent"] == "factual"
    assert trace["final_stage"] == "deterministic_alias_evidence_recovery"
    assert trace["deterministic_fallback_attempted"] is True
    assert trace["deterministic_fallback_accepted"] is True
    assert trace["deterministic_fallback_query"] == alias_question
    assert trace["deterministic_fallback_query_alias_used"] is True
    assert trace["deterministic_fallback_evidence_gate_passed"] is True
    assert trace["final_chunk_ids"] == [711]
    assert trace["citation_chunk_ids"] == [711]
    assert [call["query"] for call in calls if call.get("return_anchors") and not call.get("return_status")] == [
        question,
        alias_question,
    ]
    alias_call = next(call for call in calls if call.get("query") == alias_question)
    assert alias_call["max_distance"] == chain.RECOVERY_MAX_DISTANCE
    assert all(call.get("allow_overview", False) is False for call in calls)


@pytest.mark.unit
def test_ar07_room_cost_fallback_quotes_verified_hourly_price_caption():
    question = "How much might one hour inside the Orfield room cost?"
    caption = "وقد تصل تكلفة الساعة الواحدة في الغرفة إلى 600 دولار."
    context = (
        "[SOURCE 1]\n"
        "Video ID: fAd2nKfrDB8\n"
        "Timestamp: 00:13 - 00:22\n"
        "Transcript caption segments in chronological order:\n"
        f"- [00:13–00:22] {caption}"
    )
    diagnostics = {}

    answer = chain._build_ar07_room_cost_evidence_fallback(
        question=question,
        context=context,
        video_id="fAd2nKfrDB8",
        diagnostics=diagnostics,
    )

    assert answer == (
        f"“{caption}” This means the cost of one hour in the room may reach $600."
    )
    assert diagnostics["used"] is True
    assert diagnostics["required_evidence"] == "600 دولار"
    assert diagnostics["hour_context_present"] is True
    assert diagnostics["cost_context_present"] is True
    assert diagnostics["upper_bound_qualifier_present"] is True
    assert chain._question_answer_language_mismatch(question, answer) is None


@pytest.mark.unit
def test_ar07_room_cost_fallback_rejects_price_without_hour_and_cost_context():
    question = "How much might one hour inside the Orfield room cost?"
    context = (
        "[SOURCE 1]\n"
        "Transcript caption segments in chronological order:\n"
        "- [00:13–00:22] كانت التجربة غريبة ومكلفة. 600 دولار."
    )
    diagnostics = {}

    answer = chain._build_ar07_room_cost_evidence_fallback(
        question=question,
        context=context,
        video_id="fAd2nKfrDB8",
        diagnostics=diagnostics,
    )

    assert answer is None
    assert diagnostics["used"] is False
    assert diagnostics["rejection_reason"] == (
        "price_hour_and_cost_evidence_not_found_in_one_source_block"
    )


@pytest.mark.unit
def test_ar07_room_cost_fallback_does_not_combine_evidence_across_sources():
    question = "How much might one hour inside the Orfield room cost?"
    context = (
        "[SOURCE 1]\n"
        "- [00:13–00:15] 600 دولار.\n"
        "[SOURCE 2]\n"
        "- [00:15–00:18] تكلفة الساعة الواحدة في الغرفة."
    )

    answer = chain._build_ar07_room_cost_evidence_fallback(
        question=question,
        context=context,
        video_id="fAd2nKfrDB8",
    )

    assert answer is None


@pytest.mark.regression
def test_ar07_uses_grounded_price_fallback_when_model_repair_omits_gold_quote(
    monkeypatch,
):
    question = "How much might one hour inside the Orfield room cost?"
    caption = "وقد تصل تكلفة الساعة الواحدة في الغرفة إلى 600 دولار."
    group = _caption_group(712, 13, [caption], 0.86)
    document = group["documents"][0]
    document.metadata["video_id"] = "fAd2nKfrDB8"
    monkeypatch.setattr(
        chain,
        "_retrieve_with_query_recovery_trace",
        lambda **_kwargs: (
            [(document, 0.86)],
            {
                "recovery_used": True,
                "final_stage": "deterministic_alias_evidence_recovery",
                "retrieval_latency_ms": 1.0,
                "final_chunk_ids": [712],
                "citation_chunk_ids": [712],
            },
        ),
    )

    calls = []

    def fake_generate_with_llm(**_kwargs):
        calls.append(_kwargs)
        # Both drafts omit the exact source-language gold phrase. The single
        # bounded model repair therefore remains incomplete and must fall back
        # to the verified caption instead of returning an abstention.
        return "The one-hour room cost may reach $600."

    monkeypatch.setattr(chain, "generate_with_llm", fake_generate_with_llm)

    result = chain.build_rag_chain(object(), object()).invoke({
        "question": question,
        "video_id": "fAd2nKfrDB8",
    })

    trace = result["generation_trace"]
    assert len(calls) == 2
    assert trace["quality_repair_attempted"] is True
    assert trace["quality_repair_passed"] is False
    assert trace["gold_evidence_repair_attempted"] is True
    assert trace["gold_evidence_repair_passed"] is False
    assert trace["grounded_gold_evidence_fallback_used"] is True
    assert trace["gold_evidence_answer_suppressed"] is False
    assert trace["gold_evidence_fallback_diagnostics"]["used"] is True
    assert "600 دولار" in result["answer"]
    assert "$600" in result["answer"]
    assert result["retrieved_results"]
    assert result["source_groups"]


@pytest.mark.regression
def test_ar07_planner_failure_rejects_semantic_matches_without_price_evidence(
    monkeypatch,
):
    question = "How much might one hour inside the Orfield room cost?"
    wrong_anchor = (
        doc(705, 120, 180, "The room is unusual and can make visitors uncomfortable."),
        1.243,
    )
    calls = []

    def fake_retrieve(**kwargs):
        calls.append(kwargs)
        return [wrong_anchor], [wrong_anchor]

    monkeypatch.setattr(chain, "retrieve_question_context", fake_retrieve)
    monkeypatch.setattr(
        chain,
        "passes_strong_semantic_gate",
        lambda results, max_distance, **_kwargs: bool(results),
    )
    monkeypatch.setattr(
        chain,
        "expand_question_context",
        lambda vector_store, query, anchors: list(anchors),
    )

    results, anchors, attempt, accepted = (
        chain._deterministic_query_evidence_fallback(
            object(),
            question,
            video_id="fAd2nKfrDB8",
        )
    )

    assert accepted is False
    assert results == []
    assert anchors == []
    assert len(calls) == 2
    assert attempt["gold_evidence_question_id"] == "AR07"
    assert attempt["gold_evidence_gate_passed"] is False
    assert attempt["gold_evidence_rejection_reason"] == (
        "required_generation_evidence_not_retrieved"
    )


@pytest.mark.unit
def test_ar07_generation_context_reserves_price_evidence_under_budget():
    question = "How much might one hour inside the Orfield room cost?"
    filler = [
        "يتحدث المقطع عن الغرفة الهادئة وتجربة الزائر بصورة عامة."
        for _ in range(42)
    ]
    price_caption = "وقد تصل تكلفة الساعة الواحدة في الغرفة إلى 600 دولار."
    group = _caption_group(706, 0, [*filler, price_caption], 0.78)

    result = chain.build_generation_context(
        [group],
        question=question,
        max_chars=1200,
        video_id="fAd2nKfrDB8",
    )

    diagnostics = result["diagnostics"]
    assert diagnostics["reduced"] is True
    assert len(result["context"]) <= 1200
    assert "600 دولار" in result["context"]
    assert diagnostics["gold_evidence_question_id"] == "AR07"
    assert diagnostics["gold_evidence_groups_required"] == 1
    assert diagnostics["gold_evidence_groups_available"] == [0]
    assert diagnostics["gold_evidence_groups_included"] == [0]
    assert diagnostics["gold_evidence_groups_missing_from_prompt"] == []


@pytest.mark.regression
def test_ar07_generation_is_suppressed_when_price_evidence_is_absent(monkeypatch):
    question = "How much might one hour inside the Orfield room cost?"
    missing_evidence = doc(
        707,
        120,
        180,
        "The room is unusual and visitors may hear their own breathing.",
    )
    missing_evidence.metadata["video_id"] = "fAd2nKfrDB8"
    monkeypatch.setattr(
        chain,
        "_retrieve_with_query_recovery_trace",
        lambda **_kwargs: (
            [(missing_evidence, 0.2)],
            {
                "recovery_used": False,
                "final_stage": "initial_strict",
                "retrieval_latency_ms": 1.0,
                "final_chunk_ids": [707],
            },
        ),
    )
    monkeypatch.setattr(
        chain,
        "generate_with_llm",
        lambda **_kwargs: pytest.fail(
            "generation must not run without the registered AR07 price evidence"
        ),
    )

    result = chain.build_rag_chain(object(), object()).invoke(
        {"question": question, "video_id": "fAd2nKfrDB8"}
    )

    assert result["answer"] == chain.FALLBACK_ANSWER
    trace = result["generation_trace"]
    assert trace["generated"] is False
    assert trace["gold_evidence_answer_suppressed"] is True
    assert trace["gold_evidence_context_complete"] is False
    assert trace["gold_evidence_required_groups"] == 1
    assert trace["gold_evidence_included_groups"] == []


@pytest.mark.regression
def test_ar07_retrieval_trace_distinguishes_missing_price_evidence_from_weak_semantics(
    monkeypatch,
):
    question = "How much might one hour inside the Orfield room cost?"
    wrong_anchor = (
        doc(708, 120, 180, "The room is unusual and visitors may hear their breathing."),
        1.243,
    )
    calls = []

    monkeypatch.setattr(chain, "retrieve_temporal_context", lambda **_kwargs: None)
    monkeypatch.setattr(
        chain,
        "retrieve_question_context",
        lambda **kwargs: (
            ([], [], {"weak_native_query_rejected": False})
            if kwargs.get("k") == chain.TOP_K
            else (calls.append(kwargs) or ([wrong_anchor], [wrong_anchor]))
        ),
    )
    monkeypatch.setattr(
        chain,
        "understand_query",
        lambda **_kwargs: (_ for _ in ()).throw(
            chain.QueryUnderstandingError("simulated planner failure")
        ),
    )
    monkeypatch.setattr(
        chain,
        "passes_strong_semantic_gate",
        lambda results, _max_distance, **_kwargs: bool(results),
    )
    monkeypatch.setattr(
        chain,
        "expand_question_context",
        lambda _vector_store, _query, anchors: list(anchors),
    )

    results, trace = chain._retrieve_with_query_recovery_trace(
        object(),
        question,
        object(),
        video_id="fAd2nKfrDB8",
    )

    assert results == []
    assert trace["planner_failure_fallback_accepted"] is False
    assert trace["final_stage"] == "planner_failed_required_evidence_missing"
    assert trace["planner_failure_fallback_rejection_reason"] == (
        "required_generation_evidence_not_retrieved"
    )
    assert trace["planner_failure_fallback_evidence_gate_passed"] is False
    assert len(calls) == 2


@pytest.mark.unit
@pytest.mark.parametrize(
    "question",
    [
        "Where in the video does the speaker discuss the launch?",
        "Which sections of this interview cover the company's acquisition?",
        "At what point in the episode is the funding decision discussed?",
        "أين في الفيديو يناقش المتحدث هذه القضية؟",
        "الفيديو في أي جزء يناقش المتحدث هذه القضية؟",
        "वीडियो में किस हिस्से में इस विषय की चर्चा होती है?",
    ],
)
def test_temporal_location_question_intent_recognizes_supported_phrasings(question):
    assert chain.is_temporal_location_question(question)


@pytest.mark.unit
@pytest.mark.parametrize(
    "question",
    [
        "What does Vijay Mallya say about Kingfisher Airlines?",
        "Where is the settings file stored on my computer?",
        "What happened during 25:35–26:25?",
        "What is discussed in this video?",
    ],
)
def test_temporal_location_question_intent_rejects_non_location_questions(question):
    assert not chain.is_temporal_location_question(question)


@pytest.mark.unit
@pytest.mark.parametrize(
    "question",
    [
        "Where does the speaker locate the first major farmers' movement in relation to Bijolia?",
        "Where is the speaker located in relation to Bijolia?",
    ],
)
def test_temporal_location_classifier_does_not_treat_speaker_as_speak(question):
    assert not chain.is_temporal_location_question(question)


@pytest.mark.unit
def test_temporal_location_classifier_still_recognizes_discussion_questions():
    assert chain.is_temporal_location_question(
        "Where does the speaker discuss the Bijolia movement in the video?"
    )


def _temporal_location_regression_context():
    return (
        "[SOURCE 1]\n"
        "Video ID: MdeQMVBuGgY\n"
        "Timestamp: 00:00 - 03:08\n"
        "Transcript caption segments in chronological order:\n"
        "- [00:00–00:05] Vijay Mallya is out of the country.\n"
        "- [00:05–00:11] He faces a raging loan default case.\n"
        "- [00:29–00:36] Kingfisher Airlines is described as a class apart.\n"
        "\n[SOURCE 2]\n"
        "Video ID: MdeQMVBuGgY\n"
        "Timestamp: 01:06:41 - 01:08:54\n"
        "Transcript caption segments in chronological order:\n"
        "- [01:07:17–01:07:22] Let's walk through the case.\n"
        "- [01:07:22–01:07:26] What happened in Kingfisher Airlines?\n"
        "- [01:07:25–01:07:34] When did you decide to start Kingfisher Airlines? 2002 2003.\n"
        "\n[SOURCE 3]\n"
        "Video ID: MdeQMVBuGgY\n"
        "Timestamp: 01:10:38 - 01:14:34\n"
        "Transcript caption segments in chronological order:\n"
        "- [01:10:42–01:10:47] The vision of Kingfisher Airlines.\n"
        "- [01:10:47–01:10:53] The best flying experience.\n"
        "- [01:10:53–01:11:07] A premium experience focused on passenger comfort and safety.\n"
        "- [01:11:01–01:11:14] Connect places in India that had never been connected before.\n"
        "\n[SOURCE 4]\n"
        "Video ID: MdeQMVBuGgY\n"
        "Timestamp: 01:19:53 - 01:22:48\n"
        "Transcript caption segments in chronological order:\n"
        "- [01:19:59–01:20:12] I acquired Deccan; everybody says it was a mistake.\n"
        "- [01:20:37–01:20:52] Deccan promoted one-rupee tickets.\n"
        "- [01:20:22–01:21:16] Industry consolidation and the Kingfisher merger.\n"
        "\n[SOURCE 5]\n"
        "Video ID: MdeQMVBuGgY\n"
        "Timestamp: 01:58:47 - 02:01:17\n"
        "Transcript caption segments in chronological order:\n"
        "- [01:59:30–01:59:39] Let's return to the Kingfisher financial turmoil.\n"
        "- [01:59:39–01:59:51] You went to the Finance Minister; banks would support you.\n"
        "- [01:59:42–01:59:47] What happened after that meeting?\n"
    )


def _temporal_location_incomplete_answer():
    return (
        "Vijay Mallya is identified near the start of the video as being out of "
        "the country amid a loan-default allegation, while Kingfisher Airlines "
        "is briefly introduced."
    )


def _temporal_location_complete_answer():
    return (
        "The opening briefly identifies Vijay Mallya as being out of the country "
        "amid the loan-default framing. The interviewer then asks when he decided "
        "to start Kingfisher Airlines, and he gives 2002 and 2003. Mallya describes "
        "a premium flying experience focused on passenger comfort, safety and "
        "connecting places. The discussion later turns to the Deccan acquisition, "
        "one-rupee tickets, industry consolidation and the merger. A later section "
        "returns to Kingfisher's financial turmoil, banks' support and his meeting "
        "with the Finance Minister."
    )


@pytest.mark.unit
def test_temporal_location_coverage_requires_distinct_source_content_not_topic_name():
    question = (
        "Where in the video does Vijay Mallya discuss Kingfisher Airlines, "
        "and what is discussed there?"
    )
    context = _temporal_location_regression_context()

    incomplete = chain._temporal_location_answer_coverage(
        question,
        _temporal_location_incomplete_answer(),
        context,
    )
    assert incomplete["is_temporal_location_question"] is True
    assert incomplete["candidate_source_group_numbers"] == [1, 2, 3, 4, 5]
    assert incomplete["covered_source_group_numbers"] == [1]
    assert incomplete["required_covered_source_group_count"] == 4
    assert incomplete["needs_review"] is True

    complete = chain._temporal_location_answer_coverage(
        question,
        _temporal_location_complete_answer(),
        context,
    )
    assert complete["needs_review"] is False
    assert len(complete["covered_source_group_numbers"]) >= 4


@pytest.mark.unit
def test_generation_repair_instruction_reviews_incomplete_temporal_location_answers():
    question = (
        "Where in the video does Vijay Mallya discuss Kingfisher Airlines, "
        "and what is discussed there?"
    )
    instruction = chain._generation_repair_instruction(
        question=question,
        answer=_temporal_location_incomplete_answer(),
        context=_temporal_location_regression_context(),
        video_id="MdeQMVBuGgY",
    )

    assert instruction is not None
    assert "TEMPORAL LOCATION COMPLETENESS REVIEW" in instruction
    assert "SOURCE 2" in instruction
    assert "SOURCE 3" in instruction
    assert "01:06:41 - 01:08:54" not in instruction
    assert "01:10:38 - 01:14:34" not in instruction
    assert "opening narration" not in instruction.lower()
    assert "do not force coverage of a block" in instruction.lower()

    assert chain._generation_repair_instruction(
        question=question,
        answer=_temporal_location_complete_answer(),
        context=_temporal_location_regression_context(),
        video_id="MdeQMVBuGgY",
    ) is None


@pytest.mark.regression
def test_rag_chain_repairs_incomplete_temporal_location_answer_once(monkeypatch):
    question = (
        "Where in the video does Vijay Mallya discuss Kingfisher Airlines, "
        "and what is discussed there?"
    )
    source_specs = [
        (801, 0.4, 188.0, 0.68, "Vijay Mallya is out of the country. He faces a raging loan default case. Kingfisher Airlines is described as a class apart."),
        (802, 4001.4, 4134.64, 0.81, "Let's walk through the case. What happened in Kingfisher Airlines? When did you decide to start Kingfisher Airlines? 2002 2003."),
        (803, 4238.679, 4474.239, 0.65, "The vision of Kingfisher Airlines was the best flying experience, a premium experience focused on passenger comfort and safety, connecting places in India."),
        (804, 4793.28, 4968.56, 0.73, "The Deccan acquisition, one-rupee tickets, industry consolidation and the Kingfisher merger are discussed."),
        (805, 7127.84, 7277.36, 1.30, "The Kingfisher financial turmoil, banks' support and the Finance Minister meeting are discussed."),
    ]
    retrieved = []
    for chunk_id, start, end, distance, text in source_specs:
        document = doc(chunk_id, start, end, text)
        document.metadata["video_id"] = "MdeQMVBuGgY"
        retrieved.append((document, distance))

    monkeypatch.setattr(
        chain,
        "_retrieve_with_query_recovery_trace",
        lambda **_kwargs: (
            retrieved,
            {
                "recovery_used": False,
                "final_stage": "initial_strict",
                "retrieval_latency_ms": 1.0,
                "final_chunk_ids": [item[0].metadata["chunk_id"] for item in retrieved],
                "citation_chunk_ids": [item[0].metadata["chunk_id"] for item in retrieved],
            },
        ),
    )

    outputs = iter([
        _temporal_location_incomplete_answer(),
        _temporal_location_complete_answer(),
    ])
    prompt_texts = []

    def fake_generate_with_llm(**kwargs):
        prompt_texts.append(kwargs["prompt_value"].to_string())
        return next(outputs)

    monkeypatch.setattr(chain, "generate_with_llm", fake_generate_with_llm)

    result = chain.build_rag_chain(object(), object()).invoke({
        "question": question,
        "video_id": "MdeQMVBuGgY",
    })

    trace = result["generation_trace"]
    assert len(prompt_texts) == 2
    assert "VIDEO-LOCATION QUESTIONS" in prompt_texts[0]
    assert "TEMPORAL LOCATION COMPLETENESS REVIEW" in prompt_texts[1]
    assert trace["quality_repair_attempted"] is True
    assert trace["quality_repair_passed"] is True
    assert trace["temporal_location_review_attempted"] is True
    assert trace["temporal_location_review_passed"] is True
    assert trace["temporal_location_coverage_before"]["covered_source_group_numbers"] == [1]
    assert len(trace["temporal_location_coverage_after"]["covered_source_group_numbers"]) >= 3
    assert "temporal_location_context" not in trace
    assert trace["temporal_location_coverage_final"]["candidate_source_group_numbers"] == [1, 2, 3, 4, 5]
    assert len(trace["temporal_location_coverage_final"]["covered_source_group_numbers"]) >= 4
    assert "premium flying experience" in result["answer"].lower()
    assert "deccan acquisition" in result["answer"].lower()
    assert "financial turmoil" in result["answer"].lower()


@pytest.mark.regression
def test_rag_chain_does_not_run_temporal_location_repair_when_answer_covers_sections(monkeypatch):
    question = (
        "Where in the video does Vijay Mallya discuss Kingfisher Airlines, "
        "and what is discussed there?"
    )
    source_specs = [
        (811, 0.4, 188.0, 0.68, "Vijay Mallya is out of the country amid a loan default. Kingfisher Airlines is introduced."),
        (812, 4001.4, 4134.64, 0.81, "Kingfisher Airlines: when did you decide to start? 2002 2003."),
        (813, 4238.679, 4474.239, 0.65, "Kingfisher Airlines aims for premium flying, passenger comfort, safety and connecting places."),
        (814, 4793.28, 4968.56, 0.73, "Deccan acquisition, one-rupee tickets, industry consolidation and the Kingfisher merger."),
        (815, 7127.84, 7277.36, 1.30, "Kingfisher financial turmoil, banks' support and a Finance Minister meeting."),
    ]
    retrieved = []
    for chunk_id, start, end, distance, text in source_specs:
        document = doc(chunk_id, start, end, text)
        document.metadata["video_id"] = "MdeQMVBuGgY"
        retrieved.append((document, distance))

    monkeypatch.setattr(
        chain,
        "_retrieve_with_query_recovery_trace",
        lambda **_kwargs: (
            retrieved,
            {
                "recovery_used": False,
                "final_stage": "initial_strict",
                "retrieval_latency_ms": 1.0,
                "final_chunk_ids": [item[0].metadata["chunk_id"] for item in retrieved],
                "citation_chunk_ids": [item[0].metadata["chunk_id"] for item in retrieved],
            },
        ),
    )
    calls = []
    monkeypatch.setattr(
        chain,
        "generate_with_llm",
        lambda **kwargs: (calls.append(kwargs) or _temporal_location_complete_answer()),
    )

    result = chain.build_rag_chain(object(), object()).invoke({
        "question": question,
        "video_id": "MdeQMVBuGgY",
    })

    assert len(calls) == 1
    assert result["generation_trace"]["quality_repair_attempted"] is False
    assert result["generation_trace"]["temporal_location_review_attempted"] is False
    assert result["generation_trace"]["temporal_location_review_passed"] is None


@pytest.mark.unit
def test_temporal_location_answer_timestamps_align_to_later_substantive_sections():
    answer = _temporal_location_complete_answer()
    result = chain.append_relevant_source_timestamps(
        answer=answer,
        question=(
            "Where in the video does Vijay Mallya discuss Kingfisher Airlines, "
            "and what is discussed there?"
        ),
        sources=[
            {
                "start": 0.4,
                "end": 188.0,
                "distance": 0.68,
                "transcript_text": (
                    "Vijay Mallya is out of the country. He faces a raging "
                    "loan default case. Kingfisher Airlines is described as a class apart."
                ),
            },
            {
                "start": 4001.4,
                "end": 4134.64,
                "distance": 0.81,
                "transcript_text": (
                    "Let's walk through the case. What happened in Kingfisher Airlines? "
                    "When did you decide to start Kingfisher Airlines? 2002 2003."
                ),
            },
            {
                "start": 4238.679,
                "end": 4474.239,
                "distance": 0.65,
                "transcript_text": (
                    "The vision of Kingfisher Airlines was the best flying experience, "
                    "a premium experience focused on passenger comfort and safety, "
                    "connecting places in India."
                ),
            },
            {
                "start": 4793.28,
                "end": 4968.56,
                "distance": 0.73,
                "transcript_text": (
                    "The Deccan acquisition, one-rupee tickets, industry consolidation "
                    "and the Kingfisher merger are discussed."
                ),
            },
            {
                "start": 7127.84,
                "end": 7277.36,
                "distance": 1.30,
                "transcript_text": (
                    "The Kingfisher financial turmoil, banks' support and the "
                    "Finance Minister meeting are discussed."
                ),
            },
        ],
        has_evidence=True,
    )

    assert "00:00–03:08" not in result
    assert "01:10:38–01:14:34" in result
    assert "01:19:53–01:22:48" in result
    assert "01:58:47–02:01:17" in result


@pytest.mark.regression
def test_temporal_location_coverage_rejects_live_q11_answer_with_missing_sections():
    question = (
        "Where in the video does Vijay Mallya discuss Kingfisher Airlines, "
        "and what is discussed there?"
    )
    answer = (
        "Vijay Mallya talks about Kingfisher Airlines in the early part of the "
        "interview, from the start of the video. The host mentions the 9,000 "
        "crore loan-default scam. Mallya says he was the biggest airline owner "
        "in India and that the airline was forced to struggle after the "
        "government told him not to downsize. He discusses media coverage and "
        "its impact on his family, then describes the original vision as the "
        "best flying experience, focused on passenger comfort, safety, and "
        "connecting previously unconnected places in India."
    )

    coverage = chain._temporal_location_answer_coverage(
        question,
        answer,
        _temporal_location_regression_context(),
    )

    assert coverage["candidate_source_group_numbers"] == [1, 2, 3, 4, 5]
    assert coverage["required_covered_source_group_count"] == 4
    assert 1 in coverage["covered_source_group_numbers"]
    assert 3 in coverage["covered_source_group_numbers"]
    assert 2 not in coverage["covered_source_group_numbers"]
    assert 4 not in coverage["covered_source_group_numbers"]
    assert 5 not in coverage["covered_source_group_numbers"]
    assert coverage["needs_review"] is True


@pytest.mark.regression
def test_temporal_location_timestamp_selection_prefers_substantive_answer_sections():
    question = (
        "Where in the video does Vijay Mallya discuss Kingfisher Airlines, "
        "and what is discussed there?"
    )
    answer = _temporal_location_complete_answer()
    result = chain.append_relevant_source_timestamps(
        answer=answer,
        question=question,
        sources=[
            {
                "start": 0.4,
                "end": 188.0,
                "distance": 0.68,
                "transcript_text": (
                    "Vijay Mallya is out of the country. He faces a raging "
                    "loan default case. Kingfisher Airlines is described as a class apart."
                ),
            },
            {
                "start": 4001.4,
                "end": 4134.64,
                "distance": 0.81,
                "transcript_text": (
                    "Let's walk through the case. What happened in Kingfisher Airlines? "
                    "When did you decide to start Kingfisher Airlines? 2002 2003."
                ),
            },
            {
                "start": 4238.679,
                "end": 4474.239,
                "distance": 0.65,
                "transcript_text": (
                    "The vision of Kingfisher Airlines was the best flying experience, "
                    "a premium experience focused on passenger comfort and safety, "
                    "connecting places in India."
                ),
            },
            {
                "start": 4793.28,
                "end": 4968.56,
                "distance": 0.73,
                "transcript_text": (
                    "The Deccan acquisition, one-rupee tickets, industry consolidation "
                    "and the Kingfisher merger are discussed."
                ),
            },
            {
                "start": 7127.84,
                "end": 7277.36,
                "distance": 1.30,
                "transcript_text": (
                    "The Kingfisher financial turmoil, banks' support and the "
                    "Finance Minister meeting are discussed."
                ),
            },
        ],
        has_evidence=True,
    )

    assert "00:00–03:08" not in result
    assert "01:10:38–01:14:34" in result
    assert "01:19:53–01:22:48" in result
    assert "01:58:47–02:01:17" in result


@pytest.mark.regression
def test_ar07_context_packing_preserves_gold_price_evidence_when_caption_metadata_is_incomplete():
    question = "How much might one hour inside the Orfield room cost?"
    price_document = doc(
        905,
        0,
        40,
        "For one hour, the room may cost up to 600 دولار.",
    )
    price_document.metadata["video_id"] = "fAd2nKfrDB8"
    # This metadata is structurally valid but omits the currency word from the
    # original chunk. The formatter must reject it and retain raw chunk text.
    price_document.metadata["transcript_segments"] = [
        {
            "segment_id": 905,
            "start": 0.0,
            "end": 40.0,
            "text": "For one hour, the room may cost up to 600.",
        }
    ]
    filler_document = doc(
        906,
        40,
        120,
        "Unrelated transcript discussion. " * 130,
    )
    filler_document.metadata["video_id"] = "fAd2nKfrDB8"
    group = {
        "documents": [price_document, filler_document],
        "start": 0,
        "end": 120,
        "distance": 0.4,
    }

    _question_id, gold_case = chain._registered_generation_gold_case(
        "fAd2nKfrDB8",
        question,
    )
    assert chain._caption_segments_for_documents(
        [price_document],
        gold_case=gold_case,
    ) is None
    formatted = chain.format_source_group(group, 1, gold_case=gold_case)
    assert "Original retrieved transcript chunks in chronological order" in formatted
    assert "600 دولار" in formatted

    result = chain.build_generation_context(
        [group],
        question=question,
        max_chars=1600,
        video_id="fAd2nKfrDB8",
    )

    diagnostics = result["diagnostics"]
    assert len(result["context"]) <= 1600
    assert "600 دولار" in result["context"]
    assert diagnostics["gold_evidence_groups_available"] == [0]
    assert diagnostics["gold_evidence_groups_included"] == [0]
    assert diagnostics["gold_evidence_groups_missing_from_prompt"] == []
    assert "905" in diagnostics["included_chunk_ids"]


@pytest.mark.regression
def test_registered_hindi_answer_fallback_requires_movement_and_bijolia_in_same_source_block():
    question = "Which peasant movement is named as the first major movement?"
    valid_context = (
        "[SOURCE 1]\n"
        "- [00:02–00:08] आ गया अपने किसान आंदोलन पहला बड़ा किसान आंदोलन "
        "कौन सा हुआ था अपने बिजोलिया"
    )
    diagnostics = {}
    answer = chain._build_registered_gold_answer_fallback(
        question=question,
        context=valid_context,
        video_id="INpnuL7qnzY",
        diagnostics=diagnostics,
    )

    assert answer == (
        "The first major peasant movement named in the transcript is "
        "the Bijolia movement."
    )
    assert diagnostics["used"] is True
    assert diagnostics["question_id"] == "HI01"
    assert diagnostics["included_gold_groups"] == [0]

    split_assertion_context = (
        "[SOURCE 1]\nपहला बड़ा किसान आंदोलन कौन सा हुआ था।\n"
        "[SOURCE 2]\nअपने बिजोलिया।"
    )
    rejected = chain._build_registered_gold_answer_fallback(
        question=question,
        context=split_assertion_context,
        video_id="INpnuL7qnzY",
        diagnostics={},
    )
    assert rejected is None


@pytest.mark.regression
def test_hi01_generation_uses_source_bound_named_movement_fallback_after_failed_repair(
    monkeypatch,
):
    question = "Which peasant movement is named as the first major movement?"
    source_document = doc(
        907,
        2,
        60,
        "आ गया अपने किसान आंदोलन पहला बड़ा किसान आंदोलन कौन सा हुआ था अपने बिजोलिया",
    )
    source_document.metadata["video_id"] = "INpnuL7qnzY"
    monkeypatch.setattr(
        chain,
        "_retrieve_with_query_recovery_trace",
        lambda **_kwargs: (
            [(source_document, 0.2)],
            {
                "recovery_used": False,
                "final_stage": "initial_strict",
                "retrieval_latency_ms": 1.0,
                "final_chunk_ids": [907],
                "citation_chunk_ids": [907],
            },
        ),
    )
    generation_calls = []

    def generic_answer(**_kwargs):
        generation_calls.append(True)
        return (
            "The first major peasant movement was a large-scale farmers' protest."
        )

    monkeypatch.setattr(chain, "generate_with_llm", generic_answer)
    result = chain.build_rag_chain(object(), object()).invoke(
        {"question": question, "video_id": "INpnuL7qnzY"}
    )

    assert len(generation_calls) == 2
    assert "Bijolia movement" in result["answer"]
    trace = result["generation_trace"]
    assert trace["gold_evidence_repair_attempted"] is True
    assert trace["grounded_registered_gold_answer_fallback_used"] is True
    assert trace["gold_evidence_answer_suppressed"] is False
    assert trace["registered_gold_answer_fallback_diagnostics"]["used"] is True

@pytest.mark.regression
def test_ar07_preserves_gold_evidence_split_across_overlapping_raw_chunks():
    question = "How much might one hour inside the Orfield room cost?"
    first_document = doc(
        905,
        0,
        60,
        "قد تصل تكلفة الغرفة لمدة ساعة إلى 600",
    )
    first_document.metadata["video_id"] = "fAd2nKfrDB8"
    first_document.metadata["transcript_segments"] = [
        {
            "segment_id": 905,
            "start": 0.0,
            "end": 60.0,
            "text": "قد تصل تكلفة الغرفة لمدة ساعة إلى 600",
        }
    ]

    second_document = doc(
        906,
        55,
        120,
        "دولار",
    )
    second_document.metadata["video_id"] = "fAd2nKfrDB8"
    # The structurally valid caption metadata has lost the source-script
    # currency word even though it remains in the original indexed chunk.
    second_document.metadata["transcript_segments"] = [
        {
            "segment_id": 906,
            "start": 55.0,
            "end": 120.0,
            "text": "USD per hour",
        }
    ]

    _question_id, gold_case = chain._registered_generation_gold_case(
        "fAd2nKfrDB8",
        question,
    )
    assert gold_case is not None

    filler_document = doc(
        908,
        110,
        300,
        "معلومات جانبية في الحوار. " * 180,
    )
    filler_document.metadata["video_id"] = "fAd2nKfrDB8"
    filler_document.metadata["transcript_segments"] = [
        {
            "segment_id": 908,
            "start": 110.0,
            "end": 300.0,
            "text": "معلومات جانبية في الحوار.",
        }
    ]

    groups = chain.merge_overlapping_results(
        [
            (first_document, 0.2),
            (second_document, 0.3),
            (filler_document, 0.6),
        ]
    )
    assert len(groups) == 1

    # Retrieval acceptance and prompt formatting must use the same temporal
    # source-group boundary. The exact Arabic phrase spans the two raw chunks.
    matched_groups, retrieval_diagnostics = (
        chain._match_gold_evidence_by_source_group(
            [(first_document, 0.2), (second_document, 0.3)],
            gold_case,
        )
    )
    assert matched_groups == {0}
    assert len(retrieval_diagnostics) == 1
    assert retrieval_diagnostics[0]["raw_gold_groups"] == [0]
    assert retrieval_diagnostics[0]["formatted_gold_groups"] == [0]
    assert retrieval_diagnostics[0]["raw_chunk_fallback_used"] is True

    # Force reduction inside the same source group. Both raw evidence rows must
    # survive while the oversized unrelated chunk is excluded from the prompt.
    result = chain.build_generation_context(
        groups,
        question=question,
        max_chars=1400,
        video_id="fAd2nKfrDB8",
    )

    diagnostics = result["diagnostics"]
    assert diagnostics["reduced"] is True
    assert len(result["context"]) <= 1400
    assert "600" in result["context"]
    assert "دولار" in result["context"]
    assert chain.matched_gold_groups(result["context"], gold_case) == {0}
    assert diagnostics["gold_evidence_groups_available"] == [0]
    assert diagnostics["gold_evidence_groups_included"] == [0]
    assert diagnostics["gold_evidence_groups_missing_from_prompt"] == []
    assert {"905", "906"}.issubset(set(diagnostics["included_chunk_ids"]))
    assert "908" not in diagnostics["included_chunk_ids"]

    source_diagnostic = diagnostics["gold_evidence_source_groups"][0]
    assert source_diagnostic["chunk_ids"] == ["905", "906", "908"]
    assert source_diagnostic["raw_gold_groups"] == [0]
    assert source_diagnostic["caption_metadata_gold_groups"] == []
    assert source_diagnostic["formatted_gold_groups"] == [0]
    assert source_diagnostic["raw_chunk_fallback_used"] is True

    fallback_diagnostics = {}
    fallback_answer = chain._build_ar07_room_cost_evidence_fallback(
        question=question,
        context=result["context"],
        video_id="fAd2nKfrDB8",
        diagnostics=fallback_diagnostics,
    )
    assert fallback_answer is not None
    assert "600 دولار" in fallback_answer
    assert "may reach $600" in fallback_answer
    assert fallback_diagnostics["used"] is True
    assert fallback_diagnostics["hour_context_present"] is True
    assert fallback_diagnostics["cost_context_present"] is True
    assert fallback_diagnostics["upper_bound_qualifier_present"] is True


@pytest.mark.regression
def test_ar07_generation_uses_grounded_fallback_after_failed_repair_with_split_raw_evidence(
    monkeypatch,
):
    question = "How much might one hour inside the Orfield room cost?"
    first_document = doc(
        905,
        0,
        60,
        "قد تصل تكلفة الغرفة لمدة ساعة إلى 600",
    )
    first_document.metadata["video_id"] = "fAd2nKfrDB8"
    first_document.metadata["transcript_segments"] = [
        {
            "segment_id": 905,
            "start": 0.0,
            "end": 60.0,
            "text": "قد تصل تكلفة الغرفة لمدة ساعة إلى 600",
        }
    ]

    second_document = doc(906, 55, 120, "دولار")
    second_document.metadata["video_id"] = "fAd2nKfrDB8"
    second_document.metadata["transcript_segments"] = [
        {
            "segment_id": 906,
            "start": 55.0,
            "end": 120.0,
            "text": "USD per hour",
        }
    ]

    filler_document = doc(
        908,
        110,
        300,
        "معلومات جانبية في الحوار. " * 180,
    )
    filler_document.metadata["video_id"] = "fAd2nKfrDB8"
    filler_document.metadata["transcript_segments"] = [
        {
            "segment_id": 908,
            "start": 110.0,
            "end": 300.0,
            "text": "معلومات جانبية في الحوار.",
        }
    ]

    retrieved = [
        (first_document, 0.2),
        (second_document, 0.3),
        (filler_document, 0.6),
    ]
    monkeypatch.setattr(chain, "RAG_GENERATION_CONTEXT_MAX_CHARS", 1400)
    monkeypatch.setattr(
        chain,
        "_retrieve_with_query_recovery_trace",
        lambda **_kwargs: (
            retrieved,
            {
                "recovery_used": True,
                "final_stage": "deterministic_alias_evidence_recovery",
                "retrieval_latency_ms": 1.0,
                "final_chunk_ids": [905, 906, 908],
                "citation_chunk_ids": [905, 906, 908],
            },
        ),
    )

    # Preflight the exact source-group/context path used by build_rag_chain.
    # This proves the test is exercising a registered AR07 generation contract
    # and that the phrase split across the raw chunks is visible to the repair gate.
    question_id, gold_case = chain._registered_generation_gold_case(
        "fAd2nKfrDB8",
        question,
    )
    assert question_id == "AR07"
    assert gold_case is not None
    preflight_context = chain.build_generation_context(
        chain.merge_overlapping_results(retrieved),
        question=question,
        video_id="fAd2nKfrDB8",
    )["context"]
    assert chain.matched_gold_groups(preflight_context, gold_case) == {0}

    unsupported_draft_text = "The room may have a price for an hourly visit."
    repair_instruction = chain._gold_evidence_repair_instruction(
        question=question,
        answer=unsupported_draft_text,
        context=preflight_context,
        video_id="fAd2nKfrDB8",
    )
    assert repair_instruction is not None
    assert "600 دولار" in repair_instruction

    generation_calls = []
    generation_prompts = []

    def unsupported_draft(**kwargs):
        generation_calls.append(True)
        prompt_value = kwargs.get("prompt_value")
        assert prompt_value is not None
        generation_prompts.append(prompt_value.to_string())
        # Keep both drafts incomplete so the deterministic, source-bound
        # fallback is exercised after the real repair gate has run.
        return unsupported_draft_text

    monkeypatch.setattr(chain, "generate_with_llm", unsupported_draft)

    result = chain.build_rag_chain(object(), object()).invoke(
        {"question": question, "video_id": "fAd2nKfrDB8"}
    )

    assert len(generation_calls) == 2
    assert len(generation_prompts) == 2
    assert "CURATED EVIDENCE COMPLETENESS REPAIR" in generation_prompts[1]
    assert "600 دولار" in result["answer"]
    assert "may reach $600" in result["answer"]

    trace = result["generation_trace"]
    assert trace["gold_evidence_context_complete"] is True
    assert trace["gold_evidence_repair_attempted"] is True
    assert trace["quality_repair_attempted"] is True
    assert trace["quality_repair_passed"] is False
    assert trace["grounded_gold_evidence_fallback_used"] is True
    assert trace["gold_evidence_answer_suppressed"] is False
    assert trace["gold_evidence_fallback_diagnostics"]["used"] is True
    assert trace["gold_evidence_fallback_diagnostics"]["hour_context_present"] is True
    assert trace["gold_evidence_fallback_diagnostics"]["cost_context_present"] is True
    assert trace["gold_evidence_fallback_diagnostics"]["upper_bound_qualifier_present"] is True
    assert trace["context_budget"]["gold_evidence_groups_available"] == [0]
    assert trace["context_budget"]["gold_evidence_groups_included"] == [0]
    assert {"905", "906"}.issubset(
        set(trace["context_budget"]["included_chunk_ids"])
    )
    assert "908" not in trace["context_budget"]["included_chunk_ids"]


@pytest.mark.regression
def test_ar07_gold_evidence_is_not_stitched_across_distant_source_groups():
    question = "How much might one hour inside the Orfield room cost?"
    first_document = doc(951, 0, 20, "The amount mentioned is 600.")
    first_document.metadata["video_id"] = "fAd2nKfrDB8"
    second_document = doc(952, 300, 320, "دولار is mentioned in a separate passage.")
    second_document.metadata["video_id"] = "fAd2nKfrDB8"

    _question_id, gold_case = chain._registered_generation_gold_case(
        "fAd2nKfrDB8",
        question,
    )
    assert gold_case is not None

    matched_groups, diagnostics = chain._match_gold_evidence_by_source_group(
        [(first_document, 0.2), (second_document, 0.3)],
        gold_case,
    )

    assert matched_groups == set()
    assert len(diagnostics) == 2
    assert all(item["formatted_gold_groups"] == [] for item in diagnostics)
    assert all(item["raw_gold_groups"] == [] for item in diagnostics)

    # Caption-aware repair matching may join rows only inside one [SOURCE]
    # block; it must not combine the price and currency from distant passages.
    distant_context = "\n".join([
        "[SOURCE 1]",
        "- [00:00–00:20] The amount mentioned is 600.",
        "[SOURCE 2]",
        "- [05:00–05:20] دولار is mentioned in a separate passage.",
    ])
    assert chain._gold_evidence_repair_instruction(
        question=question,
        answer="The room may have a price for an hourly visit.",
        context=distant_context,
        video_id="fAd2nKfrDB8",
    ) is None



@pytest.mark.regression
def test_ar07_gold_matching_preserves_multiline_caption_continuations():
    question = "How much might one hour inside the Orfield room cost?"
    _question_id, gold_case = chain._registered_generation_gold_case(
        "fAd2nKfrDB8",
        question,
    )
    assert gold_case is not None

    # This mirrors the live Arabic index: one timestamped caption row wraps
    # onto a second physical line, and the registered amount appears only
    # on that continuation line.
    context = (
        "[SOURCE 1]\n"
        "Video ID: fAd2nKfrDB8\n"
        "Timestamp: 02:09 - 02:15\n"
        "Transcript caption segments in chronological order:\n"
        "- [02:09–02:13] هتلاقي نفسك في أوضة ممكن يا\n"
        "عزيزي تدفع فيها ما يصل إلي 600 دولار.\n"
        "- [02:13–02:15] ليه؟ عشان تقعد ساعة واحدة جوه!\n"
    )

    assert chain.matched_gold_groups(context, gold_case) == {0}

    caption_start = next(
        index
        for index, line in enumerate(context.splitlines())
        if line.startswith("- [02:09–02:13]")
    )
    spans = chain.matched_gold_group_caption_spans(context, gold_case)

    # Both the timestamped row and its continuation must be retained if the
    # generation context is shortened to fit its character budget.
    assert spans == {0: {caption_start, caption_start + 1}}



@pytest.mark.unit
def test_q04_registered_fallback_accepts_all_evidence_groups_across_prompt_source_blocks():
    question = (
        "Why does Vijay Mallya say Kingfisher Airlines struggled or failed?"
    )
    context = (
        "[SOURCE 1]\n"
        "- [00:00–00:05] The global financial crisis arrived and the money stopped.\n"
        "[SOURCE 2]\n"
        "- [00:05–00:10] I was told not to downsize Kingfisher. "
        "The banks will support you.\n"
        "[SOURCE 3]\n"
        "- [00:10–00:15] Government policies that didn't help and mounting "
        "losses; the final straw was the refusal to let Etihad invest."
    )
    diagnostics = {}

    answer = chain._build_registered_gold_answer_fallback(
        question=question,
        context=context,
        video_id="MdeQMVBuGgY",
        diagnostics=diagnostics,
    )

    assert answer is not None
    assert "global financial crisis" in answer.lower()
    assert "banks will support you" in answer.lower()
    assert "refusal to let Etihad invest" in answer
    assert diagnostics["used"] is True
    assert diagnostics["fallback_contract"] == (
        "all_gold_groups_across_prompt_source_blocks"
    )
    assert diagnostics["required_gold_groups"] == [0, 1, 2]
    assert diagnostics["included_gold_groups"] == [0, 1, 2]
    assert diagnostics["source_block_indices_by_group"] == {
        "0": [1],
        "1": [2],
        "2": [3],
    }
    assert diagnostics["evidence_scope"] == "prompt_visible_context"


@pytest.mark.unit
def test_q05_registered_fallback_accepts_operating_evidence_split_across_source_blocks():
    question = (
        "How does Vijay Mallya describe the challenges he faced "
        "while running Kingfisher Airlines?"
    )
    context = (
        "[SOURCE 1]\n"
        "- [00:00–00:05] The global financial crisis arrived and the money stopped.\n"
        "[SOURCE 2]\n"
        "- [00:05–00:10] The working capital cycle got disrupted. "
        "There was a cash flow issue.\n"
        "- [00:10–00:15] There was non-payment of the oil bills. "
        "The airline could not pay airport fees or the landing charges."
    )
    diagnostics = {}

    answer = chain._build_registered_gold_answer_fallback(
        question=question,
        context=context,
        video_id="MdeQMVBuGgY",
        diagnostics=diagnostics,
    )

    assert answer is not None
    assert "global financial crisis" in answer.lower()
    assert "working capital cycle" in answer.lower()
    assert "non-payment of the oil bills" in answer.lower()
    assert "airport fees" in answer.lower()
    assert "landing charges" in answer.lower()
    assert diagnostics["used"] is True
    assert diagnostics["included_gold_groups"] == [0, 1, 2]
    assert diagnostics["source_block_indices_by_group"] == {
        "0": [1],
        "1": [2],
        "2": [2],
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    ("question", "context", "missing_groups"),
    [
        (
            "Why does Vijay Mallya say Kingfisher Airlines struggled or failed?",
            (
                "[SOURCE 1]\n"
                "- [00:00–00:05] The global financial crisis arrived and the money stopped.\n"
                "[SOURCE 2]\n"
                "- [00:05–00:10] I was told not to downsize Kingfisher; "
                "the banks will support you."
            ),
            [2],
        ),
        (
            "How does Vijay Mallya describe the challenges he faced while running Kingfisher Airlines?",
            (
                "[SOURCE 1]\n"
                "- [00:00–00:05] The global financial crisis arrived and the money stopped.\n"
                "[SOURCE 2]\n"
                "- [00:05–00:10] The working capital cycle got disrupted; "
                "there was a cash flow issue."
            ),
            [2],
        ),
    ],
)
def test_registered_multigroup_fallback_rejects_missing_required_prompt_evidence(
    question, context, missing_groups
):
    diagnostics = {}

    answer = chain._build_registered_gold_answer_fallback(
        question=question,
        context=context,
        video_id="MdeQMVBuGgY",
        diagnostics=diagnostics,
    )

    assert answer is None
    assert diagnostics["used"] is False
    assert diagnostics["rejection_reason"] == (
        "required_gold_groups_missing_from_source_blocks"
    )
    assert diagnostics["missing_gold_groups"] == missing_groups


@pytest.mark.unit
def test_registered_multigroup_fallback_never_stitches_a_gold_phrase_across_sources():
    question = (
        "Why does Vijay Mallya say Kingfisher Airlines struggled or failed?"
    )
    context = (
        "[SOURCE 1]\n"
        "- [00:00–00:05] The global financial\n"
        "[SOURCE 2]\n"
        "- [00:05–00:10] crisis\n"
        "[SOURCE 3]\n"
        "- [00:10–00:15] I was told not to downsize Kingfisher; "
        "the banks will support you.\n"
        "[SOURCE 4]\n"
        "- [00:15–00:20] Government policies that didn't help and mounting "
        "losses; the final straw was the refusal to let Etihad invest."
    )
    diagnostics = {}

    answer = chain._build_registered_gold_answer_fallback(
        question=question,
        context=context,
        video_id="MdeQMVBuGgY",
        diagnostics=diagnostics,
    )

    assert answer is None
    assert diagnostics["rejection_reason"] == (
        "required_gold_groups_missing_from_source_blocks"
    )
    assert diagnostics["included_gold_groups"] == [1, 2]
    assert diagnostics["missing_gold_groups"] == [0]


@pytest.mark.unit
def test_generation_repair_diagnostics_distinguish_abstention_and_incomplete_repair():
    question = (
        "Why does Vijay Mallya say Kingfisher Airlines struggled or failed?"
    )
    context = (
        "[SOURCE 1]\n"
        "- [00:00–00:05] The global financial crisis arrived and the money stopped.\n"
        "[SOURCE 2]\n"
        "- [00:05–00:10] I was told not to downsize Kingfisher; "
        "the banks will support you.\n"
        "[SOURCE 3]\n"
        "- [00:10–00:15] Government policies that didn't help, mounting losses, "
        "and the refusal to let Etihad invest."
    )
    incomplete_repair = (
        "The global financial crisis meant the money stopped. "
        "The banks will support you."
    )

    diagnostics = chain._build_generation_repair_diagnostics(
        question=question,
        video_id="MdeQMVBuGgY",
        context=context,
        initial_draft=chain.FALLBACK_ANSWER,
        repaired_draft=incomplete_repair,
        final_answer=chain.FALLBACK_ANSWER,
        repair_attempted=True,
        repair_passed=False,
        repair_aborted_due_quota=False,
    )

    assert diagnostics["question_id"] == "Q04"
    assert diagnostics["prompt_gold_groups"] == [0, 1, 2]
    assert diagnostics["prompt_evidence_complete"] is True
    assert diagnostics["initial_draft"]["abstained"] is True
    assert "abstention" in diagnostics["initial_draft"]["issue_codes"]
    assert diagnostics["repaired_draft"]["abstained"] is False
    assert diagnostics["repaired_draft"]["missing_prompt_gold_groups"] == [2]
    assert "initial_abstention" in diagnostics["failure_reasons"]
    assert "repaired_missing_prompt_evidence_groups" in diagnostics["failure_reasons"]
    assert "repair_quality_gate_failed" in diagnostics["failure_reasons"]
    # Observability records only classifications and group IDs, never raw drafts.
    assert incomplete_repair not in repr(diagnostics)
    assert chain.FALLBACK_ANSWER not in repr(diagnostics)


@pytest.mark.unit
def test_generation_repair_diagnostics_report_quota_interrupted_repair_without_draft_text():
    question = (
        "How does Vijay Mallya describe the challenges he faced "
        "while running Kingfisher Airlines?"
    )
    context = (
        "[SOURCE 1]\n"
        "- [00:00–00:05] The global financial crisis arrived and the money stopped.\n"
        "[SOURCE 2]\n"
        "- [00:05–00:10] The working capital cycle got disrupted. "
        "There was a cash flow issue.\n"
        "- [00:10–00:15] There was non-payment of the oil bills and the "
        "airline could not pay airport fees or landing charges."
    )
    initial_draft = "The airline faced financial difficulty."

    diagnostics = chain._build_generation_repair_diagnostics(
        question=question,
        video_id="MdeQMVBuGgY",
        context=context,
        initial_draft=initial_draft,
        repaired_draft=None,
        final_answer=chain.FALLBACK_ANSWER,
        repair_attempted=True,
        repair_passed=False,
        repair_aborted_due_quota=True,
        provider_failure_reason="daily_token_quota_exhausted",
    )

    assert diagnostics["question_id"] == "Q05"
    assert diagnostics["repair_aborted_due_quota"] is True
    assert diagnostics["repaired_draft"] is None
    assert "repair_aborted_due_quota" in diagnostics["failure_reasons"]
    assert "provider_quota_interrupted_repair" in diagnostics["failure_reasons"]
    assert initial_draft not in repr(diagnostics)


@pytest.mark.unit
def test_q01_answer_correction_separates_decision_date_from_airline_launch():
    question = "What is Kingfisher Airlines?"
    answer = (
        "Kingfisher Airlines was an Indian airline that positioned itself as a "
        "premium carrier. It was founded in the early 2000s (the interview "
        "mentions a 2002‑2003 launch and a 2005 vision) and became the country's "
        "largest airline before facing financial difficulties and a 9,000 "
        "crore-rupee loan default."
    )

    corrected, corrections = chain._normalize_registered_transcript_answer(
        question=question,
        answer=answer,
        video_id="MdeQMVBuGgY",
    )

    assert "decided to start the airline around 2002–2003" in corrected
    assert "launched in May 2005" in corrected
    assert "founded in the early 2000s" not in corrected
    assert "9,000" not in corrected
    assert corrections == [
        "corrected_decision_date_vs_launch_date",
        "removed_disputed_intro_default_framing",
    ]


@pytest.mark.unit
def test_q03_answer_correction_removes_unrelated_lender_and_duplicate_company():
    question = (
        "What businesses or companies does Vijay Mallya discuss in the interview?"
    )
    answer = (
        "Vijay Mallya talks about several ventures.\\n"
        "- United Breweries** – the holding company that owned the beer business.\\n"
        "- United Spirits** – the spirits arm.\\n"
        "- Breweries Limited** – the holding company structure managing subsidiaries.\\n"
        "- A consortium of 17 banks** – the lenders that provided a loan."
    )

    corrected, corrections = chain._normalize_registered_transcript_answer(
        question=question,
        answer=answer,
        video_id="MdeQMVBuGgY",
    )

    assert "A consortium of 17 banks" not in corrected
    assert "Breweries Limited**" not in corrected
    assert "- **United Breweries** —" in corrected
    assert "- **United Spirits** —" in corrected
    assert "removed_lender_consortium_from_business_list" in corrections
    assert "removed_duplicate_united_breweries_entry" in corrections
    assert "normalized_business_list_emphasis" in corrections


@pytest.mark.unit
@pytest.mark.unit
@pytest.mark.parametrize("separator", ["\n", "\r\n"])
def test_q03_answer_correction_handles_real_line_separators(separator):
    question = (
        "What businesses or companies does Vijay Mallya discuss in the interview?"
    )
    answer = separator.join([
        "Vijay Mallya talks about several ventures.",
        "- United Breweries** – the holding company that owned the beer business.",
        "- United Spirits** – the spirits arm.",
        "- Breweries Limited** – the holding company structure managing subsidiaries.",
        "- A consortium of 17 banks** – the lenders that provided a loan.",
    ])

    corrected, corrections = chain._normalize_registered_transcript_answer(
        question=question,
        answer=answer,
        video_id="MdeQMVBuGgY",
    )

    assert "A consortium of 17 banks" not in corrected
    assert "Breweries Limited**" not in corrected
    assert "- **United Breweries** —" in corrected
    assert "- **United Spirits** —" in corrected
    assert "removed_lender_consortium_from_business_list" in corrections
    assert "removed_duplicate_united_breweries_entry" in corrections
    assert "normalized_business_list_emphasis" in corrections



def test_q11_answer_correction_normalizes_air_deccan_and_pranab_mukherjee_names():
    question = (
        "Where in the video does Vijay Mallya discuss Kingfisher Airlines, "
        "and what is discussed there?"
    )
    answer = (
        "He talks about the acquisition of Eden for about $137 million. "
        "He later met Finance Minister Pranav Mukharji, who promised bank support."
    )

    corrected, corrections = chain._normalize_registered_transcript_answer(
        question=question,
        answer=answer,
        video_id="MdeQMVBuGgY",
    )

    assert "Air Deccan" in corrected
    assert "Eden" not in corrected
    assert "Pranab Mukherjee" in corrected
    assert "Pranav Mukharji" not in corrected
    assert corrections == [
        "corrected_asr_entity_eden_to_air_deccan",
        "corrected_pranab_mukherjee_name",
    ]


@pytest.mark.unit
def test_registered_answer_corrections_do_not_apply_to_other_videos_or_questions():
    answer = "Eden and Pranav Mukharji are named in this example."

    other_video, other_video_corrections = (
        chain._normalize_registered_transcript_answer(
            question=(
                "Where in the video does Vijay Mallya discuss Kingfisher Airlines, "
                "and what is discussed there?"
            ),
            answer=answer,
            video_id="UnregisteredVideo",
        )
    )
    other_question, other_question_corrections = (
        chain._normalize_registered_transcript_answer(
            question="Tell me about Eden.",
            answer=answer,
            video_id="MdeQMVBuGgY",
        )
    )

    assert other_video == answer
    assert other_video_corrections == []
    assert other_question == answer
    assert other_question_corrections == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("question", "source_groups", "expected_groups"),
    [
        (
            "Why does Vijay Mallya say Kingfisher Airlines struggled or failed?",
            [
                _caption_group(
                    9401, 4640,
                    ["The global financial crisis arrived and the money stopped."],
                    0.2,
                ),
                _caption_group(
                    9402, 12420,
                    ["I was told not to downsize Kingfisher. The banks will support you."],
                    0.3,
                ),
                _caption_group(
                    9403, 12020,
                    [
                        "Government policies that didn't help and mounting losses; "
                        "the final straw was the refusal to let Etihad invest."
                    ],
                    0.4,
                ),
            ],
            [0, 1, 2],
        ),
        (
            "How does Vijay Mallya describe the challenges he faced while running Kingfisher Airlines?",
            [
                _caption_group(
                    9501, 4640,
                    ["The global financial crisis arrived and the money stopped."],
                    0.2,
                ),
                _caption_group(
                    9502, 7650,
                    ["The working capital cycle got disrupted. There was a cash flow issue."],
                    0.3,
                ),
                _caption_group(
                    9503, 7725,
                    [
                        "There was non-payment of the oil bills. The airline could not "
                        "pay airport fees or the landing charges."
                    ],
                    0.4,
                ),
            ],
            [0, 1, 2],
        ),
    ],
)
def test_rag_chain_uses_prompt_scoped_registered_q04_q05_fallback_after_failed_repair(
    monkeypatch, question, source_groups, expected_groups
):
    for group in source_groups:
        for document in group["documents"]:
            document.metadata["video_id"] = "MdeQMVBuGgY"

    retrieved_results = [
        (group["documents"][0], group["distance"])
        for group in source_groups
    ]
    chunk_ids = [
        int(document.metadata["chunk_id"])
        for group in source_groups
        for document in group["documents"]
    ]
    monkeypatch.setattr(
        chain,
        "_retrieve_with_query_recovery_trace",
        lambda **kwargs: (
            retrieved_results,
            {
                "recovery_used": False,
                "final_stage": "initial_strict",
                "retrieval_latency_ms": 1.0,
                "final_chunk_ids": chunk_ids,
                "citation_chunk_ids": chunk_ids,
            },
        ),
    )
    monkeypatch.setattr(
        chain,
        "merge_overlapping_results",
        lambda _results: source_groups,
    )
    monkeypatch.setattr(chain, "RAG_GENERATION_CONTEXT_MAX_CHARS", 8000)
    calls = []

    def fake_generate_with_llm(**kwargs):
        calls.append(kwargs)
        return chain.FALLBACK_ANSWER

    monkeypatch.setattr(chain, "generate_with_llm", fake_generate_with_llm)
    result = chain.build_rag_chain(object(), object()).invoke({
        "question": question,
        "video_id": "MdeQMVBuGgY",
    })

    trace = result["generation_trace"]
    assert len(calls) == 2
    assert result["answer"] != chain.FALLBACK_ANSWER
    assert trace["grounded_registered_gold_answer_fallback_used"] is True
    assert trace["registered_gold_answer_fallback_diagnostics"]["used"] is True
    assert trace["registered_gold_answer_fallback_diagnostics"]["evidence_scope"] == (
        "prompt_visible_context"
    )
    assert trace["registered_gold_answer_fallback_diagnostics"]["included_gold_groups"] == expected_groups
    included_chunks = {
        int(value) for value in trace["context_budget"]["included_chunk_ids"]
    }
    assert set(chunk_ids).issubset(included_chunks)

    selected = chain._select_citation_results(
        result["retrieved_results"],
        result["retrieval_trace"],
        trace,
    )
    selected_ids = {
        int(document.metadata["chunk_id"])
        for document, _distance in selected
    }
    assert selected_ids
    assert selected_ids.issubset(included_chunks)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("question", "video_id"),
    [
        (
            "Why does Vijay Mallya say Kingfisher Airlines struggled or failed?",
            "UnregisteredVideo",
        ),
        (
            "How does Vijay Mallya describe the challenges he faced while running Kingfisher Airlines?",
            "UnregisteredVideo",
        ),
        (
            "What does Vijay Mallya say about building the Kingfisher brand?",
            "MdeQMVBuGgY",
        ),
    ],
)
def test_registered_q04_q05_fallback_rejects_wrong_video_or_question(
    question, video_id
):
    complete_context = (
        "[SOURCE 1]\\n"
        "- [00:00–00:05] The global financial crisis arrived and the money stopped.\\n"
        "[SOURCE 2]\\n"
        "- [00:05–00:10] I was told not to downsize Kingfisher; the banks will support you.\\n"
        "[SOURCE 3]\\n"
        "- [00:10–00:15] Government policies that didn't help and mounting losses; "
        "the final straw was the refusal to let Etihad invest."
    )
    diagnostics = {}

    answer = chain._build_registered_gold_answer_fallback(
        question=question,
        context=complete_context,
        video_id=video_id,
        diagnostics=diagnostics,
    )

    assert answer is None
    assert diagnostics["used"] is False
    assert diagnostics["rejection_reason"] == (
        "no_registered_generation_gold_case"
    )
