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

    assert chain.append_requested_temporal_timestamp(
        answer=answer,
        question="Cite the relevant timestamp for 25:35–26:25.",
        retrieval_trace=trace,
        has_evidence=True,
    ) == "A factual answer.\n\nRelevant timestamp: 25:35–26:24"


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
def test_prompt_requires_answer_in_question_language_even_when_context_is_english():
    question = "ما الأصوات التي قد يبدأ الناس بسماعها من أجسامهم داخل الغرفة؟"
    prompt = chain.RAG_PROMPT.invoke({
        "context": "The transcript is in English and discusses heartbeat and breathing.",
        "question": question,
    })
    messages = chain.prompt_to_messages(prompt)
    normalized_system_prompt = " ".join(messages[0]["content"].split()).lower()

    assert question in messages[1]["content"]
    assert "answer in the same language as the user's question" in normalized_system_prompt
    assert "do not switch to english merely because the transcript" in normalized_system_prompt
    assert "for an arabic question, answer in arabic" in normalized_system_prompt


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
def test_select_citation_results_can_include_context_neighbors_in_experiment(monkeypatch):
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

    assert [document.metadata["chunk_id"] for document, _ in selected] == [1, 2, 3]


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
def test_query_recovery_rejects_lexical_only_rewrites_without_overview_fallback(
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
def test_query_recovery_uses_representative_context_as_final_fallback(
    monkeypatch,
):
    overview = [
        (doc(1, 0, 10, "opening context"), chain.MAX_DISTANCE),
        (doc(2, 20, 30, "representative context"), chain.MAX_DISTANCE),
    ]

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
            standalone_question="which side is good and which is bad",
            search_queries=[
                "arguments for each side",
                "positions of opposing sides",
            ],
        ),
    )
    monkeypatch.setattr(
        chain,
        "retrieve_overview",
        lambda vector_store: overview,
    )

    assert chain.retrieve_with_query_recovery(
        object(),
        "Which side is good and which is bad?",
        object(),
    ) == overview


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
    monkeypatch.setattr(
        chain,
        "create_groq_client",
        lambda: sentinel,
    )

    assert chain.create_llm_client() is sentinel


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
    captured_clients = {}

    monkeypatch.setattr(
        chain,
        "create_llm_client",
        lambda: configured_client,
    )

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
    assert captured_clients["generation"] is configured_client

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
        doc(1, 10, 40, "selected answer anchor"),
        doc(2, 20, 50, "context-only neighbor"),
        doc(3, 60, 80, "second selected anchor"),
    ]
    expanded_results = [
        (expanded_documents[0], 0.2),
        (expanded_documents[1], chain.MAX_DISTANCE),
        (expanded_documents[2], 0.5),
    ]
    retrieval_result = {
        "answer": "A grounded answer.",
        "retrieved_results": expanded_results,
        "source_groups": chain.merge_overlapping_results(expanded_results),
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

    assert result["retrieval_config"]["citation_policy"] == "expanded_context_experiment"
    assert result["source_segments"] == 3
    assert [source["chunk_ids"] for source in result["sources"]] == [[1], [2], [3]]
    # Chunk 2 overlaps anchor 1 in time, but the neighbor link stays precise.
    assert [(source["start"], source["end"]) for source in result["sources"]] == [
        (10.0, 40.0),
        (20.0, 50.0),
        (60.0, 80.0),
    ]


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
def test_generation_repair_instruction_detects_incomplete_arabic_bodily_sound_list():
    question = "ما الأصوات التي قد يبدأ الناس بسماعها من أجسامهم داخل الغرفة؟"
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
    assert "Arabic" in instruction
    assert "حركة أو جريان الدم في العروق" in instruction
    assert "أصوات الرئتين أو التنفس" in instruction
    assert "أصوات المعدة" in instruction
    assert "رمش الجفون" in instruction


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
    assert "14,100 crores recovered from me." in fallback
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
    source_groups = [
        {
            "documents": [doc(108, 7267, 7351, "I invested 3,000 crore rupees into Kingfisher Airlines.")],
            "start": 7267, "end": 7351, "distance": 0.2,
        },
        {
            "documents": [doc(124, 8314, 8428, "The DRT judgment debt was 4,999 crore rupees plus interest.")],
            "start": 8314, "end": 8428, "distance": 0.25,
        },
        {
            "documents": [doc(221, 14939, 15015, "The debt certificate gives a figure of 6203.")],
            "start": 14939, "end": 15015, "distance": 0.3,
        },
        {
            "documents": [doc(159, 10630, 10698, "A Ministry of Finance report says 14,100 crore rupees were recovered from me.")],
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
    monkeypatch.setattr(chain, "RAG_GENERATION_CONTEXT_MAX_CHARS", 1200)

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
    assert len(prompt_human_messages) == 2
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
    assert "3,000 crore rupees" in answer
    assert "4,999 crore rupees" in answer
    assert "debt certificate gives a figure of 6,203" in answer
    assert "14,100 crore rupees" in answer
    assert "does not clearly explain how the DRT judgment and debt-certificate figures relate" in answer


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
    assert "certificate gives a figure of 6,203" in answer
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
    assert "4,999 crore rupees" in fallback
    assert "debt certificate gives a figure of 6,203" in fallback
    assert re.search(
        r"(?:Ministry of Finance report|Finance report|report).{0,80}14,100 crore rupees",
        fallback,
        re.IGNORECASE,
    )
    assert "14,100 crore rupees" in fallback
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
    assert "4,999 crore rupees" in fallback
    assert "debt certificate gives a figure of 6,203" in fallback
    assert re.search(
        r"(?:Ministry of Finance report|Finance report|report).{0,80}14,100 crore rupees",
        fallback,
        re.IGNORECASE,
    )


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

    fallback = chain._build_financial_evidence_fallback(
        "What debt and money amounts are mentioned in the Kingfisher case?",
        context,
    )

    # An attribution from another returned source is not evidence for this
    # recovery claim; the extractor must decline rather than join unrelated
    # source blocks into one assertion.
    assert fallback is None


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

