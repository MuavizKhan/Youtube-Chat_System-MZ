import pytest

from Backend.rag_system.timestamp_utils import (
    contains_unverified_timestamp_claims,
    find_timestamp_range_texts,
    find_timestamp_ranges,
    parse_timestamp_range,
    strip_unverified_timestamp_claims,
)


@pytest.mark.parametrize(
    "value,start,end",
    [
        ("00:00 to 01:08", 0, 68),
        ("01:10 and 01:22", 70, 82),
        ("01:10:38–01:12:04", 4238, 4324),
        ("01:10:38\u201101:12:04", 4238, 4324),
        ("01:10:38\u221201:12:04", 4238, 4324),
    ],
)
def test_shared_timestamp_parser_accepts_common_formats(value, start, end):
    parsed = parse_timestamp_range(value)

    assert parsed is not None
    assert parsed.start_seconds == start
    assert parsed.end_seconds == end


@pytest.mark.parametrize(
    "value",
    [
        "01:75–02:00",
        "02:00–01:00",
        "01:10:75–01:12:04",
        "01:60:00–02:00:00",
        "1:02:03:04–1:02:03:05",
    ],
)
def test_shared_timestamp_parser_rejects_invalid_or_reversed_ranges(value):
    assert parse_timestamp_range(value) is None


def test_range_shape_detection_retains_malformed_expressions_for_auditing():
    answer = "The model claims the segment is 01:75 to 02:00."

    assert find_timestamp_range_texts(answer) == ["01:75 to 02:00"]
    assert find_timestamp_ranges(answer) == []


def test_timestamp_finder_extracts_inline_ranges_and_word_separators():
    answer = (
        "Opening at 00:00 to 01:08, then another section at "
        "01:10 and 01:22."
    )

    assert [item.raw for item in find_timestamp_ranges(answer)] == [
        "00:00 to 01:08",
        "01:10 and 01:22",
    ]


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("The discussion happens around 01:20.", True),
        ("The relevant passage is in 01:20.", True),
        ("This ranges from 00:10 to 00:30.", True),
        ("The speaker gives a 30-second structure and a 90-minute talk.", False),
        ("A grounded answer.\nRelevant timestamps: 00:10–00:30", False),
    ],
)
def test_unverified_timestamp_claim_detector_distinguishes_video_locations(
    answer,
    expected,
):
    assert contains_unverified_timestamp_claims(answer) is expected


def test_unverified_timestamp_claim_sanitizer_preserves_prose_around_ranges():
    answer = (
        "The speaker explains the airline's launch. "
        "The discussion runs from 00:00 to 01:08. "
        "The later detail appears between 01:10 and 01:22. "
        "He also describes a 30-second structure and a 90-minute talk."
    )

    cleaned = strip_unverified_timestamp_claims(answer)

    assert cleaned == (
        "The speaker explains the airline's launch. "
        "The discussion runs. "
        "The later detail appears. "
        "He also describes a 30-second structure and a 90-minute talk."
    )


def test_unverified_timestamp_claim_sanitizer_also_removes_invalid_ranges():
    answer = (
        "A grounded answer. The model claims the section runs "
        "from 01:75 to 02:00."
    )

    assert strip_unverified_timestamp_claims(answer) == (
        "A grounded answer. The model claims the section runs."
    )


@pytest.mark.parametrize("marker", ["-", "*", "+", "•"])
def test_timestamp_sanitizer_preserves_markdown_list_markers(marker):
    answer = (
        f"{marker} Investment: the speaker put in 3,000 crore of UB Group money.\\n"
        f"{marker} DRT debt: the transcript identifies 4,999 crore as debt."
    )

    assert strip_unverified_timestamp_claims(answer) == answer


@pytest.mark.parametrize("marker", ["-", "*", "+", "•"])
def test_timestamp_sanitizer_preserves_list_marker_when_removing_timestamp(marker):
    answer = f"{marker} The speaker explains the issue at 02:25–04:33."

    assert strip_unverified_timestamp_claims(answer) == (
        f"{marker} The speaker explains the issue."
    )


def test_unverified_timestamp_claim_sanitizer_removes_labelled_footer_and_single_time_cue():
    answer = (
        "A grounded answer.\n"
        "The event occurs around 01:20.\n"
        "Relevant timestamps: 00:00–01:08"
    )

    assert strip_unverified_timestamp_claims(answer) == (
        "A grounded answer.\nThe event occurs."
    )


@pytest.mark.parametrize(
    "heading",
    [
        "Timestamps: 00:10–00:20",
        "- Relevant timestamps: 00:10–00:20",
        "> Relevant timestamps: 00:10–00:20",
        "Time stamps: 00:10–00:20",
    ],
)
def test_unverified_timestamp_sanitizer_removes_model_timestamp_heading_lines(heading):
    answer = f"A grounded answer.\n{heading}"

    assert strip_unverified_timestamp_claims(answer) == "A grounded answer."


def test_unverified_timestamp_sanitizer_preserves_answer_with_inline_range():
    answer = (
        "The speaker offers a 30-second structure for explaining your work "
        "at 02:25–04:33."
    )

    cleaned = strip_unverified_timestamp_claims(answer)

    assert cleaned == (
        "The speaker offers a 30-second structure for explaining your work."
    )


@pytest.mark.parametrize(
    "answer,expected",
    [
        (
            "The interview discusses the airline's launch, from the start of the video up to about 01:08 minutes.",
            "The interview discusses the airline's launch, from the start of the video.",
        ),
        (
            "The event occurs at around 01:20 minutes.",
            "The event occurs.",
        ),
    ],
)
def test_unverified_timestamp_sanitizer_removes_natural_language_time_units(
    answer,
    expected,
):
    assert strip_unverified_timestamp_claims(answer) == expected


def test_unverified_timestamp_sanitizer_removes_bracketed_time_without_losing_answer():
    answer = (
        "The speaker offers a 30-second structure for explaining your work "
        "[02:25–04:33]."
    )

    assert strip_unverified_timestamp_claims(answer) == (
        "The speaker offers a 30-second structure for explaining your work."
    )


def test_unverified_timestamp_sanitizer_removes_parenthetical_time_without_losing_answer():
    answer = (
        "The speaker offers a 30-second structure for explaining your work "
        "(02:25–04:33)."
    )

    assert strip_unverified_timestamp_claims(answer) == (
        "The speaker offers a 30-second structure for explaining your work."
    )


def test_unverified_timestamp_sanitizer_removes_mixed_point_and_range_claims():
    answer = (
        "A grounded answer. The event occurs around 00:45, and the discussion "
        "continues from 01:00 to 02:00."
    )

    assert strip_unverified_timestamp_claims(answer) == (
        "A grounded answer. The event occurs, and the discussion continues."
    )
