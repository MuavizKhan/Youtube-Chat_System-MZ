import pytest

from Backend.rag_system.timestamp_utils import (
    find_timestamp_ranges,
    contains_unverified_timestamp_claims,
    find_timestamp_range_texts,
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


def test_unverified_timestamp_claim_sanitizer_removes_time_sentences_not_answers():
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


def test_unverified_timestamp_claim_sanitizer_removes_labelled_footer_and_single_time_cue():
    answer = (
        "A grounded answer.\n"
        "The event occurs around 01:20.\n"
        "Relevant timestamps: 00:00–01:08"
    )


def test_unverified_timestamp_sanitizer_preserves_answer_with_inline_range():
    answer = (
        "The speaker offers a 30-second structure for explaining your work "
        "at 02:25–04:33."
    )

    cleaned = strip_unverified_timestamp_claims(answer)

    assert cleaned == (
        "The speaker offers a 30-second structure for explaining your work."
    )


    assert strip_unverified_timestamp_claims(answer) == (
        "A grounded answer.\\nThe event occurs."
    )
