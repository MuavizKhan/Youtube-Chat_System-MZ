import pytest

from Backend.rag_system.timestamp_utils import (
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


def test_timestamp_finder_extracts_inline_ranges_and_word_separators():
    answer = (
        "Opening at 00:00 to 01:08, then another section at "
        "01:10 and 01:22."
    )

    assert [item.raw for item in find_timestamp_ranges(answer)] == [
        "00:00 to 01:08",
        "01:10 and 01:22",
    ]


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
        "He also describes a 30-second structure and a 90-minute talk."
    )


def test_unverified_timestamp_claim_sanitizer_removes_labelled_footer_and_single_time_cue():
    answer = (
        "A grounded answer.\n"
        "The event occurs around 01:20.\n"
        "Relevant timestamps: 00:00–01:08"
    )

    assert strip_unverified_timestamp_claims(answer) == "A grounded answer."
