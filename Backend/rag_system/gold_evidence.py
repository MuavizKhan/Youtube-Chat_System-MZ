"""
Curated gold-evidence anchors for the retrieval-quality benchmark.

The annotations are intentionally transcript-text based rather than
chunk-id based. Chunk IDs can change when chunking is tuned, while the
underlying evidence language remains stable.

These annotations cover the content-retrieval cases Q01-Q10 for the
benchmark video used by the regression suite. Q11 is a temporal question
and remains covered by the existing retrieval regression contract; a
separate timestamp-grounded annotation set is intentionally deferred.
"""

import re
import unicodedata
from dataclasses import dataclass


GOLD_EVIDENCE_VIDEO_ID = "MdeQMVBuGgY"
GOLD_EVIDENCE_SOURCE_URL = (
    "https://youtubetotranscript.com/transcript"
    "?current_language_code=en&v=MdeQMVBuGgY"
)


@dataclass(frozen=True)
class GoldEvidenceCase:
    """Human-curated evidence groups for one retrieval question.

    Each group contains alternative transcript phrases that represent the
    same evidence concept. A case can require coverage of several concepts.
    """

    question_id: str
    groups: tuple[tuple[str, ...], ...]
    min_group_coverage: float
    min_context_group_coverage: float | None = None
    # Optional stricter requirement for the excerpts actually sent to generation.
    # This is separate from retrieval-context evaluation so prompt packing can
    # reserve the evidence needed to answer the registered question.
    min_generation_context_group_coverage: float | None = None
    # Optional paraphrases accepted only when checking generated answers.
    # Retrieval and prompt-context matching continue to use exact source phrases.
    generation_answer_aliases: tuple[tuple[str, ...], ...] = ()
    # Optional deterministic response, permitted only when every configured
    # source assertion is present in one retrieved source block and the gold
    # evidence contract is also satisfied.
    generation_fallback_answer: str | None = None
    generation_fallback_source_phrases: tuple[str, ...] = ()


GOLD_EVIDENCE_CASES = (
    GoldEvidenceCase(
        question_id="Q01",
        groups=(
            ("single class airline", "low cost",),
            (
                "best flying experience that India had ever seen",
                "best flying experience",
                "premium experience",
            ),
            (
                "inflight entertainment",
                "it had inflight entertainment",
                "offered meals",
                "good food",
            ),
        ),
        min_group_coverage=2 / 3,
    ),
    GoldEvidenceCase(
        question_id="Q02",
        groups=(
            (
                "Formula 1",
                "pinnacle of motorsport",
                "dream to have a team on the F1 grid",
            ),
        ),
        min_group_coverage=1.0,
    ),
    GoldEvidenceCase(
        question_id="Q03",
        groups=(
            ("United Breweries", "United Breweries Limited", "United Spirits"),
            (
                "merger of Kingfisher and",
                "Kingfisher Red",
                "eliminate the disruptor",
            ),
            (
                "Royal Challengers Bangalore",
                "Royal Challenge",
                "RCB was born",
            ),
        ),
        min_group_coverage=2 / 3,
        min_context_group_coverage=1.0,
    ),
    GoldEvidenceCase(
        question_id="Q04",
        groups=(
            (
                "global financial crisis",
                "money stopped",
                "depressed economic circumstances",
            ),
            (
                "banks will support you",
                "not to downsize Kingfisher",
                "I was told not to downsize Kingfisher",
            ),
            (
                "government policies that didn't help",
                "mounting losses",
                "refusal to let Etihad invest",
                "final straw",
            ),
        ),
        min_group_coverage=2 / 3,
        min_generation_context_group_coverage=1.0,
    ),
    GoldEvidenceCase(
        question_id="Q05",
        groups=(
            ("global financial crisis", "money stopped",),
            (
                "working capital cycle got disrupted",
                "cash flow issue",
            ),
            (
                "non-payment of the oil bills",
                "any money in the account of the airline was frozen",
                "airport fees",
                "landing charges",
            ),
        ),
        min_group_coverage=2 / 3,
        min_generation_context_group_coverage=1.0,
        generation_answer_aliases=(
            (),
            (
                "cash flow nightmare",
                "cash flow crisis",
                "cash flow problem",
                "working capital cycle broke",
                "broke the working capital cycle",
                "broken working capital cycle",
                "working capital cycle was disrupted",
            ),
            (),
        ),
    ),
    GoldEvidenceCase(
        question_id="Q06",
        groups=(
            ("a brand needs a personality", "gave brands their personalities"),
            ("surrogate advertising", "authoring surrogate advertising"),
            (
                "full water business around Kingfisher",
                "water business around Kingfisher",
                "advertise the core brand",
            ),
        ),
        min_group_coverage=2 / 3,
    ),
    GoldEvidenceCase(
        question_id="Q07",
        groups=(
            ("4,999 crores", "4999 crores"),
            ("6203 crores", "6,203 crores"),
            ("14,100 crores", "14100 crores"),
        ),
        min_group_coverage=2 / 3,
    ),
    GoldEvidenceCase(
        question_id="Q08",
        groups=(
            (
                "14,100 crores recovered from me",
                "government has recovered 14,100 crores",
            ),
            (
                "banks have been fully paid",
                "the banks have been fully paid",
            ),
            ("statement of account", "statement in parliament"),
        ),
        min_group_coverage=2 / 3,
    ),
    GoldEvidenceCase(
        question_id="Q09",
        groups=(
            (
                "I asked only for policy changes",
                "policy changes",
            ),
            (
                "cost of crude",
                "aviation turbine fuel",
                "ad valorum sales tax",
            ),
            (
                "banks will support you",
                "not to downsize Kingfisher",
                "government policies",
            ),
        ),
        min_group_coverage=2 / 3,
        min_context_group_coverage=1.0,
        min_generation_context_group_coverage=1.0,
    ),
    GoldEvidenceCase(
        question_id="Q10",
        groups=(
            (
                "global financial crisis",
                "money stopped",
            ),
            (
                "government policies that didn't help",
                "mounting losses",
            ),
            (
                "final straw",
                "refusal to let Etihad invest",
                "run out of options",
            ),
        ),
        min_group_coverage=2 / 3,
        min_generation_context_group_coverage=1.0,
    ),
)


_GOLD_BY_ID = {
    case.question_id: case
    for case in GOLD_EVIDENCE_CASES
}


def get_gold_evidence_case(question_id: str) -> GoldEvidenceCase:
    """Return the curated gold-evidence annotation for a question."""

    try:
        return _GOLD_BY_ID[question_id]
    except KeyError as error:
        raise ValueError(
            f"No gold-evidence annotation exists for {question_id}."
        ) from error


def normalize_evidence_text(text: str) -> str:
    """Normalize transcript text while preserving letters across scripts.

    Unicode compatibility normalization and case folding make equivalent
    spellings more consistent. Punctuation becomes a word boundary, format
    controls such as zero-width joiners are ignored, and optional Arabic
    combining marks are removed. Other combining marks (including Indic
    vowel signs) are preserved because they carry meaning in those scripts.
    """

    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = (
        normalized
        .replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("\u0640", "")  # Arabic tatweel
    )

    characters = []
    for character in normalized:
        category = unicodedata.category(character)
        unicode_name = unicodedata.name(character, "")

        # Arabic vowel/recitation marks are optional in many transcript
        # renderings; do not remove combining marks from other scripts.
        if category.startswith("M") and unicode_name.startswith("ARABIC "):
            continue

        # Ignore zero-width formatting controls, which may appear in Arabic
        # and Urdu text without changing the visible words.
        if category == "Cf":
            continue

        if category[0] in {"L", "N", "M"}:
            characters.append(character)
        else:
            characters.append(" ")

    return " ".join("".join(characters).split())


def _source_scoped_evidence_texts(text: str) -> list[str]:
    """Return transcript text per source block, joining adjacent caption rows."""
    if not isinstance(text, str) or not text:
        return []

    source_pattern = re.compile(r"(?m)^\s*\[SOURCE\s+\d+\]\s*$")
    caption_pattern = re.compile(
        r"^\s*-\s*\[(?:\d{1,2}:)?\d{2}:\d{2}\s*[–—-]\s*"
        r"(?:\d{1,2}:)?\d{2}:\d{2}\]\s*(?P<text>.*)$"
    )
    blocks = source_pattern.split(text)
    evidence_texts = []
    for block in blocks:
        if not block.strip():
            continue
        caption_texts = []
        for line in block.splitlines():
            match = caption_pattern.match(line)
            if match and match.group("text").strip():
                caption_texts.append(match.group("text").strip())
        evidence_texts.append(" ".join(caption_texts) if caption_texts else block)
    return evidence_texts


def matched_gold_group_caption_spans(
    text: str,
    case: GoldEvidenceCase,
) -> dict[int, set[int]]:
    """Map matched gold groups to the exact caption rows containing their words.

    Returned indexes refer to text.splitlines(). Matching is restricted to
    one source block at a time, and normalized-caption character offsets
    identify only the rows actually touched by each matched phrase.
    """
    if not isinstance(text, str) or not text:
        return {}

    source_pattern = re.compile(r"(?m)^\s*\[SOURCE\s+\d+\]\s*$")
    caption_pattern = re.compile(
        r"^\s*-\s*\[(?:\d{1,2}:)?\d{2}:\d{2}\s*[–—-]\s*"
        r"(?:\d{1,2}:)?\d{2}:\d{2}\]\s*(?P<text>.*)$"
    )
    lines = text.splitlines()
    marker_indices = [
        index for index, line in enumerate(lines)
        if source_pattern.match(line.strip())
    ]
    source_ranges = (
        [
            (
                marker + 1,
                marker_indices[index + 1]
                if index + 1 < len(marker_indices)
                else len(lines),
            )
            for index, marker in enumerate(marker_indices)
        ]
        if marker_indices
        else [(0, len(lines))]
    )

    matched_spans: dict[int, set[int]] = {}
    for range_start, range_end in source_ranges:
        captions = []
        for line_index in range(range_start, range_end):
            match = caption_pattern.match(lines[line_index])
            if match and match.group("text").strip():
                normalized_caption = normalize_evidence_text(
                    match.group("text").strip()
                )
                if normalized_caption:
                    captions.append((line_index, normalized_caption))

        if not captions:
            continue

        joined_parts = []
        caption_offsets = []
        cursor = 0
        for line_index, normalized_caption in captions:
            start_offset = cursor
            end_offset = start_offset + len(normalized_caption)
            caption_offsets.append((line_index, start_offset, end_offset))
            joined_parts.append(normalized_caption)
            cursor = end_offset + 1
        joined_captions = " ".join(joined_parts)

        for group_index, alternatives in enumerate(case.groups):
            for alternative in alternatives:
                phrase = normalize_evidence_text(alternative)
                if not phrase:
                    continue

                search_start = 0
                while True:
                    match_start = joined_captions.find(phrase, search_start)
                    if match_start < 0:
                        break
                    match_end = match_start + len(phrase)
                    matched_spans.setdefault(group_index, set()).update(
                        line_index
                        for line_index, caption_start, caption_end in caption_offsets
                        if caption_start < match_end and caption_end > match_start
                    )
                    search_start = match_start + 1

    return matched_spans


def matched_gold_groups(
    text: str,
    case: GoldEvidenceCase,
) -> set[int]:
    """Return evidence groups present in transcript text.

    Timestamped caption rows are joined within each source block. Distinct
    source blocks are never concatenated to manufacture a phrase.
    """
    normalized_blocks = [
        normalize_evidence_text(block)
        for block in _source_scoped_evidence_texts(text)
    ]
    return {
        group_index
        for group_index, alternatives in enumerate(case.groups)
        if any(
            normalize_evidence_text(phrase) in normalized_block
            for phrase in alternatives
            for normalized_block in normalized_blocks
        )
    }


def matched_gold_answer_alias_groups(
    text: str,
    case: GoldEvidenceCase,
) -> set[int]:
    """Match only explicitly registered answer aliases, never source quotations."""
    if not isinstance(text, str) or not text or not case.generation_answer_aliases:
        return set()

    normalized_answer = normalize_evidence_text(text)
    padded_answer = f" {normalized_answer} "
    matched = set()
    for group_index, aliases in enumerate(case.generation_answer_aliases):
        for alias in aliases:
            normalized_alias = normalize_evidence_text(alias)
            if normalized_alias and f" {normalized_alias} " in padded_answer:
                matched.add(group_index)
                break
    return matched


def matched_gold_answer_groups(
    text: str,
    case: GoldEvidenceCase,
) -> set[int]:
    """Match generated-answer evidence, including curated answer-only aliases.

    This intentionally does not alter matched_gold_groups(), which remains the
    strict transcript/context matcher used by retrieval and prompt packing.
    """
    matched = matched_gold_groups(text, case)
    if not isinstance(text, str) or not text or not case.generation_answer_aliases:
        return matched

    normalized_answer = normalize_evidence_text(text)
    for group_index, aliases in enumerate(case.generation_answer_aliases):
        if group_index >= len(case.groups):
            break
        if any(
            normalize_evidence_text(alias) in normalized_answer
            for alias in aliases
        ):
            matched.add(group_index)

    return matched
