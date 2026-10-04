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
from dataclasses import dataclass


GOLD_EVIDENCE_VIDEO_ID = "MdeQMVBuGgY"


@dataclass(frozen=True)
class GoldEvidenceCase:
    """Human-curated evidence groups for one retrieval question.

    Each group contains alternative transcript phrases that represent the
    same evidence concept. A case can require coverage of several concepts.
    """

    question_id: str
    groups: tuple[tuple[str, ...], ...]
    min_group_coverage: float


GOLD_EVIDENCE_CASES = (
    GoldEvidenceCase(
        question_id="Q01",
        groups=(
            ("single class airline",),
            ("best flying experience", "premium experience"),
            ("in flight, entertainment", "offered meals"),
        ),
        min_group_coverage=2 / 3,
    ),
    GoldEvidenceCase(
        question_id="Q02",
        groups=(
            ("Formula 1", "Formula One car"),
        ),
        min_group_coverage=1.0,
    ),
    GoldEvidenceCase(
        question_id="Q03",
        groups=(
            ("United Breweries", "United Spirits"),
            ("Kingfisher Airlines", "Air Deccan"),
            ("Royal Challengers Bangalore", "Royal Challenge"),
        ),
        min_group_coverage=2 / 3,
    ),
    GoldEvidenceCase(
        question_id="Q04",
        groups=(
            ("Lehman Brothers", "global financial crisis"),
            ("banks will support you", "not to downsize Kingfisher"),
            (
                "government policies that didn't help",
                "mounting losses",
                "Etihad invest in Kingfisher",
            ),
        ),
        min_group_coverage=2 / 3,
    ),
    GoldEvidenceCase(
        question_id="Q05",
        groups=(
            ("global financial crisis", "Lehman Brothers"),
            ("working capital cycle got disrupted", "cash flow issue"),
            (
                "any money in the account of the airline was frozen",
                "oil bills",
            ),
        ),
        min_group_coverage=2 / 3,
    ),
    GoldEvidenceCase(
        question_id="Q06",
        groups=(
            ("brand needs a personality",),
            ("surrogate advertising",),
            ("water business around Kingfisher",),
        ),
        min_group_coverage=2 / 3,
    ),
    GoldEvidenceCase(
        question_id="Q07",
        groups=(
            ("4,999 crores", "4999 crores"),
            ("6203 crores",),
            ("14,100 crores",),
        ),
        min_group_coverage=2 / 3,
    ),
    GoldEvidenceCase(
        question_id="Q08",
        groups=(
            ("recovery of 14,100 crores", "14,100 crores recovered"),
            ("banks have been fully paid", "banks had been fully paid"),
            ("statement of account",),
        ),
        min_group_coverage=2 / 3,
    ),
    GoldEvidenceCase(
        question_id="Q09",
        groups=(
            ("banks will support you", "not to downsize Kingfisher"),
            ("asked only for policy changes",),
            ("government policies that didn't help",),
        ),
        min_group_coverage=2 / 3,
    ),
    GoldEvidenceCase(
        question_id="Q10",
        groups=(
            ("global financial crisis", "Lehman Brothers"),
            ("government policies that didn't help", "mounting losses"),
            ("final straw", "Etihad invest in Kingfisher"),
        ),
        min_group_coverage=2 / 3,
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
    """Normalize transcript text for deterministic phrase matching."""

    normalized = text.casefold()
    normalized = normalized.replace("’", "'")
    normalized = normalized.replace("“", '"')
    normalized = normalized.replace("”", '"')
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def matched_gold_groups(
    text: str,
    case: GoldEvidenceCase,
) -> set[int]:
    """Return group indexes whose evidence phrase occurs in text."""

    normalized = normalize_evidence_text(text)
    matched = set()

    for group_index, alternatives in enumerate(case.groups):
        if any(
            normalize_evidence_text(phrase) in normalized
            for phrase in alternatives
        ):
            matched.add(group_index)

    return matched
