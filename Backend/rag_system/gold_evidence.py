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
