"""Capture reproducible /chat answers and compare transcript evidence with citations.

Run from the repository root while the local FastAPI backend is running:
    python -m scripts.chat_answer_source_audit

The response body is decoded from UTF-8 bytes explicitly. Effective retrieval
settings are copied from the running API response, not from this process's
environment (which may differ from the backend process).
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import re
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from Backend.rag_system.benchmark_registry import (
    BENCHMARK_CURATION_BLOCKERS,
    VIDEO_BENCHMARKS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation_reports" / "chat_answer_source_audit.json"
REPORT_ID = "chat_answer_source_audit_v1"

AUDIT_RATE_LIMIT_RETRIES = 2
AUDIT_RATE_LIMIT_BUFFER_SECONDS = 1.0
AUDIT_RATE_LIMIT_FALLBACK_WAIT_SECONDS = 30.0

# Resolve live audit questions from the canonical video-scoped registry so
# prompts cannot silently drift from retrieval acceptance annotations.
def _registry_question(video_id: str, question_id: str) -> str:
    benchmark = VIDEO_BENCHMARKS.get(video_id)
    if benchmark is None:
        raise ValueError(f"No registered benchmark for video {video_id!r}.")
    for item in benchmark.question_cases:
        if item.get("id") == question_id:
            return str(item["question"])
    raise ValueError(
        f"No question {question_id!r} is registered for video {video_id!r}."
    )


def _make_audit_case(
    question_id: str,
    video_id: str,
    *,
    expected_behavior: str,
    answer_language: str | None = None,
    answer_requirements: tuple[dict[str, str], ...] = (),
    evidence_groups: tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    if expected_behavior not in {"answerable", "unanswerable"}:
        raise ValueError(f"Unsupported expected behavior: {expected_behavior!r}")
    case: dict[str, Any] = {
        "id": question_id,
        "video_id": video_id,
        "question": _registry_question(video_id, question_id),
        "expected_behavior": expected_behavior,
        "answer_requirements": list(answer_requirements),
        "evidence_groups": list(evidence_groups),
    }
    if answer_language is not None:
        case["answer_language"] = answer_language
    return case


# Phase 10.10 closure suite retained for reproducible regression comparisons.
# The three chunk annotations were verified against the isolated MiniLM
# acceptance namespace. Do not copy them to another namespace without checking
# the actual indexed caption chunks.
AUDIT_CASES: tuple[dict[str, Any], ...] = (
    _make_audit_case(
        "Q07", "MdeQMVBuGgY",
        expected_behavior="answerable",
        answer_language="en",
        evidence_groups=(
            {"label": "Kingfisher investment: 3,000 crore", "candidate_chunk_ids": [108]},
            {"label": "DRT judgment debt figure: 4,999 crore", "candidate_chunk_ids": [124]},
            {"label": "debt-certificate figure: 6,203 crore", "candidate_chunk_ids": [221]},
            {
                "label": "Ministry of Finance recovery figure: 14,100 crore",
                "candidate_chunk_ids": [153, 159, 221],
            },
        ),
    ),
    _make_audit_case(
        "AR03", "fAd2nKfrDB8",
        expected_behavior="answerable",
        answer_language="en",
        evidence_groups=(
            {"label": "heartbeat, blood movement, lungs and stomach phrase", "candidate_chunk_ids": [2]},
            {"label": "eyelid-blinking phrase", "candidate_chunk_ids": [2]},
        ),
    ),
    _make_audit_case(
        "AR08", "fAd2nKfrDB8",
        expected_behavior="answerable",
        answer_language="ar",
        evidence_groups=(
            {"label": "heartbeat, blood movement, lungs and stomach phrase", "candidate_chunk_ids": [2]},
            {"label": "eyelid-blinking phrase", "candidate_chunk_ids": [2]},
        ),
    ),
)


# Phase 10.11 is a representative live answer/source suite across the five
# registered pilot videos. All question text comes from benchmark_registry;
# answer requirements use the already-curated gold phrases/concepts.
PHASE10_11_AUDIT_CASES: tuple[dict[str, Any], ...] = (
    _make_audit_case(
        "Q07", "MdeQMVBuGgY",
        expected_behavior="answerable",
        answer_language="en",
        evidence_groups=AUDIT_CASES[0]["evidence_groups"],
    ),
    _make_audit_case(
        "Q13", "MdeQMVBuGgY",
        expected_behavior="unanswerable",
    ),
    _make_audit_case(
        "EW01", "pOnl-Zndcls",
        expected_behavior="answerable",
        answer_language="en",
        answer_requirements=(
            # This direct-fact question asks for the count. Require the number
            # and its unit; don't force the model to restate the question's
            # "express yourself" wording when "Three ways." is a complete answer.
            {"name": "three_ways", "pattern": r"\b(?:three|3)\s+ways?\b"},
        ),
    ),
    _make_audit_case(
        "EW05", "pOnl-Zndcls",
        expected_behavior="unanswerable",
    ),
    _make_audit_case(
        "AI01", "A-JBaNvv3Tk",
        expected_behavior="answerable",
        answer_language="en",
        answer_requirements=(
            {"name": "hackability", "pattern": r"\bhack\w*\b"},
            {"name": "ease_prediction", "pattern": r"\beas(?:y|ier|ily)\b"},
        ),
    ),
    _make_audit_case(
        "AI05", "A-JBaNvv3Tk",
        expected_behavior="unanswerable",
    ),
    _make_audit_case(
        "HI04", "INpnuL7qnzY",
        expected_behavior="answerable",
        answer_language="en",
        answer_requirements=(
            {
                "name": "named_chavri_tax",
                "pattern": r"\bch(?:avri|awri|owri|avari|awari)\b|चवरी|चावरी",
            },
            {"name": "tax_concept", "pattern": r"\b(?:tax|taxes|levy|kar)\b|कर|टैक्स"},
        ),
    ),
    _make_audit_case(
        "HI05", "INpnuL7qnzY",
        expected_behavior="unanswerable",
    ),
    _make_audit_case(
        "HI09", "INpnuL7qnzY",
        expected_behavior="answerable",
        answer_language="hi",
        answer_requirements=(
            {"name": "named_chavri_tax", "pattern": r"चवरी|चावरी|चवरी|\bchavri\b|\bchawri\b"},
            {"name": "tax_concept", "pattern": r"कर|टैक्स|\btax\b"},
        ),
    ),
    _make_audit_case(
        "HI10", "INpnuL7qnzY",
        expected_behavior="unanswerable",
    ),
    _make_audit_case(
        "AR03", "fAd2nKfrDB8",
        expected_behavior="answerable",
        answer_language="en",
        evidence_groups=AUDIT_CASES[1]["evidence_groups"],
    ),
    _make_audit_case(
        "AR04", "fAd2nKfrDB8",
        expected_behavior="answerable",
        answer_language="en",
        answer_requirements=(
            {"name": "sound_role", "pattern": r"\bsound\b"},
            {"name": "perception_or_compass", "pattern": r"\b(?:compass|percept\w*|perceiv\w*|guide|orient\w*|direction|navigat\w*)\b"},
        ),
    ),
    _make_audit_case(
        "AR05", "fAd2nKfrDB8",
        expected_behavior="unanswerable",
    ),
    _make_audit_case(
        "AR08", "fAd2nKfrDB8",
        expected_behavior="answerable",
        answer_language="ar",
        evidence_groups=AUDIT_CASES[2]["evidence_groups"],
    ),
    _make_audit_case(
        "AR09", "fAd2nKfrDB8",
        expected_behavior="answerable",
        answer_language="ar",
        answer_requirements=(
            {"name": "sound_subject", "pattern": r"الصوت|صوت"},
            {"name": "perception_role", "pattern": r"بوصلة|إدراك|ادراك|توجيه"},
        ),
    ),
    _make_audit_case(
        "AR10", "fAd2nKfrDB8",
        expected_behavior="unanswerable",
    ),
)

# Phase 11.0 broadens the accepted live audit from the representative 16-case
# pilot to every question in the five-video registry. The Phase 10.11 tuple
# above remains unchanged as a frozen regression suite.
PHASE11_0_ADDITIONAL_CASES: tuple[dict[str, Any], ...] = (
    _make_audit_case(
        "Q01", "MdeQMVBuGgY",
        expected_behavior="answerable", answer_language="en",
        answer_requirements=({"name": "airline_differentiator", "pattern": r"\bsingle.class\b|\blow.cost\b|\bpremium\b|\binflight entertainment\b|\bmeals\b|\bgood food\b"},),
    ),
    _make_audit_case(
        "Q02", "MdeQMVBuGgY",
        expected_behavior="answerable", answer_language="en",
        answer_requirements=({"name": "formula_one_connection", "pattern": r"\bformula\s*(?:one|1)\b|\bf1\b|\bmotorsport\b"},),
    ),
    _make_audit_case(
        "Q03", "MdeQMVBuGgY",
        expected_behavior="answerable", answer_language="en",
        answer_requirements=({"name": "business_or_team", "pattern": r"\b(?:royal challengers|united breweries|united spirits|kingfisher red)\b"},),
    ),
    _make_audit_case(
        "Q04", "MdeQMVBuGgY",
        expected_behavior="answerable", answer_language="en",
        answer_requirements=({"name": "failure_reason", "pattern": r"\b(?:financial crisis|government polic|etihad|mounting losses|money stopped)\b"},),
    ),
    _make_audit_case(
        "Q05", "MdeQMVBuGgY",
        expected_behavior="answerable", answer_language="en",
        answer_requirements=({"name": "operating_challenge", "pattern": r"\b(?:working capital|cash flow|oil bills|airport fees|landing charges)\b"},),
    ),
    _make_audit_case(
        "Q06", "MdeQMVBuGgY",
        expected_behavior="answerable", answer_language="en",
        answer_requirements=({"name": "brand_strategy", "pattern": r"\b(?:brand|surrogate advertising|personality)\b"},),
    ),
    _make_audit_case(
        "Q08", "MdeQMVBuGgY",
        expected_behavior="answerable", answer_language="en",
        answer_requirements=({"name": "bank_recovery", "pattern": r"\b14[, ]?100\b|\bfully paid\b|\bstatement of account\b|\bstatement in parliament\b"},),
    ),
    _make_audit_case(
        "Q09", "MdeQMVBuGgY",
        expected_behavior="answerable", answer_language="en",
        answer_requirements=({"name": "policy_detail", "pattern": r"\b(?:policy changes|aviation turbine fuel|crude|sales tax)\b"},),
    ),
    _make_audit_case(
        "Q10", "MdeQMVBuGgY",
        expected_behavior="answerable", answer_language="en",
        answer_requirements=({"name": "failure_explanation", "pattern": r"\b(?:financial crisis|government polic|etihad|mounting losses|run out of options)\b"},),
    ),
    _make_audit_case(
        "Q11", "MdeQMVBuGgY",
        expected_behavior="answerable", answer_language="en",
    ),
    # Q12 is marked unanswerable in the canonical registry. Its retrieval
    # expectation is may_retrieve, so this live case tests answer abstention
    # and source suppression rather than requiring retrieval to return zero
    # candidates before generation.
    _make_audit_case(
        "Q12", "MdeQMVBuGgY",
        expected_behavior="unanswerable",
    ),
    _make_audit_case(
        "EW02", "pOnl-Zndcls",
        expected_behavior="answerable", answer_language="en",
        answer_requirements=({"name": "work_explanation_structure", "pattern": r"\b30[- ]second\b|\bthirty[- ]second\b"},),
    ),
    _make_audit_case(
        "EW03", "pOnl-Zndcls",
        expected_behavior="answerable", answer_language="en",
        answer_requirements=({"name": "small_talk_difficulty", "pattern": r"\b(?:small talk|strangers|conversation)\b"},),
    ),
    _make_audit_case(
        "EW04", "pOnl-Zndcls",
        expected_behavior="answerable", answer_language="en",
        answer_requirements=(
            {"name": "email_structure", "pattern": r"\bemail\b"},
            {"name": "email_outcome", "pattern": r"\brepl(?:y|ies)\b|\bresponse\b"},
        ),
    ),
    _make_audit_case(
        "AI02", "A-JBaNvv3Tk",
        expected_behavior="answerable", answer_language="en",
        answer_requirements=({"name": "open_source_alignment_concern", "pattern": r"\bopen.source\b|\balignment\b"},),
    ),
    _make_audit_case(
        "AI03", "A-JBaNvv3Tk",
        expected_behavior="answerable", answer_language="en",
        answer_requirements=({"name": "unified_system", "pattern": r"\b(?:central system|one system|unified system)\b"},),
    ),
    _make_audit_case(
        "AI04", "A-JBaNvv3Tk",
        expected_behavior="answerable", answer_language="en",
        answer_requirements=({"name": "shipping_pace", "pattern": r"\bship(?:ping|ped)?\b"},),
    ),
    _make_audit_case(
        "HI01", "INpnuL7qnzY",
        expected_behavior="answerable", answer_language="en",
        answer_requirements=({"name": "named_movement", "pattern": r"\bbijoli?ya\b|\bbijolia\b|बिजोलिया"},),
    ),
    _make_audit_case(
        "HI02", "INpnuL7qnzY",
        expected_behavior="answerable", answer_language="en",
        answer_requirements=({"name": "movement_location", "pattern": r"\bbijoli?ya\b|\bbijolia\b|बिजोलिया"},),
    ),
    _make_audit_case(
        "HI03", "INpnuL7qnzY",
        expected_behavior="answerable", answer_language="en",
        answer_requirements=({"name": "tax_count", "pattern": r"\b84\b|\beighty.four\b|८४"},),
    ),
    _make_audit_case(
        "HI06", "INpnuL7qnzY",
        expected_behavior="answerable", answer_language="hi",
        answer_requirements=({"name": "named_movement", "pattern": r"बिजोलिया|पहला बड़ा किसान आंदोलन"},),
    ),
    _make_audit_case(
        "HI07", "INpnuL7qnzY",
        expected_behavior="answerable", answer_language="hi",
        answer_requirements=({"name": "movement_location", "pattern": r"बिजोलिया"},),
    ),
    _make_audit_case(
        "HI08", "INpnuL7qnzY",
        expected_behavior="answerable", answer_language="hi",
        answer_requirements=({"name": "tax_count", "pattern": r"84|८४"},),
    ),
    _make_audit_case(
        "AR01", "fAd2nKfrDB8",
        expected_behavior="answerable", answer_language="en",
        answer_requirements=({"name": "silence_and_rest", "pattern": r"\b(?:quiet|silence|rest)\b"},),
    ),
    _make_audit_case(
        "AR02", "fAd2nKfrDB8",
        expected_behavior="answerable", answer_language="en",
        answer_requirements=({"name": "room_cost", "pattern": r"\b600\b|\bsix hundred\b"},),
    ),
    _make_audit_case(
        "AR06", "fAd2nKfrDB8",
        expected_behavior="answerable", answer_language="ar",
        answer_requirements=({"name": "silence_and_rest", "pattern": r"الهدوء|الصمت|راحة"},),
    ),
    _make_audit_case(
        "AR07", "fAd2nKfrDB8",
        expected_behavior="answerable", answer_language="ar",
        answer_requirements=({"name": "room_cost", "pattern": r"600|٦٠٠|دولار"},),
    ),
)

PHASE11_0_AUDIT_CASES: tuple[dict[str, Any], ...] = (
    *PHASE10_11_AUDIT_CASES,
    *PHASE11_0_ADDITIONAL_CASES,
)
PHASE11_0_REPORT_ID = "phase11_0_full_registry_answer_source_audit_v1"
PHASE11_0_DEFAULT_OUTPUT = (
    PROJECT_ROOT / "evaluation_reports" / "phase11_0_full_registry_answer_source_audit.json"
)

PHASE10_11_REPORT_ID = "phase10_11_multivideo_answer_source_audit_v1"
PHASE10_11_DEFAULT_OUTPUT = (
    PROJECT_ROOT / "evaluation_reports" / "phase10_11_multivideo_answer_source_audit.json"
)
SUPPORTED_LIVE_LANGUAGES = ("en", "hi", "ar")


# These marker sequences are strong indicators that non-ASCII answer text has
# already been decoded with the wrong character encoding. They are warnings,
# not an automatic repair, because the client must not rewrite model output.
_MOJIBAKE_MARKERS = (
    "Ø", "Ù", "â", "Ã", "\ufffd",
    # Common Windows-1252 spellings of UTF-8 punctuation; "â" catches
    # a different Latin-1/control-code variant.
    "â€“", "â€”", "â€™", "â€œ", "â€", "â€˜", "â€¦",
)
_TIMESTAMP_RANGE_PATTERN = re.compile(
    r"(?<!\d)(?:\d{1,2}:)?\d{1,2}:\d{2}\s*[–—-]\s*"
    r"(?:\d{1,2}:)?\d{1,2}:\d{2}(?!\d)"
)


def decode_json_response(body: bytes) -> dict[str, Any]:
    """Decode a JSON HTTP body as UTF-8 and require an object payload."""
    text = body.decode("utf-8", errors="strict")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Expected the API JSON response to be an object.")
    return payload


def _chunk_ids(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return list(dict.fromkeys(str(value) for value in values if value is not None))


def _format_timestamp(seconds: Any) -> str | None:
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return None
    if value < 0:
        return None

    total = int(value)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _normalize_range(value: str) -> tuple[str, str] | None:
    parts = re.split(r"\s*[–—-]\s*", value.strip(), maxsplit=1)
    if len(parts) != 2:
        return None
    return parts[0].strip(), parts[1].strip()


def _timestamp_to_seconds(value: str) -> int | None:
    """Convert a displayed MM:SS or HH:MM:SS value into whole seconds."""
    try:
        parts = [int(part) for part in value.strip().split(":")]
    except (AttributeError, TypeError, ValueError):
        return None

    if len(parts) == 2:
        minutes, seconds = parts
        if seconds >= 60:
            return None
        return minutes * 60 + seconds

    if len(parts) == 3:
        hours, minutes, seconds = parts
        if minutes >= 60 or seconds >= 60:
            return None
        return hours * 3600 + minutes * 60 + seconds

    return None


def _timestamp_range_bounds(value: str) -> tuple[int, int] | None:
    """Return inclusive whole-second bounds for a displayed timestamp range."""
    normalized = _normalize_range(value)
    if normalized is None:
        return None
    start = _timestamp_to_seconds(normalized[0])
    end = _timestamp_to_seconds(normalized[1])
    if start is None or end is None or end < start:
        return None
    return start, end


def _timestamp_range_covered_by_sources(
    timestamp_range: str,
    source_ranges: list[dict[str, Any]],
) -> bool:
    """Require every second in an answer range to be covered by returned sources."""
    target = _timestamp_range_bounds(timestamp_range)
    if target is None:
        return False
    target_start, target_end = target

    intervals = []
    for source in source_ranges:
        display_range = source.get("display_range")
        if not isinstance(display_range, str):
            continue
        bounds = _timestamp_range_bounds(display_range)
        if bounds is not None:
            intervals.append(bounds)

    intervals.sort()
    covered_until = target_start
    for source_start, source_end in intervals:
        if source_end < covered_until:
            continue
        if source_start > covered_until + 1:
            return False
        covered_until = max(covered_until, source_end)
        if covered_until >= target_end:
            return True
    return False


def _answer_timestamp_ranges(answer: str) -> list[str]:
    """Extract displayed timestamp ranges whether inline or in the footer."""
    return [
        item.group(0).strip()
        for item in _TIMESTAMP_RANGE_PATTERN.finditer(answer)
    ]


def _substantive_answer(answer: str) -> str:
    """Remove the application timestamp footer before checking answer content."""
    return re.sub(
        r"(?im)^[ \t]*Relevant timestamps?[ \t]*:[^\r\n]*(?:\r?\n|$)",
        "",
        answer,
    ).strip()


def _normalize_answer_requirement_text(answer: str) -> str:
    """Remove inline Markdown wrappers before matching answer requirements."""
    text = _substantive_answer(answer)
    # Do not let presentation markup split otherwise valid phrases such as
    # "**three** ways" and cause a false negative in the acceptance gate.
    text = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\w)([*_])([^*_]+?)\1(?!\w)", r"\2", text)
    text = re.sub(r"`+([^`\n]+?)`+", r"\1", text)
    # Unicode no-break spaces (including U+202F) are valid word separators.
    # Normalize whitespace so content checks are formatting-agnostic.
    return re.sub(r"\s+", " ", text).strip()


def _quality_check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _amount_clauses(answer: str, pattern: str) -> list[str]:
    clauses = [
        clause.strip()
        for clause in re.split(r"(?<=[.!?;])\s+|[\r\n]+", _substantive_answer(answer))
        if clause.strip()
    ]
    return [clause for clause in clauses if re.search(pattern, clause, re.IGNORECASE)]


def _q07_answer_checks(answer: str) -> list[dict[str, Any]]:
    checks = []
    verification_abstention = "couldn't reliably verify the financial figures" in (
        _substantive_answer(answer).casefold()
    )
    checks.append(_quality_check(
        "financial_answer_not_abstained",
        not verification_abstention,
        (
            "The answer must provide source-supported financial claims rather than "
            "returning the safety abstention."
            if verification_abstention
            else "The answer did not trigger the financial verification abstention."
        ),
    ))
    claim_specs = [
        (
            "investment_3000",
            r"(?<!\d)3[ ,\u202f]?000(?!\d)",
            r"\b(invest(?:ed|ment)?|injected|infused|put)\b",
            "The 3,000-crore figure must be attributed to an investment into Kingfisher.",
        ),
        (
            "drt_4999",
            r"(?<!\d)4[ ,\u202f]?999(?!\d)",
            r"\b(drt|debt|tribunal|judg(?:e)?ment)\b",
            "The 4,999-crore figure must remain tied to the DRT/debt passage.",
        ),
        (
            "certificate_6203",
            r"(?<!\d)6[ ,\u202f]?203(?!\d)",
            r"\b(certificate|drt|debt)\b",
            "The 6,203-crore figure must remain tied to the debt-certificate passage.",
        ),
        (
            "recovery_14100",
            r"(?<!\d)14[ ,\u202f]?100(?!\d)",
            r"\b(recover(?:ed|y|ies)?|recovery)\b",
            "The 14,100-crore figure must be described as a recovery, not debt.",
        ),
    ]
    for name, amount_pattern, attribution_pattern, detail in claim_specs:
        matching_clauses = _amount_clauses(answer, amount_pattern)
        attributed = any(
            re.search(attribution_pattern, clause, re.IGNORECASE)
            for clause in matching_clauses
        )
        checks.append(_quality_check(name, attributed, detail))

    recovery_clauses = _amount_clauses(answer, r"(?<!\d)14[ ,\u202f]?100(?!\d)")
    has_report_attribution = any(
        re.search(r"\b(ministry|finance|report)\b", clause, re.IGNORECASE)
        and re.search(r"\b(recover(?:ed|y|ies)?|recovery)\b", clause, re.IGNORECASE)
        for clause in recovery_clauses
    )
    debt_total_misattributed = any(
        re.search(
            r"(?:company(?:['’]s)?|kingfisher(?: airlines?)?)\s+(?:company\s+)?debt.{0,35}(?:total|totals|amount|14[ ,\u202f]?100)"
            r"|(?:total debt|company debt|debt total).{0,35}14[ ,\u202f]?100",
            clause,
            re.IGNORECASE,
        )
        for clause in recovery_clauses
    )
    checks.append(_quality_check(
        "recovery_source_attribution",
        has_report_attribution and not debt_total_misattributed,
        "The recovery amount must be attributed to the Ministry/Finance report and not asserted as the company debt total.",
    ))

    has_4999 = bool(_amount_clauses(answer, r"(?<!\d)4[ ,\u202f]?999(?!\d)"))
    has_6203 = bool(_amount_clauses(answer, r"(?<!\d)6[ ,\u202f]?203(?!\d)"))
    uncertainty_terms = (
        r"\bunclear\b|\bnot clear\b|"
        r"does not (?:clearly )?(?:explain|establish|reconcile)|"
        r"doesn't (?:clearly )?(?:explain|establish|reconcile)|not explained|"
        r"cannot reconcile|can't reconcile|cannot determine|can't determine|"
        r"uncertain relationship|unresolved relationship|relationship.{0,30}(?:unclear|not clear|unexplained)|"
        r"not enough information to determine"
    )
    uncertainty_preserved = (
        has_4999
        and has_6203
        and bool(re.search(uncertainty_terms, _substantive_answer(answer), re.IGNORECASE))
    )
    checks.append(_quality_check(
        "unclear_relationship_preserved",
        uncertainty_preserved,
        "The 4,999- and 6,203-crore figures must not be silently reconciled when the captions do not clearly explain their relationship.",
    ))

    # These malformed figures and the "official 9,000-crore recovery" claim
    # were observed in live Q07 generations. Keep the audit from passing just
    # because the four expected claims are also present in the same answer.
    known_unsupported_patterns = (
        ("malformed_623", r"(?<!\d)6\s*,\s*23(?!\d)"),
        ("malformed_62003", r"(?<!\d)6\s*,?\s*2003(?!\d)"),
        ("malformed_60002", r"(?<!\d)6\s*,?\s*0002(?!\d)"),
        (
            "unsupported_official_9000_recovery",
            r"(?:(?:official|debt[\s\-‑–—]*recovery|recovery).{0,60}"
            r"9[ ,\u202f]?000(?!\d)|"
            r"9[ ,\u202f]?000(?!\d).{0,60}"
            r"(?:official|debt[\s\-‑–—]*recovery|recovery))",
        ),
    )
    unsupported_found = [
        label
        for label, pattern in known_unsupported_patterns
        if re.search(pattern, _substantive_answer(answer), re.IGNORECASE)
    ]
    checks.append(_quality_check(
        "malformed_or_unsupported_financial_amounts_absent",
        not unsupported_found,
        (
            "No known malformed extra financial amounts or unsupported official "
            "9,000-crore recovery claim were detected."
            if not unsupported_found
            else "Unsupported financial claims detected: " + ", ".join(unsupported_found)
        ),
    ))
    return checks


def _bodily_sound_checks(case_id: str, answer: str) -> list[dict[str, Any]]:
    text = _substantive_answer(answer)
    if case_id == "AR03":
        requirements = [
            ("heartbeat", r"heart\s*beat|heartbeat"),
            ("blood_movement", r"blood.{0,35}(?:vein|arter|flow|rush|move)|(?:vein|arter).{0,35}blood"),
            ("lungs_or_breathing", r"lung|breath"),
            ("stomach", r"stomach|rumbl|belly"),
            ("eyelids_blinking", r"eyelid|blink|eye.{0,25}(?:blink|lid)"),
        ]
        checks = [
            _quality_check(
                f"sound_{name}",
                bool(re.search(pattern, text, re.IGNORECASE)),
                f"The answer must mention the transcript-supported {name.replace('_', ' ')} sound.",
            )
            for name, pattern in requirements
        ]
        checks.append(_quality_check(
            "question_language",
            not bool(re.search(r"[\u0600-\u06ff]", text)),
            "The English question must receive an English answer.",
        ))
        return checks

    requirements = [
        ("heartbeat", r"نبض.{0,20}قلب|قلب.{0,20}نبض"),
        ("blood_movement", r"(?:حركة|تدفق|جريان|سريان).{0,25}الدم|الدم.{0,30}(?:حركة|تدفق|جريان|سريان|عروق|شرايين|أوعية)"),
        ("lungs_or_breathing", r"رئتين|الرئتين|الرئة|تنفس"),
        ("stomach", r"المعدة|معدتك|صوت المعدة"),
        ("eyelids_blinking", r"(?:عين|عيني|جفن|جفون|الجفون|الجفن).{0,30}(?:ترمش|تومض|ترف|رمش|تحرك)|(?:ترمش|تومض|ترف|رمش|تحرك).{0,30}(?:عين|جفن|جفون|الجفون)"),
    ]
    checks = [
        _quality_check(
            f"sound_{name}",
            bool(re.search(pattern, text, re.IGNORECASE)),
            f"The Arabic answer must mention the transcript-supported {name.replace('_', ' ')} sound.",
        )
        for name, pattern in requirements
    ]
    arabic_chars = len(re.findall(r"[\u0600-\u06ff]", text))
    latin_chars = len(re.findall(r"[A-Za-z]", text))
    checks.append(_quality_check(
        "question_language",
        arabic_chars >= 12 and arabic_chars > latin_chars,
        "The Arabic question must receive a substantive Arabic answer, not an English response.",
    ))
    return checks


_ABSTENTION_PATTERN = re.compile(
    r"couldn['’]?t find enough information|may not be covered|"
    r"not covered by (?:this|the) video|not mentioned in (?:this|the) video|"
    r"not enough information|unable to answer|cannot answer|can['’]?t answer|"
    r"no information (?:is )?(?:available|provided)|not specified|"
    r"वीडियो में.{0,45}नहीं|जानकारी.{0,25}नहीं|"
    r"(?:नहीं|नही).{0,30}(?:बताया|उल्लेख|मिला|उपलब्ध|मौजूद)|"
    r"इस वीडियो में.{0,35}(?:नहीं|उल्लेख)|"
    r"(?:لا|لم)\s+أجد.{0,55}(?:معلومات|معلومة)|"
    r"قد\s+لا\s+يكون.{0,60}(?:مغطى|مذكور|موجود|متناول|متوفر)|"
    r"(?:لا|لم).{0,40}(?:يذكر|تذكر|يتناول|تحتوي|توجد|يوجد|تتوفر|أستطيع|يمكن)|"
    r"غير مذكور|غير موجود|لا توجد معلومات|لا تتوفر معلومات",
    re.IGNORECASE,
)


# Exact fallback forms emitted by the backend. Keep this deliberately narrower
# than _ABSTENTION_PATTERN: a substantive answer can contain a caveat such as
# "the video does not specify the date" and still answer the question.
_ABSTENTION_ONLY_PATTERNS = (
    re.compile(
        r"I couldn't find enough information about that in the video"
        r"(?:\.\s*It may not be covered by this video)?[.!]?",
        re.IGNORECASE,
    ),
    re.compile(
        r"वीडियो में इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं मिली[।.!]?"
    ),
    re.compile(
        r"لم أجد في الفيديو معلومات كافية للإجابة عن هذا السؤال[.!؟]?"
    ),
)


def _is_abstention_only_answer(answer: str) -> bool:
    """Return true for canonical fallback-only answers, not caveated answers."""
    text = _substantive_answer(answer).strip()
    return any(pattern.fullmatch(text) for pattern in _ABSTENTION_ONLY_PATTERNS)


def _answer_language_matches(answer: str, language: str) -> bool:
    """Use script proportions as a conservative language guard for live answers."""
    text = _substantive_answer(answer)
    latin = len(re.findall(r"[A-Za-z]", text))
    devanagari = len(re.findall(r"[\u0900-\u097f]", text))
    arabic = len(re.findall(r"[\u0600-\u06ff]", text))
    if language == "en":
        return latin >= 8 and latin > devanagari and latin > arabic
    if language == "hi":
        return devanagari >= 6 and devanagari > latin and devanagari > arabic
    if language == "ar":
        return arabic >= 12 and arabic > latin and arabic > devanagari
    return False


def _answer_quality_checks(
    case_id: str | dict[str, Any],
    answer: str,
) -> list[dict[str, Any]]:
    case = case_id if isinstance(case_id, dict) else {"id": case_id}
    normalized_case_id = str(case.get("id", ""))
    expected_behavior = case.get("expected_behavior", "answerable")
    substantive = _substantive_answer(answer)

    if expected_behavior == "unanswerable":
        return [
            _quality_check(
                "unanswerable_case_abstained",
                bool(_ABSTENTION_PATTERN.search(substantive)),
                "The unrelated/unanswerable question must receive an explicit abstention, not a guessed factual answer.",
            )
        ]

    checks = [
        _quality_check(
            "substantive_answer",
            bool(substantive) and not _is_abstention_only_answer(substantive),
            "The answer must contain a substantive response rather than the fallback.",
        )
    ]
    if normalized_case_id == "Q07":
        checks.extend(_q07_answer_checks(answer))
    elif normalized_case_id in {"AR03", "AR08"}:
        checks.extend(_bodily_sound_checks(normalized_case_id, answer))

    requirement_text = _normalize_answer_requirement_text(answer)
    for requirement in case.get("answer_requirements", []):
        if not isinstance(requirement, dict):
            continue
        name = str(requirement.get("name", "answer_requirement"))
        pattern = requirement.get("pattern")
        passed = isinstance(pattern, str) and bool(
            re.search(pattern, requirement_text, flags=re.IGNORECASE)
        )
        checks.append(_quality_check(
            f"answer_requirement_{name}",
            passed,
            (
                f"The answer must satisfy the registered {name.replace('_', ' ')} requirement."
                if passed
                else f"The answer did not satisfy the registered {name.replace('_', ' ')} requirement."
            ),
        ))

    answer_language = case.get("answer_language")
    if isinstance(answer_language, str):
        passed = _answer_language_matches(answer, answer_language)
        checks.append(_quality_check(
            "question_language",
            passed,
            (
                f"The answer uses the expected {answer_language} script."
                if passed
                else f"The answer does not use the expected {answer_language} script."
            ),
        ))
    return checks


def _citation_provenance_diagnostics(
    *,
    citation_policy: Any,
    retrieval: dict[str, Any],
    generation: dict[str, Any],
    returned_source_ids: list[str],
) -> dict[str, Any]:
    """Prove that each returned source was prompt-visible or an anchor fallback."""
    policy = citation_policy if isinstance(citation_policy, str) else None
    source_set = set(returned_source_ids)
    raw_prompt_ids = None
    context_budget = generation.get("context_budget")
    if isinstance(context_budget, dict):
        candidate_ids = context_budget.get("included_chunk_ids")
        if isinstance(candidate_ids, list):
            raw_prompt_ids = _chunk_ids(candidate_ids)

    raw_anchor_ids = retrieval.get("citation_chunk_ids")
    anchors_available = isinstance(raw_anchor_ids, list)
    anchor_ids = _chunk_ids(raw_anchor_ids) if anchors_available else []

    if policy == "prompt_included_chunks":
        if raw_prompt_ids is not None:
            allowed_ids = set(raw_prompt_ids)
            mode = "generation_context"
            fallback_used = False
            trace_available = True
        elif anchors_available:
            allowed_ids = set(anchor_ids)
            mode = "selected_anchor_fallback"
            fallback_used = True
            trace_available = True
        else:
            allowed_ids = set()
            mode = "unverifiable_missing_prompt_and_anchor_trace"
            fallback_used = False
            trace_available = False
    elif policy == "selected_anchors":
        allowed_ids = set(anchor_ids)
        mode = "selected_anchors"
        fallback_used = False
        trace_available = anchors_available
    else:
        # Unknown/absent policy appears in older synthetic fixtures; the live
        # Phase 10.11 suite separately requires the exact expected policy.
        allowed_ids = set()
        mode = "not_applicable"
        fallback_used = False
        trace_available = False

    outside_ids = sorted(source_set - allowed_ids)
    if policy not in {"prompt_included_chunks", "selected_anchors"}:
        passed = True
    else:
        passed = not outside_ids and (trace_available or not source_set)

    return {
        "citation_policy": policy,
        "mode": mode,
        "used_selected_anchor_fallback": fallback_used,
        "prompt_inclusion_trace_available": raw_prompt_ids is not None,
        "selected_anchor_trace_available": anchors_available,
        "allowed_chunk_ids": sorted(allowed_ids),
        "returned_source_chunk_ids": sorted(source_set),
        "source_chunk_ids_outside_allowed_set": outside_ids,
        "passed": passed,
    }


def analyze_response(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    """Compare annotated candidate evidence with context, anchors, sources, and timestamp line."""
    observability = response.get("observability")
    if not isinstance(observability, dict):
        observability = {}
    retrieval = observability.get("retrieval")
    if not isinstance(retrieval, dict):
        retrieval = {}

    generation = observability.get("generation")
    if not isinstance(generation, dict):
        generation = {}
    context_budget = generation.get("context_budget")
    if (
        isinstance(context_budget, dict)
        and isinstance(context_budget.get("included_chunk_ids"), list)
    ):
        # Prefer chunks actually selected for prompt packing. Older backend
        # versions have no such diagnostic and retain the retrieval-trace fallback.
        final_context_ids = _chunk_ids(context_budget.get("included_chunk_ids"))
    else:
        final_context_ids = _chunk_ids(retrieval.get("final_chunk_ids"))
    anchor_ids = _chunk_ids(retrieval.get("citation_chunk_ids"))
    source_objects = response.get("sources")
    if not isinstance(source_objects, list):
        source_objects = []

    returned_source_ids: list[str] = []
    source_ranges: list[dict[str, Any]] = []
    for source in source_objects:
        if not isinstance(source, dict):
            continue
        ids = _chunk_ids(source.get("chunk_ids"))
        returned_source_ids.extend(ids)
        start = source.get("start")
        end = source.get("end")
        start_label = _format_timestamp(start)
        end_label = _format_timestamp(end)
        source_ranges.append(
            {
                "source_id": source.get("source_id"),
                "chunk_ids": ids,
                "start": start,
                "end": end,
                "display_range": (
                    f"{start_label}–{end_label}"
                    if start_label is not None and end_label is not None
                    else None
                ),
            }
        )

    returned_source_ids = list(dict.fromkeys(returned_source_ids))
    answer = response.get("answer")
    if not isinstance(answer, str):
        answer = ""

    timestamp_ranges = _answer_timestamp_ranges(answer)
    normalized_summary_ranges = [
        bounds
        for item in timestamp_ranges
        if (bounds := _timestamp_range_bounds(item)) is not None
    ]
    summary_source_ids: list[str] = []
    for source in source_ranges:
        source_bounds = (
            _timestamp_range_bounds(source["display_range"])
            if isinstance(source.get("display_range"), str)
            else None
        )
        if source_bounds is None:
            continue
        source_start, source_end = source_bounds
        if any(
            source_start <= summary_end and source_end >= summary_start
            for summary_start, summary_end in normalized_summary_ranges
        ):
            summary_source_ids.extend(source["chunk_ids"])
    summary_source_ids = list(dict.fromkeys(summary_source_ids))

    context_set = set(final_context_ids)
    anchor_set = set(anchor_ids)
    source_set = set(returned_source_ids)
    summary_set = set(summary_source_ids)
    retrieval_config = response.get("retrieval_config")
    if not isinstance(retrieval_config, dict):
        retrieval_config = {}
    citation_provenance = _citation_provenance_diagnostics(
        citation_policy=retrieval_config.get("citation_policy"),
        retrieval=retrieval,
        generation=generation,
        returned_source_ids=returned_source_ids,
    )

    evidence_group_results = []
    for group in case.get("evidence_groups", []):
        candidates = _chunk_ids(group.get("candidate_chunk_ids"))
        candidate_set = set(candidates)
        evidence_group_results.append(
            {
                "label": group.get("label"),
                "candidate_chunk_ids": candidates,
                "present_in_final_context": bool(candidate_set & context_set),
                "present_in_selected_anchors": bool(candidate_set & anchor_set),
                "present_in_returned_sources": bool(candidate_set & source_set),
                "represented_in_plain_text_timestamp_summary": bool(
                    candidate_set & summary_set
                ),
                "context_chunk_ids": [
                    chunk_id for chunk_id in candidates if chunk_id in context_set
                ],
                "anchor_chunk_ids": [
                    chunk_id for chunk_id in candidates if chunk_id in anchor_set
                ],
                "source_chunk_ids": [
                    chunk_id for chunk_id in candidates if chunk_id in source_set
                ],
                "timestamp_summary_chunk_ids": [
                    chunk_id for chunk_id in candidates if chunk_id in summary_set
                ],
            }
        )

    mojibake_markers = [marker for marker in _MOJIBAKE_MARKERS if marker in answer]
    quality_checks = _answer_quality_checks(case, answer)
    if citation_provenance["citation_policy"] in {"prompt_included_chunks", "selected_anchors"}:
        quality_checks.append(_quality_check(
            "citation_prompt_provenance",
            citation_provenance["passed"],
            (
                "Every returned source chunk is within the prompt-visible citation set."
                if citation_provenance["passed"]
                else (
                    "Returned source chunks are not permitted by the active citation policy: "
                    + ", ".join(citation_provenance["source_chunk_ids_outside_allowed_set"])
                )
            ),
        ))

    generation_used_fallback = generation.get("grounded_financial_fallback_used") is True
    if str(case.get("id", "")) == "Q07" and generation_used_fallback:
        required_fallback_labels = (
            ("Investment", r"(?im)^\s*-\s*Investment:"),
            ("DRT debt", r"(?im)^\s*-\s*DRT debt:"),
            ("Debt certificate", r"(?im)^\s*-\s*Debt certificate:"),
            ("Ministry of Finance recovery report",
             r"(?im)^\s*-\s*Ministry of Finance recovery report:"),
        )
        missing_fallback_labels = [
            label for label, pattern in required_fallback_labels
            if not re.search(pattern, _substantive_answer(answer))
        ]
        quality_checks.append(_quality_check(
            "financial_fallback_statements_are_structured",
            not missing_fallback_labels,
            (
                "The deterministic financial fallback uses complete, topic-labeled statements."
                if not missing_fallback_labels
                else "Missing complete financial statement labels: "
                + ", ".join(missing_fallback_labels)
            ),
        ))

    missing_context_groups = [
        group["label"]
        for group in evidence_group_results
        if not group["present_in_final_context"]
    ]
    quality_checks.append(_quality_check(
        "annotated_evidence_in_retrieved_context",
        not missing_context_groups,
        (
            "All annotated evidence groups are represented in the retrieved context."
            if not missing_context_groups
            else "Evidence groups missing from the retrieved context: "
            + ", ".join(missing_context_groups)
        ),
    ))

    timestamp_alignment_failures = []
    for group in evidence_group_results:
        if not group["present_in_final_context"]:
            continue
        if not group["present_in_returned_sources"]:
            timestamp_alignment_failures.append(
                f"{group['label']}: no returned clickable source"
            )
        elif not group["represented_in_plain_text_timestamp_summary"]:
            timestamp_alignment_failures.append(
                f"{group['label']}: not represented by the timestamp summary"
            )
    quality_checks.append(_quality_check(
        "evidence_source_and_timestamp_alignment",
        not timestamp_alignment_failures,
        (
            "Annotated answer evidence is represented by returned sources and the timestamp summary."
            if not timestamp_alignment_failures
            else "; ".join(timestamp_alignment_failures)
        ),
    ))
    quality_checks.append(_quality_check(
        "encoding_integrity",
        not bool(mojibake_markers),
        (
            "No configured mojibake markers were detected."
            if not mojibake_markers
            else "Potential mojibake markers: " + ", ".join(mojibake_markers)
        ),
    ))

    generation = observability.get("generation")
    if not isinstance(generation, dict):
        generation = {}
    budget = generation.get("context_budget")
    budget_valid = (
        isinstance(budget, dict)
        and isinstance(budget.get("budget_chars"), int)
        and isinstance(budget.get("final_chars"), int)
        and 0 < budget["final_chars"] <= budget["budget_chars"]
    )
    quality_checks.append(_quality_check(
        "generation_context_budget",
        budget_valid,
        (
            f"Generation context is {budget['final_chars']} / {budget['budget_chars']} characters."
            if budget_valid
            else "Generation context-budget diagnostics are missing or exceed the configured budget."
        ),
    ))

    expected_behavior = case.get("expected_behavior", "answerable")
    if expected_behavior == "unanswerable":
        no_sources = len(returned_source_ids) == 0
        no_timestamps = len(timestamp_ranges) == 0
        quality_checks.append(_quality_check(
            "unanswerable_sources_suppressed",
            no_sources and no_timestamps,
            (
                "The unrelated/unanswerable question has no clickable sources or source timestamp claims."
                if no_sources and no_timestamps
                else "The unrelated/unanswerable question returned sources or timestamp claims."
            ),
        ))
    else:
        sources_present = len(returned_source_ids) > 0
        timestamp_alignment_ok = (
            len(timestamp_ranges) > 0
            and all(
                _timestamp_range_covered_by_sources(item, source_ranges)
                for item in timestamp_ranges
            )
        )
        quality_checks.append(_quality_check(
            "answerable_sources_present",
            sources_present,
            (
                "The answerable question returns at least one clickable source."
                if sources_present
                else "The answerable question did not return a clickable source."
            ),
        ))
        quality_checks.append(_quality_check(
            "timestamp_summary_matches_returned_sources",
            timestamp_alignment_ok,
            (
                "Every displayed timestamp range is fully covered by returned source intervals."
                if timestamp_alignment_ok
                else "The timestamp summary is missing or at least one displayed interval extends beyond the returned source intervals."
            ),
        ))

    quality = {
        "passed": all(item["passed"] for item in quality_checks),
        "checks": quality_checks,
        "failed_checks": [item["name"] for item in quality_checks if not item["passed"]],
    }
    return {
        "effective_retrieval_config": response.get("retrieval_config"),
        "retrieval_method": response.get("retrieval_method"),
        "citation_provenance": citation_provenance,
        "final_context_chunk_ids": final_context_ids,
        "selected_anchor_chunk_ids": anchor_ids,
        "context_only_chunk_ids": [
            chunk_id for chunk_id in final_context_ids if chunk_id not in anchor_set
        ],
        "returned_source_chunk_ids": returned_source_ids,
        "plain_text_timestamp_ranges": timestamp_ranges,
        "source_ranges": source_ranges,
        "answer_text_encoding_suspect": bool(mojibake_markers),
        "possible_mojibake_markers": mojibake_markers,
        "annotated_evidence_groups": evidence_group_results,
        "quality": quality,
    }


def build_coverage_manifest(
    cases: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> dict[str, Any]:
    """Account for every registered video/question and disclose unevaluated cases."""
    audited_by_video: dict[str, list[dict[str, Any]]] = {}
    unregistered_cases: list[dict[str, str]] = []
    seen_keys: set[tuple[str, str]] = set()
    duplicate_cases: list[dict[str, str]] = []

    for case in cases:
        video_id = str(case.get("video_id", ""))
        question_id = str(case.get("id", ""))
        benchmark = VIDEO_BENCHMARKS.get(video_id)
        question_ids = (
            {str(item.get("id")) for item in benchmark.question_cases}
            if benchmark is not None
            else set()
        )
        if benchmark is None or question_id not in question_ids:
            unregistered_cases.append({
                "video_id": video_id,
                "question_id": question_id,
            })
            continue
        key = (video_id, question_id)
        if key in seen_keys:
            duplicate_cases.append({
                "video_id": video_id,
                "question_id": question_id,
            })
        seen_keys.add(key)
        audited_by_video.setdefault(video_id, []).append(case)

    video_rows = []
    unassessed_question_cases = []
    for video_id, benchmark in VIDEO_BENCHMARKS.items():
        audited_cases = audited_by_video.get(video_id, [])
        audited_ids = {str(case.get("id")) for case in audited_cases}
        registered_ids = [str(case.get("id")) for case in benchmark.question_cases]
        not_audited = [
            question_id for question_id in registered_ids
            if question_id not in audited_ids
        ]
        for question_id in not_audited:
            unassessed_question_cases.append({
                "video_id": video_id,
                "transcript_language": benchmark.transcript_language,
                "question_id": question_id,
                "status": "retrieval_benchmark_only_not_live_answer_source_audited",
            })
        positive_count = sum(
            case.get("expected_behavior") == "answerable"
            for case in audited_cases
        )
        negative_count = sum(
            case.get("expected_behavior") == "unanswerable"
            for case in audited_cases
        )
        video_rows.append({
            "video_id": video_id,
            "title": benchmark.title,
            "transcript_language": benchmark.transcript_language,
            "video_url": benchmark.video_url,
            "transcript_source_url": benchmark.transcript_source_url,
            "caption_track_kind": benchmark.caption_track_kind,
            "transcript_source_kind": benchmark.transcript_source_kind,
            "annotation_source_urls": list(benchmark.annotation_source_urls),
            "caption_validation_note": benchmark.caption_validation_note,
            "registered_question_case_count": len(registered_ids),
            "live_audit_question_ids": [
                str(case.get("id")) for case in audited_cases
            ],
            "live_audit_case_count": len(audited_cases),
            "live_answerable_case_count": positive_count,
            "live_unanswerable_control_count": negative_count,
            "unevaluated_question_ids": not_audited,
            "representative_live_coverage_passed": (
                positive_count >= 1 and negative_count >= 1
            ),
        })

    blocker = BENCHMARK_CURATION_BLOCKERS
    blocked_languages = []
    if isinstance(blocker, dict) and blocker:
        blocked_languages.append(dict(blocker))

    expected_video_ids = set(VIDEO_BENCHMARKS)
    covered_video_ids = set(audited_by_video)
    all_videos_have_representative_coverage = (
        covered_video_ids == expected_video_ids
        and all(row["representative_live_coverage_passed"] for row in video_rows)
    )
    return {
        "supported_live_languages": list(SUPPORTED_LIVE_LANGUAGES),
        "registered_video_count": len(VIDEO_BENCHMARKS),
        "registered_question_case_count": sum(
            len(benchmark.question_cases)
            for benchmark in VIDEO_BENCHMARKS.values()
        ),
        "live_audit_case_count": len(cases),
        "live_audit_video_count": len(covered_video_ids),
        "all_registered_videos_have_answerable_and_unanswerable_live_cases": (
            all_videos_have_representative_coverage
        ),
        "unregistered_live_cases": unregistered_cases,
        "duplicate_live_cases": duplicate_cases,
        "blocked_languages": blocked_languages,
        "videos": video_rows,
        "question_case_accounting": {
            "live_audited_count": len(seen_keys),
            "not_live_audited_count": len(unassessed_question_cases),
            "not_live_audited_cases": unassessed_question_cases,
            "accounting_complete": (
                not unregistered_cases
                and not duplicate_cases
                and len(seen_keys) + len(unassessed_question_cases)
                == sum(len(benchmark.question_cases) for benchmark in VIDEO_BENCHMARKS.values())
            ),
        },
    }


def _read_git_state() -> dict[str, Any]:
    def run_git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        except OSError:
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    status = run_git("status", "--porcelain")
    return {
        "runner_git_commit": run_git("rev-parse", "HEAD"),
        "runner_working_tree_clean": status == "",
    }


def _post_chat(
    base_url: str,
    case: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    request_body = json.dumps(
        {"video_id": case["video_id"], "question": case["question"]},
        ensure_ascii=False,
    ).encode("utf-8")

    status_code: int | None = None
    content_type: str | None = None
    raw_body = b""
    transport_error = None
    request_attempts = 0
    rate_limit_retry_wait_seconds = 0.0

    for attempt in range(AUDIT_RATE_LIMIT_RETRIES + 1):
        request_attempts += 1
        request = Request(
            f"{base_url.rstrip('/')}/chat",
            data=request_body,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/json",
            },
            method="POST",
        )
        response_headers = None
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                status_code = response.status
                content_type = response.headers.get("Content-Type")
                response_headers = response.headers
                raw_body = response.read()
        except HTTPError as error:
            status_code = error.code
            response_headers = error.headers
            content_type = (
                error.headers.get("Content-Type") if error.headers else None
            )
            raw_body = error.read()
        except (URLError, TimeoutError, OSError) as error:
            transport_error = str(error)
            break

        if status_code != 429 or attempt >= AUDIT_RATE_LIMIT_RETRIES:
            break

        retry_after: float | None = None
        if response_headers is not None:
            try:
                raw_retry_after = response_headers.get("Retry-After")
                if raw_retry_after is not None:
                    candidate = float(raw_retry_after)
                    if math.isfinite(candidate) and candidate >= 0:
                        retry_after = candidate
            except (AttributeError, TypeError, ValueError):
                retry_after = None

        if retry_after is None:
            retry_after = AUDIT_RATE_LIMIT_FALLBACK_WAIT_SECONDS

        wait_seconds = retry_after + AUDIT_RATE_LIMIT_BUFFER_SECONDS
        rate_limit_retry_wait_seconds += wait_seconds
        time.sleep(wait_seconds)

    payload: dict[str, Any] | None = None
    parse_error = None
    if raw_body:
        try:
            payload = decode_json_response(raw_body)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            parse_error = str(error)

    success = (
        status_code is not None
        and 200 <= status_code < 300
        and isinstance(payload, dict)
        and isinstance(payload.get("answer"), str)
    )

    result: dict[str, Any] = {
        "case_id": case["id"],
        "video_id": case["video_id"],
        "question": case["question"],
        "status": "success" if success else "error",
        "http_status": status_code,
        "content_type": content_type,
        "response_body_sha256": hashlib.sha256(raw_body).hexdigest() if raw_body else None,
        "transport_error": transport_error,
        "response_parse_error": parse_error,
        "request_attempts": request_attempts,
        "rate_limit_retry_count": max(0, request_attempts - 1),
        "rate_limit_retry_wait_seconds": round(rate_limit_retry_wait_seconds, 3),
        "response": payload,
    }
    if isinstance(payload, dict) and isinstance(payload.get("answer"), str):
        result["audit"] = analyze_response(case, payload)
    elif raw_body and payload is None:
        result["unparsed_response_body"] = raw_body.decode("utf-8", errors="replace")
    return result


def run_audit(
    *,
    base_url: str,
    output_path: Path,
    timeout_seconds: float,
    cases: tuple[dict[str, Any], ...] = AUDIT_CASES,
    report_id: str = REPORT_ID,
    expected_configuration: dict[str, Any] | None = None,
    require_full_pilot_coverage: bool = False,
    require_full_registry_coverage: bool = False,
) -> dict[str, Any]:
    results = [
        _post_chat(base_url, case, timeout_seconds)
        for case in cases
    ]

    successful_audits = [
        item.get("audit")
        for item in results
        if item.get("status") == "success" and isinstance(item.get("audit"), dict)
    ]
    configs = [
        audit.get("effective_retrieval_config")
        for audit in successful_audits
        if isinstance(audit.get("effective_retrieval_config"), dict)
    ]
    effective_config = configs[0] if configs else None
    # Require one effective config for every attempted case. A subset of
    # matching configurations must not masquerade as cross-case consistency.
    configs_consistent = (
        len(configs) == len(cases)
        and bool(configs)
        and all(config == configs[0] for config in configs)
    )

    config_matches_expected: bool | None = None
    if expected_configuration is not None:
        config_matches_expected = (
            isinstance(effective_config, dict)
            and all(
                effective_config.get(key) == value
                for key, value in expected_configuration.items()
            )
        )

    cases_succeeded = sum(item.get("status") == "success" for item in results)
    cases_failed = sum(item.get("status") != "success" for item in results)
    quality_failures = [
        {
            "case_id": item.get("case_id"),
            "failed_checks": item.get("audit", {}).get("quality", {}).get("failed_checks", []),
        }
        for item in results
        if item.get("status") != "success"
        or not item.get("audit", {}).get("quality", {}).get("passed", False)
    ]
    if not configs_consistent:
        quality_failures.append({
            "case_id": "ALL",
            "failed_checks": ["effective_server_configuration_consistency"],
        })
    if config_matches_expected is False:
        quality_failures.append({
            "case_id": "ALL",
            "failed_checks": ["effective_server_configuration_matches_expected"],
        })

    coverage_manifest = build_coverage_manifest(cases)
    coverage_passed = (
        coverage_manifest["all_registered_videos_have_answerable_and_unanswerable_live_cases"]
        and coverage_manifest["unregistered_live_cases"] == []
        and coverage_manifest["duplicate_live_cases"] == []
        and coverage_manifest["question_case_accounting"]["accounting_complete"]
    )
    registry_accounting = coverage_manifest["question_case_accounting"]
    full_registry_coverage_passed = (
        coverage_manifest["unregistered_live_cases"] == []
        and coverage_manifest["duplicate_live_cases"] == []
        and registry_accounting["accounting_complete"]
        and registry_accounting["live_audited_count"]
        == coverage_manifest["registered_question_case_count"]
        and registry_accounting["not_live_audited_count"] == 0
    )
    if require_full_pilot_coverage and not coverage_passed:
        quality_failures.append({
            "case_id": "ALL",
            "failed_checks": ["representative_five_video_live_coverage"],
        })
    if require_full_registry_coverage and not full_registry_coverage_passed:
        quality_failures.append({
            "case_id": "ALL",
            "failed_checks": ["all_registered_question_cases_live_audited"],
        })

    quality_passed = (
        cases_succeeded == len(cases)
        and cases_failed == 0
        and configs_consistent
        and config_matches_expected is not False
        and (coverage_passed or not require_full_pilot_coverage)
        and (full_registry_coverage_passed or not require_full_registry_coverage)
        and not quality_failures
    )

    report = {
        "report_schema_version": "1.2.0",
        "report_id": report_id,
        "generated_at": datetime.now().astimezone().isoformat(),
        "backend_base_url": base_url.rstrip("/"),
        **_read_git_state(),
        "live_suite_case_ids": [str(case.get("id")) for case in cases],
        "expected_server_configuration": expected_configuration,
        "effective_server_configuration": effective_config,
        "effective_server_configuration_consistent_across_cases": configs_consistent,
        "effective_server_configuration_matches_expected": config_matches_expected,
        "coverage_manifest": coverage_manifest,
        "full_registry_coverage_passed": full_registry_coverage_passed,
        "cases_attempted": len(cases),
        "cases_succeeded": cases_succeeded,
        "cases_failed": cases_failed,
        "quality_passed": quality_passed,
        "quality_failures": quality_failures,
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report

def select_audit_cases(
    cases: tuple[dict[str, Any], ...],
    requested_ids: list[str],
) -> tuple[dict[str, Any], ...]:
    """Select an ordered, unique subset of suite cases by exact case ID."""
    available = {str(case.get("id")): case for case in cases}
    duplicates = sorted({
        case_id for case_id in requested_ids
        if requested_ids.count(case_id) > 1
    })
    if duplicates:
        raise ValueError(
            "Duplicate case IDs are not allowed: " + ", ".join(duplicates)
        )

    unknown = [case_id for case_id in requested_ids if case_id not in available]
    if unknown:
        raise ValueError(
            "Unknown case IDs for the selected suite: " + ", ".join(unknown)
        )
    if not requested_ids:
        raise ValueError("At least one case ID must be selected.")

    return tuple(available[case_id] for case_id in requested_ids)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the live answer/source audit against a local FastAPI backend."
    )
    parser.add_argument(
        "--suite",
        choices=("phase10_10", "phase10_11", "phase11_0"),
        default="phase10_11",
        help=(
            "Use the three-case closure audit, frozen 16-case pilot suite, "
            "or Phase 11.0 full 43-question registry suite."
        ),
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL of the running local FastAPI service.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON report path. A suite-specific safe default is used when omitted.",
    )
    parser.add_argument(
        "--case-ids",
        default=None,
        help=(
            "Optional comma-separated subset of case IDs from the selected suite, "
            "for a small controlled model-comparison pilot. When used, --output "
            "is required and the full-suite coverage gate is disabled."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--expected-namespace",
        default="phase10-10-acceptance-minilm-20261023",
        help="Expected isolated index namespace for the Phase 10.11 live suite.",
    )
    parser.add_argument(
        "--expected-embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Expected embedding model for the Phase 10.11 live suite.",
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero.")
    if args.case_ids is not None and args.output is None:
        parser.error("--output is required when --case-ids is used.")

    requested_case_ids = None
    if args.case_ids is not None:
        requested_case_ids = [
            item.strip().upper()
            for item in args.case_ids.split(",")
            if item.strip()
        ]
        if not requested_case_ids:
            parser.error("--case-ids must contain at least one case ID.")

    expected_configuration = None
    require_full_pilot_coverage = False
    require_full_registry_coverage = False

    if args.suite == "phase10_10":
        cases = AUDIT_CASES
        report_id = REPORT_ID
        output_path = args.output or DEFAULT_OUTPUT
    elif args.suite == "phase10_11":
        cases = PHASE10_11_AUDIT_CASES
        report_id = PHASE10_11_REPORT_ID
        output_path = args.output or PHASE10_11_DEFAULT_OUTPUT
        require_full_pilot_coverage = True
    else:
        cases = PHASE11_0_AUDIT_CASES
        report_id = PHASE11_0_REPORT_ID
        output_path = args.output or PHASE11_0_DEFAULT_OUTPUT
        require_full_pilot_coverage = True
        require_full_registry_coverage = True

    if args.suite in {"phase10_11", "phase11_0"}:
        expected_configuration = {
            "embedding_model": args.expected_embedding_model,
            "index_namespace": args.expected_namespace,
            "citation_policy": "prompt_included_chunks",
            "top_k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.7,
            "max_distance": 1.3,
            "context_expansion_chunks": 1,
            "context_max_chunks": 12,
            "dense_semantic_k": 24,
            "dense_semantic_fetch_k": 32,
            "dense_anchor_limit": 12,
            "dense_anchor_min_chunk_gap": 2,
            "dense_context_max_chunks": 16,
            "dense_facet_rerank_weight": 0.5,
            "rerank_enabled": False,
            "rerank_model": "cross-encoder/ms-marco-MiniLM-L6-v2",
            "rerank_candidate_k": 24,
            "rerank_batch_size": 16,
            "rerank_max_length": 512,
        }

    if requested_case_ids is not None:
        try:
            cases = select_audit_cases(cases, requested_case_ids)
        except ValueError as error:
            parser.error(str(error))
        report_id = f"{report_id}_subset"
        require_full_pilot_coverage = False
        require_full_registry_coverage = False

    report = run_audit(
        base_url=args.base_url,
        output_path=output_path,
        timeout_seconds=args.timeout,
        cases=cases,
        report_id=report_id,
        expected_configuration=expected_configuration,
        require_full_pilot_coverage=require_full_pilot_coverage,
        require_full_registry_coverage=require_full_registry_coverage,
    )
    print(
        json.dumps(
            {
                "report_path": str(output_path),
                "report_id": report["report_id"],
                "report_schema_version": report["report_schema_version"],
                "cases_attempted": report["cases_attempted"],
                "cases_succeeded": report["cases_succeeded"],
                "cases_failed": report["cases_failed"],
                "quality_passed": report["quality_passed"],
                "full_registry_coverage_passed": report["full_registry_coverage_passed"],
                "quality_failures": report["quality_failures"],
                "effective_server_configuration": report["effective_server_configuration"],
                "effective_server_configuration_consistent_across_cases": (
                    report["effective_server_configuration_consistent_across_cases"]
                ),
                "effective_server_configuration_matches_expected": (
                    report["effective_server_configuration_matches_expected"]
                ),
                "coverage_manifest": report["coverage_manifest"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["quality_passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
