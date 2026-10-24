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
import re
import subprocess
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
            {"name": "three_ways", "pattern": r"\b(?:three|3)\b"},
            {"name": "self_expression", "pattern": r"\bexpress\w*\b"},
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
                "pattern": r"\bch(?:a|o|u)?v?ri\b|चवरी|चावरी|चवरी",
            },
            {"name": "tax_concept", "pattern": r"\b(?:tax|taxes|levy)\b|कर|टैक्स"},
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
            {"name": "perception_or_compass", "pattern": r"\b(?:compass|perception|guide|orient\w*|direction)\b"},
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


def _answer_timestamp_ranges(answer: str) -> list[str]:
    match = re.search(
        r"Relevant timestamps?:\s*([^\r\n]+)",
        answer,
        flags=re.IGNORECASE,
    )
    if not match:
        return []
    return [item.group(0).strip() for item in _TIMESTAMP_RANGE_PATTERN.finditer(match.group(1))]


def _substantive_answer(answer: str) -> str:
    """Remove the application timestamp footer before checking answer content."""
    return re.sub(
        r"(?im)^[ \t]*Relevant timestamps?[ \t]*:[^\r\n]*(?:\r?\n|$)",
        "",
        answer,
    ).strip()


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
    r"(?:لا|لم).{0,40}(?:يذكر|تذكر|يتناول|تحتوي|توجد|يوجد|تتوفر|أستطيع|يمكن)|"
    r"غير مذكور|غير موجود|لا توجد معلومات|لا تتوفر معلومات|",
    re.IGNORECASE,
)


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
            bool(substantive) and substantive != (
                "I couldn't find enough information about that in the video. It may not be covered by this video."
            ),
            "The answer must contain a substantive response rather than the fallback.",
        )
    ]
    if normalized_case_id == "Q07":
        checks.extend(_q07_answer_checks(answer))
    elif normalized_case_id in {"AR03", "AR08"}:
        checks.extend(_bodily_sound_checks(normalized_case_id, answer))

    for requirement in case.get("answer_requirements", []):
        if not isinstance(requirement, dict):
            continue
        name = str(requirement.get("name", "answer_requirement"))
        pattern = requirement.get("pattern")
        passed = isinstance(pattern, str) and bool(
            re.search(pattern, substantive, flags=re.IGNORECASE)
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
    normalized_summary_ranges = {
        normalized
        for item in timestamp_ranges
        if (normalized := _normalize_range(item)) is not None
    }
    summary_source_ids: list[str] = []
    for source in source_ranges:
        normalized = (
            _normalize_range(source["display_range"])
            if isinstance(source.get("display_range"), str)
            else None
        )
        if normalized in normalized_summary_ranges:
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
                (normalized := _normalize_range(item)) is not None
                and normalized in {
                    _normalize_range(source["display_range"])
                    for source in source_ranges
                    if isinstance(source.get("display_range"), str)
                }
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
                "Every displayed timestamp range exactly matches a returned source interval."
                if timestamp_alignment_ok
                else "The timestamp summary is missing or contains an interval not present in returned source metadata."
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
            "caption_track_kind": benchmark.caption_track_kind,
            "transcript_source_kind": benchmark.transcript_source_kind,
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
    request = Request(
        f"{base_url.rstrip('/')}/chat",
        data=request_body,
        headers={"Content-Type": "application/json; charset=utf-8", "Accept": "application/json"},
        method="POST",
    )

    status_code: int | None = None
    content_type: str | None = None
    raw_body = b""
    transport_error = None
    try:
        with urlopen(request, timeout=timeout_seconds) as result:
            status_code = result.status
            content_type = result.headers.get("Content-Type")
            raw_body = result.read()
    except HTTPError as error:
        status_code = error.code
        content_type = error.headers.get("Content-Type") if error.headers else None
        raw_body = error.read()
    except (URLError, TimeoutError, OSError) as error:
        transport_error = str(error)

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
    if require_full_pilot_coverage and not coverage_passed:
        quality_failures.append({
            "case_id": "ALL",
            "failed_checks": ["representative_five_video_live_coverage"],
        })

    quality_passed = (
        cases_succeeded == len(cases)
        and cases_failed == 0
        and configs_consistent
        and config_matches_expected is not False
        and (coverage_passed or not require_full_pilot_coverage)
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

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture UTF-8 /chat answers and audit evidence-to-source coverage."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL of the running local FastAPI service.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"JSON report path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=90.0,
        help="Per-request timeout in seconds.",
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero.")

    report = run_audit(
        base_url=args.base_url,
        output_path=args.output,
        timeout_seconds=args.timeout,
    )
    print(
        json.dumps(
            {
                "report_path": str(args.output),
                "cases_attempted": report["cases_attempted"],
                "cases_succeeded": report["cases_succeeded"],
                "cases_failed": report["cases_failed"],
                "quality_passed": report["quality_passed"],
                "quality_failures": report["quality_failures"],
                "effective_server_configuration": report["effective_server_configuration"],
                "effective_server_configuration_consistent_across_cases": (
                    report["effective_server_configuration_consistent_across_cases"]
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["quality_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
