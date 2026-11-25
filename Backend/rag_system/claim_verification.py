"""Generic, fail-closed assessment of generated claims against transcript passages."""
from __future__ import annotations

import logging
import math
import re
from typing import Any, Sequence

from .evidence_assessment import AssessmentRunStatus, EvidencePassage, assess_evidence_support
from .evidence_contracts import EvidenceAspect, EvidenceReference, QuestionAnalysis, QuestionIntent
from .overview_summarization import OVERVIEW_SAMPLING_DISCLOSURE

MAX_CLAIMS = 8
MAX_PASSAGES = 12
MAX_PASSAGE_CHARS = 640
MAX_INPUT_CHARS = 8_000
logger = logging.getLogger(__name__)
_STOP = set(
    "a an and are as at be been but by can did do does for from had has have "
    "he her his how i if in into is it its may might not of on or our she so "
    "than that the their them then there these they this those to was we were "
    "what when where which while who why will with would you your".split()
)


def _meta(document: Any) -> dict:
    value = getattr(document, "metadata", {})
    return value if isinstance(value, dict) else {}


def _terms(text: str) -> set[str]:
    return {
        item.casefold()
        for item in re.findall(r"[^\W_]+", text, flags=re.UNICODE)
        if len(item) > 1 and item.casefold() not in _STOP
    }


def _answer_disclosures(answer: str) -> list[str]:
    """Return system-generated notices that must not be judged as transcript claims."""
    if not isinstance(answer, str):
        return []
    expected = OVERVIEW_SAMPLING_DISCLOSURE.casefold()
    return list(dict.fromkeys(
        " ".join(line.split())
        for line in answer.splitlines()
        if " ".join(line.split()).casefold() == expected
    ))


def _without_answer_disclosures(answer: str) -> str:
    expected = OVERVIEW_SAMPLING_DISCLOSURE.casefold()
    return "\n".join(
        line for line in answer.splitlines()
        if " ".join(line.split()).casefold() != expected
    ).strip()


def _append_answer_disclosures(answer: str, disclosures: Sequence[str]) -> str:
    if not disclosures:
        return answer
    return answer.rstrip() + "\n\n" + "\n".join(disclosures)


def split_answer_claims(
    answer: str,
    *,
    max_claims: int | None = MAX_CLAIMS,
) -> list[str]:
    """Split paragraph and bullet answers into a bounded sequence of claims."""
    if not isinstance(answer, str) or not answer.strip():
        return []
    text = _without_answer_disclosures(answer)
    text = re.sub(r"(?im)^\s*Relevant timestamps?\s*:.*$", "", text).strip()
    claims = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^(?:[-*•]\s+|\d{1,2}[.)]\s+)", "", line).strip()
        if not line or (line.endswith(":") and len(line) < 120):
            continue
        claims.extend(
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])", line)
            if len(part.strip()) >= 8
        )
    unique = list(dict.fromkeys(claims))
    if max_claims is None:
        return unique
    if isinstance(max_claims, bool) or not isinstance(max_claims, int) or max_claims < 0:
        raise ValueError("max_claims must be a non-negative integer or None.")
    return unique[:max_claims]


def should_verify_answer(answer: str, question: str) -> bool:
    """Audit compound or fact-sensitive output without adding calls to trivial replies."""
    if not isinstance(answer, str) or not answer.strip():
        return False
    if len(split_answer_claims(answer)) >= 2 or len(answer) >= 240:
        return True
    if re.search(
        r"(?:₹|\$|€|£|\b(?:INR|USD|EUR|GBP)\b|\b\d[\d,.]*\s*(?:%|percent|million|billion|"
        r"thousand|crore|lakh|years?|months?|weeks?|days?|hours?|minutes?)\b|\b(?:19|20)\d{2}\b)",
        answer,
        re.IGNORECASE,
    ):
        return True
    if re.search(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,3}\b", answer):
        return True
    return bool(re.search(
        r"\b(?:study|studies|source|sources|research|authentic|genuine|statistics?|"
        r"evidence|released|release date|who is|which person|which company)\b",
        question,
        re.IGNORECASE,
    ))


def _excerpt(
    text: str,
    terms: set[str],
    *,
    max_chars: int = MAX_PASSAGE_CHARS,
) -> str:
    """Choose a bounded, claim-relevant excerpt from text already shown to generation."""
    content = " ".join(text.split())
    if len(content) <= max_chars:
        return content
    step = max(40, max_chars // 3)
    windows = [
        content[start : start + max_chars]
        for start in range(0, max(1, len(content) - max_chars + 1), step)
    ]
    if not windows:
        return content[:max_chars]
    return max(windows, key=lambda value: len(terms & _terms(value)))


def _visible_excerpt(text: str, prompt_context: str, terms: set[str]) -> str | None:
    """Return an exact normalized excerpt that was actually shown to generation."""
    source = " ".join(text.split())
    visible = " ".join(prompt_context.split())
    if not source or not visible:
        return None
    if source in visible:
        return _excerpt(source, terms)
    for length in (MAX_PASSAGE_CHARS, 400, 320, 240, 160, 100, 64):
        if len(source) < length:
            continue
        step = max(24, length // 4)
        windows = [
            source[start : start + length]
            for start in range(0, max(1, len(source) - length + 1), step)
        ]
        matches = [window for window in windows if window in visible]
        if matches:
            return max(matches, key=lambda value: len(terms & _terms(value)))
    return None


def _select_documents(
    results: Sequence[Any],
    included_ids: Sequence[Any] | None,
    claims: Sequence[str],
    prompt_visible_context: str | None,
) -> list[tuple[Any, str]]:
    if not isinstance(included_ids, (list, tuple, set)):
        return []
    allowed = {str(value) for value in included_ids if value is not None}
    if not allowed or not isinstance(prompt_visible_context, str) or not prompt_visible_context.strip():
        return []
    candidates, seen = [], set()
    for rank, pair in enumerate(results):
        if not isinstance(pair, (tuple, list)) or not pair:
            continue
        document = pair[0]
        meta = _meta(document)
        chunk_id = meta.get("chunk_id")
        text = getattr(document, "page_content", None)
        if chunk_id is None or not isinstance(text, str) or not text.strip():
            continue
        if str(chunk_id) not in allowed:
            continue
        visible_text = _visible_excerpt(text, prompt_visible_context, _terms(" ".join(claims)))
        if not visible_text:
            continue
        key = (str(meta.get("video_id", "")), str(chunk_id))
        if key in seen:
            continue
        seen.add(key)
        content_terms = _terms(text)
        scores = [len(_terms(claim) & content_terms) for claim in claims]
        candidates.append({
            "document": document,
            "visible_text": visible_text,
            "rank": rank,
            "scores": scores,
            "score": max(scores, default=0),
        })
    if len(candidates) <= MAX_PASSAGES:
        return [(item["document"], item["visible_text"]) for item in candidates]

    selected, selected_ids = [], set()
    for claim_index in range(len(claims)):
        best = max(
            candidates,
            key=lambda item: (item["scores"][claim_index], item["score"], -item["rank"]),
        )
        chunk_id = str(_meta(best["document"]).get("chunk_id"))
        if best["scores"][claim_index] and chunk_id not in selected_ids:
            selected.append((best["document"], best["visible_text"]))
            selected_ids.add(chunk_id)
            if len(selected) == MAX_PASSAGES:
                return selected
    for item in sorted(candidates, key=lambda value: (value["score"], -value["rank"]), reverse=True):
        chunk_id = str(_meta(item["document"]).get("chunk_id"))
        if chunk_id not in selected_ids:
            selected.append((item["document"], item["visible_text"]))
            selected_ids.add(chunk_id)
            if len(selected) == MAX_PASSAGES:
                break
    return selected


def verify_answer_claims(
    *,
    answer: str,
    question: str,
    video_id: str,
    retrieved_results: Sequence[Any],
    included_chunk_ids: Sequence[Any] | None,
    prompt_visible_context: str | None,
    client: Any,
    fallback_answer: str,
) -> dict:
    """Check claims only against passages that were visible to answer generation.

    Provider/schema failures are reported as unavailable and do not masquerade
    as successful verification. A completed check removes unsupported claims.
    """
    disclosures = _answer_disclosures(answer)
    all_claims = split_answer_claims(answer, max_claims=None)
    claims = all_claims[:MAX_CLAIMS]
    truncated_claim_count = max(0, len(all_claims) - len(claims))
    if (
        not should_verify_answer(answer, question)
        or (
            isinstance(fallback_answer, str)
            and isinstance(answer, str)
            and answer.strip() == fallback_answer.strip()
        )
    ):
        return {
            "answer": answer, "performed": False, "status": "not_required",
            "claim_count": len(claims), "truncated_claim_count": truncated_claim_count, "supported_claim_count": 0,
            "partial_claim_count": 0, "removed_claim_count": 0,
            "supported_chunk_ids": [], "error_code": None,
        }

    documents = _select_documents(
        retrieved_results,
        included_chunk_ids,
        claims,
        prompt_visible_context,
    )
    if not claims or not documents:
        return {
            "answer": _append_answer_disclosures(fallback_answer, disclosures), "performed": True,
            "status": "no_prompt_visible_evidence", "claim_count": len(claims), "truncated_claim_count": truncated_claim_count,
            "supported_claim_count": 0, "partial_claim_count": 0,
            "removed_claim_count": len(claims), "supported_chunk_ids": [],
            "error_code": "no_prompt_visible_evidence",
        }

    terms = _terms(question + " " + answer)
    aspects = tuple(
        EvidenceAspect(
            aspect_id=f"claim_{index:02d}",
            description=claim[:900],
            required=True,
        )
        for index, claim in enumerate(claims, start=1)
    )
    analysis = QuestionAnalysis(
        intent=QuestionIntent.DIRECT_FACT,
        language="en",
        aspects=aspects,
        video_id=video_id,
        subject=question[:240],
        analysis_source="generated_answer_claim_audit",
    )

    # The assessor budget includes its fixed system instructions, the entire
    # claim list, passage metadata and transcript excerpts. A set of eight
    # otherwise valid passages can therefore exceed the hard 8,000-character
    # limit. Repack only the already prompt-visible excerpts into progressively
    # smaller exact windows; do not drop claims, admit hidden transcript text,
    # or raise the assessor's hard budget.
    excerpt_limits = tuple(dict.fromkeys((
        MAX_PASSAGE_CHARS,
        560,
        480,
        400,
        320,
        240,
        192,
        160,
        128,
        96,
        64,
    )))
    result = None
    selected_excerpt_limit = MAX_PASSAGE_CHARS
    assessment_compaction_attempts = 0
    for excerpt_limit in excerpt_limits:
        selected_excerpt_limit = excerpt_limit
        assessment_compaction_attempts += 1
        passages = []
        for index, (document, visible_text) in enumerate(documents, start=1):
            meta = _meta(document)
            chunk_id = meta.get("chunk_id")
            try:
                start, end = float(meta.get("start")), float(meta.get("end"))
                if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
                    start = end = None
            except (TypeError, ValueError, OverflowError):
                start = end = None
            passages.append(EvidencePassage(
                passage_id=f"A{index:02d}",
                text=_excerpt(visible_text, terms, max_chars=excerpt_limit),
                references=(EvidenceReference(
                    video_id=video_id,
                    chunk_id=str(chunk_id),
                    start_seconds=start,
                    end_seconds=end,
                ),),
            ))

        try:
            result = assess_evidence_support(
                analysis,
                passages,
                client,
                max_input_chars=MAX_INPUT_CHARS,
                max_passages=MAX_PASSAGES,
                max_aspects=MAX_CLAIMS,
                compact_prompt=True,
            )
        except Exception:
            logger.exception("Unexpected answer claim assessment failure.")
            return {
                "answer": answer,
                "performed": True,
                "status": AssessmentRunStatus.PROVIDER_ERROR.value,
                "claim_count": len(claims),
                "truncated_claim_count": truncated_claim_count,
                "supported_claim_count": 0,
                "partial_claim_count": 0,
                "removed_claim_count": 0,
                "supported_chunk_ids": [],
                "assessment_compaction_attempts": assessment_compaction_attempts,
                "assessment_passage_excerpt_limit": excerpt_limit,
                "error_code": "unexpected_assessment_error",
            }
        diagnostic = result.diagnostics
        if not (
            diagnostic.status is AssessmentRunStatus.BUDGET_EXCEEDED
            and diagnostic.error_code == "assessment_input_character_budget_exceeded"
        ):
            break

    diagnostic = result.diagnostics
    assessment_diagnostics = {
        "assessment_input_chars": getattr(diagnostic, "input_chars", None),
        "assessment_input_limit_chars": getattr(diagnostic, "input_limit_chars", MAX_INPUT_CHARS),
        "assessment_passage_count": getattr(diagnostic, "passage_count", len(documents)),
        "assessment_aspect_count": getattr(diagnostic, "aspect_count", len(claims)),
        "assessment_request_count": getattr(diagnostic, "request_count", None),
        "assessment_compaction_attempts": assessment_compaction_attempts,
        "assessment_passage_excerpt_limit": selected_excerpt_limit,
    }
    if diagnostic.status is not AssessmentRunStatus.COMPLETED:
        return {
            "answer": answer, "performed": True, "status": diagnostic.status.value,
            "claim_count": len(claims), "truncated_claim_count": truncated_claim_count,
            "supported_claim_count": 0, "partial_claim_count": 0, "removed_claim_count": 0,
            "supported_chunk_ids": [], "error_code": diagnostic.error_code,
            **assessment_diagnostics,
        }

    by_id = {item.aspect_id: item for item in result.assessments}
    kept, chunk_ids, claim_evidence = [], [], []
    supported = partial = removed = conflicted = 0
    for index, claim in enumerate(claims, start=1):
        item = by_id.get(f"claim_{index:02d}")
        if item is None:
            removed += 1
            claim_evidence.append({
                "claim_index": index,
                "status": "unknown",
                "supporting_chunk_ids": [],
            })
            continue
        references = (
            item.supporting_evidence
            if item.status.value in {"supported", "partial"}
            else ()
        )
        claim_chunk_ids = list(dict.fromkeys(
            str(reference.chunk_id)
            for reference in references
            if reference.video_id == video_id
        ))
        claim_evidence.append({
            "claim_index": index,
            "status": item.status.value,
            "supporting_chunk_ids": claim_chunk_ids,
        })
        if item.status.value == "supported":
            kept.append(claim)
            supported += 1
        elif item.status.value == "partial":
            kept.append("The transcript only partially supports this detail: " + claim)
            partial += 1
        else:
            removed += 1
            if item.status.value in {"conflicted", "contradicted"}:
                conflicted += 1
        chunk_ids.extend(claim_chunk_ids)

    chunk_ids = list(dict.fromkeys(chunk_ids))
    if not kept:
        verified = fallback_answer
    else:
        verified = "\n".join(
            f"{index}. {claim}" for index, claim in enumerate(kept, start=1)
        )
        if removed:
            verified += "\n\nI left out details that the available transcript did not clearly establish."
        if conflicted:
            verified += "\nThe transcript contains conflicting information about at least one omitted point."
    if truncated_claim_count:
        verified += (
            "\n\nAdditional details were omitted because the answer exceeded "
            f"the {MAX_CLAIMS}-claim verification limit."
        )
    verified = _append_answer_disclosures(verified, disclosures)

    return {
        "answer": verified, "performed": True, "status": "completed",
        "claim_count": len(claims), "truncated_claim_count": truncated_claim_count,
        "supported_claim_count": supported,
        "partial_claim_count": partial, "removed_claim_count": removed,
        "supported_chunk_ids": chunk_ids,
        "claim_evidence": claim_evidence,
        "error_code": None,
        **assessment_diagnostics,
    }
