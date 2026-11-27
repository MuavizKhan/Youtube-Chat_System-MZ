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
MAX_TOTAL_CLAIMS = 13
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
    claim_evidence_ids: Sequence[Sequence[Any]] | None = None,
) -> list[tuple[Any, str]]:
    """Select prompt-visible sources while preserving point-to-source links."""
    if not isinstance(included_ids, (list, tuple, set)):
        return []
    allowed = {str(value) for value in included_ids if value is not None}
    if not allowed or not isinstance(prompt_visible_context, str) or not prompt_visible_context.strip():
        return []

    claim_ids: list[list[str]] = []
    for index in range(len(claims)):
        values = (
            claim_evidence_ids[index]
            if isinstance(claim_evidence_ids, (list, tuple))
            and index < len(claim_evidence_ids)
            and isinstance(claim_evidence_ids[index], (list, tuple, set))
            else ()
        )
        claim_ids.append(list(dict.fromkeys(
            str(value) for value in values if value is not None
        )))

    claim_indexes_by_chunk: dict[str, list[int]] = {}
    for claim_index, chunk_ids in enumerate(claim_ids):
        for chunk_id in chunk_ids:
            claim_indexes_by_chunk.setdefault(chunk_id, []).append(claim_index)

    all_claim_terms = _terms(" ".join(claims))
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
        chunk_key = str(chunk_id)
        if chunk_key not in allowed:
            continue
        key = (str(meta.get("video_id", "")), chunk_key)
        if key in seen:
            continue
        seen.add(key)

        direct_claim_indexes = claim_indexes_by_chunk.get(chunk_key, [])
        excerpt_terms = _terms(" ".join(
            claims[index] for index in direct_claim_indexes
        )) if direct_claim_indexes else all_claim_terms
        visible_text = _visible_excerpt(
            text,
            prompt_visible_context,
            excerpt_terms or all_claim_terms,
        )
        if not visible_text:
            continue
        content_terms = _terms(text)
        scores = [
            len(_terms(claim) & content_terms)
            for claim in claims
        ]
        candidates.append({
            "document": document,
            "visible_text": visible_text,
            "rank": rank,
            "scores": scores,
        })

    candidate_by_id = {
        str(_meta(item["document"]).get("chunk_id")): item
        for item in candidates
    }

    if claim_evidence_ids is not None and claim_ids:
        direct_matches = [
            [
                chunk_id
                for chunk_id in chunk_ids
                if chunk_id in candidate_by_id
            ]
            for chunk_ids in claim_ids
        ]
        if all(direct_matches):
            selected, selected_ids = [], set()

            def add_direct_candidate(chunk_id: str) -> None:
                if chunk_id in selected_ids:
                    return
                candidate = candidate_by_id[chunk_id]
                selected.append((candidate["document"], candidate["visible_text"]))
                selected_ids.add(chunk_id)

            # Preserve at least one exact passage per point. Extra citations are
            # included round-robin within the fixed passage ceiling.
            for matches in direct_matches:
                add_direct_candidate(matches[0])
                if len(selected) >= MAX_PASSAGES:
                    return selected

            max_direct_ids = max((len(values) for values in direct_matches), default=0)
            for source_offset in range(1, max_direct_ids):
                for matches in direct_matches:
                    if source_offset < len(matches):
                        add_direct_candidate(matches[source_offset])
                        if len(selected) >= MAX_PASSAGES:
                            return selected
            return selected

    if len(candidates) <= MAX_PASSAGES:
        return [(item["document"], item["visible_text"]) for item in candidates]

    selected, selected_ids = [], set()

    def add_candidate(chunk_id: str) -> bool:
        candidate = candidate_by_id.get(chunk_id)
        if candidate is None or chunk_id in selected_ids:
            return False
        selected.append((candidate["document"], candidate["visible_text"]))
        selected_ids.add(chunk_id)
        return True

    # Reserve at least one directly cited, prompt-visible source per claim.
    for chunk_ids in claim_ids:
        for chunk_id in chunk_ids:
            if add_candidate(chunk_id):
                break
        if len(selected) >= MAX_PASSAGES:
            return selected

    # Then include further cited sources fairly across claims.
    max_direct_ids = max((len(values) for values in claim_ids), default=0)
    for source_offset in range(1, max_direct_ids):
        for chunk_ids in claim_ids:
            if source_offset < len(chunk_ids):
                add_candidate(chunk_ids[source_offset])
                if len(selected) >= MAX_PASSAGES:
                    return selected

    # Claims without a usable direct source receive the best lexical fallback.
    for claim_index in range(len(claims)):
        selected_supports_claim = any(
            _terms(claims[claim_index]) & _terms(
                getattr(document, "page_content", "")
            )
            for document, _visible_text in selected
        )
        if selected_supports_claim:
            continue
        ranked = sorted(
            candidates,
            key=lambda item: (
                item["scores"][claim_index],
                max(item["scores"], default=0),
                -item["rank"],
            ),
            reverse=True,
        )
        for item in ranked:
            chunk_id = str(_meta(item["document"]).get("chunk_id"))
            if item["scores"][claim_index] > 0 and chunk_id not in selected_ids:
                add_candidate(chunk_id)
                break
        if len(selected) >= MAX_PASSAGES:
            return selected

    for item in sorted(
        candidates,
        key=lambda value: (max(value["scores"], default=0), -value["rank"]),
        reverse=True,
    ):
        chunk_id = str(_meta(item["document"]).get("chunk_id"))
        add_candidate(chunk_id)
        if len(selected) >= MAX_PASSAGES:
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
    structured_claims: Sequence[dict] | None = None,
) -> dict:
    """Verify claims against exact prompt-visible sources.

    Overview points can carry their own source chunk IDs. Other answers use
    bounded lexical source selection. Up to ten claims are assessed in batches
    of at most eight aspects per provider request.
    """
    disclosures = _answer_disclosures(answer)
    claim_evidence_ids: list[list[str]] | None = None

    if isinstance(structured_claims, (list, tuple)):
        normalized_claims = []
        normalized_evidence_ids = []
        for item in structured_claims:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            evidence_ids = item.get("evidence_chunk_ids", [])
            if not isinstance(text, str) or not text.strip():
                continue
            if not isinstance(evidence_ids, (list, tuple, set)):
                evidence_ids = []
            normalized_claims.append(" ".join(text.split()))
            normalized_evidence_ids.append(list(dict.fromkeys(
                str(value) for value in evidence_ids if value is not None
            )))
        all_claims = normalized_claims
        claim_evidence_ids = normalized_evidence_ids
        structured_mode = True
    else:
        all_claims = split_answer_claims(answer, max_claims=None)
        structured_mode = False

    claims = all_claims[:MAX_TOTAL_CLAIMS]
    if claim_evidence_ids is not None:
        claim_evidence_ids = claim_evidence_ids[:MAX_TOTAL_CLAIMS]
    truncated_claim_count = max(0, len(all_claims) - len(claims))

    if (
        (not structured_mode and not should_verify_answer(answer, question))
        or (
            isinstance(fallback_answer, str)
            and isinstance(answer, str)
            and answer.strip() == fallback_answer.strip()
        )
    ):
        return {
            "answer": answer,
            "performed": False,
            "status": "not_required",
            "claim_count": len(claims),
            "truncated_claim_count": truncated_claim_count,
            "supported_claim_count": 0,
            "partial_claim_count": 0,
            "removed_claim_count": 0,
            "supported_chunk_ids": [],
            "error_code": None,
        }

    all_claim_evidence: list[dict] = []
    all_supported_chunk_ids: list[str] = []
    batch_diagnostics = []

    for batch_start in range(0, len(claims), MAX_CLAIMS):
        batch_claims = claims[batch_start:batch_start + MAX_CLAIMS]
        batch_evidence_ids = (
            claim_evidence_ids[batch_start:batch_start + MAX_CLAIMS]
            if claim_evidence_ids is not None
            else None
        )
        documents = _select_documents(
            retrieved_results,
            included_chunk_ids,
            batch_claims,
            prompt_visible_context,
            claim_evidence_ids=batch_evidence_ids,
        )
        if not batch_claims or not documents:
            return {
                "answer": _append_answer_disclosures(fallback_answer, disclosures),
                "performed": True,
                "status": "no_prompt_visible_evidence",
                "claim_count": len(claims),
                "truncated_claim_count": truncated_claim_count,
                "supported_claim_count": 0,
                "partial_claim_count": 0,
                "removed_claim_count": len(claims),
                "supported_chunk_ids": [],
                "assessment_selected_chunk_ids": [],
                "error_code": "no_prompt_visible_evidence",
            }

        terms = _terms(question + " " + " ".join(batch_claims))
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
        compaction_attempts = 0
        selected_documents_for_batch = []
        for excerpt_limit in excerpt_limits:
            selected_excerpt_limit = excerpt_limit
            compaction_attempts += 1
            passages = []
            selected_documents_for_batch = []
            for index, (document, visible_text) in enumerate(documents, start=1):
                meta = _meta(document)
                chunk_id = str(meta.get("chunk_id"))
                direct_indexes = [
                    claim_index
                    for claim_index, ids in enumerate(batch_evidence_ids or [])
                    if chunk_id in ids
                ]
                passage_terms = _terms(" ".join(
                    batch_claims[claim_index] for claim_index in direct_indexes
                )) if direct_indexes else terms
                passage_text = _excerpt(
                    visible_text,
                    passage_terms or terms,
                    max_chars=excerpt_limit,
                )
                try:
                    start, end = float(meta.get("start")), float(meta.get("end"))
                    if (
                        not math.isfinite(start)
                        or not math.isfinite(end)
                        or start < 0
                        or end <= start
                    ):
                        start = end = None
                except (TypeError, ValueError, OverflowError):
                    start = end = None
                passages.append(EvidencePassage(
                    passage_id=f"A{index:02d}",
                    text=passage_text,
                    references=(EvidenceReference(
                        video_id=video_id,
                        chunk_id=chunk_id,
                        start_seconds=start,
                        end_seconds=end,
                    ),),
                ))
                selected_documents_for_batch.append(document)

            aspects = tuple(
                EvidenceAspect(
                    aspect_id=f"claim_{index:02d}",
                    description=claim[:900],
                    required=True,
                )
                for index, claim in enumerate(batch_claims, start=1)
            )
            analysis = QuestionAnalysis(
                intent=QuestionIntent.DIRECT_FACT,
                language="en",
                aspects=aspects,
                video_id=video_id,
                subject=question[:240],
                analysis_source="generated_answer_claim_audit",
            )

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
                    "assessment_batch_count": len(batch_diagnostics) + 1,
                    "assessment_compaction_attempts": compaction_attempts,
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
        if diagnostic.status is not AssessmentRunStatus.COMPLETED:
            return {
                "answer": answer,
                "performed": True,
                "status": diagnostic.status.value,
                "claim_count": len(claims),
                "truncated_claim_count": truncated_claim_count,
                "supported_claim_count": 0,
                "partial_claim_count": 0,
                "removed_claim_count": 0,
                "supported_chunk_ids": [],
                "error_code": diagnostic.error_code,
                "assessment_batch_count": len(batch_diagnostics) + 1,
                "assessment_request_count": (
                    sum(item["request_count"] for item in batch_diagnostics)
                    + (getattr(diagnostic, "request_count", 0) or 0)
                ),
                "assessment_elapsed_ms": (
                    sum(item["elapsed_ms"] for item in batch_diagnostics)
                    + (getattr(diagnostic, "elapsed_ms", 0.0) or 0.0)
                ),
                "assessment_input_chars": getattr(diagnostic, "input_chars", 0),
                "assessment_input_chars_total": (
                    sum(item["input_chars"] for item in batch_diagnostics)
                    + (getattr(diagnostic, "input_chars", 0) or 0)
                ),
                "assessment_input_limit_chars": getattr(
                    diagnostic,
                    "input_limit_chars",
                    MAX_INPUT_CHARS,
                ),
                "assessment_passage_count": getattr(
                    diagnostic,
                    "passage_count",
                    len(documents),
                ),
                "assessment_aspect_count": getattr(
                    diagnostic,
                    "aspect_count",
                    len(batch_claims),
                ),
                "assessment_input_tokens": (
                    sum(item["input_tokens"] for item in batch_diagnostics)
                    + (getattr(diagnostic, "input_tokens", 0) or 0)
                ),
                "assessment_output_tokens": (
                    sum(item["output_tokens"] for item in batch_diagnostics)
                    + (getattr(diagnostic, "output_tokens", 0) or 0)
                ),
                "assessment_output_chars": (
                    sum(item["output_chars"] for item in batch_diagnostics)
                    + (getattr(diagnostic, "output_chars", 0) or 0)
                ),
                "assessment_compaction_attempts": (
                    sum(item["compaction_attempts"] for item in batch_diagnostics)
                    + compaction_attempts
                ),
                "assessment_passage_excerpt_limit": selected_excerpt_limit,
                "assessment_finish_reason": getattr(diagnostic, "finish_reason", None),
                "assessment_validation_error_code": getattr(
                    diagnostic,
                    "validation_error_code",
                    None,
                ),
                "assessment_selected_chunk_ids": list(dict.fromkeys(
                    [
                        chunk_id
                        for item in batch_diagnostics
                        for chunk_id in item["selected_chunk_ids"]
                    ]
                    + [
                        str(_meta(document).get("chunk_id"))
                        for document, _visible_text in documents
                    ]
                )),
            }

        batch_diagnostics.append({
            "input_chars": getattr(diagnostic, "input_chars", 0),
            "passage_count": getattr(diagnostic, "passage_count", len(documents)),
            "aspect_count": getattr(diagnostic, "aspect_count", len(batch_claims)),
            "request_count": getattr(diagnostic, "request_count", 0),
            "elapsed_ms": getattr(diagnostic, "elapsed_ms", 0.0),
            "input_tokens": getattr(diagnostic, "input_tokens", 0) or 0,
            "output_tokens": getattr(diagnostic, "output_tokens", 0) or 0,
            "output_chars": getattr(diagnostic, "output_chars", 0) or 0,
            "finish_reason": getattr(diagnostic, "finish_reason", None),
            "validation_error_code": getattr(
                diagnostic,
                "validation_error_code",
                None,
            ),
            "compaction_attempts": compaction_attempts,
            "excerpt_limit": selected_excerpt_limit,
            "selected_chunk_ids": [
                str(_meta(document).get("chunk_id"))
                for document in selected_documents_for_batch
            ],
        })

        by_id = {item.aspect_id: item for item in result.assessments}
        for local_index, _claim in enumerate(batch_claims, start=1):
            global_index = batch_start + local_index
            item = by_id.get(f"claim_{local_index:02d}")
            if item is None:
                all_claim_evidence.append({
                    "claim_index": global_index,
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
            all_claim_evidence.append({
                "claim_index": global_index,
                "status": item.status.value,
                "supporting_chunk_ids": claim_chunk_ids,
            })
            all_supported_chunk_ids.extend(claim_chunk_ids)

    all_supported_chunk_ids = list(dict.fromkeys(all_supported_chunk_ids))
    evidence_by_index = {
        item["claim_index"]: item for item in all_claim_evidence
    }
    kept = []
    supported = partial = removed = conflicted = 0
    for index, claim in enumerate(claims, start=1):
        evidence = evidence_by_index.get(index, {})
        status = evidence.get("status", "unknown")
        if status == "supported":
            kept.append(claim)
            supported += 1
        elif status == "partial":
            kept.append("The transcript only partially supports this detail: " + claim)
            partial += 1
        else:
            removed += 1
            if status in {"conflicted", "contradicted"}:
                conflicted += 1

    if not kept:
        verified = fallback_answer
    else:
        verified = "\n".join(
            f"{index}. {claim}" for index, claim in enumerate(kept, start=1)
        )
        if removed:
            verified += (
                "\n\nI left out details that the available transcript did not "
                "clearly establish."
            )
        if conflicted:
            verified += (
                "\nThe transcript contains conflicting information about at "
                "least one omitted point."
            )
    if truncated_claim_count:
        verified += (
            "\n\nAdditional details were omitted because the answer exceeded "
            f"the {MAX_TOTAL_CLAIMS}-claim verification limit."
        )
    verified = _append_answer_disclosures(verified, disclosures)

    return {
        "answer": verified,
        "performed": True,
        "status": "completed",
        "claim_count": len(claims),
        "truncated_claim_count": truncated_claim_count,
        "supported_claim_count": supported,
        "partial_claim_count": partial,
        "removed_claim_count": removed,
        "supported_chunk_ids": all_supported_chunk_ids,
        "claim_evidence": all_claim_evidence,
        "error_code": None,
        "assessment_batch_count": len(batch_diagnostics),
        "assessment_request_count": sum(item["request_count"] for item in batch_diagnostics),
        "assessment_elapsed_ms": sum(item["elapsed_ms"] for item in batch_diagnostics),
        "assessment_input_chars": max(
            (item["input_chars"] for item in batch_diagnostics),
            default=0,
        ),
        "assessment_input_chars_total": sum(
            item["input_chars"] for item in batch_diagnostics
        ),
        "assessment_input_limit_chars": MAX_INPUT_CHARS,
        "assessment_passage_count": max(
            (item["passage_count"] for item in batch_diagnostics),
            default=0,
        ),
        "assessment_passage_total_count": sum(
            item["passage_count"] for item in batch_diagnostics
        ),
        "assessment_aspect_count": max(
            (item["aspect_count"] for item in batch_diagnostics),
            default=0,
        ),
        "assessment_assessed_claim_count": len(claims),
        "assessment_max_aspects_per_request": max(
            (item["aspect_count"] for item in batch_diagnostics),
            default=0,
        ),
        "assessment_input_tokens": sum(item["input_tokens"] for item in batch_diagnostics),
        "assessment_output_tokens": sum(item["output_tokens"] for item in batch_diagnostics),
        "assessment_output_chars": sum(item["output_chars"] for item in batch_diagnostics),
        "assessment_finish_reason": (
            batch_diagnostics[-1]["finish_reason"] if batch_diagnostics else None
        ),
        "assessment_validation_error_code": next(
            (
                item["validation_error_code"]
                for item in batch_diagnostics
                if item["validation_error_code"]
            ),
            None,
        ),
        "assessment_compaction_attempts": sum(
            item["compaction_attempts"] for item in batch_diagnostics
        ),
        "assessment_passage_excerpt_limit": min(
            (item["excerpt_limit"] for item in batch_diagnostics),
            default=MAX_PASSAGE_CHARS,
        ),
        "assessment_selected_chunk_ids": list(dict.fromkeys(
            chunk_id
            for item in batch_diagnostics
            for chunk_id in item["selected_chunk_ids"]
        )),
        "assessment_batches": batch_diagnostics,
    }

