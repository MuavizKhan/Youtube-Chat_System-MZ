"""Video-wide map/reduce summarization with validated transcript references."""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Callable, Sequence

from .overview_core import (
    OverviewError,
    build_sections,
    parse_points,
    requested_count,
    sort_key,
)
from .overview_prompts import (
    OVERVIEW_FINAL_PROMPT,
    OVERVIEW_MAP_PROMPT,
    OVERVIEW_REDUCE_PROMPT,
)


def _normalized_point_text(value: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", value.casefold()).split())


def _duplicate_point(candidate: str, existing: Sequence[dict]) -> bool:
    normalized = _normalized_point_text(candidate)
    if not normalized:
        return True
    for item in existing:
        current = _normalized_point_text(str(item.get("text", "")))
        if not current:
            continue
        if normalized == current or SequenceMatcher(None, normalized, current).ratio() >= 0.88:
            return True
    return False


def _numbered_previous_points(value: str) -> list[dict]:
    points = []
    for line in str(value or "").splitlines():
        match = re.match(r"^\s*\d{1,2}[.)]\s+(.+)$", line.strip())
        if match:
            points.append({"text": match.group(1).strip(), "evidence_ids": []})
    return points


def summarize_video_overview(
    *,
    documents: Sequence[Any],
    question: str,
    generate_text: Callable[[Any], str],
    exclude_points: str = "",
    requested_points: int | None = None,
    start_index: int = 1,
) -> dict:
    """Summarize the full indexed timeline, or sample it evenly under a hard budget."""
    sections, lookup, stats = build_sections(documents)
    if requested_points is None:
        requested = requested_count(question)
    elif isinstance(requested_points, bool) or not isinstance(requested_points, int) or requested_points < 1:
        raise ValueError("requested_points must be a positive integer or None.")
    else:
        requested = min(requested_points, 10)
    if isinstance(start_index, bool) or not isinstance(start_index, int) or start_index < 1:
        raise ValueError("start_index must be a positive integer.")
    exclude_points = str(exclude_points or "")[:4_000]
    previous_points = _numbered_previous_points(exclude_points)
    notes = []
    map_findings = []
    note_evidence_ids = set()
    prompt_context_parts = []
    map_calls = 0

    for number, section in enumerate(sections, start=1):
        context = "\n".join(item["line"] for item in section)
        prompt_context_parts.append(context)
        allowed = {item["id"] for item in section}
        if len(sections) == 1:
            points = parse_points(
                generate_text(OVERVIEW_FINAL_PROMPT.invoke({
                    "question": question,
                    "count": requested,
                    "exclude_points": exclude_points,
                    "context": context,
                })),
                allowed,
                requested,
            )
            break

        findings = parse_points(
            generate_text(OVERVIEW_MAP_PROMPT.invoke({
                "question": question,
                "number": number,
                "total": len(sections),
                "exclude_points": exclude_points,
                "context": context,
            })),
            allowed,
            3,
            allow_empty=True,
        )
        map_findings.extend(findings)
        notes.extend(
            f"Section {number}: {item['text']} "
            f"[evidence_ids: {', '.join(item['evidence_ids'])}]"
            for item in findings
        )
        for item in findings:
            note_evidence_ids.update(item["evidence_ids"])
        map_calls += 1
    else:
        if not notes:
            raise OverviewError("No section findings were generated.")
        points = parse_points(
            generate_text(OVERVIEW_REDUCE_PROMPT.invoke({
                "question": question,
                "count": requested,
                "exclude_points": exclude_points,
                "notes": "\n".join(notes),
            })),
            note_evidence_ids,
            requested,
        )

    reducer_point_count = len(points)
    map_fallback_points_added = 0
    if len(points) < requested and map_findings:
        duplicate_guard = [*previous_points, *points]
        for finding in map_findings:
            if _duplicate_point(finding["text"], duplicate_guard):
                continue
            points.append(finding)
            duplicate_guard.append(finding)
            map_fallback_points_added += 1
            if len(points) >= requested:
                break

    used_ids = list(dict.fromkeys(
        evidence_id for point in points for evidence_id in point["evidence_ids"]
    ))
    used_documents = [lookup[value] for value in used_ids if value in lookup]
    used_documents.sort(key=sort_key)
    if not used_documents:
        raise OverviewError("Overview did not cite any indexed transcript chunks.")

    answer = "\n".join(
        f"{index}. {point['text']}"
        for index, point in enumerate(points, start=start_index)
    )
    if stats["coverage_mode"] == "timeline_stratified":
        answer = (
            "The transcript is long, so this overview samples sections across "
            "the full timeline; some fine details may be omitted.\n\n" + answer
        )
    chunk_ids = list(dict.fromkeys(
        str(document.metadata["chunk_id"]) for document in used_documents
    ))
    stats.update({
        "map_calls": map_calls,
        "final_synthesis_calls": 1,
        "requested_point_count": requested,
        "returned_point_count": len(points),
        "reducer_returned_point_count": reducer_point_count,
        "map_fallback_points_added": map_fallback_points_added,
        "point_count_satisfied": len(points) >= requested,
    })
    return {
        "answer": answer,
        "prompt_visible_context": "\n\n".join(prompt_context_parts),
        "retrieved_results": [(document, 0.0) for document in used_documents],
        "retrieval_trace": {
            "route": "video_wide_overview",
            "final_stage": "video_wide_map_reduce",
            "recovery_used": False,
            "overview": stats,
            "final_chunk_ids": chunk_ids,
            "citation_chunk_ids": chunk_ids,
        },
        "generation_trace": {
            "generated": True,
            "fallback_used": False,
            "overview_summary": stats,
            "context_budget": {
                "budget_chars": 48_000,
                "original_chars": stats["indexed_source_chars"],
                "final_chars": stats["selected_source_chars"],
                "groups_available": stats["indexed_document_count"],
                "groups_included": len(used_documents),
                "included_chunk_ids": chunk_ids,
                "reduced": stats["coverage_mode"] == "timeline_stratified",
            },
        },
    }
