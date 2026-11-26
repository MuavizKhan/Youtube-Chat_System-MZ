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


OVERVIEW_SAMPLING_DISCLOSURE = (
    "The transcript is long, so this overview samples sections across "
    "the full timeline; some fine details may be omitted."
)


def _normalized_point_text(value: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", value.casefold()).split())


def _duplicate_point(candidate: str, existing: Sequence[dict]) -> bool:
    normalized = _normalized_point_text(candidate)
    if not normalized:
        return True
    candidate_numbers = re.findall(r"\b\d+(?:[.,]\d+)*\b", normalized)
    for item in existing:
        current = _normalized_point_text(str(item.get("text", "")))
        if not current:
            continue
        current_numbers = re.findall(r"\b\d+(?:[.,]\d+)*\b", current)
        # Different numeric claims are materially distinct evidence, even when
        # the surrounding wording is almost identical. Preserve them for the
        # transcript claim verifier rather than fuzzy-collapsing their values.
        if candidate_numbers != current_numbers:
            continue
        if normalized == current or SequenceMatcher(None, normalized, current).ratio() >= 0.88:
            return True
    return False



def _timeline_bucket_index(
    seconds: float,
    timeline_start: float,
    timeline_end: float,
    bucket_count: int,
) -> int:
    duration = timeline_end - timeline_start
    if duration <= 0 or bucket_count < 1:
        return 0
    position = (seconds - timeline_start) / duration
    return min(bucket_count - 1, max(0, int(position * bucket_count)))


def _point_timeline_buckets(
    point: dict,
    lookup: dict,
    timeline_start: float,
    timeline_end: float,
    bucket_count: int,
) -> tuple[set[int], list[float]]:
    """Assign each point to one bucket using its median cited source time."""
    duration = timeline_end - timeline_start
    if duration <= 0 or bucket_count < 1:
        return set(), []

    midpoints = []
    for evidence_id in point.get("evidence_ids", []):
        document = lookup.get(str(evidence_id))
        if document is None:
            continue
        start, end, _ = sort_key(document)
        midpoints.append((start + end) / 2.0)

    if not midpoints:
        return set(), []
    midpoints.sort()
    anchor_time = midpoints[len(midpoints) // 2]
    return {
        _timeline_bucket_index(
            anchor_time,
            timeline_start,
            timeline_end,
            bucket_count,
        )
    }, midpoints


def _balance_points_over_timeline(
    reducer_points: Sequence[dict],
    map_findings: Sequence[dict],
    lookup: dict,
    previous_points: Sequence[dict],
    requested: int,
) -> tuple[list[dict], dict]:
    """Select source-grounded points across the complete available timeline.

    Reducer wording is preferred within a time bucket. When the reducer
    clusters points in a few sections, validated map findings from uncovered
    sections fill the missing intervals. No transcript claim or evidence ID
    is created by this selection step.
    """
    if requested < 2 or previous_points or not map_findings or not lookup:
        return list(reducer_points), {
            "balancing_applied": False,
            "map_points_selected": 0,
        }

    documents = list(lookup.values())
    timeline_start = min(sort_key(document)[0] for document in documents)
    timeline_end = max(sort_key(document)[1] for document in documents)
    if timeline_end <= timeline_start:
        return list(reducer_points), {
            "balancing_applied": False,
            "map_points_selected": 0,
        }

    candidates = []
    duplicate_guard = list(previous_points)
    for origin, items in (
        ("reducer", reducer_points),
        ("map", map_findings),
    ):
        for order, point in enumerate(items):
            text = str(point.get("text", "")).strip()
            if not text or _duplicate_point(text, duplicate_guard):
                continue
            buckets, midpoints = _point_timeline_buckets(
                point,
                lookup,
                timeline_start,
                timeline_end,
                requested,
            )
            if not buckets or not midpoints:
                continue
            duplicate_guard.append(point)
            candidates.append({
                "point": point,
                "origin": origin,
                "order": order,
                "buckets": buckets,
                "midpoints": midpoints,
            })

    if not candidates:
        return list(reducer_points), {
            "balancing_applied": False,
            "map_points_selected": 0,
        }

    selected = []
    selected_indexes = set()
    covered_buckets = set()
    duration = timeline_end - timeline_start

    for bucket in range(requested):
        if bucket in covered_buckets:
            continue
        bucket_center = timeline_start + (
            duration * (bucket + 0.5) / requested
        )
        options = [
            (index, candidate)
            for index, candidate in enumerate(candidates)
            if index not in selected_indexes
            and bucket in candidate["buckets"]
        ]
        if not options:
            continue

        index, candidate = min(
            options,
            key=lambda pair: (
                pair[1]["origin"] != "reducer",
                min(
                    abs(midpoint - bucket_center)
                    for midpoint in pair[1]["midpoints"]
                    if _timeline_bucket_index(
                        midpoint,
                        timeline_start,
                        timeline_end,
                        requested,
                    ) == bucket
                ),
                pair[1]["order"],
            ),
        )
        selected.append(candidate)
        selected_indexes.add(index)
        covered_buckets.update(candidate["buckets"])

    # If a section has no distinct finding for one or more time buckets, keep
    # filling from the remaining grounded reducer/map findings. The coverage
    # diagnostic below makes any unrepresented intervals explicit.
    for index, candidate in enumerate(candidates):
        if len(selected) >= requested:
            break
        if index in selected_indexes:
            continue
        selected.append(candidate)
        selected_indexes.add(index)

    selected.sort(
        key=lambda candidate: (
            min(candidate["midpoints"]),
            candidate["origin"] != "reducer",
            candidate["order"],
        )
    )
    selected_points = [candidate["point"] for candidate in selected[:requested]]
    return selected_points, {
        "balancing_applied": (
            [point.get("text") for point in selected_points]
            != [point.get("text") for point in reducer_points]
        ),
        "map_points_selected": sum(
            candidate["origin"] == "map"
            for candidate in selected[:requested]
        ),
    }


def _timeline_coverage_stats(
    points: Sequence[dict],
    lookup: dict,
    bucket_count: int,
) -> dict:
    """Report which equal-duration timeline intervals have selected evidence."""
    if bucket_count < 1 or not lookup:
        return {}

    documents = list(lookup.values())
    timeline_start = min(sort_key(document)[0] for document in documents)
    timeline_end = max(sort_key(document)[1] for document in documents)
    if timeline_end <= timeline_start:
        return {}

    covered = set()
    for point in points:
        buckets, _midpoints = _point_timeline_buckets(
            point,
            lookup,
            timeline_start,
            timeline_end,
            bucket_count,
        )
        covered.update(buckets)

    return {
        "bucket_count": bucket_count,
        "covered_buckets": [value + 1 for value in sorted(covered)],
        "missing_buckets": [
            value + 1
            for value in range(bucket_count)
            if value not in covered
        ],
        "coverage_fraction": round(len(covered) / bucket_count, 4),
        "start_seconds": round(timeline_start, 2),
        "end_seconds": round(timeline_end, 2),
    }

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
    points, timeline_balance = _balance_points_over_timeline(
        reducer_points=points,
        map_findings=map_findings,
        lookup=lookup,
        previous_points=previous_points,
        requested=requested,
    )
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
        answer = OVERVIEW_SAMPLING_DISCLOSURE + "\n\n" + answer
    chunk_ids = list(dict.fromkeys(
        str(document.metadata["chunk_id"]) for document in used_documents
    ))
    # The reducer receives every map finding, not only the evidence IDs it
    # chooses to attach to its final points. Retain all map-cited documents in
    # a separate internal pool so the claim auditor can check every summary
    # point against evidence that was visible during the map/reduce process.
    verification_ids = list(dict.fromkeys(
        evidence_id
        for finding in map_findings
        for evidence_id in finding["evidence_ids"]
    )) or used_ids
    verification_documents = [
        lookup[value] for value in verification_ids if value in lookup
    ]
    verification_documents.sort(key=sort_key)
    verification_chunk_ids = list(dict.fromkeys(
        str(document.metadata["chunk_id"])
        for document in verification_documents
    ))
    stats.update({
        "map_calls": map_calls,
        "final_synthesis_calls": 1,
        "requested_point_count": requested,
        "returned_point_count": len(points),
        "reducer_returned_point_count": reducer_point_count,
        "map_fallback_points_added": map_fallback_points_added,
        "point_count_satisfied": len(points) >= requested,
        "verification_document_count": len(verification_documents),
        "timeline_balancing_applied": timeline_balance["balancing_applied"],
        "timeline_map_points_selected": timeline_balance["map_points_selected"],
        "timeline_coverage": _timeline_coverage_stats(
            points,
            lookup,
            requested,
        ),
    })
    return {
        "answer": answer,
        "prompt_visible_context": "\n\n".join(prompt_context_parts),
        "retrieved_results": [(document, 0.0) for document in used_documents],
        "verification_results": [
            (document, 0.0) for document in verification_documents
        ],
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
                "verification_chunk_ids": verification_chunk_ids,
                "reduced": stats["coverage_mode"] == "timeline_stratified",
            },
        },
    }
