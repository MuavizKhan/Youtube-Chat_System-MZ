"""Video-wide map/reduce summarization with validated transcript references."""
from __future__ import annotations

import math
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


def _select_backup_points(
    primary_points: Sequence[dict],
    map_findings: Sequence[dict],
    lookup: dict,
    requested: int,
    maximum_backups: int,
) -> list[dict]:
    """Reserve verified map alternatives to replace unsupported primary points."""
    if maximum_backups < 1 or not map_findings or not lookup:
        return []

    documents = list(lookup.values())
    timeline_start = min(sort_key(document)[0] for document in documents)
    timeline_end = max(sort_key(document)[1] for document in documents)
    if timeline_end <= timeline_start:
        return []

    duplicate_guard = list(primary_points)
    candidates = []
    for order, finding in enumerate(map_findings):
        text = str(finding.get("text", "")).strip()
        if not text or _duplicate_point(text, duplicate_guard):
            continue
        buckets, midpoints = _point_timeline_buckets(
            finding,
            lookup,
            timeline_start,
            timeline_end,
            requested,
        )
        if not buckets or not midpoints:
            continue
        duplicate_guard.append(finding)
        candidates.append({
            "point": finding,
            "order": order,
            "buckets": buckets,
            "midpoints": midpoints,
        })

    if not candidates:
        return []

    covered_buckets = set()
    for point in primary_points:
        buckets, _midpoints = _point_timeline_buckets(
            point,
            lookup,
            timeline_start,
            timeline_end,
            requested,
        )
        covered_buckets.update(buckets)

    selected = []
    selected_indexes = set()
    duration = timeline_end - timeline_start
    for bucket in range(requested):
        if bucket in covered_buckets or len(selected) >= maximum_backups:
            continue
        bucket_center = timeline_start + duration * (bucket + 0.5) / requested
        options = [
            (index, candidate)
            for index, candidate in enumerate(candidates)
            if index not in selected_indexes and bucket in candidate["buckets"]
        ]
        if not options:
            continue
        index, candidate = min(
            options,
            key=lambda pair: (
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

    for index, candidate in enumerate(candidates):
        if len(selected) >= maximum_backups:
            break
        if index in selected_indexes:
            continue
        selected.append(candidate)
        selected_indexes.add(index)

    selected.sort(
        key=lambda candidate: (
            min(candidate["midpoints"]),
            candidate["order"],
        )
    )
    return [candidate["point"] for candidate in selected[:maximum_backups]]


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


def select_verified_overview_points(
    verification_claims: Sequence[dict],
    claim_evidence: Sequence[dict],
    documents: Sequence[Any],
    *,
    requested_count: int,
    timeline_start: float | None = None,
    timeline_end: float | None = None,
    bucket_count: int | None = None,
) -> list[dict]:
    """Keep verified primary points and fill gaps with verified map alternatives.

    Timeline bucket balancing is used when trustworthy timestamp bounds exist.
    When timestamps are missing or degenerate, the selector still enforces the
    requested output cap and uses stable chronological/source order.
    """
    if (
        isinstance(requested_count, bool)
        or not isinstance(requested_count, int)
        or requested_count < 1
    ):
        return []

    document_values = [
        document for document in documents
        if isinstance(getattr(document, "metadata", None), dict)
    ]
    by_chunk_id = {
        str(getattr(document, "metadata", {}).get("chunk_id")): document
        for document in document_values
        if getattr(document, "metadata", {}).get("chunk_id") is not None
    }
    if not by_chunk_id:
        return []

    try:
        derived_start = min(sort_key(document)[0] for document in document_values)
        derived_end = max(sort_key(document)[1] for document in document_values)
        start_value = (
            float(timeline_start)
            if timeline_start is not None
            else derived_start
        )
        end_value = (
            float(timeline_end)
            if timeline_end is not None
            else derived_end
        )
    except (TypeError, ValueError, OverflowError):
        start_value = derived_start
        end_value = derived_end

    valid_buckets = (
        isinstance(bucket_count, int)
        and not isinstance(bucket_count, bool)
        and bucket_count > 0
        and math.isfinite(start_value)
        and math.isfinite(end_value)
        and end_value > start_value
    )

    evidence_by_index = {
        item.get("claim_index"): item
        for item in claim_evidence
        if isinstance(item, dict)
    }
    primary, backups = [], []
    for claim_index, claim in enumerate(verification_claims, start=1):
        if not isinstance(claim, dict):
            continue
        evidence = evidence_by_index.get(claim_index, {})
        status = evidence.get("status")
        if status not in {"supported", "partial"}:
            continue
        chunk_ids = list(dict.fromkeys(
            str(value)
            for value in evidence.get("supporting_chunk_ids", [])
            if str(value) in by_chunk_id
        ))
        source_times = []
        for chunk_id in chunk_ids:
            source_start, source_end, _ = sort_key(by_chunk_id[chunk_id])
            source_times.append((source_start + source_end) / 2.0)
        if not source_times:
            continue
        source_times.sort()
        anchor_time = source_times[len(source_times) // 2]
        bucket = (
            _timeline_bucket_index(
                anchor_time,
                start_value,
                end_value,
                bucket_count,
            )
            if valid_buckets
            else None
        )
        candidate = {
            "claim_index": claim_index,
            "text": claim.get("text", ""),
            "status": status,
            "supporting_chunk_ids": chunk_ids,
            "anchor_seconds": anchor_time,
            "bucket": bucket,
            "is_backup": bool(claim.get("is_backup")),
        }
        (backups if candidate["is_backup"] else primary).append(candidate)

    selected = sorted(
        primary,
        key=lambda item: (item["anchor_seconds"], item["claim_index"]),
    )[:requested_count]
    backup_pool = sorted(
        backups,
        key=lambda item: (item["anchor_seconds"], item["claim_index"]),
    )

    if valid_buckets:
        covered = {item["bucket"] for item in selected}
        duration = end_value - start_value
        for bucket in range(bucket_count):
            if len(selected) >= requested_count:
                break
            if bucket in covered:
                continue
            options = [
                item for item in backup_pool
                if item["bucket"] == bucket
            ]
            if not options:
                continue
            bucket_center = (
                start_value + duration * (bucket + 0.5) / bucket_count
            )
            chosen = min(
                options,
                key=lambda item: (
                    abs(item["anchor_seconds"] - bucket_center),
                    item["claim_index"],
                ),
            )
            selected.append(chosen)
            backup_pool.remove(chosen)
            covered.add(bucket)

    for candidate in backup_pool:
        if len(selected) >= requested_count:
            break
        selected.append(candidate)

    return sorted(
        selected,
        key=lambda item: (item["anchor_seconds"], item["claim_index"]),
    )[:requested_count]

def timeline_coverage_from_claim_evidence(
    claim_evidence: Sequence[dict],
    documents: Sequence[Any],
    *,
    timeline_start: float,
    timeline_end: float,
    bucket_count: int,
) -> dict:
    """Measure final-answer coverage from evidence attached to verified claims."""
    if (
        isinstance(bucket_count, bool)
        or not isinstance(bucket_count, int)
        or bucket_count < 1
        or not math.isfinite(float(timeline_start))
        or not math.isfinite(float(timeline_end))
        or timeline_end <= timeline_start
    ):
        return {}

    by_chunk_id = {
        str(getattr(document, "metadata", {}).get("chunk_id")): document
        for document in documents
        if isinstance(getattr(document, "metadata", None), dict)
        and getattr(document, "metadata", {}).get("chunk_id") is not None
    }
    covered = set()
    verified_point_count = 0
    for item in claim_evidence:
        if not isinstance(item, dict) or item.get("status") not in {"supported", "partial"}:
            continue
        source_times = []
        for raw_chunk_id in item.get("supporting_chunk_ids", []):
            document = by_chunk_id.get(str(raw_chunk_id))
            if document is None:
                continue
            start, end, _chunk_id = sort_key(document)
            source_times.append((start + end) / 2.0)
        if not source_times:
            continue
        source_times.sort()
        anchor_time = source_times[len(source_times) // 2]
        covered.add(_timeline_bucket_index(
            anchor_time,
            timeline_start,
            timeline_end,
            bucket_count,
        ))
        verified_point_count += 1

    return {
        "bucket_count": bucket_count,
        "covered_buckets": [value + 1 for value in sorted(covered)],
        "missing_buckets": [
            value + 1 for value in range(bucket_count) if value not in covered
        ],
        "coverage_fraction": round(len(covered) / bucket_count, 4),
        "start_seconds": round(float(timeline_start), 2),
        "end_seconds": round(float(timeline_end), 2),
        "verified_point_count": verified_point_count,
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
    if requested_points is None:
        requested = requested_count(question)
    elif isinstance(requested_points, bool) or not isinstance(requested_points, int) or requested_points < 1:
        raise ValueError("requested_points must be a positive integer or None.")
    else:
        requested = min(requested_points, 10)
    sections, lookup, stats = build_sections(
        documents,
        preferred_sections=requested,
    )
    if isinstance(start_index, bool) or not isinstance(start_index, int) or start_index < 1:
        raise ValueError("start_index must be a positive integer.")
    exclude_points = str(exclude_points or "")[:4_000]
    previous_points = _numbered_previous_points(exclude_points)
    notes = []
    map_findings = []
    note_evidence_ids = set()
    prompt_context_parts = []
    map_calls = 0
    map_section_point_counts = []

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
        map_section_point_counts.append(len(findings))
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
    # Count map findings selected for timeline balancing as additions too;
    # preserve the existing diagnostic's meaning across both selection paths.
    map_fallback_points_added = timeline_balance["map_points_selected"]
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
        "map_finding_count": len(map_findings),
        "map_section_point_counts": map_section_point_counts,
        "map_finding_timeline_coverage": _timeline_coverage_stats(
            map_findings,
            lookup,
            requested,
        ),
        "candidate_point_timeline_coverage": _timeline_coverage_stats(
            [*map_findings, *points],
            lookup,
            requested,
        ),
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
    maximum_backup_points = 3
    backup_points = _select_backup_points(
        primary_points=points,
        map_findings=map_findings,
        lookup=lookup,
        requested=requested,
        maximum_backups=maximum_backup_points,
    )
    verification_claims = []
    for point, is_backup in (
        *((point, False) for point in points),
        *((point, True) for point in backup_points),
    ):
        evidence_chunk_ids = []
        for evidence_id in point.get("evidence_ids", []):
            document = lookup.get(str(evidence_id))
            if document is None:
                continue
            chunk_id = getattr(document, "metadata", {}).get("chunk_id")
            if chunk_id is not None:
                evidence_chunk_ids.append(str(chunk_id))
        verification_claims.append({
            "text": point["text"],
            "evidence_chunk_ids": list(dict.fromkeys(evidence_chunk_ids)),
            "is_backup": is_backup,
        })
    stats["backup_candidate_point_count"] = len(backup_points)

    return {
        "answer": answer,
        "prompt_visible_context": "\n\n".join(prompt_context_parts),
        "retrieved_results": [(document, 0.0) for document in used_documents],
        "verification_results": [
            (document, 0.0) for document in verification_documents
        ],
        "verification_claims": verification_claims,
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
