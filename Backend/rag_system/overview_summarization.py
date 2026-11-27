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
    OVERVIEW_TERMINAL_END_CAP_PROMPT,
    OVERVIEW_TERMINAL_PROMPT,
)


# Audit two alternative developments per requested timeline bucket. The verifier
# processes at most eight aspects per provider request, so 21 candidates are
# handled as three bounded requests for a seven-point overview.
MAX_OVERVIEW_VERIFICATION_CLAIMS = 21
MAX_BACKUP_POINTS = MAX_OVERVIEW_VERIFICATION_CLAIMS - 1
MAX_BACKUP_POINTS_PER_BUCKET = 2
MAX_TERMINAL_TAIL_CONTEXT_CHARS = 9_000
MAX_TERMINAL_TAIL_FINDINGS = 3
MAX_TERMINAL_END_CAP_FINDINGS = 1


OVERVIEW_SAMPLING_DISCLOSURE = (
    "The transcript is long, so this overview samples sections across "
    "the full timeline; some fine details may be omitted."
)


def _normalized_point_text(value: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", value.casefold()).split())


def _point_evidence_ids(point: dict) -> set[str]:
    values = []
    for key in ("evidence_ids", "evidence_chunk_ids", "supporting_chunk_ids"):
        raw = point.get(key, [])
        if isinstance(raw, (list, tuple, set)):
            values.extend(raw)
    return {str(value) for value in values if value is not None}


def _point_terms(value: str) -> set[str]:
    stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
        "for", "from", "had", "has", "have", "he", "her", "his", "how",
        "in", "into", "is", "it", "its", "of", "on", "or", "she", "so",
        "that", "the", "their", "them", "then", "there", "they", "this",
        "to", "until", "was", "we", "were", "when", "which", "while", "who", "with",
    }
    return {
        token.casefold()
        for token in re.findall(r"[^\W_]+", value, flags=re.UNICODE)
        if len(token) > 2 and token.casefold() not in stop_words
    }


def _duplicate_point(candidate: str | dict, existing: Sequence[dict]) -> bool:
    candidate_item = candidate if isinstance(candidate, dict) else {}
    candidate_text = (
        str(candidate_item.get("text", ""))
        if candidate_item
        else str(candidate)
    )
    normalized = _normalized_point_text(candidate_text)
    if not normalized:
        return True
    candidate_numbers = re.findall(r"\b\d+(?:[.,]\d+)*\b", normalized)
    candidate_terms = _point_terms(normalized)
    candidate_sources = _point_evidence_ids(candidate_item)

    for item in existing:
        current = _normalized_point_text(str(item.get("text", "")))
        if not current:
            continue
        current_numbers = re.findall(r"\b\d+(?:[.,]\d+)*\b", current)
        # Different numeric claims are materially distinct evidence, even when
        # the surrounding wording is almost identical. Preserve their values.
        if candidate_numbers != current_numbers:
            continue
        if normalized == current or SequenceMatcher(None, normalized, current).ratio() >= 0.88:
            return True

        current_sources = _point_evidence_ids(item)
        shared_sources = candidate_sources.intersection(current_sources)
        if not shared_sources:
            continue
        current_terms = _point_terms(current)
        shared_terms = candidate_terms.intersection(current_terms)
        if not candidate_terms or not current_terms:
            continue
        overlap_coefficient = len(shared_terms) / min(
            len(candidate_terms),
            len(current_terms),
        )
        jaccard_similarity = len(shared_terms) / len(
            candidate_terms.union(current_terms)
        )
        # When two paraphrases rely on the same transcript chunk, use both
        # lexical overlap measures to catch redundant restatements without
        # collapsing merely related events that share a broader topic.
        if (
            len(shared_terms) >= 4
            and overlap_coefficient >= 0.70
            and jaccard_similarity >= 0.50
        ):
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


TERMINAL_TAIL_MIN_SECONDS = 60.0
TERMINAL_TAIL_MAX_SECONDS = 420.0
TERMINAL_TAIL_FRACTION = 0.075


def _terminal_tail_start(
    timeline_start: float,
    timeline_end: float,
) -> float | None:
    """Return the start of a bounded, independently measured transcript tail."""
    duration = float(timeline_end) - float(timeline_start)
    if not math.isfinite(duration) or duration <= 0:
        return None
    window_seconds = min(
        TERMINAL_TAIL_MAX_SECONDS,
        max(TERMINAL_TAIL_MIN_SECONDS, duration * TERMINAL_TAIL_FRACTION),
    )
    # Avoid designating most of a short video as its terminal tail.
    if window_seconds >= duration * 0.5:
        return None
    return float(timeline_end) - window_seconds


def _terminal_overlap_center(
    document: Any,
    tail_start: float | None,
    timeline_end: float,
) -> float | None:
    """Return a timestamp only when the entire cited chunk is inside the tail."""
    if tail_start is None:
        return None
    start, end, _ = sort_key(document)
    if start < tail_start or end > timeline_end or end <= start:
        return None
    return (start + end) / 2.0


def _point_terminal_tail_times(
    point: dict,
    lookup: dict,
    tail_start: float | None,
    timeline_end: float,
) -> list[float]:
    if tail_start is None:
        return []
    times = []
    for evidence_id in point.get("evidence_ids", []):
        document = lookup.get(str(evidence_id))
        if document is None:
            continue
        overlap_center = _terminal_overlap_center(
            document,
            tail_start,
            timeline_end,
        )
        if overlap_center is not None:
            times.append(overlap_center)
    return sorted(times)


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
    terminal_tail_start = _terminal_tail_start(
        timeline_start,
        timeline_end,
    )
    terminal_tail_times = _point_terminal_tail_times(
        point,
        lookup,
        terminal_tail_start,
        timeline_end,
    )
    anchor_bucket = (
        bucket_count - 1
        if terminal_tail_times
        else _timeline_bucket_index(
            anchor_time,
            timeline_start,
            timeline_end,
            bucket_count,
        )
    )
    return {anchor_bucket}, midpoints


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

    terminal_tail_start = _terminal_tail_start(timeline_start, timeline_end)
    candidates = []
    duplicate_guard = list(previous_points)
    for origin, items in (
        ("reducer", reducer_points),
        ("map", map_findings),
    ):
        for order, point in enumerate(items):
            text = str(point.get("text", "")).strip()
            if not text or _duplicate_point(point, duplicate_guard):
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
                "terminal_tail_times": _point_terminal_tail_times(
                    point,
                    lookup,
                    terminal_tail_start,
                    timeline_end,
                ),
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

        def candidate_rank(pair):
            candidate = pair[1]
            in_terminal_tail = (
                bucket == requested - 1
                and bool(candidate["terminal_tail_times"])
            )
            if bucket == requested - 1 and terminal_tail_start is not None:
                distance_to_video_end = (
                    timeline_end - max(candidate["terminal_tail_times"])
                    if in_terminal_tail
                    else min(
                        abs(midpoint - bucket_center)
                        for midpoint in candidate["midpoints"]
                        if _timeline_bucket_index(
                            midpoint,
                            timeline_start,
                            timeline_end,
                            requested,
                        ) == bucket
                    )
                )
                return (
                    not in_terminal_tail,
                    distance_to_video_end,
                    candidate["origin"] != "reducer",
                    candidate["order"],
                )
            return (
                candidate["origin"] != "reducer",
                min(
                    abs(midpoint - bucket_center)
                    for midpoint in candidate["midpoints"]
                    if _timeline_bucket_index(
                        midpoint,
                        timeline_start,
                        timeline_end,
                        requested,
                    ) == bucket
                ),
                candidate["order"],
            )

        index, candidate = min(options, key=candidate_rank)
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
    timeline_valid = timeline_end > timeline_start
    terminal_tail_start = (
        _terminal_tail_start(timeline_start, timeline_end)
        if timeline_valid
        else None
    )

    duplicate_guard = list(primary_points)
    candidates = []
    for order, finding in enumerate(map_findings):
        text = str(finding.get("text", "")).strip()
        if not text or _duplicate_point(finding, duplicate_guard):
            continue
        if timeline_valid:
            buckets, midpoints = _point_timeline_buckets(
                finding,
                lookup,
                timeline_start,
                timeline_end,
                requested,
            )
        else:
            midpoints = []
            for evidence_id in finding.get("evidence_ids", []):
                document = lookup.get(str(evidence_id))
                if document is None:
                    continue
                start, end, _ = sort_key(document)
                midpoints.append((start + end) / 2.0)
            buckets = {0} if midpoints else set()
        if not buckets or not midpoints:
            continue
        duplicate_guard.append(finding)
        candidates.append({
            "point": finding,
            "order": order,
            "buckets": buckets,
            "midpoints": midpoints,
            "terminal_tail_times": _point_terminal_tail_times(
                finding,
                lookup,
                terminal_tail_start,
                timeline_end,
            ),
        })

    if not candidates:
        return []

    selected = []
    selected_indexes = set()
    duration = timeline_end - timeline_start
    bucket_count = requested if timeline_valid else 1

    # Reserve alternatives across the entire timeline, even where a primary
    # point currently appears to cover a bucket. Primary points are not trusted
    # until the claim audit completes; any one of them may be partial.
    if maximum_backups >= bucket_count:
        target_buckets = list(range(bucket_count))
    elif maximum_backups == 1:
        target_buckets = [bucket_count // 2]
    else:
        target_buckets = sorted({
            round(index * (bucket_count - 1) / (maximum_backups - 1))
            for index in range(maximum_backups)
        })

    reserve_rounds = max(
        1,
        math.ceil(maximum_backups / max(1, len(target_buckets))),
    )
    for _round in range(reserve_rounds):
        for bucket in target_buckets:
            if len(selected) >= maximum_backups:
                break
            bucket_center = (
                timeline_start + duration * (bucket + 0.5) / bucket_count
                if timeline_valid
                else timeline_start
            )
            options = [
                (index, candidate)
                for index, candidate in enumerate(candidates)
                if index not in selected_indexes and bucket in candidate["buckets"]
            ]
            if not options:
                continue
            def candidate_rank(pair):
                candidate = pair[1]
                in_terminal_tail = (
                    bucket == bucket_count - 1
                    and bool(candidate["terminal_tail_times"])
                )
                if bucket == bucket_count - 1 and terminal_tail_start is not None:
                    distance_to_video_end = (
                        timeline_end - max(candidate["terminal_tail_times"])
                        if in_terminal_tail
                        else min(
                            abs(midpoint - bucket_center)
                            for midpoint in candidate["midpoints"]
                            if _timeline_bucket_index(
                                midpoint,
                                timeline_start,
                                timeline_end,
                                bucket_count,
                            ) == bucket
                        )
                    )
                    return (
                        not in_terminal_tail,
                        distance_to_video_end,
                        candidate["order"],
                    )
                return (
                    min(
                        abs(midpoint - bucket_center)
                        for midpoint in candidate["midpoints"]
                        if _timeline_bucket_index(
                            midpoint,
                            timeline_start,
                            timeline_end,
                            bucket_count,
                        ) == bucket
                    ),
                    candidate["order"],
                )

            index, candidate = min(options, key=candidate_rank)
            selected.append(candidate)
            selected_indexes.add(index)
        if len(selected) >= maximum_backups:
            break

    # Sparse timelines may not have a candidate in every target bucket. Fill
    # the remaining reserve from distinct, source-linked map findings.
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
    """Select distinct, fully supported points across the available timeline.

    Primary and reserve claims compete for the same timeline buckets. The
    selector never pads a covered bucket with a duplicate or a second point
    while another usable interval is available. If an interval has no distinct
    fully supported candidate, the caller reports the resulting shortfall.
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

    terminal_tail_start = (
        _terminal_tail_start(start_value, end_value)
        if valid_buckets
        else None
    )
    evidence_by_index = {
        item.get("claim_index"): item
        for item in claim_evidence
        if isinstance(item, dict)
    }
    candidates = []
    for claim_index, claim in enumerate(verification_claims, start=1):
        if not isinstance(claim, dict):
            continue
        evidence = evidence_by_index.get(claim_index, {})
        if evidence.get("status") != "supported":
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
        terminal_tail_times = [
            overlap_center
            for chunk_id in chunk_ids
            if (
                overlap_center := _terminal_overlap_center(
                    by_chunk_id[chunk_id],
                    terminal_tail_start,
                    end_value,
                )
            ) is not None
        ]
        candidates.append({
            "claim_index": claim_index,
            "text": str(claim.get("text", "")).strip(),
            "status": "supported",
            "supporting_chunk_ids": chunk_ids,
            "anchor_seconds": anchor_time,
            "terminal_tail_times": terminal_tail_times,
            "bucket": (
                bucket_count - 1
                if valid_buckets and terminal_tail_times
                else (
                    _timeline_bucket_index(
                        anchor_time,
                        start_value,
                        end_value,
                        bucket_count,
                    )
                    if valid_buckets
                    else None
                )
            ),
            "is_backup": bool(claim.get("is_backup")),
        })

    # Keep primary wording in preference to reserve wording when two claims
    # repeat the same development. A same-source paraphrase is considered a
    # duplicate only when its informative terms substantially overlap.
    distinct_candidates = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            item["is_backup"],
            item["anchor_seconds"],
            item["claim_index"],
        ),
    ):
        if not candidate["text"] or _duplicate_point(candidate, distinct_candidates):
            continue
        distinct_candidates.append(candidate)

    if not valid_buckets:
        return sorted(
            distinct_candidates,
            key=lambda item: (item["anchor_seconds"], item["claim_index"]),
        )[:requested_count]

    if requested_count >= bucket_count:
        target_buckets = list(range(bucket_count))
    elif requested_count == 1:
        target_buckets = [bucket_count // 2]
    else:
        target_buckets = sorted({
            round(index * (bucket_count - 1) / (requested_count - 1))
            for index in range(requested_count)
        })

    selected = []
    selected_indexes = set()
    duration = end_value - start_value
    for bucket in target_buckets:
        if len(selected) >= requested_count:
            break
        options = [
            item for item in distinct_candidates
            if item["bucket"] == bucket
            and item["claim_index"] not in selected_indexes
        ]
        if not options:
            continue
        bucket_center = start_value + (
            duration * (bucket + 0.5) / bucket_count
        )
        def verified_candidate_rank(item):
            if bucket == bucket_count - 1 and terminal_tail_start is not None:
                in_terminal_tail = bool(item["terminal_tail_times"])
                return (
                    not in_terminal_tail,
                    (
                        end_value - max(item["terminal_tail_times"])
                        if in_terminal_tail
                        else abs(item["anchor_seconds"] - bucket_center)
                    ),
                    item["is_backup"],
                    item["claim_index"],
                )
            return (
                item["is_backup"],
                abs(item["anchor_seconds"] - bucket_center),
                item["claim_index"],
            )

        chosen = min(options, key=verified_candidate_rank)
        selected.append(chosen)
        selected_indexes.add(chosen["claim_index"])

    # If one of the evenly spaced target buckets has no verified finding,
    # use another unrepresented bucket before returning a shortfall. Never
    # take a second point from a bucket already represented.
    remaining_buckets = [
        bucket for bucket in range(bucket_count)
        if bucket not in target_buckets
    ]
    for bucket in remaining_buckets:
        if len(selected) >= requested_count:
            break
        options = [
            item for item in distinct_candidates
            if item["bucket"] == bucket
            and item["claim_index"] not in selected_indexes
        ]
        if not options:
            continue
        bucket_center = start_value + (
            duration * (bucket + 0.5) / bucket_count
        )
        def verified_candidate_rank(item):
            if bucket == bucket_count - 1 and terminal_tail_start is not None:
                in_terminal_tail = bool(item["terminal_tail_times"])
                return (
                    not in_terminal_tail,
                    (
                        end_value - max(item["terminal_tail_times"])
                        if in_terminal_tail
                        else abs(item["anchor_seconds"] - bucket_center)
                    ),
                    item["is_backup"],
                    item["claim_index"],
                )
            return (
                item["is_backup"],
                abs(item["anchor_seconds"] - bucket_center),
                item["claim_index"],
            )

        chosen = min(options, key=verified_candidate_rank)
        selected.append(chosen)
        selected_indexes.add(chosen["claim_index"])

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
        if not isinstance(item, dict) or item.get("status") != "supported":
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
        terminal_times = [
            overlap_center
            for chunk_id in item.get("supporting_chunk_ids", [])
            if (
                document := by_chunk_id.get(str(chunk_id))
            ) is not None
            and (
                overlap_center := _terminal_overlap_center(
                    document,
                    _terminal_tail_start(timeline_start, timeline_end),
                    timeline_end,
                )
            ) is not None
        ]
        covered.add(
            bucket_count - 1
            if terminal_times
            else _timeline_bucket_index(
                anchor_time,
                timeline_start,
                timeline_end,
                bucket_count,
            )
        )
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

def terminal_tail_coverage_from_claim_evidence(
    claim_evidence: Sequence[dict],
    documents: Sequence[Any],
    *,
    timeline_start: float | None = None,
    timeline_end: float | None = None,
    end_cap_chunk_ids: Sequence[str] | None = None,
) -> dict:
    """Measure verified evidence in the closing window and exact final passages."""
    document_values = [
        document for document in documents
        if isinstance(getattr(document, "metadata", None), dict)
    ]
    if not document_values:
        return {}

    evidence_start = min(sort_key(document)[0] for document in document_values)
    evidence_end = max(sort_key(document)[1] for document in document_values)
    timeline_start = (
        evidence_start if timeline_start is None else float(timeline_start)
    )
    timeline_end = (
        evidence_end if timeline_end is None else float(timeline_end)
    )
    tail_start = _terminal_tail_start(timeline_start, timeline_end)
    if tail_start is None:
        return {
            "eligible": False,
            "covered": None,
            "reason": "video_too_short_for_a_separate_terminal_window",
            "start_seconds": round(timeline_start, 2),
            "end_seconds": round(timeline_end, 2),
        }

    by_chunk_id = {
        str(getattr(document, "metadata", {}).get("chunk_id")): document
        for document in document_values
        if getattr(document, "metadata", {}).get("chunk_id") is not None
    }
    requested_end_cap_ids = {
        str(value)
        for value in (end_cap_chunk_ids or ())
        if value is not None
    }
    # The end-cap IDs come from the full transcript passages already checked
    # for terminal-window containment during overview construction. Keep
    # eligibility independent from the verifier's narrower evidence pool:
    # those exact documents may be absent when no claim cites them.
    eligible_end_cap_ids = requested_end_cap_ids
    end_cap_eligible = bool(eligible_end_cap_ids)
    supporting_chunk_ids = []
    end_cap_supporting_chunk_ids = []
    latest_supported_midpoint = None
    latest_end_cap_supported_midpoint = None
    for item in claim_evidence:
        if not isinstance(item, dict) or item.get("status") != "supported":
            continue
        for raw_chunk_id in item.get("supporting_chunk_ids", []):
            chunk_id = str(raw_chunk_id)
            document = by_chunk_id.get(chunk_id)
            if document is None:
                continue
            overlap_center = _terminal_overlap_center(
                document,
                tail_start,
                timeline_end,
            )
            if overlap_center is not None:
                supporting_chunk_ids.append(chunk_id)
                latest_supported_midpoint = (
                    overlap_center
                    if latest_supported_midpoint is None
                    else max(latest_supported_midpoint, overlap_center)
                )
            end_cap_center = (
                overlap_center
                if chunk_id in eligible_end_cap_ids
                else None
            )
            if end_cap_center is not None:
                end_cap_supporting_chunk_ids.append(chunk_id)
                latest_end_cap_supported_midpoint = (
                    end_cap_center
                    if latest_end_cap_supported_midpoint is None
                    else max(latest_end_cap_supported_midpoint, end_cap_center)
                )

    unique_chunk_ids = list(dict.fromkeys(supporting_chunk_ids))
    unique_end_cap_chunk_ids = list(dict.fromkeys(end_cap_supporting_chunk_ids))
    return {
        "eligible": True,
        "covered": bool(unique_chunk_ids),
        "reason": (
            None
            if unique_chunk_ids
            else "no_selected_supported_claim_in_terminal_window"
        ),
        "window_seconds": round(timeline_end - tail_start, 2),
        "start_seconds": round(tail_start, 2),
        "end_seconds": round(timeline_end, 2),
        "supporting_chunk_ids": unique_chunk_ids,
        "latest_supported_midpoint_seconds": (
            round(latest_supported_midpoint, 2)
            if latest_supported_midpoint is not None
            else None
        ),
        "end_cap_eligible": end_cap_eligible,
        "end_cap_covered": (
            bool(unique_end_cap_chunk_ids) if end_cap_eligible else None
        ),
        "end_cap_chunk_ids": sorted(eligible_end_cap_ids),
        "end_cap_supporting_chunk_ids": unique_end_cap_chunk_ids,
        "latest_end_cap_supported_midpoint_seconds": (
            round(latest_end_cap_supported_midpoint, 2)
            if latest_end_cap_supported_midpoint is not None
            else None
        ),
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
    backup_findings = []
    note_evidence_ids = set()
    prompt_context_parts = []
    map_calls = 0
    map_section_point_counts = []
    terminal_points = []

    for number, section in enumerate(sections, start=1):
        context = "\n".join(item["line"] for item in section)
        prompt_context_parts.append(context)
        allowed = {item["id"] for item in section}
        if len(sections) == 1:
            single_section_candidates = parse_points(
                generate_text(OVERVIEW_FINAL_PROMPT.invoke({
                    "question": question,
                    "count": requested + min(
                        MAX_BACKUP_POINTS_PER_BUCKET * requested,
                        MAX_OVERVIEW_VERIFICATION_CLAIMS - requested,
                    ),
                    "exclude_points": exclude_points,
                    "context": context,
                })),
                allowed,
                requested + min(
                    MAX_BACKUP_POINTS_PER_BUCKET * requested,
                    MAX_OVERVIEW_VERIFICATION_CLAIMS - requested,
                ),
            )
            points = single_section_candidates[:requested]
            backup_findings.extend(single_section_candidates[requested:])
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
    # The actual ending is audited independently from the broad timeline
    # buckets. This prevents an earlier passage in the final bucket from
    # standing in for substantive developments in the transcript's closing
    # minutes. The scan reuses the existing overview verification budget.
    selected_documents = list(lookup.values())
    timeline_start = min(sort_key(document)[0] for document in selected_documents)
    timeline_end = max(sort_key(document)[1] for document in selected_documents)
    terminal_tail_start = _terminal_tail_start(timeline_start, timeline_end)
    terminal_tail_scan = {
        "status": "not_applicable",
        "finding_count": 0,
        "start_seconds": None,
        "end_seconds": round(timeline_end, 2),
        "passage_ids": [],
    }
    if terminal_tail_start is not None:
        terminal_items = [
            item
            for section in sections
            for item in section
            if _terminal_overlap_center(
                item["document"],
                terminal_tail_start,
                timeline_end,
            ) is not None
        ]
        terminal_items.sort(
            key=lambda item: sort_key(item["document"])
        )
        # Keep the most recent passages under a fixed prompt budget. The
        # ending itself is never removed to make room for older context.
        bounded_terminal_items = []
        terminal_chars = 0
        for item in reversed(terminal_items):
            line_length = len(item["line"])
            if (
                bounded_terminal_items
                and terminal_chars + line_length > MAX_TERMINAL_TAIL_CONTEXT_CHARS
            ):
                break
            if not bounded_terminal_items and line_length > MAX_TERMINAL_TAIL_CONTEXT_CHARS:
                item = {
                    **item,
                    "line": item["line"][-MAX_TERMINAL_TAIL_CONTEXT_CHARS:],
                }
                line_length = len(item["line"])
            bounded_terminal_items.append(item)
            terminal_chars += line_length
        bounded_terminal_items.reverse()
        if bounded_terminal_items:
            terminal_context = "\n".join(
                item["line"] for item in bounded_terminal_items
            )
            terminal_allowed_ids = {
                item["id"] for item in bounded_terminal_items
            }
            terminal_passage_ids = [
                item["id"] for item in bounded_terminal_items
            ]
            terminal_source_ids = list(dict.fromkeys(
                str(item["document"].metadata.get("chunk_id"))
                for item in bounded_terminal_items
            ))
            try:
                terminal_response = generate_text(
                    OVERVIEW_TERMINAL_PROMPT.invoke({
                        "question": question,
                        "exclude_points": (
                            exclude_points
                            + "\n"
                            + "\n".join(item["text"] for item in map_findings)
                        )[-4_000:],
                        "context": terminal_context,
                    })
                )
                terminal_points = parse_points(
                    terminal_response,
                    terminal_allowed_ids,
                    MAX_TERMINAL_TAIL_FINDINGS,
                    allow_empty=True,
                )
            except Exception as error:
                # A provider throttle or malformed response must not discard
                # an otherwise complete, evidence-verified overview.
                terminal_points = []
                terminal_tail_scan = {
                    "status": "unavailable",
                    "failure_type": type(error).__name__,
                    "finding_count": 0,
                    "start_seconds": round(terminal_tail_start, 2),
                    "end_seconds": round(timeline_end, 2),
                    "passage_ids": terminal_passage_ids,
                    "source_chunk_ids": terminal_source_ids,
                    "context_chars": terminal_chars,
                }
            else:
                terminal_tail_scan = {
                    "status": "completed",
                    "finding_count": len(terminal_points),
                    "start_seconds": round(terminal_tail_start, 2),
                    "end_seconds": round(timeline_end, 2),
                    "passage_ids": terminal_passage_ids,
                    "source_chunk_ids": terminal_source_ids,
                    "context_chars": terminal_chars,
                }

            # A single broad-tail prompt can prefer the first part of its
            # closing window. If neither it nor the section maps cites either
            # of the final two eligible passages, inspect those exact passages
            # separately and pass any distinct finding through normal audit.
            end_cap_items = bounded_terminal_items[-2:]
            end_cap_passage_ids = [item["id"] for item in end_cap_items]
            end_cap_allowed_ids = set(end_cap_passage_ids)
            end_cap_source_ids = list(dict.fromkeys(
                str(item["document"].metadata.get("chunk_id"))
                for item in end_cap_items
            ))
            end_cap_start_seconds = (
                round(sort_key(end_cap_items[0]["document"])[0], 2)
                if end_cap_items
                else None
            )
            end_cap_scan = {
                "status": "not_applicable",
                "finding_count": 0,
                "start_seconds": end_cap_start_seconds,
                "end_seconds": round(timeline_end, 2),
                "passage_ids": end_cap_passage_ids,
                "source_chunk_ids": end_cap_source_ids,
                "source": "none",
            }
            if end_cap_items:
                end_cap_candidates_already_found = [
                    point
                    for point in [*map_findings, *terminal_points]
                    if _point_evidence_ids(point).intersection(end_cap_allowed_ids)
                ]
                if end_cap_candidates_already_found:
                    end_cap_scan.update({
                        "status": "completed",
                        "finding_count": len(end_cap_candidates_already_found),
                        "source": (
                            "terminal_tail_scan"
                            if any(
                                point in terminal_points
                                for point in end_cap_candidates_already_found
                            )
                            else "section_map"
                        ),
                    })
                elif terminal_tail_scan.get("status") == "completed":
                    end_cap_context = "\n".join(
                        item["line"] for item in end_cap_items
                    )
                    try:
                        end_cap_response = generate_text(
                            OVERVIEW_TERMINAL_END_CAP_PROMPT.invoke({
                                "question": question,
                                "exclude_points": (
                                    exclude_points
                                    + "\n"
                                    + "\n".join(
                                        item["text"]
                                        for item in [*map_findings, *terminal_points]
                                    )
                                )[-4_000:],
                                "context": end_cap_context,
                            })
                        )
                        end_cap_candidates = parse_points(
                            end_cap_response,
                            end_cap_allowed_ids,
                            MAX_TERMINAL_END_CAP_FINDINGS,
                            allow_empty=True,
                        )
                    except Exception as error:
                        end_cap_scan.update({
                            "status": "unavailable",
                            "failure_type": type(error).__name__,
                        })
                    else:
                        appended_end_cap_count = 0
                        for candidate in end_cap_candidates:
                            if _duplicate_point(
                                candidate, [*map_findings, *terminal_points]
                            ):
                                continue
                            terminal_points.append(candidate)
                            appended_end_cap_count += 1
                        end_cap_scan.update({
                            "status": "completed",
                            "finding_count": appended_end_cap_count,
                            "source": "dedicated_end_cap_scan",
                        })
                else:
                    end_cap_scan.update({
                        "status": "skipped_terminal_scan_unavailable",
                        "failure_type": terminal_tail_scan.get("failure_type"),
                    })

            terminal_tail_scan["base_finding_count"] = (
                terminal_tail_scan.get("finding_count", 0)
            )
            terminal_tail_scan["finding_count"] = len(terminal_points)
            terminal_tail_scan["end_cap_scan"] = end_cap_scan
            if terminal_tail_scan.get("status") == "completed":
                map_findings.extend(terminal_points)
                for item in terminal_points:
                    note_evidence_ids.update(item["evidence_ids"])
                    notes.append(
                        f"Terminal closing development: {item['text']} "
                        f"[evidence_ids: {', '.join(item['evidence_ids'])}]"
                    )
        else:
            terminal_tail_scan = {
                "status": "no_terminal_passages_selected",
                "finding_count": 0,
                "start_seconds": round(terminal_tail_start, 2),
                "end_seconds": round(timeline_end, 2),
                "passage_ids": [],
                "end_cap_scan": {
                    "status": "not_applicable",
                    "finding_count": 0,
                    "start_seconds": None,
                    "end_seconds": round(timeline_end, 2),
                    "passage_ids": [],
                    "source_chunk_ids": [],
                    "source": "none",
                },
            }

    if len(sections) > 1:
        if not notes:
            raise OverviewError("No section or terminal findings were generated.")
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
            if _duplicate_point(finding, duplicate_guard):
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
    verification_findings = [*map_findings, *backup_findings]
    verification_ids = list(dict.fromkeys([
        *used_ids,
        *(
            evidence_id
            for finding in verification_findings
            for evidence_id in finding["evidence_ids"]
        ),
    ]))
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
        "map_finding_count": len(map_findings) - len(terminal_points),
        "terminal_tail_scan": terminal_tail_scan,
        "terminal_tail_finding_timeline_coverage": _timeline_coverage_stats(
            terminal_points,
            lookup,
            requested,
        ),
        "map_section_point_counts": map_section_point_counts,
        "map_finding_timeline_coverage": _timeline_coverage_stats(
            map_findings[:-len(terminal_points)] if terminal_points else map_findings,
            lookup,
            requested,
        ),
        "single_section_backup_candidate_count": len(backup_findings),
        "backup_finding_timeline_coverage": _timeline_coverage_stats(
            backup_findings,
            lookup,
            requested,
        ),
        "candidate_point_timeline_coverage": _timeline_coverage_stats(
            [*map_findings, *backup_findings, *points],
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
    maximum_backup_points = min(
        MAX_BACKUP_POINTS,
        MAX_BACKUP_POINTS_PER_BUCKET * requested,
        max(0, MAX_OVERVIEW_VERIFICATION_CLAIMS - len(points)),
    )
    backup_points = _select_backup_points(
        primary_points=points,
        map_findings=[*map_findings, *backup_findings],
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
