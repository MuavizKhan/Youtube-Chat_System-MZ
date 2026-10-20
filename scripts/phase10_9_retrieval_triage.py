"""Run a reproducible, targeted Phase 10.9 retrieval-failure triage.

This runner packages the existing stage-level retrieval diagnostics. It does not
change retrieval behavior, call the answer-generation model, alter indexes, or
declare an answer-quality pass. Its purpose is to show where indexed gold
evidence is lost for the three previously observed problem videos.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from Backend.rag_system import config as rag_config
from Backend.rag_system.benchmark_registry import VIDEO_BENCHMARKS
from Backend.rag_system.evaluation import diagnose_video
from Backend.rag_system.indexing import get_index_state
from Backend.rag_system.multi_video_evaluation import _build_experiment_metadata


REPORT_ID = "phase10_9_retrieval_failure_triage"
REPORT_SCHEMA_VERSION = "1.0.0"

# These are investigation targets from the latest recorded multi-video pilot.
# The IDs select gold-evidence questions, not expected outcomes: a target may
# pass on a later run, and the report must preserve what the trace actually says.
DEFAULT_DIAGNOSTIC_TARGETS: dict[str, tuple[str, ...]] = {
    "MdeQMVBuGgY": ("Q03",),
    "A-JBaNvv3Tk": ("AI01",),
    "fAd2nKfrDB8": ("AR01", "AR02", "AR03", "AR04"),
}

_INDEX_METADATA_FIELDS = (
    "metadata_version",
    "transcript_segment_schema_version",
    "video_id",
    "embedding_model",
    "chunk_size",
    "chunk_overlap",
    "chunk_count",
    "transcript_language",
    "transcript_language_code",
    "transcript_generated",
    "requested_languages",
)

_STAGE_RANK_FIELDS = {
    "raw_faiss": "raw_faiss_rank",
    "semantic_mmr": "semantic_mmr_rank",
    "semantic_fusion": "semantic_fused_rank",
    "lexical": "lexical_rank",
    "hybrid_fusion": "hybrid_fused_rank",
    "facet_rerank": "facet_soft_reranked_rank",
    "anchor_selection": "anchor_rank",
    "production_context": "final_context_rank",
}


def _project_relative_label(path: Path) -> str:
    """Return a portable path label without leaking an external local username."""
    try:
        return path.resolve().relative_to(
            Path(rag_config.PROJECT_ROOT).resolve()
        ).as_posix()
    except ValueError:
        return "external-configured-index-path"


def _sha256_file(path: Path) -> str:
    """Hash a file without loading the entire index into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_index_snapshot(video_id: str) -> dict[str, Any]:
    """Capture the persisted index state and identity without exposing transcript text."""
    index_dir = Path(rag_config.VECTOR_STORE_ROOT) / video_id
    metadata_path = index_dir / "metadata.json"

    state_value = get_index_state(video_id)
    state = getattr(state_value, "value", str(state_value))
    files: dict[str, dict[str, Any]] = {}

    for filename in ("index.faiss", "index.pkl", "metadata.json"):
        path = index_dir / filename
        record: dict[str, Any] = {"exists": path.is_file()}
        if path.is_file():
            try:
                record["size_bytes"] = path.stat().st_size
                record["sha256"] = _sha256_file(path)
            except OSError:
                record["hash_error"] = True
        files[filename] = record

    metadata_summary = None
    metadata_sha256 = None
    metadata_error = None
    if metadata_path.is_file():
        try:
            raw = metadata_path.read_bytes()
            metadata_sha256 = hashlib.sha256(raw).hexdigest()
            parsed = json.loads(raw.decode("utf-8"))
            if isinstance(parsed, dict):
                metadata_summary = {
                    key: parsed[key]
                    for key in _INDEX_METADATA_FIELDS
                    if key in parsed
                }
            else:
                metadata_error = "metadata_root_is_not_an_object"
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            metadata_error = type(error).__name__

    return {
        "state": state,
        "directory_label": _project_relative_label(index_dir),
        "files": files,
        "metadata_sha256": metadata_sha256,
        "metadata": metadata_summary,
        "metadata_error": metadata_error,
    }


def _index_blocker(
    index_snapshot: dict[str, Any],
    expected_language: str,
) -> str | None:
    """Prevent a trace from being misinterpreted when its index is not comparable."""
    if index_snapshot.get("state") != "valid":
        return "index_state_not_valid"

    metadata = index_snapshot.get("metadata")
    if not isinstance(metadata, dict):
        return "index_metadata_unavailable"

    actual_language = metadata.get("transcript_language_code")
    if not isinstance(actual_language, str) or not actual_language.strip():
        return "index_transcript_language_unverifiable"

    normalized_language = (
        actual_language.strip().casefold().replace("_", "-").split("-", 1)[0]
    )
    if normalized_language != expected_language.casefold():
        return "index_transcript_language_mismatch"

    return None


def _group_summaries(diagnostics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compact each gold-evidence group to stage-presence counts and the existing diagnosis."""
    summaries: list[dict[str, Any]] = []
    for question in diagnostics:
        question_id = question.get("id")
        for group in question.get("groups") or []:
            candidates = [
                candidate
                for candidate in (group.get("candidates") or [])
                if isinstance(candidate, dict)
            ]
            stage_counts = {
                stage: sum(
                    1
                    for candidate in candidates
                    if candidate.get(rank_field) is not None
                )
                for stage, rank_field in _STAGE_RANK_FIELDS.items()
            }
            summaries.append(
                {
                    "question_id": question_id,
                    "group_index": group.get("group"),
                    "matched_index_phrase": group.get("matched_index_phrase"),
                    "indexed_candidate_count": group.get("candidate_count"),
                    "candidate_chunk_ids": list(group.get("candidate_chunk_ids") or []),
                    "raw_anchor_top_k_status": group.get("raw_anchor_top_k_status"),
                    "existing_group_diagnosis": group.get("diagnosis"),
                    "returned_gold_candidates_by_stage": stage_counts,
                }
            )
    return summaries


def _validate_target(video_id: str) -> tuple[Any, list[str]]:
    if video_id not in DEFAULT_DIAGNOSTIC_TARGETS:
        raise ValueError(
            f"{video_id!r} is not a configured Phase 10.9 triage target. "
            f"Choose one of: {', '.join(DEFAULT_DIAGNOSTIC_TARGETS)}."
        )

    benchmark = VIDEO_BENCHMARKS.get(video_id)
    if benchmark is None:
        raise ValueError(
            f"No benchmark is registered for Phase 10.9 target {video_id!r}."
        )

    question_ids = list(DEFAULT_DIAGNOSTIC_TARGETS[video_id])
    gold_question_ids = set(benchmark.gold_evidence_by_question)
    evaluation_question_ids = {
        case["id"] for case in benchmark.question_cases_for_evaluation()
    }
    invalid = [
        question_id
        for question_id in question_ids
        if question_id not in gold_question_ids
        or question_id not in evaluation_question_ids
    ]
    if invalid:
        raise ValueError(
            f"Phase 10.9 target {video_id!r} refers to questions that are not "
            f"available for gold diagnostics: {invalid}."
        )

    return benchmark, question_ids


def run_phase10_9_triage(
    *,
    video_ids: Sequence[str] | None = None,
    diagnostic_runner: Callable[..., dict[str, Any]] | None = None,
    provenance_builder: Callable[[], dict[str, Any]] | None = None,
    index_snapshot_builder: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run targeted retrieval traces and return a versioned, local report."""
    selected_ids = (
        list(DEFAULT_DIAGNOSTIC_TARGETS)
        if video_ids is None
        else list(video_ids)
    )
    if not selected_ids:
        raise ValueError("At least one Phase 10.9 video target must be selected.")
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Phase 10.9 video targets must not contain duplicates.")

    # Validate the entire selection before performing any index reads.
    targets = {
        video_id: _validate_target(video_id)
        for video_id in selected_ids
    }
    if diagnostic_runner is None:
        diagnostic_runner = diagnose_video
    if provenance_builder is None:
        provenance_builder = _build_experiment_metadata
    if index_snapshot_builder is None:
        index_snapshot_builder = build_index_snapshot

    videos: list[dict[str, Any]] = []
    for video_id in selected_ids:
        benchmark, question_ids = targets[video_id]
        base = {
            "video_id": video_id,
            "title": benchmark.title,
            "expected_transcript_language": benchmark.transcript_language,
            "video_url": benchmark.video_url,
            "target_question_ids": question_ids,
        }
        try:
            index_snapshot = index_snapshot_builder(video_id)
            blocker = _index_blocker(
                index_snapshot,
                benchmark.transcript_language,
            )
            metadata = index_snapshot.get("metadata")
            if (
                isinstance(metadata, dict)
                and metadata.get("video_id") != video_id
            ):
                blocker = "index_video_identity_mismatch"

            if blocker:
                videos.append(
                    {
                        **base,
                        "status": "blocked",
                        "failure_reasons": [blocker],
                        "index_snapshot": index_snapshot,
                        "diagnostics": [],
                        "group_summaries": [],
                    }
                )
                continue

            trace = diagnostic_runner(
                benchmark.video_url,
                question_ids,
                display=False,
            )
            if not isinstance(trace, dict) or not isinstance(
                trace.get("diagnostics"), list
            ):
                raise ValueError("Diagnostic runner returned an invalid trace object.")

            observed_question_ids = [
                item.get("id")
                for item in trace["diagnostics"]
                if isinstance(item, dict)
            ]
            if observed_question_ids != question_ids:
                raise ValueError(
                    "Diagnostic trace question IDs did not exactly match the "
                    f"requested order: expected {question_ids}, got {observed_question_ids}."
                )

            for item in trace["diagnostics"]:
                question_id = item["id"]
                expected_group_count = len(
                    benchmark.gold_evidence_by_question[question_id].groups
                )
                observed_groups = item.get("groups")
                if not isinstance(observed_groups, list) or len(
                    observed_groups
                ) != expected_group_count:
                    raise ValueError(
                        f"Diagnostic trace for {question_id} did not include all "
                        f"gold groups: expected {expected_group_count}."
                    )

            videos.append(
                {
                    **base,
                    "status": "completed",
                    "failure_reasons": [],
                    "index_snapshot": index_snapshot,
                    "diagnostics": trace["diagnostics"],
                    "group_summaries": _group_summaries(trace["diagnostics"]),
                }
            )
        except Exception as error:
            videos.append(
                {
                    **base,
                    "status": "error",
                    "failure_reasons": ["diagnostic_execution_failed"],
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                    "diagnostics": [],
                    "group_summaries": [],
                }
            )

    completed = [item for item in videos if item["status"] == "completed"]
    blocked = [item for item in videos if item["status"] == "blocked"]
    errors = [item for item in videos if item["status"] == "error"]
    return {
        "report_id": REPORT_ID,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "experiment_metadata": provenance_builder(),
        "purpose": (
            "Stage-level retrieval diagnosis only. A completed trace is not an "
            "answer-quality pass, and this report does not change production behavior."
        ),
        "summary": {
            "selected_video_count": len(videos),
            "completed_video_count": len(completed),
            "blocked_video_count": len(blocked),
            "error_video_count": len(errors),
            "diagnosed_question_count": sum(
                len(item["diagnostics"]) for item in completed
            ),
            "diagnosed_group_count": sum(
                len(item["group_summaries"]) for item in completed
            ),
            "diagnostics_completed": len(completed) == len(videos),
        },
        "videos": videos,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase 10.9 retrieval diagnostics for the known failure targets. "
            "This command does not generate chatbot answers or rebuild indexes."
        )
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        choices=sorted(DEFAULT_DIAGNOSTIC_TARGETS),
        help="Optional subset of the configured triage video IDs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "JSON report path. Defaults to a timestamped file under "
            "evaluation_reports/."
        ),
    )
    args = parser.parse_args(argv)

    try:
        report = run_phase10_9_triage(video_ids=args.videos)
    except ValueError as error:
        parser.error(str(error))

    output_path = args.output
    if output_path is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = Path("evaluation_reports") / (
            f"phase10_9_retrieval_triage_{timestamp}.json"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    output_path.write_text(serialized + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "report_path": output_path.as_posix(),
                **report["summary"],
                "interpretation": (
                    "Diagnostics completed; inspect group diagnoses before proposing a fix."
                    if report["summary"]["diagnostics_completed"]
                    else "One or more targets were blocked or errored; resolve that before interpreting quality."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["summary"]["diagnostics_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
