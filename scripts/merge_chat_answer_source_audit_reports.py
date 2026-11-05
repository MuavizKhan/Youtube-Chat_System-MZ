"""Consolidate resumable Phase 11.0 full-registry live answer/source audits.

Subset runs intentionally disable the full-registry release gate. This tool
combines saved subset reports, validates their provenance and frozen
configuration, and recomputes the complete 43-case registry acceptance gate.
It never calls the backend or an inference provider.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from scripts.chat_answer_source_audit import (
    PHASE11_0_AUDIT_CASES,
    PHASE11_0_REPORT_ID,
    build_coverage_manifest,
)


SOURCE_REPORT_SCHEMA_VERSION = "1.2.0"
CONSOLIDATED_REPORT_SCHEMA_VERSION = "1.0.0"
CONSOLIDATED_REPORT_ID = (
    "phase11_6_resumable_full_registry_answer_source_audit_v1"
)
SUBSET_REPORT_ID = f"{PHASE11_0_REPORT_ID}_subset"

FROZEN_ACCEPTANCE_CONFIGURATION: dict[str, Any] = {
    "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "index_namespace": "phase10-10-acceptance-minilm-20261023",
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

_CASES_BY_ID = {str(case["id"]): case for case in PHASE11_0_AUDIT_CASES}
_CASE_ORDER = {
    str(case["id"]): index
    for index, case in enumerate(PHASE11_0_AUDIT_CASES)
}
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _require_mapping(value: Any, field: str, source_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{source_name}: {field} must be a JSON object.")
    return value


def _case_quality_passed(result: dict[str, Any]) -> bool:
    if result.get("status") != "success":
        return False
    audit = result.get("audit")
    if not isinstance(audit, dict):
        return False
    quality = audit.get("quality")
    return isinstance(quality, dict) and quality.get("passed") is True


def _validate_subset_report(
    report: dict[str, Any],
    source_name: str,
    expected_runner_commit: str | None,
) -> tuple[str, str, list[dict[str, Any]]]:
    """Validate one report before accepting any of its case results."""
    if report.get("report_schema_version") != SOURCE_REPORT_SCHEMA_VERSION:
        raise ValueError(
            f"{source_name}: expected source schema "
            f"{SOURCE_REPORT_SCHEMA_VERSION!r}."
        )
    if report.get("report_id") != SUBSET_REPORT_ID:
        raise ValueError(
            f"{source_name}: expected a Phase 11.0 subset report "
            f"({SUBSET_REPORT_ID!r})."
        )

    if report.get("expected_server_configuration") != FROZEN_ACCEPTANCE_CONFIGURATION:
        raise ValueError(
            f"{source_name}: expected_server_configuration does not match "
            "the frozen Phase 10.11 acceptance configuration."
        )

    effective = report.get("effective_server_configuration")
    if effective is not None and effective != FROZEN_ACCEPTANCE_CONFIGURATION:
        raise ValueError(
            f"{source_name}: effective_server_configuration differs from "
            "the frozen acceptance configuration."
        )

    if report.get("runner_working_tree_clean") is not True:
        raise ValueError(
            f"{source_name}: runner working tree was not confirmed clean."
        )

    commit = report.get("runner_git_commit")
    if not isinstance(commit, str) or not _COMMIT_PATTERN.fullmatch(commit):
        raise ValueError(
            f"{source_name}: runner_git_commit must be a full 40-character SHA."
        )
    if expected_runner_commit is not None and commit != expected_runner_commit:
        raise ValueError(
            f"{source_name}: runner commit {commit} does not match required "
            f"commit {expected_runner_commit}."
        )

    base_url = report.get("backend_base_url")
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError(f"{source_name}: backend_base_url is missing.")

    requested_ids = report.get("live_suite_case_ids")
    results = report.get("results")
    if not isinstance(requested_ids, list) or not requested_ids:
        raise ValueError(f"{source_name}: live_suite_case_ids must be non-empty.")
    if not isinstance(results, list) or not results:
        raise ValueError(f"{source_name}: results must be a non-empty array.")
    if any(not isinstance(case_id, str) for case_id in requested_ids):
        raise ValueError(f"{source_name}: case IDs must be strings.")
    if len(requested_ids) != len(set(requested_ids)):
        raise ValueError(f"{source_name}: duplicate case IDs within one report.")

    result_ids = [
        result.get("case_id") if isinstance(result, dict) else None
        for result in results
    ]
    if result_ids != requested_ids:
        raise ValueError(
            f"{source_name}: results must match live_suite_case_ids exactly and "
            "in the same order."
        )

    actual_succeeded = 0
    for result in results:
        case_id = result["case_id"]
        case_definition = _CASES_BY_ID.get(case_id)
        if case_definition is None:
            raise ValueError(
                f"{source_name}: {case_id!r} is not part of the canonical "
                "43-case registry suite."
            )
        if result.get("video_id") != case_definition.get("video_id"):
            raise ValueError(
                f"{source_name}: video_id mismatch for case {case_id}."
            )
        if not isinstance(result.get("status"), str):
            raise ValueError(
                f"{source_name}: result status is missing for case {case_id}."
            )
        if result.get("status") == "success":
            actual_succeeded += 1
            audit = _require_mapping(result.get("audit"), "audit", source_name)
            effective_case_config = audit.get("effective_retrieval_config")
            if (
                effective_case_config is not None
                and effective_case_config != FROZEN_ACCEPTANCE_CONFIGURATION
            ):
                raise ValueError(
                    f"{source_name}: effective retrieval configuration drift "
                    f"for case {case_id}."
                )

    if report.get("cases_attempted") != len(results):
        raise ValueError(f"{source_name}: cases_attempted does not match results.")
    if report.get("cases_succeeded") != actual_succeeded:
        raise ValueError(f"{source_name}: cases_succeeded does not match results.")
    if report.get("cases_failed") != len(results) - actual_succeeded:
        raise ValueError(f"{source_name}: cases_failed does not match results.")

    return commit, base_url.rstrip("/"), results


def build_consolidated_report(
    source_reports: Sequence[dict[str, Any]],
    *,
    source_names: Sequence[str] | None = None,
    source_sha256: Sequence[str] | None = None,
    expected_runner_commit: str | None = None,
) -> dict[str, Any]:
    """Build a registry-wide report from validated, independently saved batches.

    Partial inputs are allowed so the output can disclose missing cases and
    produce a retry plan. Acceptance remains false until all 43 cases have
    passed transport and answer/source quality checks.
    """
    if isinstance(source_reports, (str, bytes)) or not isinstance(
        source_reports, Sequence
    ):
        raise TypeError("source_reports must be a sequence of report objects.")
    if not source_reports:
        raise ValueError("At least one source report is required.")

    names = (
        [f"batch-{index + 1}.json" for index in range(len(source_reports))]
        if source_names is None
        else list(source_names)
    )
    if len(names) != len(source_reports):
        raise ValueError("source_names must have one entry per source report.")

    hashes = (
        [
            hashlib.sha256(
                json.dumps(report, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            for report in source_reports
        ]
        if source_sha256 is None
        else list(source_sha256)
    )
    if len(hashes) != len(source_reports):
        raise ValueError("source_sha256 must have one entry per source report.")

    runner_commit: str | None = None
    backend_base_url: str | None = None
    selected_results: dict[str, dict[str, Any]] = {}
    selected_source: dict[str, str] = {}
    attempt_history: dict[str, list[dict[str, Any]]] = {}
    replaced_failed_attempts: list[dict[str, str]] = []
    retained_success_over_later_failure: list[dict[str, str]] = []
    batch_summaries = []

    for report, source_name, digest in zip(
        source_reports, names, hashes, strict=True
    ):
        _require_mapping(report, "report", source_name)
        commit, base_url, results = _validate_subset_report(
            report,
            source_name,
            expected_runner_commit,
        )
        if runner_commit is None:
            runner_commit = commit
        elif runner_commit != commit:
            raise ValueError(
                f"{source_name}: runner commit differs from earlier batches. "
                "Do not consolidate reports produced by different code versions."
            )
        if backend_base_url is None:
            backend_base_url = base_url
        elif backend_base_url != base_url:
            raise ValueError(
                f"{source_name}: backend_base_url differs from earlier batches."
            )

        batch_summaries.append(
            {
                "file_name": Path(source_name).name,
                "sha256": digest,
                "generated_at": report.get("generated_at"),
                "case_ids": [result["case_id"] for result in results],
            }
        )

        for result in results:
            case_id = result["case_id"]
            status = result["status"]
            attempt_history.setdefault(case_id, []).append(
                {
                    "source_report": Path(source_name).name,
                    "status": status,
                    "response_body_sha256": result.get("response_body_sha256"),
                    "quality_passed": _case_quality_passed(result),
                }
            )

            previous = selected_results.get(case_id)
            if previous is None:
                selected_results[case_id] = result
                selected_source[case_id] = Path(source_name).name
                continue

            previous_status = previous.get("status")
            previous_source = selected_source[case_id]
            if previous_status == "success" and status == "success":
                raise ValueError(
                    f"Case {case_id} has multiple HTTP-success attempts "
                    f"({previous_source}, {Path(source_name).name}). Resolve the "
                    "answer-quality difference explicitly; the consolidator will "
                    "not select a more favorable stochastic answer."
                )
            if previous_status != "success" and status == "success":
                selected_results[case_id] = result
                selected_source[case_id] = Path(source_name).name
                replaced_failed_attempts.append(
                    {
                        "case_id": case_id,
                        "previous_source_report": previous_source,
                        "replacement_source_report": Path(source_name).name,
                    }
                )
            elif previous_status == "success" and status != "success":
                retained_success_over_later_failure.append(
                    {
                        "case_id": case_id,
                        "retained_source_report": previous_source,
                        "ignored_later_failure_report": Path(source_name).name,
                    }
                )
            else:
                selected_results[case_id] = result
                selected_source[case_id] = Path(source_name).name

    ordered_ids = sorted(selected_results, key=lambda case_id: _CASE_ORDER[case_id])
    ordered_results = [selected_results[case_id] for case_id in ordered_ids]
    ordered_definitions = [_CASES_BY_ID[case_id] for case_id in ordered_ids]

    effective_configs = []
    for result in ordered_results:
        audit = result.get("audit")
        if isinstance(audit, dict):
            case_config = audit.get("effective_retrieval_config")
            if isinstance(case_config, dict):
                effective_configs.append(case_config)

    effective_config = effective_configs[0] if effective_configs else None
    configurations_consistent = (
        len(effective_configs) == len(ordered_results)
        and bool(effective_configs)
        and all(config == effective_configs[0] for config in effective_configs)
    )
    configuration_matches_expected = (
        bool(effective_configs)
        and all(
            config == FROZEN_ACCEPTANCE_CONFIGURATION
            for config in effective_configs
        )
    )

    cases_succeeded = sum(
        result.get("status") == "success" for result in ordered_results
    )
    cases_failed = len(ordered_results) - cases_succeeded

    transport_failed_ids = [
        result["case_id"]
        for result in ordered_results
        if result.get("status") != "success"
    ]
    quality_failed_ids = [
        result["case_id"]
        for result in ordered_results
        if result.get("status") == "success"
        and not _case_quality_passed(result)
    ]
    missing_ids = [
        str(case["id"])
        for case in PHASE11_0_AUDIT_CASES
        if str(case["id"]) not in selected_results
    ]

    retry_id_set = set(missing_ids) | set(transport_failed_ids)
    recommended_retry_ids = [
        str(case["id"])
        for case in PHASE11_0_AUDIT_CASES
        if str(case["id"]) in retry_id_set
    ]

    quality_failures = []
    for result in ordered_results:
        case_id = result["case_id"]
        audit = result.get("audit")
        quality = audit.get("quality") if isinstance(audit, dict) else None
        if result.get("status") != "success":
            failed_checks = (
                list(quality.get("failed_checks") or [])
                if isinstance(quality, dict)
                else []
            )
            if not failed_checks:
                failed_checks = ["http_or_transport_failure"]
            quality_failures.append(
                {"case_id": case_id, "failed_checks": failed_checks}
            )
        elif not _case_quality_passed(result):
            failed_checks = (
                list(quality.get("failed_checks") or [])
                if isinstance(quality, dict)
                else []
            )
            if not failed_checks:
                failed_checks = ["answer_source_quality_not_passed"]
            quality_failures.append(
                {"case_id": case_id, "failed_checks": failed_checks}
            )

    if not configurations_consistent:
        quality_failures.append(
            {
                "case_id": "ALL",
                "failed_checks": ["effective_server_configuration_consistency"],
            }
        )
    if not configuration_matches_expected:
        quality_failures.append(
            {
                "case_id": "ALL",
                "failed_checks": ["effective_server_configuration_matches_expected"],
            }
        )

    coverage_manifest = build_coverage_manifest(ordered_definitions)
    registry_accounting = coverage_manifest["question_case_accounting"]
    full_registry_coverage_passed = (
        coverage_manifest["unregistered_live_cases"] == []
        and coverage_manifest["duplicate_live_cases"] == []
        and registry_accounting["accounting_complete"]
        and registry_accounting["live_audited_count"]
        == coverage_manifest["registered_question_case_count"]
        and registry_accounting["not_live_audited_count"] == 0
    )
    representative_coverage_passed = (
        coverage_manifest[
            "all_registered_videos_have_answerable_and_unanswerable_live_cases"
        ]
        and coverage_manifest["unregistered_live_cases"] == []
        and coverage_manifest["duplicate_live_cases"] == []
        and registry_accounting["accounting_complete"]
    )
    if not representative_coverage_passed:
        quality_failures.append(
            {
                "case_id": "ALL",
                "failed_checks": ["representative_five_video_live_coverage"],
            }
        )
    if not full_registry_coverage_passed:
        quality_failures.append(
            {
                "case_id": "ALL",
                "failed_checks": ["all_registered_question_cases_live_audited"],
            }
        )

    quality_passed = (
        full_registry_coverage_passed
        and representative_coverage_passed
        and cases_succeeded == len(PHASE11_0_AUDIT_CASES)
        and cases_failed == 0
        and configurations_consistent
        and configuration_matches_expected
        and not quality_failures
    )

    return {
        "report_schema_version": CONSOLIDATED_REPORT_SCHEMA_VERSION,
        "report_id": CONSOLIDATED_REPORT_ID,
        "generated_at": datetime.now().astimezone().isoformat(),
        "source_report_schema_version": SOURCE_REPORT_SCHEMA_VERSION,
        "backend_base_url": backend_base_url,
        "runner_git_commit": runner_commit,
        "runner_working_tree_clean": True,
        "expected_server_configuration": dict(FROZEN_ACCEPTANCE_CONFIGURATION),
        "effective_server_configuration": effective_config,
        "effective_server_configuration_consistent_across_cases": (
            configurations_consistent
        ),
        "effective_server_configuration_matches_expected": (
            configuration_matches_expected
        ),
        "coverage_manifest": coverage_manifest,
        "representative_five_video_coverage_passed": representative_coverage_passed,
        "full_registry_coverage_passed": full_registry_coverage_passed,
        "live_suite_case_ids": ordered_ids,
        "cases_attempted": len(ordered_results),
        "cases_succeeded": cases_succeeded,
        "cases_failed": cases_failed,
        "quality_passed": quality_passed,
        "quality_failures": quality_failures,
        "results": ordered_results,
        "batch_consolidation": {
            "source_report_count": len(batch_summaries),
            "source_reports": batch_summaries,
            "case_attempt_count": sum(
                len(item["case_ids"]) for item in batch_summaries
            ),
            "case_attempt_history": attempt_history,
            "replaced_failed_attempts": replaced_failed_attempts,
            "retained_success_over_later_failure": (
                retained_success_over_later_failure
            ),
            "missing_case_ids": missing_ids,
            "transport_failed_case_ids": transport_failed_ids,
            "quality_failed_case_ids": quality_failed_ids,
            "recommended_retry_case_ids": recommended_retry_ids,
            "manual_review_case_ids": quality_failed_ids,
            "runner_git_commit": runner_commit,
            "duplicate_policy": (
                "A later success may replace an earlier transport failure. "
                "A later transport failure cannot replace an earlier success. "
                "Multiple successful attempts for one case are rejected."
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Consolidate Phase 11.0 subset reports into one auditable "
            "43-case live answer/source acceptance report."
        )
    )
    parser.add_argument(
        "--reports",
        nargs="+",
        required=True,
        type=Path,
        help="One or more JSON reports produced by --suite phase11_0 --case-ids.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination path for the consolidated JSON report.",
    )
    parser.add_argument(
        "--expected-runner-commit",
        default=None,
        help="Optional exact 40-character commit required for every batch.",
    )
    args = parser.parse_args(argv)

    if args.expected_runner_commit is not None and not _COMMIT_PATTERN.fullmatch(
        args.expected_runner_commit
    ):
        parser.error("--expected-runner-commit must be a full 40-character SHA.")

    try:
        output_resolved = args.output.resolve()
        reports = []
        names = []
        hashes = []
        for report_path in args.reports:
            if report_path.resolve() == output_resolved:
                raise ValueError("The output path must not overwrite a source report.")
            raw = report_path.read_bytes()
            report = json.loads(raw.decode("utf-8"))
            if not isinstance(report, dict):
                raise ValueError(f"{report_path}: report root must be an object.")
            reports.append(report)
            names.append(report_path.name)
            hashes.append(hashlib.sha256(raw).hexdigest())

        consolidated = build_consolidated_report(
            reports,
            source_names=names,
            source_sha256=hashes,
            expected_runner_commit=args.expected_runner_commit,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(consolidated, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))

    summary = {
        "report_path": str(args.output),
        "report_id": consolidated["report_id"],
        "cases_attempted": consolidated["cases_attempted"],
        "cases_succeeded": consolidated["cases_succeeded"],
        "cases_failed": consolidated["cases_failed"],
        "full_registry_coverage_passed": consolidated[
            "full_registry_coverage_passed"
        ],
        "quality_passed": consolidated["quality_passed"],
        "missing_case_ids": consolidated["batch_consolidation"][
            "missing_case_ids"
        ],
        "transport_failed_case_ids": consolidated["batch_consolidation"][
            "transport_failed_case_ids"
        ],
        "quality_failed_case_ids": consolidated["batch_consolidation"][
            "quality_failed_case_ids"
        ],
        "recommended_retry_case_ids": consolidated["batch_consolidation"][
            "recommended_retry_case_ids"
        ],
        "manual_review_case_ids": consolidated["batch_consolidation"][
            "manual_review_case_ids"
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if consolidated["quality_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
