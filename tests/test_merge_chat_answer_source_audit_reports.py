from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.chat_answer_source_audit import (
    PHASE11_0_AUDIT_CASES,
    PHASE11_0_REPORT_ID,
)
from scripts.merge_chat_answer_source_audit_reports import (
    FROZEN_ACCEPTANCE_CONFIGURATION,
    SUBSET_REPORT_ID,
    build_consolidated_report,
)


RUNNER_COMMIT = "10b248c09411166d8c44890c6d39a371ef4fd95b"


def _make_subset_report(cases):
    results = []
    for case in cases:
        results.append(
            {
                "case_id": case["id"],
                "video_id": case["video_id"],
                "status": "success",
                "response_body_sha256": f"hash-{case['id']}",
                "audit": {
                    "effective_retrieval_config": dict(
                        FROZEN_ACCEPTANCE_CONFIGURATION
                    ),
                    "quality": {
                        "passed": True,
                        "failed_checks": [],
                    },
                },
            }
        )

    return {
        "report_schema_version": "1.2.0",
        "report_id": SUBSET_REPORT_ID,
        "generated_at": "2026-11-05T20:00:00+05:30",
        "backend_base_url": "http://127.0.0.1:8000",
        "runner_git_commit": RUNNER_COMMIT,
        "runner_working_tree_clean": True,
        "live_suite_case_ids": [case["id"] for case in cases],
        "expected_server_configuration": dict(FROZEN_ACCEPTANCE_CONFIGURATION),
        "effective_server_configuration": dict(FROZEN_ACCEPTANCE_CONFIGURATION),
        "effective_server_configuration_consistent_across_cases": True,
        "effective_server_configuration_matches_expected": True,
        "cases_attempted": len(results),
        "cases_succeeded": len(results),
        "cases_failed": 0,
        "quality_passed": True,
        "quality_failures": [],
        "results": results,
    }


def test_consolidates_all_43_cases_across_multiple_batches():
    cases = list(PHASE11_0_AUDIT_CASES)
    batches = [
        cases[0:5],
        cases[5:10],
        cases[10:15],
        cases[15:20],
        cases[20:25],
        cases[25:30],
        cases[30:35],
        cases[35:40],
        cases[40:43],
    ]
    reports = [_make_subset_report(batch) for batch in batches]

    result = build_consolidated_report(
        reports,
        source_names=[f"batch-{index + 1}.json" for index in range(len(reports))],
        expected_runner_commit=RUNNER_COMMIT,
    )

    assert result["report_id"] == (
        "phase11_6_resumable_full_registry_answer_source_audit_v1"
    )
    assert result["cases_attempted"] == 43
    assert result["cases_succeeded"] == 43
    assert result["cases_failed"] == 0
    assert result["full_registry_coverage_passed"] is True
    assert result["representative_five_video_coverage_passed"] is True
    assert result["effective_server_configuration_consistent_across_cases"] is True
    assert result["effective_server_configuration_matches_expected"] is True
    assert result["quality_passed"] is True
    assert result["live_suite_case_ids"] == [
        case["id"] for case in PHASE11_0_AUDIT_CASES
    ]
    assert result["batch_consolidation"]["recommended_retry_case_ids"] == []


def test_partial_batches_report_missing_ids_and_do_not_pass_release_gate():
    cases = list(PHASE11_0_AUDIT_CASES)
    reports = [
        _make_subset_report(cases[:10]),
        _make_subset_report(cases[10:25]),
    ]

    result = build_consolidated_report(reports)

    expected_missing = [case["id"] for case in cases[25:]]
    assert result["cases_attempted"] == 25
    assert result["full_registry_coverage_passed"] is False
    assert result["quality_passed"] is False
    assert result["batch_consolidation"]["missing_case_ids"] == expected_missing
    assert result["batch_consolidation"]["recommended_retry_case_ids"] == (
        expected_missing
    )


def test_transport_failure_is_retryable_but_not_counted_as_a_case_success():
    cases = list(PHASE11_0_AUDIT_CASES[:2])
    failed_report = _make_subset_report(cases)
    failed_report["results"][0]["status"] = "failed"
    failed_report["results"][0].pop("audit")
    failed_report["cases_succeeded"] = 1
    failed_report["cases_failed"] = 1
    failed_report["effective_server_configuration_consistent_across_cases"] = False

    result = build_consolidated_report([failed_report])

    assert result["cases_succeeded"] == 1
    assert result["cases_failed"] == 1
    assert result["effective_server_configuration_consistent_across_cases"] is False
    assert result["batch_consolidation"]["transport_failed_case_ids"] == [
        cases[0]["id"]
    ]
    assert result["batch_consolidation"]["recommended_retry_case_ids"] == [
        cases[0]["id"],
        *[case["id"] for case in PHASE11_0_AUDIT_CASES[2:]],
    ]


def test_quality_failure_requires_review_not_blind_provider_retry():
    cases = list(PHASE11_0_AUDIT_CASES[:1])
    report = _make_subset_report(cases)
    report["results"][0]["audit"]["quality"] = {
        "passed": False,
        "failed_checks": ["answer_requirement_example"],
    }

    result = build_consolidated_report([report])

    assert result["cases_succeeded"] == 1
    assert result["cases_failed"] == 0
    assert result["batch_consolidation"]["quality_failed_case_ids"] == [
        cases[0]["id"]
    ]
    assert cases[0]["id"] not in result["batch_consolidation"][
        "recommended_retry_case_ids"
    ]
    assert result["batch_consolidation"]["manual_review_case_ids"] == [
        cases[0]["id"]
    ]


def test_later_success_can_replace_a_prior_transport_failure():
    case = PHASE11_0_AUDIT_CASES[0]
    failed = _make_subset_report([case])
    failed["results"][0]["status"] = "failed"
    failed["results"][0].pop("audit")
    failed["cases_succeeded"] = 0
    failed["cases_failed"] = 1
    failed["effective_server_configuration"] = None
    failed["effective_server_configuration_consistent_across_cases"] = False
    failed["effective_server_configuration_matches_expected"] = False

    retried = _make_subset_report([case])
    result = build_consolidated_report(
        [failed, retried],
        source_names=["first-attempt.json", "retry.json"],
    )

    assert result["cases_succeeded"] == 1
    assert result["cases_failed"] == 0
    assert result["results"][0]["status"] == "success"
    assert result["batch_consolidation"]["replaced_failed_attempts"] == [
        {
            "case_id": case["id"],
            "previous_source_report": "first-attempt.json",
            "replacement_source_report": "retry.json",
        }
    ]


def test_multiple_successful_attempts_for_same_case_are_rejected():
    case = PHASE11_0_AUDIT_CASES[0]
    with pytest.raises(ValueError, match="multiple HTTP-success attempts"):
        build_consolidated_report(
            [_make_subset_report([case]), _make_subset_report([case])]
        )


def test_configuration_drift_is_rejected():
    case = PHASE11_0_AUDIT_CASES[0]
    report = _make_subset_report([case])
    report["expected_server_configuration"]["top_k"] = 9

    with pytest.raises(ValueError, match="frozen Phase 10.11 acceptance"):
        build_consolidated_report([report])


def test_mixed_runner_commits_are_rejected():
    cases = list(PHASE11_0_AUDIT_CASES[:2])
    first = _make_subset_report(cases[:1])
    second = _make_subset_report(cases[1:])
    second["runner_git_commit"] = "a" * 40

    with pytest.raises(ValueError, match="different code versions"):
        build_consolidated_report([first, second])


def test_dirty_runner_report_is_rejected():
    case = PHASE11_0_AUDIT_CASES[0]
    report = _make_subset_report([case])
    report["runner_working_tree_clean"] = False

    with pytest.raises(ValueError, match="not confirmed clean"):
        build_consolidated_report([report])


def test_expected_runner_commit_is_enforced():
    case = PHASE11_0_AUDIT_CASES[0]
    report = _make_subset_report([case])

    with pytest.raises(ValueError, match="does not match required commit"):
        build_consolidated_report(
            [report],
            expected_runner_commit="b" * 40,
        )


def test_case_video_identity_must_match_the_canonical_registry():
    case = PHASE11_0_AUDIT_CASES[0]
    report = _make_subset_report([case])
    report["results"][0]["video_id"] = "different-video"

    with pytest.raises(ValueError, match="video_id mismatch"):
        build_consolidated_report([report])


def test_duplicate_case_ids_inside_one_source_report_are_rejected():
    case = PHASE11_0_AUDIT_CASES[0]
    report = _make_subset_report([case])
    report["live_suite_case_ids"].append(case["id"])
    report["results"].append(deepcopy(report["results"][0]))
    report["cases_attempted"] = 2
    report["cases_succeeded"] = 2

    with pytest.raises(ValueError, match="duplicate case IDs within one report"):
        build_consolidated_report([report])


def test_report_suite_id_must_be_the_phase_11_0_subset():
    case = PHASE11_0_AUDIT_CASES[0]
    report = _make_subset_report([case])
    report["report_id"] = PHASE11_0_REPORT_ID

    with pytest.raises(ValueError, match="subset report"):
        build_consolidated_report([report])
