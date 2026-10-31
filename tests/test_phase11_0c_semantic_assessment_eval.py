"""Regression tests for the isolated Phase 11.0C assessment evaluation runner."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from Backend.rag_system.evidence_assessment import AssessmentRunStatus
from Backend.rag_system.evidence_contracts import EvidenceReference, EvidenceSupportStatus
from scripts.phase11_0c_semantic_assessment_eval import (
    DEFAULT_DATASET,
    ThresholdLockError,
    _quality_gate_eligibility,
    _wilson_interval,
    build_metrics,
    load_manifest,
    load_threshold_lock,
    run_evaluation,
    validate_manifest,
)


def _load_seed():
    return load_manifest(DEFAULT_DATASET)


def test_seed_manifest_is_balanced_and_split_disjoint():
    manifest, digest = _load_seed()

    assert len(digest) == 64
    assert len(manifest["cases"]) == 24
    for split in ("development", "held_out"):
        cases = [case for case in manifest["cases"] if case["split"] == split]
        assert len(cases) == 12
        assert {case["language_group"] for case in cases} == {"en", "hi", "ar"}
        assert {
            label["expected_status"]
            for case in cases
            for label in case["labels"]
        } == {
            "supported", "partial", "missing",
            "contradicted", "conflicted", "unknown",
        }


def test_manifest_rejects_independence_group_leakage_between_splits():
    manifest, _ = _load_seed()
    changed = deepcopy(manifest)
    changed["cases"][-1]["independence_group"] = (
        changed["cases"][0]["independence_group"]
    )

    with pytest.raises(ValueError, match="crosses splits"):
        validate_manifest(changed)


def test_manifest_rejects_video_reuse_across_splits():
    manifest, _ = _load_seed()
    changed = deepcopy(manifest)
    changed["cases"][-1]["question_analysis"]["video_id"] = (
        changed["cases"][0]["question_analysis"]["video_id"]
    )
    changed["cases"][-1]["passages"][0]["references"][0]["video_id"] = (
        changed["cases"][0]["question_analysis"]["video_id"]
    )

    with pytest.raises(ValueError, match="occurs in multiple splits"):
        validate_manifest(changed)


def test_manifest_rejects_unknown_gold_passage_reference():
    manifest, _ = _load_seed()
    changed = deepcopy(manifest)
    label = changed["cases"][0]["labels"][0]
    label["supporting_passage_ids"] = ["passage-not-in-case"]

    with pytest.raises(ValueError, match="absent from the case"):
        validate_manifest(changed)


def test_manifest_rejects_status_and_reference_inconsistency():
    manifest, _ = _load_seed()
    changed = deepcopy(manifest)
    label = changed["cases"][0]["labels"][0]
    label["supporting_passage_ids"] = []
    label["expected_status"] = "supported"

    with pytest.raises(ValueError, match="inconsistent with supported"):
        validate_manifest(changed)


def test_manifest_requires_every_aspect_to_have_one_label():
    manifest, _ = _load_seed()
    changed = deepcopy(manifest)
    changed["cases"][0]["labels"] = []

    with pytest.raises(ValueError, match="exactly one reference label"):
        validate_manifest(changed)


def test_class_metrics_produce_complete_six_way_confusion_matrix():
    statuses = (
        "supported", "partial", "missing",
        "contradicted", "conflicted", "unknown",
    )
    rows = [
        {
            "expected_status": status,
            "predicted_status": status,
            "case_id": f"case-{index}",
            "language_group": "en",
            "expected_supporting_passage_ids": [],
            "predicted_supporting_passage_ids": [],
            "expected_contradicting_passage_ids": [],
            "predicted_contradicting_passage_ids": [],
        }
        for index, status in enumerate(statuses)
    ]

    result = build_metrics([
        {
            "case_id": row["case_id"],
            "language_group": row["language_group"],
            "assessment_run_status": "completed",
            "request_count": 1,
            "elapsed_ms": 10.0,
            "input_tokens": 40,
            "output_tokens": 20,
            "error_code": None,
            "assessments": [row],
        }
        for row in rows
    ])

    assert result["semantic"]["observations"] == 6
    assert result["semantic"]["accuracy"] == 1.0
    assert result["semantic"]["macro_f1"] == 1.0
    assert all(
        result["semantic"]["confusion_matrix"][status][status] == 1
        for status in statuses
    )


def test_supported_binary_metrics_count_false_supported_predictions():
    rows = [
        {"expected_status": "supported", "predicted_status": "supported"},
        {"expected_status": "missing", "predicted_status": "supported"},
        {"expected_status": "contradicted", "predicted_status": "partial"},
        {"expected_status": "conflicted", "predicted_status": "contradicted"},
    ]
    from scripts.phase11_0c_semantic_assessment_eval import _binary_supported_metrics

    metrics = _binary_supported_metrics(rows)
    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 0
    assert metrics["true_negative"] == 2
    assert metrics["supported_precision"] == 0.5
    assert metrics["unsupported_specificity"] == pytest.approx(2 / 3)
    assert metrics["false_supported_rate_on_unsupported"] == pytest.approx(1 / 3)
    assert metrics["supported_precision_wilson_95"]["lower_95"] < 0.5


def test_wilson_interval_handles_zero_and_nonzero_denominators():
    assert _wilson_interval(0, 0) is None
    interval = _wilson_interval(10, 10)
    assert interval is not None
    assert 0.0 < interval["lower_95"] < 1.0
    assert interval["upper_95"] == pytest.approx(1.0)


def test_provisional_synthetic_labels_never_qualify_for_heldout_gate():
    manifest, _ = _load_seed()
    held_out = [case for case in manifest["cases"] if case["split"] == "held_out"]
    records = [
        {
            "case_id": case["case_id"],
            "language_group": case["language_group"],
            "assessments": [
                {
                    "expected_status": case["labels"][0]["expected_status"],
                    "predicted_status": case["labels"][0]["expected_status"],
                }
            ],
        }
        for case in held_out
    ]

    result = _quality_gate_eligibility(
        manifest, "held_out", "completed", records
    )
    assert result["eligible"] is False
    assert "labels_are_not_independently_adjudicated" in result["reasons"]
    assert "held_out_aspects_below_minimum_100" in result["reasons"]


def test_threshold_lock_requires_matching_completed_development_report(tmp_path):
    manifest, dataset_sha256 = _load_seed()
    development_report = {
        "split": "development",
        "run_status": "completed",
        "dataset_id": manifest["dataset_id"],
        "dataset_sha256": dataset_sha256,
        "minimum_support_score": 0.75,
        "annotation_status": "adjudicated",
    }
    report_path = tmp_path / "development.json"
    report_path.write_text(json.dumps(development_report), encoding="utf-8")
    report_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()

    lock = {
        "schema_version": "1.0.0",
        "dataset_id": manifest["dataset_id"],
        "dataset_sha256": dataset_sha256,
        "minimum_support_score": 0.75,
        "annotation_status": "adjudicated",
        "development_report_path": "development.json",
        "development_report_sha256": report_sha,
        "selected_by": "reviewer-1",
        "selection_rationale": "Selected after reviewing development-only metrics.",
    }
    lock_path = tmp_path / "threshold-lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    threshold, metadata = load_threshold_lock(
        lock_path,
        dataset_id=manifest["dataset_id"],
        dataset_sha256=dataset_sha256,
        annotation_status="adjudicated",
    )
    assert threshold == 0.75
    assert metadata["development_report_sha256"] == report_sha


def test_threshold_lock_rejects_development_report_hash_drift(tmp_path):
    manifest, dataset_sha256 = _load_seed()
    report_path = tmp_path / "development.json"
    report_path.write_text(json.dumps({
        "split": "development",
        "run_status": "completed",
        "dataset_id": manifest["dataset_id"],
        "dataset_sha256": dataset_sha256,
        "minimum_support_score": 0.75,
        "annotation_status": "adjudicated",
    }), encoding="utf-8")
    report_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()
    lock_path = tmp_path / "threshold-lock.json"
    lock_path.write_text(json.dumps({
        "schema_version": "1.0.0",
        "dataset_id": manifest["dataset_id"],
        "dataset_sha256": dataset_sha256,
        "minimum_support_score": 0.75,
        "annotation_status": "adjudicated",
        "development_report_path": "development.json",
        "development_report_sha256": report_sha,
        "selected_by": "reviewer-1",
        "selection_rationale": "Development-only selection.",
    }), encoding="utf-8")
    report_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ThresholdLockError, match="hash"):
        load_threshold_lock(
            lock_path,
            dataset_id=manifest["dataset_id"],
            dataset_sha256=dataset_sha256,
            annotation_status="adjudicated",
        )


def test_threshold_lock_rejects_provisional_annotation_status(tmp_path):
    manifest, dataset_sha256 = _load_seed()
    lock_path = tmp_path / "lock.json"
    lock_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ThresholdLockError, match="human labels"):
        load_threshold_lock(
            lock_path,
            dataset_id=manifest["dataset_id"],
            dataset_sha256=dataset_sha256,
            annotation_status="provisional_synthetic",
        )


def test_heldout_eligibility_rejects_failed_assessment_even_if_every_case_was_attempted():
    manifest, _ = _load_seed()
    held_out = [case for case in manifest["cases"] if case["split"] == "held_out"]
    records = []
    for case in held_out:
        label = case["labels"][0]
        records.append({
            "case_id": case["case_id"],
            "language_group": case["language_group"],
            "assessment_run_status": "completed",
            "assessments": [{
                "expected_status": label["expected_status"],
                "predicted_status": label["expected_status"],
            }],
        })
    records[0]["assessment_run_status"] = "provider_error"

    result = _quality_gate_eligibility(manifest, "held_out", "completed", records)
    assert result["eligible"] is False
    assert (
        "one_or_more_held_out_assessment_calls_failed_or_returned_invalid_output"
        in result["reasons"]
    )


def test_runner_checkpoints_sanitized_partial_report_without_transcript_text(tmp_path):
    manifest, dataset_sha256 = _load_seed()
    case = manifest["cases"][0]
    passage = case["passages"][0]
    reference = passage["references"][0]
    aspect_id = case["question_analysis"]["aspects"][0]["aspect_id"]

    def fake_assessor(analysis, passages, client, **kwargs):
        assert kwargs["minimum_support_score"] == 0.75
        return SimpleNamespace(
            diagnostics=SimpleNamespace(
                status=AssessmentRunStatus.COMPLETED,
                provider="groq",
                model="test-model",
                video_id=analysis.video_id,
                request_count=1,
                elapsed_ms=5.0,
                input_chars=500,
                input_limit_chars=8000,
                passage_count=len(passages),
                aspect_count=len(analysis.aspects),
                error_code=None,
                input_tokens=100,
                output_tokens=30,
            ),
            assessments=(
                SimpleNamespace(
                    aspect_id=aspect_id,
                    status=EvidenceSupportStatus.SUPPORTED,
                    support_score=0.91,
                    supporting_evidence=(
                        EvidenceReference(
                            video_id=reference["video_id"],
                            chunk_id=reference["chunk_id"],
                            start_seconds=reference["start_seconds"],
                            end_seconds=reference["end_seconds"],
                        ),
                    ),
                    contradicting_evidence=(),
                ),
            ),
        )

    output_path = tmp_path / "partial-report.json"
    report = run_evaluation(
        manifest=manifest,
        dataset_sha256=dataset_sha256,
        split="development",
        client=object(),
        provider="groq",
        model="test-model",
        minimum_support_score=0.75,
        output_path=output_path,
        max_cases=1,
        sleep_seconds=0,
        assessor=fake_assessor,
    )

    serialized = output_path.read_text(encoding="utf-8")
    assert report["run_status"] == "partial"
    assert report["stop_reason"] == "max_cases_limit_reached_partial_run"
    assert len(report["skipped_case_ids"]) == 11
    assert report["metrics"]["semantic"]["observations"] == 1
    assert passage["text"] not in serialized
    assert "Transcript text" not in serialized


def test_runner_script_supports_documented_direct_help_invocation():
    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "scripts" / "phase11_0c_semantic_assessment_eval.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--threshold-lock" in completed.stdout
    assert "--allow-provisional-labels" in completed.stdout
    assert "--split" in completed.stdout
