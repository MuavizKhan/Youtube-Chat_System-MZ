from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import phase11_0d_annotation_workflow as workflow
from scripts.phase11_0c_semantic_assessment_eval import DEFAULT_DATASET, validate_manifest


def _source_fixture() -> tuple[dict, dict[str, list[dict]]]:
    seed = json.loads(DEFAULT_DATASET.read_text(encoding="utf-8"))
    original_labels: dict[str, list[dict]] = {}
    cases = []
    for case_index, original in enumerate(
        case for case in seed["cases"] if case["split"] == "development"
    ):
        case = deepcopy(original)
        case_id = case["case_id"]
        original_labels[case_id] = deepcopy(case.pop("labels"))
        case.pop("annotation", None)
        video_id = case["question_analysis"]["video_id"]
        case["independence_group"] = f"source-family-{video_id}"
        case["split"] = "development"
        case["question_text"] = (
            "Assess whether the transcript passages establish the requested aspects: "
            + "; ".join(item["description"] for item in case["question_analysis"]["aspects"])
        )
        case["source"] = {
            "kind": "youtube_caption",
            "source_group": video_id,
            "source_url": f"https://www.youtube.com/watch?v={video_id}",
            "transcript_track": case["transcript_language_group"],
        }
        timestamp = 10.0 + case_index * 100.0
        for passage_index, passage in enumerate(case["passages"]):
            for ref_index, reference in enumerate(passage["references"]):
                reference["start_seconds"] = timestamp + passage_index * 20 + ref_index * 5
                reference["end_seconds"] = reference["start_seconds"] + 4.0
        cases.append(case)
    return {
        "schema_version": workflow.SOURCE_SCHEMA_VERSION,
        "dataset_id": "annotation-dev-test",
        "split": "development",
        "cases": cases,
    }, original_labels


def _make_completed_review(
    packet: dict,
    labels_by_case: dict[str, list[dict]],
    *,
    rationale: str,
) -> dict:
    result = deepcopy(packet)
    result["blinding_attested"] = True
    for case in result["cases"]:
        case["labels"] = [
            {
                **deepcopy(label),
                "rationale": rationale,
            }
            for label in labels_by_case[case["case_id"]]
        ]
    return result


def _write_private_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    private = tmp_path / "evaluation_data" / "private"
    monkeypatch.setattr(workflow, "PRIVATE_DATA_DIR", private)
    source, labels = _source_fixture()
    source_path = private / "phase11_0D_development_sources.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        json.dumps(source, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest, digest = workflow.load_source_manifest(source_path)
    return private, source_path, manifest, digest, labels


def _fill_adjudication_packet(
    packet: dict,
    labels_by_case: dict[str, list[dict]],
) -> dict:
    result = deepcopy(packet)
    result["adjudication_attested"] = True
    for case in result["cases"]:
        case["adjudicated_labels"] = [
            {
                **deepcopy(label),
                "rationale": "Final adjudication confirmed against the cited transcript interval.",
            }
            for label in labels_by_case[case["case_id"]]
        ]
    return result


def test_source_manifest_validates_and_reports_language_pair_coverage(tmp_path, monkeypatch):
    _, _, manifest, digest, _ = _write_private_source(tmp_path, monkeypatch)

    validated = workflow.validate_source_manifest(manifest)
    summary = workflow._summary(validated)

    assert digest
    assert summary["case_count"] == 16
    assert summary["model_calls_made"] == 0
    assert set(summary["question_language_case_counts"]) == {"en", "hi", "ar"}
    assert set(summary["transcript_language_case_counts"]) == {"en", "hi", "ar"}
    assert len(summary["question_to_transcript_pair_case_counts"]) >= 3


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda data: data["cases"][0].update(labels=[]), "forbidden field"),
        (lambda data: data["cases"][0].update(annotation={}), "forbidden field"),
        (lambda data: data["cases"][0].update(model_predictions=[]), "forbidden field"),
        (
            lambda data: data["cases"][0]["source"].update(
                source_url="https://example.com/not-youtube"
            ),
            "HTTPS URL",
        ),
        (
            lambda data: data["cases"][0]["passages"][0]["references"][0].update(
                end_seconds=None
            ),
            "exact timestamps",
        ),
        (
            lambda data: data["cases"][0]["passages"][0]["references"][0].update(
                video_id="different-video"
            ),
            "video ID mismatch",
        ),
    ],
)
def test_source_manifest_rejects_unsafe_or_invalid_inputs(tmp_path, monkeypatch, mutation, message):
    _, _, manifest, _, _ = _write_private_source(tmp_path, monkeypatch)
    candidate = deepcopy(manifest)
    mutation(candidate)

    with pytest.raises(workflow.DatasetValidationError, match=message):
        workflow.validate_source_manifest(candidate)


def test_exported_review_packets_are_separate_and_do_not_expose_split_or_labels(tmp_path, monkeypatch):
    _, _, manifest, digest, _ = _write_private_source(tmp_path, monkeypatch)

    first = workflow.export_review_packet(manifest, digest, "reviewer-A")
    second = workflow.export_review_packet(manifest, digest, "reviewer-B")

    assert first["reviewer_id"] == "reviewer-A"
    assert second["reviewer_id"] == "reviewer-B"
    assert first["packet_id"] != second["packet_id"]
    assert first["blinding_attested"] is False
    assert "split" not in first
    assert "independence_group" not in first["cases"][0]
    assert "labels" in first["cases"][0]
    assert first["cases"][0]["labels"][0]["expected_status"] == ""
    assert "predicted_status" not in json.dumps(first)
    assert "support_score" not in json.dumps(first)


def test_review_packet_requires_blinding_and_complete_labels(tmp_path, monkeypatch):
    _, _, manifest, digest, labels = _write_private_source(tmp_path, monkeypatch)
    packet = workflow.export_review_packet(manifest, digest, "reviewer-A")
    completed = _make_completed_review(packet, labels, rationale="Checked against the excerpt.")

    completed["blinding_attested"] = False
    with pytest.raises(workflow.DatasetValidationError, match="attest"):
        workflow._validate_review_packet(completed, manifest, digest)

    completed["blinding_attested"] = True
    completed["cases"][0]["labels"].pop()
    with pytest.raises(workflow.DatasetValidationError, match="every aspect"):
        workflow._validate_review_packet(completed, manifest, digest)


def test_review_packet_cannot_change_source_text_or_timestamps(tmp_path, monkeypatch):
    _, _, manifest, digest, labels = _write_private_source(tmp_path, monkeypatch)
    packet = workflow.export_review_packet(manifest, digest, "reviewer-A")
    completed = _make_completed_review(packet, labels, rationale="Verified.")
    completed["cases"][0]["passages"][0]["text"] = "Altered passage"

    with pytest.raises(workflow.DatasetValidationError, match="source content differs"):
        workflow._validate_review_packet(completed, manifest, digest)


def test_adjudication_packet_requires_two_distinct_blinded_reviewers(tmp_path, monkeypatch):
    _, _, manifest, digest, labels = _write_private_source(tmp_path, monkeypatch)
    base = workflow.export_review_packet(manifest, digest, "reviewer-A")
    review_a = _make_completed_review(base, labels, rationale="First review.")
    review_b = _make_completed_review(
        workflow.export_review_packet(manifest, digest, "reviewer-B"),
        labels,
        rationale="Second review.",
    )

    with pytest.raises(workflow.DatasetValidationError, match="distinct"):
        workflow.build_adjudication_packet(
            manifest, digest, review_a, review_a, "adjudicator-C"
        )
    with pytest.raises(workflow.DatasetValidationError, match="distinct"):
        workflow.build_adjudication_packet(
            manifest, digest, review_a, review_b, "reviewer-A"
        )


def test_adjudication_packet_preserves_original_review_labels(tmp_path, monkeypatch):
    _, _, manifest, digest, labels = _write_private_source(tmp_path, monkeypatch)
    review_a = _make_completed_review(
        workflow.export_review_packet(manifest, digest, "reviewer-A"),
        labels,
        rationale="First.",
    )
    review_b = _make_completed_review(
        workflow.export_review_packet(manifest, digest, "reviewer-B"),
        labels,
        rationale="Second.",
    )
    adjudication = workflow.build_adjudication_packet(
        manifest, digest, review_a, review_b, "adjudicator-C"
    )
    adjudication["adjudication_attested"] = True
    adjudication["cases"][0]["reviewer_labels_by_aspect"][0]["reviewer_labels"][0][
        "label"
    ]["expected_status"] = "unknown"

    with pytest.raises(workflow.DatasetValidationError, match="altered an independent review"):
        workflow.assemble_adjudicated_manifest(
            manifest, digest, review_a, review_b, adjudication
        )


def test_adjudication_packet_cannot_change_source_excerpt(tmp_path, monkeypatch):
    _, _, manifest, digest, labels = _write_private_source(tmp_path, monkeypatch)
    review_a = _make_completed_review(
        workflow.export_review_packet(manifest, digest, "reviewer-A"), labels, rationale="A"
    )
    review_b = _make_completed_review(
        workflow.export_review_packet(manifest, digest, "reviewer-B"), labels, rationale="B"
    )
    adjudication = workflow.build_adjudication_packet(
        manifest, digest, review_a, review_b, "adjudicator-C"
    )
    adjudication["adjudication_attested"] = True
    adjudication["cases"][0]["passages"][0]["text"] = "Tampered source excerpt"

    with pytest.raises(workflow.DatasetValidationError, match="source content differs"):
        workflow.assemble_adjudicated_manifest(
            manifest, digest, review_a, review_b, adjudication
        )


def test_complete_annotation_round_trip_produces_valid_phase11_0c_manifest(tmp_path, monkeypatch):
    private, source_path, manifest, digest, labels = _write_private_source(tmp_path, monkeypatch)
    review_a = _make_completed_review(
        workflow.export_review_packet(manifest, digest, "reviewer-A"),
        labels,
        rationale="Independent review A.",
    )
    review_b = _make_completed_review(
        workflow.export_review_packet(manifest, digest, "reviewer-B"),
        labels,
        rationale="Independent review B.",
    )
    adjudication = workflow.build_adjudication_packet(
        manifest, digest, review_a, review_b, "adjudicator-C"
    )
    adjudication = _fill_adjudication_packet(adjudication, labels)

    final = workflow.assemble_adjudicated_manifest(
        manifest, digest, review_a, review_b, adjudication
    )
    validated = validate_manifest(final)

    assert validated["annotation_status"] == "adjudicated"
    assert validated["split"] == "development"
    assert len(validated["cases"]) == 16
    assert all(case["annotation"]["blinding_attested"] for case in validated["cases"])
    assert all(
        len(case["annotation"]["independent_reviews"]) == 2
        for case in validated["cases"]
    )
    assert not source_path.samefile(private / "phase11_0C_development.json") if (
        private / "phase11_0C_development.json"
    ).exists() else True


def test_assembler_rejects_source_hash_drift(tmp_path, monkeypatch):
    _, _, manifest, digest, labels = _write_private_source(tmp_path, monkeypatch)
    packet_a = _make_completed_review(
        workflow.export_review_packet(manifest, digest, "reviewer-A"), labels, rationale="A"
    )
    packet_b = _make_completed_review(
        workflow.export_review_packet(manifest, digest, "reviewer-B"), labels, rationale="B"
    )
    packet_b["source_manifest_sha256"] = "0" * 64

    with pytest.raises(workflow.DatasetValidationError, match="hash mismatch"):
        workflow._validate_review_packet(packet_b, manifest, digest)


def test_annotation_files_must_stay_inside_private_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, "PRIVATE_DATA_DIR", tmp_path / "private")
    with pytest.raises(workflow.DatasetValidationError, match="under"):
        workflow._private_path(tmp_path / "public" / "annotation.json")


def test_private_writer_refuses_to_overwrite_existing_reviews(tmp_path, monkeypatch):
    private = tmp_path / "private"
    monkeypatch.setattr(workflow, "PRIVATE_DATA_DIR", private)
    target = private / "reviewer-A.json"
    workflow._write_json_private(target, {"one": 1})

    with pytest.raises(workflow.DatasetValidationError, match="Refusing to overwrite"):
        workflow._write_json_private(target, {"two": 2})


def test_annotation_workflow_cli_supports_direct_help_invocation():
    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "scripts" / "phase11_0d_annotation_workflow.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "export-review-packets" in completed.stdout
    assert "build-adjudication-packet" in completed.stdout
