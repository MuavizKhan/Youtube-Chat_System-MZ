from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import phase11_0d_annotation_workflow as workflow
from scripts.phase11_0c_semantic_assessment_eval import (
    DEFAULT_DATASET,
    STATUS_ORDER,
    validate_manifest,
)


def _source_fixture() -> tuple[dict, dict[str, list[dict]]]:
    seed = json.loads(DEFAULT_DATASET.read_text(encoding="utf-8"))
    original_labels: dict[str, list[dict]] = {}
    cases = []
    for case_index, original in enumerate(
        case for case in seed["cases"] if case["split"] == "development"
    ):
        case = deepcopy(original)
        case_id = f"CASE-{case_index + 1:04d}"
        original_labels[case_id] = deepcopy(case.pop("labels"))
        case.pop("annotation", None)
        case["case_id"] = case_id
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
            "caption_track_kind": "auto_generated",
            "verification": {
                "video_id_confirmed": True,
                "caption_track_confirmed": True,
                "timestamps_confirmed": True,
                "verified_by": "CURATOR-01",
            },
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


def _held_out_source_fixture(*, unique_sources: bool = True) -> dict:
    source_manifest, _ = _source_fixture()
    templates = source_manifest["cases"]
    cases = []
    counter = 0
    for question_language in ("en", "hi", "ar"):
        for transcript_language in ("en", "hi", "ar"):
            for _ in range(10):
                counter += 1
                case = deepcopy(templates[(counter - 1) % len(templates)])
                case_id = f"CASE-{counter:04d}"
                video_id = (
                    f"SyntheticHeldout{counter:04d}"
                    if unique_sources else "SyntheticHeldoutShared"
                )
                source_group = (
                    f"source-group-{counter:04d}"
                    if unique_sources else "source-group-shared"
                )
                independence_group = (
                    f"independence-group-{counter:04d}"
                    if unique_sources else "independence-group-shared"
                )
                case["case_id"] = case_id
                case["split"] = "held_out"
                case["independence_group"] = independence_group
                case["question_language_group"] = question_language
                case["transcript_language_group"] = transcript_language
                case["question_text"] = f"Question-language cohort {question_language}."
                analysis = case["question_analysis"]
                analysis["video_id"] = video_id
                analysis["language"] = question_language
                if len(analysis["aspects"]) < 2:
                    analysis["aspects"].append({
                        "aspect_id": "aspect-extra",
                        "description": "An additional independently assessed information need.",
                        "required": True,
                        "weight": 1.0,
                        "retrieval_hints": [],
                    })
                for passage_index, passage in enumerate(case["passages"]):
                    passage["transcript_language"] = transcript_language
                    for ref_index, reference in enumerate(passage["references"]):
                        reference["video_id"] = video_id
                        reference["chunk_id"] = (
                            f"chunk-{counter:04d}-{passage_index}-{ref_index}"
                        )
                        reference["start_seconds"] = 10.0 + counter + passage_index * 10
                        reference["end_seconds"] = reference["start_seconds"] + 5.0
                source = case["source"]
                source["source_group"] = source_group
                source["source_url"] = f"https://www.youtube.com/watch?v={video_id}"
                source["transcript_track"] = transcript_language
                cases.append(case)
    return {
        "schema_version": workflow.SOURCE_SCHEMA_VERSION,
        "dataset_id": "annotation-heldout-test",
        "split": "held_out",
        "cases": cases,
    }


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


def _heldout_adjudicated_roundtrip(*, balanced: bool):
    manifest = workflow.validate_source_manifest(_held_out_source_fixture())
    source_sha256 = "a" * 64
    labels_by_case: dict[str, list[dict]] = {}
    observation_index = 0
    for case in manifest["cases"]:
        case_labels = []
        passage_id = case["passages"][0]["passage_id"]
        for aspect in case["question_analysis"]["aspects"]:
            status = (
                STATUS_ORDER[observation_index % len(STATUS_ORDER)]
                if balanced or observation_index < len(STATUS_ORDER)
                else STATUS_ORDER[0]
            )
            observation_index += 1
            supporting = [passage_id] if status in {
                "supported", "partial", "conflicted"
            } else []
            contradicting = [passage_id] if status in {
                "contradicted", "conflicted"
            } else []
            case_labels.append({
                "aspect_id": aspect["aspect_id"],
                "expected_status": status,
                "supporting_passage_ids": supporting,
                "contradicting_passage_ids": contradicting,
            })
        labels_by_case[case["case_id"]] = case_labels

    review_a = _make_completed_review(
        workflow.export_review_packet(manifest, source_sha256, "REVIEWER-A"),
        labels_by_case,
        rationale="Independent review A checked the cited excerpt.",
    )
    review_b = _make_completed_review(
        workflow.export_review_packet(manifest, source_sha256, "REVIEWER-B"),
        labels_by_case,
        rationale="Independent review B checked the cited excerpt.",
    )
    adjudication = workflow.build_adjudication_packet(
        manifest, source_sha256, review_a, review_b, "ADJUDICATOR-C"
    )
    adjudication = _fill_adjudication_packet(adjudication, labels_by_case)
    final = workflow.assemble_adjudicated_manifest(
        manifest, source_sha256, review_a, review_b, adjudication
    )
    return final


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
    assert summary["all_cases_within_assessor_input_budget"] is True
    assert summary["phase11_0c_input_character_limit"] == 8000
    assert summary["aspect_observations"] >= 16


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
            lambda data: data["cases"][0]["source"]["verification"].update(
                timestamps_confirmed=False
            ),
            "timestamps_confirmed=true",
        ),
        (
            lambda data: data["cases"][0]["source"].update(
                caption_track_kind="unknown"
            ),
            "caption_track_kind",
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


@pytest.mark.parametrize(
    "case_id",
    [
        "DEV-001", "DEVELOPMENT-01", "HOLDOUT-01", "HELD_OUT-02",
        "TEST-001", "CASE-EN-SUP-01", "CASE-AR-CON-02",
    ],
)
@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda case: case["question_analysis"].update(constraints="single string"),
            "constraints must be an array",
        ),
        (
            lambda case: case["question_analysis"]["aspects"][0].update(
                retrieval_hints="single string"
            ),
            "retrieval_hints must be an array",
        ),
        (
            lambda case: case["question_analysis"]["aspects"][0].update(
                retrieval_hints=["repeat", "repeat"]
            ),
            "retrieval_hints must not contain duplicate",
        ),
    ],
)
def test_source_manifest_rejects_malformed_string_array_fields(
    mutation, message, tmp_path, monkeypatch
):
    _, _, manifest, _, _ = _write_private_source(tmp_path, monkeypatch)
    candidate = deepcopy(manifest)
    mutation(candidate["cases"][0])

    with pytest.raises(workflow.DatasetValidationError, match=message):
        workflow.validate_source_manifest(candidate)


def test_source_manifest_rejects_non_opaque_case_ids(case_id, tmp_path, monkeypatch):
    _, _, manifest, _, _ = _write_private_source(tmp_path, monkeypatch)
    candidate = deepcopy(manifest)
    candidate["cases"][0]["case_id"] = case_id

    with pytest.raises(workflow.DatasetValidationError, match="opaque CASE"):
        workflow.validate_source_manifest(candidate)


def test_exported_review_packets_are_separate_and_do_not_expose_split_or_labels(tmp_path, monkeypatch):
    _, _, manifest, digest, _ = _write_private_source(tmp_path, monkeypatch)

    first = workflow.export_review_packet(manifest, digest, "REVIEWER-A")
    second = workflow.export_review_packet(manifest, digest, "REVIEWER-B")

    assert first["reviewer_id"] == "REVIEWER-A"
    assert second["reviewer_id"] == "REVIEWER-B"
    assert first["packet_id"] != second["packet_id"]
    assert first["blinding_attested"] is False
    assert "split" not in first
    assert "dataset_id" not in first
    assert "independence_group" not in first["cases"][0]
    assert "source_group" not in first["cases"][0]["source"]
    assert "verified_by" not in json.dumps(first)
    assert "retrieval_hints" not in json.dumps(first)
    assert "confidence" not in json.dumps(first)
    assert "labels" in first["cases"][0]
    assert first["cases"][0]["labels"][0]["expected_status"] == ""
    assert "predicted_status" not in json.dumps(first)
    assert "support_score" not in json.dumps(first)


@pytest.mark.parametrize(
    "primary_language, regional_track",
    [("en", "en-US"), ("hi", "hi-IN"), ("ar", "ar-SA")],
)
def test_source_manifest_accepts_regional_bcp47_caption_tracks(
    primary_language, regional_track, tmp_path, monkeypatch
):
    _, _, manifest, _, _ = _write_private_source(tmp_path, monkeypatch)
    candidate = deepcopy(manifest)
    case = next(
        item for item in candidate["cases"]
        if item["transcript_language_group"] == primary_language
    )
    case["source"]["transcript_track"] = regional_track

    validated = workflow.validate_source_manifest(candidate)

    matching = next(item for item in validated["cases"] if item["case_id"] == case["case_id"])
    assert matching["source"]["transcript_track"] == regional_track


def test_source_manifest_rejects_caption_track_with_wrong_primary_language(tmp_path, monkeypatch):
    _, _, manifest, _, _ = _write_private_source(tmp_path, monkeypatch)
    candidate = deepcopy(manifest)
    candidate["cases"][0]["source"]["transcript_track"] = "hi-IN"

    with pytest.raises(workflow.DatasetValidationError, match="primary language must match"):
        workflow.validate_source_manifest(candidate)


@pytest.mark.parametrize("annotator_id", ["MuavizKhan", "name@example.com", "JohnSmith"])
def test_annotator_identifiers_must_be_pseudonymous_role_ids(annotator_id):
    with pytest.raises(workflow.DatasetValidationError, match="pseudonymous role ID"):
        workflow._annotator_id(annotator_id, "annotator_id")


def test_supported_pseudonymous_annotator_roles_are_accepted():
    assert workflow._annotator_id("R01", "reviewer_id") == "R01"
    assert workflow._annotator_id("REVIEWER-A", "reviewer_id") == "REVIEWER-A"
    assert workflow._annotator_id("ADJUDICATOR-C", "adjudicator_id") == "ADJUDICATOR-C"
    assert workflow._annotator_id("CURATOR-01", "curator_id") == "CURATOR-01"


def test_source_manifest_rejects_passages_over_the_phase11_0b_input_budget(tmp_path, monkeypatch):
    _, _, manifest, _, _ = _write_private_source(tmp_path, monkeypatch)
    candidate = deepcopy(manifest)
    candidate["cases"][0]["passages"][0]["text"] = "x" * 9000

    with pytest.raises(workflow.DatasetValidationError, match="hard limit of 8000"):
        workflow.validate_source_manifest(candidate)


def test_source_manifest_requires_at_least_one_required_aspect(tmp_path, monkeypatch):
    _, _, manifest, _, _ = _write_private_source(tmp_path, monkeypatch)
    candidate = deepcopy(manifest)
    for aspect in candidate["cases"][0]["question_analysis"]["aspects"]:
        aspect["required"] = False

    with pytest.raises(workflow.DatasetValidationError, match="at least one evidence aspect"):
        workflow.validate_source_manifest(candidate)


@pytest.mark.parametrize("identity", ["video_id", "source_group", "independence_group"])
def test_development_and_held_out_source_manifests_must_be_disjoint(identity):
    development = workflow.validate_source_manifest(_source_fixture()[0])
    held_out_raw = _held_out_source_fixture(unique_sources=True)
    development_case = development["cases"][0]
    held_out_case = held_out_raw["cases"][0]

    if identity == "video_id":
        shared_video_id = development_case["question_analysis"]["video_id"]
        held_out_case["question_analysis"]["video_id"] = shared_video_id
        held_out_case["source"]["source_url"] = (
            f"https://www.youtube.com/watch?v={shared_video_id}"
        )
        for passage in held_out_case["passages"]:
            for reference in passage["references"]:
                reference["video_id"] = shared_video_id
    elif identity == "source_group":
        held_out_case["source"]["source_group"] = development_case["source"]["source_group"]
    else:
        held_out_case["independence_group"] = development_case["independence_group"]

    held_out = workflow.validate_source_manifest(held_out_raw)
    with pytest.raises(workflow.DatasetValidationError, match="source leakage detected"):
        workflow.validate_split_isolation(development, held_out)


def test_development_and_held_out_source_manifests_pass_when_disjoint():
    development = workflow.validate_source_manifest(_source_fixture()[0])
    held_out = workflow.validate_source_manifest(_held_out_source_fixture(unique_sources=True))

    workflow.validate_split_isolation(development, held_out)


def test_held_out_source_preflight_accepts_all_phase11_0c_source_floors():
    manifest = _held_out_source_fixture(unique_sources=True)

    validated = workflow.validate_source_manifest(manifest)
    summary = workflow._summary(validated)

    assert len(validated["cases"]) == 90
    assert summary["aspect_observations"] >= 100
    assert set(summary["question_to_transcript_pair_case_counts"].values()) == {10}
    assert summary["held_out_sample_floors_met"] is True


def test_held_out_source_preflight_rejects_repeated_cases_from_too_few_sources():
    manifest = _held_out_source_fixture(unique_sources=False)

    with pytest.raises(workflow.DatasetValidationError, match="source diversity is too low"):
        workflow.validate_source_manifest(manifest)


def test_held_out_source_preflight_rejects_underfilled_cohorts_before_annotation(
    tmp_path, monkeypatch
):
    _, _, manifest, _, _ = _write_private_source(tmp_path, monkeypatch)
    candidate = deepcopy(manifest)
    candidate["split"] = "held_out"
    for case in candidate["cases"]:
        case["split"] = "held_out"

    with pytest.raises(workflow.DatasetValidationError, match="held-out source manifest has"):
        workflow.validate_source_manifest(candidate)


def test_review_packet_requires_blinding_and_complete_labels(tmp_path, monkeypatch):
    _, _, manifest, digest, labels = _write_private_source(tmp_path, monkeypatch)
    packet = workflow.export_review_packet(manifest, digest, "REVIEWER-A")
    completed = _make_completed_review(packet, labels, rationale="Checked against the excerpt.")

    completed["blinding_attested"] = False
    with pytest.raises(workflow.DatasetValidationError, match="attest"):
        workflow._validate_review_packet(completed, manifest, digest)

    completed["blinding_attested"] = True
    completed["cases"][0]["labels"].pop()
    with pytest.raises(workflow.DatasetValidationError, match="at least 1 item"):
        workflow._validate_review_packet(completed, manifest, digest)


def test_review_packet_cannot_change_source_text_or_timestamps(tmp_path, monkeypatch):
    _, _, manifest, digest, labels = _write_private_source(tmp_path, monkeypatch)
    packet = workflow.export_review_packet(manifest, digest, "REVIEWER-A")
    completed = _make_completed_review(packet, labels, rationale="Verified.")
    completed["cases"][0]["passages"][0]["text"] = "Altered passage"

    with pytest.raises(workflow.DatasetValidationError, match="source content differs"):
        workflow._validate_review_packet(completed, manifest, digest)


def test_adjudication_packet_requires_two_distinct_blinded_reviewers(tmp_path, monkeypatch):
    _, _, manifest, digest, labels = _write_private_source(tmp_path, monkeypatch)
    base = workflow.export_review_packet(manifest, digest, "REVIEWER-A")
    review_a = _make_completed_review(base, labels, rationale="First review.")
    review_b = _make_completed_review(
        workflow.export_review_packet(manifest, digest, "REVIEWER-B"),
        labels,
        rationale="Second review.",
    )

    with pytest.raises(workflow.DatasetValidationError, match="distinct"):
        workflow.build_adjudication_packet(
            manifest, digest, review_a, review_a, "ADJUDICATOR-C"
        )
    with pytest.raises(workflow.DatasetValidationError, match="distinct"):
        workflow.build_adjudication_packet(
            manifest, digest, review_a, review_b, "REVIEWER-A"
        )


def test_adjudication_packet_preserves_original_review_labels(tmp_path, monkeypatch):
    _, _, manifest, digest, labels = _write_private_source(tmp_path, monkeypatch)
    review_a = _make_completed_review(
        workflow.export_review_packet(manifest, digest, "REVIEWER-A"),
        labels,
        rationale="First.",
    )
    review_b = _make_completed_review(
        workflow.export_review_packet(manifest, digest, "REVIEWER-B"),
        labels,
        rationale="Second.",
    )
    adjudication = workflow.build_adjudication_packet(
        manifest, digest, review_a, review_b, "ADJUDICATOR-C"
    )
    adjudication["adjudication_attested"] = True
    # Change only a human rationale so the reviewer label remains schema-valid;
    # the integrity comparison, not status/reference validation, must catch it.
    adjudication["cases"][0]["reviewer_labels_by_aspect"][0]["reviewer_labels"][0][
        "label"
    ]["rationale"] = "Tampered rationale"

    with pytest.raises(workflow.DatasetValidationError, match="altered an independent review"):
        workflow.assemble_adjudicated_manifest(
            manifest, digest, review_a, review_b, adjudication
        )


def test_held_out_adjudication_roundtrip_meets_phase11_0c_acceptance_contract():
    final = _heldout_adjudicated_roundtrip(balanced=True)

    assert final["annotation_status"] == "adjudicated"
    assert final["split"] == "held_out"
    assert len(final["cases"]) == 90
    counts = {
        status: sum(
            1
            for case in final["cases"]
            for label in case["labels"]
            if label["expected_status"] == status
        )
        for status in STATUS_ORDER
    }
    assert all(counts[status] >= 10 for status in STATUS_ORDER)


def test_held_out_assembler_rejects_status_counts_below_final_eligibility_floor():
    with pytest.raises(workflow.DatasetValidationError, match="per-status eligibility floor"):
        _heldout_adjudicated_roundtrip(balanced=False)


def test_adjudication_packet_detects_review_packet_revision_drift(tmp_path, monkeypatch):
    _, _, manifest, digest, labels = _write_private_source(tmp_path, monkeypatch)
    review_a = _make_completed_review(
        workflow.export_review_packet(manifest, digest, "REVIEWER-A"),
        labels,
        rationale="Original review A.",
    )
    review_b = _make_completed_review(
        workflow.export_review_packet(manifest, digest, "REVIEWER-B"),
        labels,
        rationale="Original review B.",
    )
    adjudication = workflow.build_adjudication_packet(
        manifest, digest, review_a, review_b, "ADJUDICATOR-C"
    )
    adjudication = _fill_adjudication_packet(adjudication, labels)
    review_a["cases"][0]["labels"][0]["rationale"] = "Changed after adjudication packet creation."

    with pytest.raises(workflow.DatasetValidationError, match="exact independent review revisions"):
        workflow.assemble_adjudicated_manifest(
            manifest, digest, review_a, review_b, adjudication
        )


def test_adjudication_packet_cannot_change_source_excerpt(tmp_path, monkeypatch):
    _, _, manifest, digest, labels = _write_private_source(tmp_path, monkeypatch)
    review_a = _make_completed_review(
        workflow.export_review_packet(manifest, digest, "REVIEWER-A"), labels, rationale="A"
    )
    review_b = _make_completed_review(
        workflow.export_review_packet(manifest, digest, "REVIEWER-B"), labels, rationale="B"
    )
    adjudication = workflow.build_adjudication_packet(
        manifest, digest, review_a, review_b, "ADJUDICATOR-C"
    )
    adjudication["adjudication_attested"] = True
    adjudication["cases"][0]["passages"][0]["text"] = "Tampered source excerpt"

    with pytest.raises(workflow.DatasetValidationError, match="source content differs"):
        workflow.assemble_adjudicated_manifest(
            manifest, digest, review_a, review_b, adjudication
        )


def test_complete_annotation_round_trip_produces_valid_phase11_0c_manifest(tmp_path, monkeypatch):
    _, _, manifest, digest, labels = _write_private_source(tmp_path, monkeypatch)
    review_a = _make_completed_review(
        workflow.export_review_packet(manifest, digest, "REVIEWER-A"),
        labels,
        rationale="Independent review A.",
    )
    review_b = _make_completed_review(
        workflow.export_review_packet(manifest, digest, "REVIEWER-B"),
        labels,
        rationale="Independent review B.",
    )
    adjudication = workflow.build_adjudication_packet(
        manifest, digest, review_a, review_b, "ADJUDICATOR-C"
    )
    adjudication = _fill_adjudication_packet(adjudication, labels)

    final = workflow.assemble_adjudicated_manifest(
        manifest, digest, review_a, review_b, adjudication
    )
    validated = validate_manifest(final)

    assert validated["annotation_status"] == "adjudicated"
    assert validated["split"] == "development"
    assert len(validated["cases"]) == 16
    workflow_provenance = validated["annotation_workflow_provenance"]
    assert workflow_provenance["source_manifest_sha256"] == digest
    assert len(workflow_provenance["reviewer_packets"]) == 2
    assert workflow_provenance["adjudication_packet"]["canonical_sha256"]
    assert workflow_provenance["workflow_tool_sha256"]
    assert all(case["annotation"]["blinding_attested"] for case in validated["cases"])
    assert all(
        len(case["annotation"]["independent_reviews"]) == 2
        for case in validated["cases"]
    )


def test_assembler_rejects_source_hash_drift(tmp_path, monkeypatch):
    _, _, manifest, digest, labels = _write_private_source(tmp_path, monkeypatch)
    packet_a = _make_completed_review(
        workflow.export_review_packet(manifest, digest, "REVIEWER-A"), labels, rationale="A"
    )
    packet_b = _make_completed_review(
        workflow.export_review_packet(manifest, digest, "REVIEWER-B"), labels, rationale="B"
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
    target = private / "REVIEWER-A.json"
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
