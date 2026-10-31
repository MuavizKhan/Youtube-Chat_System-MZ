"""Prepare blinded human annotation packets for Phase 11.0C.

This tool is local-only and deliberately has no model/provider integration. It
turns a source-only manifest into two independent reviewer packets, combines
those reviews into an adjudication packet, and assembles the final
Phase 11.0C-compatible adjudicated dataset.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Backend.rag_system.evidence_assessment import EvidencePassage  # noqa: E402
from Backend.rag_system.evidence_contracts import (  # noqa: E402
    EvidenceAspect,
    EvidenceReference,
    QuestionAnalysis,
    QuestionIntent,
    TemporalConstraint,
)

from scripts.phase11_0c_semantic_assessment_eval import (  # noqa: E402
    DATASET_SCHEMA_VERSION,
    PRIVATE_DATA_DIR,
    STATUS_ORDER,
    SUPPORTED_LANGUAGES,
    VALID_SPLITS,
    DatasetValidationError,
    _validate_status_label,
    _validate_youtube_source_url,
    validate_manifest,
)

SOURCE_SCHEMA_VERSION = "1.0.0"
PACKET_SCHEMA_VERSION = "1.0.0"
SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
FORBIDDEN_SOURCE_KEYS = {
    "labels",
    "annotation",
    "expected_status",
    "predicted_status",
    "support_score",
    "assessor_predictions",
    "model_predictions",
    "threshold_lock",
}
FORBIDDEN_REVIEW_KEYS = {
    "predicted_status",
    "support_score",
    "assessor_predictions",
    "model_predictions",
    "raw_provider_response",
    "prompt",
    "gold_labels",
}


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DatasetValidationError(f"{field} must be a JSON object.")
    return value


def _require_list(value: Any, field: str, *, minimum: int = 1) -> list[Any]:
    if not isinstance(value, list) or len(value) < minimum:
        raise DatasetValidationError(f"{field} must be an array with at least {minimum} item(s).")
    return value


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(f"{field} must be a non-empty string.")
    return value.strip()


def _safe_id(value: Any, field: str) -> str:
    result = _require_text(value, field)
    if not SAFE_ID_PATTERN.fullmatch(result):
        raise DatasetValidationError(
            f"{field} must use 1-64 letters, numbers, dots, underscores, or hyphens."
        )
    return result


def _reject_keys(value: Any, forbidden: set[str], path: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in forbidden:
                raise DatasetValidationError(f"{path}: forbidden field {key!r}.")
            _reject_keys(child, forbidden, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_keys(child, forbidden, f"{path}[{index}]")


def _private_path(path: Path | str) -> Path:
    candidate = Path(path).expanduser().resolve()
    private_root = PRIVATE_DATA_DIR.resolve()
    if not candidate.is_relative_to(private_root):
        raise DatasetValidationError(
            f"Sensitive annotation files must remain under {private_root}."
        )
    return candidate


def _read_json(path: Path | str, *, private: bool = True) -> tuple[dict[str, Any], str]:
    resolved = _private_path(path) if private else Path(path).expanduser().resolve()
    try:
        raw = resolved.read_bytes()
    except OSError as error:
        raise DatasetValidationError(f"Cannot read annotation file: {resolved}") from error
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DatasetValidationError(f"Annotation file is not valid UTF-8 JSON: {resolved}") from error
    return _require_mapping(value, str(resolved)), hashlib.sha256(raw).hexdigest()


def _write_json_private(path: Path | str, value: dict[str, Any]) -> Path:
    resolved = _private_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists():
        raise DatasetValidationError(
            f"Refusing to overwrite an existing annotation file: {resolved}"
        )
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # Recheck after writing so a concurrently-created target is never replaced.
        if resolved.exists():
            raise DatasetValidationError(
                f"Refusing to overwrite an existing annotation file: {resolved}"
            )
        os.rename(temporary, resolved)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return resolved


def _validate_source_case(case: dict[str, Any], split: str) -> dict[str, Any]:
    case_id = _require_text(case.get("case_id"), "case.case_id")
    if case.get("split") != split:
        raise DatasetValidationError(f"{case_id}: case split must match its source manifest.")
    _require_text(case.get("independence_group"), f"{case_id}.independence_group")

    question_language = _require_text(
        case.get("question_language_group"), f"{case_id}.question_language_group"
    )
    transcript_language = _require_text(
        case.get("transcript_language_group"), f"{case_id}.transcript_language_group"
    )
    if question_language not in SUPPORTED_LANGUAGES:
        raise DatasetValidationError(
            f"{case_id}: unsupported question language {question_language!r}."
        )
    if transcript_language not in SUPPORTED_LANGUAGES:
        raise DatasetValidationError(
            f"{case_id}: unsupported transcript language {transcript_language!r}."
        )
    question_text = _require_text(case.get("question_text"), f"{case_id}.question_text")

    analysis = _require_mapping(case.get("question_analysis"), f"{case_id}.question_analysis")
    video_id = _require_text(analysis.get("video_id"), f"{case_id}.question_analysis.video_id")
    if analysis.get("language", question_language) != question_language:
        raise DatasetValidationError(
            f"{case_id}: question_analysis.language must match question_language_group."
        )
    aspects_raw = _require_list(
        analysis.get("aspects"), f"{case_id}.question_analysis.aspects"
    )
    if len(aspects_raw) > 8:
        raise DatasetValidationError(f"{case_id}: at most eight aspects are supported.")
    aspect_ids: set[str] = set()
    aspects: list[EvidenceAspect] = []
    try:
        for raw_aspect in aspects_raw:
            item = _require_mapping(raw_aspect, f"{case_id}.aspect")
            aspect_id = _require_text(item.get("aspect_id"), f"{case_id}.aspect_id")
            if aspect_id in aspect_ids:
                raise DatasetValidationError(f"{case_id}: duplicate aspect ID {aspect_id!r}.")
            aspect_ids.add(aspect_id)
            aspects.append(EvidenceAspect(
                aspect_id=aspect_id,
                description=_require_text(item.get("description"), f"{case_id}/{aspect_id}.description"),
                required=item.get("required", True),
                weight=item.get("weight", 1.0),
                retrieval_hints=tuple(item.get("retrieval_hints", ())),
            ))
        temporal_raw = analysis.get("temporal_constraint")
        temporal = (
            TemporalConstraint(
                start_seconds=temporal_raw["start_seconds"],
                end_seconds=temporal_raw["end_seconds"],
            )
            if temporal_raw is not None else None
        )
        QuestionAnalysis(
            intent=QuestionIntent(analysis.get("intent", "unknown")),
            language=question_language,
            aspects=tuple(aspects),
            video_id=video_id,
            subject=analysis.get("subject"),
            constraints=tuple(analysis.get("constraints", ())),
            temporal_constraint=temporal,
            planner_version=analysis.get("planner_version", "phase11_0D_source_v1"),
            analysis_source="phase11_0D_annotation_source",
        )
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, DatasetValidationError):
            raise
        raise DatasetValidationError(f"{case_id}: invalid question analysis: {error}") from error

    passages_raw = _require_list(case.get("passages"), f"{case_id}.passages")
    if len(passages_raw) > 12:
        raise DatasetValidationError(f"{case_id}: at most twelve passages are supported.")
    passage_ids: set[str] = set()
    source_chunks: set[tuple[str, str]] = set()
    for raw_passage in passages_raw:
        passage = _require_mapping(raw_passage, f"{case_id}.passage")
        passage_id = _require_text(passage.get("passage_id"), f"{case_id}.passage_id")
        if passage_id in passage_ids:
            raise DatasetValidationError(f"{case_id}: duplicate passage ID {passage_id!r}.")
        passage_ids.add(passage_id)
        passage_language = passage.get("transcript_language", transcript_language)
        if passage_language != transcript_language:
            raise DatasetValidationError(
                f"{case_id}/{passage_id}: passage language must match transcript_language_group."
            )
        refs_raw = _require_list(
            passage.get("references"), f"{case_id}/{passage_id}.references"
        )
        references: list[EvidenceReference] = []
        for raw_ref in refs_raw:
            ref = _require_mapping(raw_ref, f"{case_id}/{passage_id}.reference")
            start = ref.get("start_seconds")
            end = ref.get("end_seconds")
            if start is None or end is None:
                raise DatasetValidationError(
                    f"{case_id}/{passage_id}: every human-review reference needs exact timestamps."
                )
            try:
                reference = EvidenceReference(
                    video_id=ref.get("video_id"),
                    chunk_id=ref.get("chunk_id"),
                    start_seconds=start,
                    end_seconds=end,
                )
            except (TypeError, ValueError) as error:
                raise DatasetValidationError(
                    f"{case_id}/{passage_id}: invalid source reference: {error}"
                ) from error
            if reference.video_id != video_id:
                raise DatasetValidationError(
                    f"{case_id}/{passage_id}: source reference video ID mismatch."
                )
            identity = (reference.video_id, str(reference.chunk_id))
            if identity in source_chunks:
                raise DatasetValidationError(
                    f"{case_id}: a source chunk may belong to only one passage."
                )
            source_chunks.add(identity)
            references.append(reference)
        try:
            EvidencePassage(
                passage_id=passage_id,
                text=_require_text(passage.get("text"), f"{case_id}/{passage_id}.text"),
                references=tuple(references),
                transcript_language=transcript_language,
            )
        except (TypeError, ValueError) as error:
            raise DatasetValidationError(
                f"{case_id}/{passage_id}: invalid passage: {error}"
            ) from error

    source = _require_mapping(case.get("source"), f"{case_id}.source")
    source_kind = _require_text(source.get("kind"), f"{case_id}.source.kind")
    if source_kind == "synthetic_fixture":
        raise DatasetValidationError(f"{case_id}: synthetic sources are not allowed.")
    source_url = _require_text(source.get("source_url"), f"{case_id}.source.source_url")
    _validate_youtube_source_url(source_url, video_id, f"{case_id}.source.source_url")
    _require_text(source.get("transcript_track"), f"{case_id}.source.transcript_track")

    return {
        "case_id": case_id,
        "question_text": question_text,
        "question_language_group": question_language,
        "transcript_language_group": transcript_language,
        "question_analysis": analysis,
        "passages": passages_raw,
        "source": {
            **source,
            "source_group": _require_text(
                source.get("source_group", video_id), f"{case_id}.source.source_group"
            ),
        },
    }


def validate_source_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate a source-only manifest before any reviewer packet is issued."""
    manifest = _require_mapping(manifest, "source_manifest")
    if manifest.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise DatasetValidationError(
            f"Unsupported source manifest schema_version: {manifest.get('schema_version')!r}."
        )
    _safe_id(manifest.get("dataset_id"), "source_manifest.dataset_id")
    split = _require_text(manifest.get("split"), "source_manifest.split")
    if split not in VALID_SPLITS:
        raise DatasetValidationError(f"split must be one of {', '.join(VALID_SPLITS)}.")
    _reject_keys(manifest, FORBIDDEN_SOURCE_KEYS, "source_manifest")
    cases_raw = _require_list(manifest.get("cases"), "source_manifest.cases")
    cases: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    for raw_case in cases_raw:
        case = _require_mapping(raw_case, "source_manifest.case")
        case_id = _require_text(case.get("case_id"), "source_manifest.case.case_id")
        if case_id in seen_case_ids:
            raise DatasetValidationError(f"Duplicate case_id: {case_id}.")
        seen_case_ids.add(case_id)
        cases.append(_validate_source_case(case, split))
    return {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "dataset_id": manifest["dataset_id"],
        "split": split,
        "cases": cases,
    }


def load_source_manifest(path: Path | str) -> tuple[dict[str, Any], str]:
    manifest, digest = _read_json(path)
    _reject_keys(manifest, FORBIDDEN_SOURCE_KEYS, "source_manifest")
    return validate_source_manifest(manifest), digest


def _case_packet_projection(case: dict[str, Any]) -> dict[str, Any]:
    # Intentionally omit split, independence_group, source_group and any labels.
    return {
        "case_id": case["case_id"],
        "question_text": case["question_text"],
        "question_language_group": case["question_language_group"],
        "transcript_language_group": case["transcript_language_group"],
        "question_analysis": case["question_analysis"],
        "passages": case["passages"],
        "source": case["source"],
    }


def export_review_packet(
    manifest: dict[str, Any],
    source_sha256: str,
    reviewer_id: str,
) -> dict[str, Any]:
    reviewer_id = _safe_id(reviewer_id, "reviewer_id")
    cases = []
    for case in manifest["cases"]:
        labels = [
            {
                "aspect_id": aspect["aspect_id"],
                "expected_status": "",
                "supporting_passage_ids": [],
                "contradicting_passage_ids": [],
                "rationale": "",
            }
            for aspect in case["question_analysis"]["aspects"]
        ]
        cases.append({**_case_packet_projection(case), "labels": labels})
    binding = f"{manifest['dataset_id']}|{source_sha256}|{reviewer_id}"
    return {
        "packet_schema_version": PACKET_SCHEMA_VERSION,
        "packet_type": "independent_review",
        "packet_id": hashlib.sha256(binding.encode("utf-8")).hexdigest(),
        "dataset_id": manifest["dataset_id"],
        "source_manifest_sha256": source_sha256,
        "reviewer_id": reviewer_id,
        "blinding_attested": False,
        "cases": cases,
    }


def _validate_packet_binding(
    packet: dict[str, Any],
    manifest: dict[str, Any],
    source_sha256: str,
    packet_type: str,
) -> None:
    if packet.get("packet_schema_version") != PACKET_SCHEMA_VERSION:
        raise DatasetValidationError("Unsupported annotation packet schema version.")
    if packet.get("packet_type") != packet_type:
        raise DatasetValidationError(f"Expected packet_type {packet_type!r}.")
    if packet.get("dataset_id") != manifest["dataset_id"]:
        raise DatasetValidationError("Annotation packet dataset_id mismatch.")
    if packet.get("source_manifest_sha256") != source_sha256:
        raise DatasetValidationError("Annotation packet source manifest hash mismatch.")


def _validate_review_packet(
    packet: dict[str, Any],
    manifest: dict[str, Any],
    source_sha256: str,
) -> tuple[str, dict[str, list[dict[str, Any]]]]:
    _validate_packet_binding(packet, manifest, source_sha256, "independent_review")
    _reject_keys(packet, FORBIDDEN_REVIEW_KEYS, "review_packet")
    reviewer_id = _safe_id(packet.get("reviewer_id"), "review_packet.reviewer_id")
    if packet.get("blinding_attested") is not True:
        raise DatasetValidationError(
            f"{reviewer_id}: reviewer must attest to independent blind annotation."
        )
    packet_cases = _require_list(packet.get("cases"), "review_packet.cases")
    expected_cases = {case["case_id"]: case for case in manifest["cases"]}
    observed_cases: dict[str, list[dict[str, Any]]] = {}
    for raw_case in packet_cases:
        review_case = _require_mapping(raw_case, f"{reviewer_id}.case")
        case_id = _require_text(review_case.get("case_id"), "review_case.case_id")
        if case_id not in expected_cases or case_id in observed_cases:
            raise DatasetValidationError(f"{reviewer_id}: unknown or duplicate case {case_id!r}.")
        source_case = expected_cases[case_id]
        projection = _case_packet_projection(source_case)
        if any(review_case.get(key) != value for key, value in projection.items()):
            raise DatasetValidationError(
                f"{reviewer_id}/{case_id}: source content differs from the issued blind packet."
            )
        if set(review_case) != set(projection) | {"labels"}:
            raise DatasetValidationError(
                f"{reviewer_id}/{case_id}: unexpected or missing reviewer packet fields."
            )
        labels = _require_list(review_case.get("labels"), f"{case_id}.labels")
        aspect_ids = {aspect["aspect_id"] for aspect in source_case["question_analysis"]["aspects"]}
        passage_ids = {passage["passage_id"] for passage in source_case["passages"]}
        seen_aspects: set[str] = set()
        for label in labels:
            label = _require_mapping(label, f"{case_id}.label")
            _validate_status_label(label, passage_ids, aspect_ids)
            aspect_id = label["aspect_id"]
            if aspect_id in seen_aspects:
                raise DatasetValidationError(f"{reviewer_id}/{case_id}: duplicate aspect label.")
            seen_aspects.add(aspect_id)
            rationale = label.get("rationale", "")
            if not isinstance(rationale, str):
                raise DatasetValidationError(
                    f"{reviewer_id}/{case_id}/{aspect_id}: rationale must be text."
                )
        if seen_aspects != aspect_ids:
            raise DatasetValidationError(
                f"{reviewer_id}/{case_id}: every aspect must have exactly one label."
            )
        observed_cases[case_id] = labels
    if set(observed_cases) != set(expected_cases):
        missing = sorted(set(expected_cases) - set(observed_cases))
        raise DatasetValidationError(f"{reviewer_id}: review packet is missing cases: {missing}.")
    return reviewer_id, observed_cases


def build_adjudication_packet(
    manifest: dict[str, Any],
    source_sha256: str,
    review_packet_a: dict[str, Any],
    review_packet_b: dict[str, Any],
    adjudicator_id: str,
) -> dict[str, Any]:
    reviewer_a, labels_a = _validate_review_packet(review_packet_a, manifest, source_sha256)
    reviewer_b, labels_b = _validate_review_packet(review_packet_b, manifest, source_sha256)
    adjudicator_id = _safe_id(adjudicator_id, "adjudicator_id")
    if reviewer_a == reviewer_b:
        raise DatasetValidationError("The two independent reviewer IDs must be distinct.")
    if adjudicator_id in {reviewer_a, reviewer_b}:
        raise DatasetValidationError("The adjudicator must be distinct from both reviewers.")

    cases = []
    for case in manifest["cases"]:
        case_id = case["case_id"]
        labels_by_aspect = []
        for aspect in case["question_analysis"]["aspects"]:
            aspect_id = aspect["aspect_id"]
            labels_by_aspect.append({
                "aspect_id": aspect_id,
                "description": aspect["description"],
                "reviewer_labels": [
                    {"reviewer_id": reviewer_a, "label": labels_a[case_id][
                        next(i for i, item in enumerate(labels_a[case_id]) if item["aspect_id"] == aspect_id)
                    ]},
                    {"reviewer_id": reviewer_b, "label": labels_b[case_id][
                        next(i for i, item in enumerate(labels_b[case_id]) if item["aspect_id"] == aspect_id)
                    ]},
                ],
            })
        cases.append({
            **_case_packet_projection(case),
            "reviewer_labels_by_aspect": labels_by_aspect,
            "adjudicated_labels": [
                {
                    "aspect_id": aspect["aspect_id"],
                    "expected_status": "",
                    "supporting_passage_ids": [],
                    "contradicting_passage_ids": [],
                    "rationale": "",
                }
                for aspect in case["question_analysis"]["aspects"]
            ],
        })
    binding = (
        f"{manifest['dataset_id']}|{source_sha256}|{reviewer_a}|{reviewer_b}|{adjudicator_id}"
    )
    return {
        "packet_schema_version": PACKET_SCHEMA_VERSION,
        "packet_type": "adjudication",
        "packet_id": hashlib.sha256(binding.encode("utf-8")).hexdigest(),
        "dataset_id": manifest["dataset_id"],
        "source_manifest_sha256": source_sha256,
        "reviewer_ids": [reviewer_a, reviewer_b],
        "adjudicator_id": adjudicator_id,
        "adjudication_attested": False,
        "cases": cases,
    }


def _validate_adjudication_packet(
    packet: dict[str, Any],
    manifest: dict[str, Any],
    source_sha256: str,
    reviewer_ids: tuple[str, str],
    reviewer_labels: dict[str, dict[str, list[dict[str, Any]]]],
) -> tuple[str, dict[str, list[dict[str, Any]]]]:
    _validate_packet_binding(packet, manifest, source_sha256, "adjudication")
    _reject_keys(packet, FORBIDDEN_REVIEW_KEYS, "adjudication_packet")
    adjudicator_id = _safe_id(packet.get("adjudicator_id"), "adjudication_packet.adjudicator_id")
    declared_reviewers = packet.get("reviewer_ids")
    if declared_reviewers != list(reviewer_ids):
        raise DatasetValidationError("Adjudication packet reviewer IDs do not match supplied reviews.")
    if adjudicator_id in set(reviewer_ids):
        raise DatasetValidationError("The adjudicator must be distinct from both reviewers.")
    if packet.get("adjudication_attested") is not True:
        raise DatasetValidationError("Adjudicator must attest that final labels were independently adjudicated.")
    expected_cases = {case["case_id"]: case for case in manifest["cases"]}
    observed: dict[str, list[dict[str, Any]]] = {}
    for raw_case in _require_list(packet.get("cases"), "adjudication_packet.cases"):
        case = _require_mapping(raw_case, "adjudication_case")
        case_id = _require_text(case.get("case_id"), "adjudication_case.case_id")
        if case_id not in expected_cases or case_id in observed:
            raise DatasetValidationError(f"Adjudication packet has unknown or duplicate case {case_id!r}.")
        expected = expected_cases[case_id]
        projection = _case_packet_projection(expected)
        if any(case.get(key) != value for key, value in projection.items()):
            raise DatasetValidationError(
                f"{case_id}: adjudication source content differs from the issued packet."
            )
        if set(case) != set(projection) | {"reviewer_labels_by_aspect", "adjudicated_labels"}:
            raise DatasetValidationError(
                f"{case_id}: unexpected or missing adjudication packet fields."
            )
        aspect_ids = {aspect["aspect_id"] for aspect in expected["question_analysis"]["aspects"]}
        aspect_descriptions = {
            aspect["aspect_id"]: aspect["description"]
            for aspect in expected["question_analysis"]["aspects"]
        }
        reviewer_rows = _require_list(
            case.get("reviewer_labels_by_aspect"),
            f"{case_id}.reviewer_labels_by_aspect",
        )
        seen_reviewer_aspects: set[str] = set()
        for row in reviewer_rows:
            row = _require_mapping(row, f"{case_id}.reviewer_label_row")
            aspect_id = _require_text(row.get("aspect_id"), f"{case_id}.reviewer_label_row.aspect_id")
            if aspect_id not in aspect_ids or aspect_id in seen_reviewer_aspects:
                raise DatasetValidationError(
                    f"{case_id}: adjudication packet has an unknown or duplicate reviewer aspect."
                )
            seen_reviewer_aspects.add(aspect_id)
            if row.get("description") != aspect_descriptions[aspect_id]:
                raise DatasetValidationError(
                    f"{case_id}/{aspect_id}: adjudication aspect description was altered."
                )
            displayed = _require_list(
                row.get("reviewer_labels"),
                f"{case_id}/{aspect_id}.reviewer_labels",
            )
            if len(displayed) != 2:
                raise DatasetValidationError(
                    f"{case_id}/{aspect_id}: adjudicator must see exactly two independent labels."
                )
            for expected_reviewer in reviewer_ids:
                matched = [
                    item for item in displayed
                    if isinstance(item, dict) and item.get("reviewer_id") == expected_reviewer
                ]
                if len(matched) != 1:
                    raise DatasetValidationError(
                        f"{case_id}/{aspect_id}: reviewer label attribution is invalid."
                    )
                source_labels = {
                    label["aspect_id"]: label
                    for label in reviewer_labels[expected_reviewer][case_id]
                }
                if matched[0].get("label") != source_labels[aspect_id]:
                    raise DatasetValidationError(
                        f"{case_id}/{aspect_id}: adjudication packet altered an independent review."
                    )
        if seen_reviewer_aspects != aspect_ids:
            raise DatasetValidationError(
                f"{case_id}: adjudication packet must show both labels for every aspect."
            )
        labels = _require_list(case.get("adjudicated_labels"), f"{case_id}.adjudicated_labels")
        passage_ids = {passage["passage_id"] for passage in expected["passages"]}
        seen: set[str] = set()
        for label in labels:
            label = _require_mapping(label, f"{case_id}.adjudicated_label")
            _validate_status_label(label, passage_ids, aspect_ids)
            aspect_id = label["aspect_id"]
            if aspect_id in seen:
                raise DatasetValidationError(f"{case_id}: duplicate adjudicated aspect label.")
            seen.add(aspect_id)
            if not isinstance(label.get("rationale", ""), str):
                raise DatasetValidationError(f"{case_id}/{aspect_id}: rationale must be text.")
        if seen != aspect_ids:
            raise DatasetValidationError(f"{case_id}: adjudicator must label every aspect exactly once.")
        observed[case_id] = labels
    if set(observed) != set(expected_cases):
        raise DatasetValidationError("Adjudication packet must contain every source case exactly once.")
    return adjudicator_id, observed


def assemble_adjudicated_manifest(
    manifest: dict[str, Any],
    source_sha256: str,
    review_packet_a: dict[str, Any],
    review_packet_b: dict[str, Any],
    adjudication_packet: dict[str, Any],
) -> dict[str, Any]:
    reviewer_a, labels_a = _validate_review_packet(review_packet_a, manifest, source_sha256)
    reviewer_b, labels_b = _validate_review_packet(review_packet_b, manifest, source_sha256)
    if reviewer_a == reviewer_b:
        raise DatasetValidationError("The two independent reviewer IDs must be distinct.")
    adjudicator_id, final_labels = _validate_adjudication_packet(
        adjudication_packet,
        manifest,
        source_sha256,
        (reviewer_a, reviewer_b),
        {reviewer_a: labels_a, reviewer_b: labels_b},
    )

    cases = []
    for source_case in manifest["cases"]:
        case_id = source_case["case_id"]
        case = dict(source_case)
        case["labels"] = final_labels[case_id]
        case["annotation"] = {
            "status": "adjudicated",
            "reviewer_ids": [reviewer_a, reviewer_b],
            "adjudicator_id": adjudicator_id,
            "blinding_attested": True,
            "independent_reviews": [
                {"reviewer_id": reviewer_a, "labels": labels_a[case_id]},
                {"reviewer_id": reviewer_b, "labels": labels_b[case_id]},
            ],
        }
        cases.append(case)
    result = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_id": manifest["dataset_id"],
        "annotation_status": "adjudicated",
        "split": manifest["split"],
        "cases": cases,
    }
    # The Phase 11.0C validator is the final authority for all labels,
    # provenance, per-split language/status coverage and annotation records.
    return validate_manifest(result)


def _summary(manifest: dict[str, Any]) -> dict[str, Any]:
    question_counts = Counter(case["question_language_group"] for case in manifest["cases"])
    transcript_counts = Counter(case["transcript_language_group"] for case in manifest["cases"])
    pair_counts = Counter(
        f"{case['question_language_group']}->{case['transcript_language_group']}"
        for case in manifest["cases"]
    )
    return {
        "dataset_id": manifest["dataset_id"],
        "split": manifest["split"],
        "case_count": len(manifest["cases"]),
        "question_language_case_counts": dict(sorted(question_counts.items())),
        "transcript_language_case_counts": dict(sorted(transcript_counts.items())),
        "question_to_transcript_pair_case_counts": dict(sorted(pair_counts.items())),
        "source_manifest_sha256": None,
        "contains_transcript_passages": True,
        "model_calls_made": 0,
    }


def _load_two_review_packets(
    paths: list[Path],
    manifest: dict[str, Any],
    source_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(paths) != 2:
        raise DatasetValidationError("Exactly two reviewer packet paths are required.")
    first, _ = _read_json(paths[0])
    second, _ = _read_json(paths[1])
    _validate_review_packet(first, manifest, source_sha256)
    _validate_review_packet(second, manifest, source_sha256)
    return first, second


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create and assemble blind human-annotation packets for Phase 11.0C."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-source", help="Validate a source-only manifest.")
    validate.add_argument("--source-manifest", type=Path, required=True)

    export = commands.add_parser("export-review-packets", help="Create two separate blind reviewer packets.")
    export.add_argument("--source-manifest", type=Path, required=True)
    export.add_argument("--reviewer-ids", nargs=2, required=True)
    export.add_argument("--output-dir", type=Path, required=True)

    adjudicate = commands.add_parser("build-adjudication-packet", help="Combine two completed blind reviews for a third adjudicator.")
    adjudicate.add_argument("--source-manifest", type=Path, required=True)
    adjudicate.add_argument("--reviewer-packets", nargs=2, type=Path, required=True)
    adjudicate.add_argument("--adjudicator-id", required=True)
    adjudicate.add_argument("--output", type=Path, required=True)

    assemble = commands.add_parser("assemble", help="Create and validate the final Phase 11.0C dataset.")
    assemble.add_argument("--source-manifest", type=Path, required=True)
    assemble.add_argument("--reviewer-packets", nargs=2, type=Path, required=True)
    assemble.add_argument("--adjudication-packet", type=Path, required=True)
    assemble.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest, source_sha256 = load_source_manifest(args.source_manifest)
        if args.command == "validate-source":
            summary = _summary(manifest)
            summary["source_manifest_sha256"] = source_sha256
            summary["validation"] = "passed"
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0

        if args.command == "export-review-packets":
            reviewer_ids = [_safe_id(item, "reviewer_id") for item in args.reviewer_ids]
            if len(set(reviewer_ids)) != 2:
                raise DatasetValidationError("Exactly two distinct reviewer IDs are required.")
            output_dir = _private_path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            created = []
            for reviewer_id in reviewer_ids:
                packet = export_review_packet(manifest, source_sha256, reviewer_id)
                target = output_dir / f"reviewer_{reviewer_id}.json"
                created.append(str(_write_json_private(target, packet)))
            print(json.dumps({
                "dataset_id": manifest["dataset_id"],
                "source_manifest_sha256": source_sha256,
                "reviewer_packet_paths": created,
                "reviewers": reviewer_ids,
                "model_calls_made": 0,
            }, ensure_ascii=False, indent=2))
            return 0

        if args.command == "build-adjudication-packet":
            packet_a, packet_b = _load_two_review_packets(
                args.reviewer_packets, manifest, source_sha256
            )
            adjudicator = _safe_id(args.adjudicator_id, "adjudicator_id")
            packet = build_adjudication_packet(
                manifest, source_sha256, packet_a, packet_b, adjudicator
            )
            output = _write_json_private(args.output, packet)
            print(json.dumps({
                "adjudication_packet_path": str(output),
                "adjudicator_id": adjudicator,
                "case_count": len(packet["cases"]),
                "model_calls_made": 0,
            }, ensure_ascii=False, indent=2))
            return 0

        if args.command == "assemble":
            packet_a, packet_b = _load_two_review_packets(
                args.reviewer_packets, manifest, source_sha256
            )
            adjudication_packet, _ = _read_json(args.adjudication_packet)
            result = assemble_adjudicated_manifest(
                manifest, source_sha256, packet_a, packet_b, adjudication_packet
            )
            output = _write_json_private(args.output, result)
            print(json.dumps({
                "dataset_path": str(output),
                "dataset_id": result["dataset_id"],
                "annotation_status": result["annotation_status"],
                "split": result["split"],
                "case_count": len(result["cases"]),
                "validation": "passed",
                "model_calls_made": 0,
            }, ensure_ascii=False, indent=2))
            return 0
        raise DatasetValidationError(f"Unsupported command: {args.command}")
    except (DatasetValidationError, OSError, TypeError, ValueError) as error:
        print(json.dumps({"validation": "failed", "error": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
