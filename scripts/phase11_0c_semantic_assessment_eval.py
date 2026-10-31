"""Leakage-aware evaluation runner for the Phase 11.0B semantic evidence assessor.

This runner is intentionally isolated from the production RAG chain. It consumes
a labelled manifest, sends only question analysis and transcript passages to the
assessor, and writes aggregate/sanitized predictions without copying transcript
text or reference labels into model prompts.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Backend.rag_system import evidence_assessment as evidence_assessment_module  # noqa: E402
from Backend.rag_system import evidence_contracts as evidence_contracts_module  # noqa: E402
from Backend.rag_system.evidence_assessment import (  # noqa: E402
    DEFAULT_MAX_ASSESSMENT_ASPECTS,
    DEFAULT_MAX_ASSESSMENT_INPUT_CHARS,
    DEFAULT_MAX_ASSESSMENT_OUTPUT_TOKENS,
    DEFAULT_MAX_ASSESSMENT_PASSAGES,
    AssessmentRunStatus,
    EvidencePassage,
    assess_evidence_support,
)
from Backend.rag_system.evidence_contracts import (  # noqa: E402
    EvidenceAspect,
    EvidenceReference,
    EvidenceSupportStatus,
    QuestionAnalysis,
    QuestionIntent,
    TemporalConstraint,
)

DEFAULT_DATASET = (
    PROJECT_ROOT / "evaluation_data" / "phase11_0C_semantic_assessment_seed.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evaluation_reports"
STATUS_ORDER = tuple(status.value for status in EvidenceSupportStatus)
SUPPORTED_LANGUAGES = ("en", "hi", "ar")
VALID_SPLITS = ("development", "held_out")
DATASET_SCHEMA_VERSION = "1.0.0"
REPORT_SCHEMA_VERSION = "1.0.0"
LOCK_SCHEMA_VERSION = "1.0.0"

# This is an eligibility floor for considering a held-out report statistically
# reviewable. It is not a claim that this sample size proves model quality.
MIN_HELD_OUT_ASPECTS = 100
MIN_HELD_OUT_PER_STATUS = 10
MIN_HELD_OUT_PER_LANGUAGE = 20


class DatasetValidationError(ValueError):
    """Raised when an evaluation dataset violates its declared contract."""


class ThresholdLockError(ValueError):
    """Raised when a held-out run is not tied to a completed development report."""


def _require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DatasetValidationError(f"{field_name} must be a JSON object.")
    return value


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _require_string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise DatasetValidationError(f"{field_name} must be an array of non-empty strings.")
    cleaned = [item.strip() for item in value]
    if len(cleaned) != len(set(cleaned)):
        raise DatasetValidationError(f"{field_name} must not contain duplicates.")
    return cleaned


def _validate_status_label(label: dict[str, Any], passage_ids: set[str], aspect_ids: set[str]) -> None:
    aspect_id = _require_text(label.get("aspect_id"), "label.aspect_id")
    if aspect_id not in aspect_ids:
        raise DatasetValidationError(f"Label references unknown aspect: {aspect_id}.")
    try:
        status = EvidenceSupportStatus(label.get("expected_status"))
    except (TypeError, ValueError) as error:
        raise DatasetValidationError(
            f"{aspect_id}: expected_status must be one of {', '.join(STATUS_ORDER)}."
        ) from error

    support_ids = _require_string_list(
        label.get("supporting_passage_ids", []),
        f"{aspect_id}.supporting_passage_ids",
    )
    contradiction_ids = _require_string_list(
        label.get("contradicting_passage_ids", []),
        f"{aspect_id}.contradicting_passage_ids",
    )
    cited_ids = set(support_ids) | set(contradiction_ids)
    if not cited_ids.issubset(passage_ids):
        unknown = sorted(cited_ids - passage_ids)
        raise DatasetValidationError(
            f"{aspect_id}: labels cite passages absent from the case: {unknown}."
        )

    if status is EvidenceSupportStatus.SUPPORTED:
        valid = bool(support_ids) and not contradiction_ids
    elif status is EvidenceSupportStatus.PARTIAL:
        valid = bool(support_ids) and not contradiction_ids
    elif status is EvidenceSupportStatus.MISSING:
        valid = not support_ids and not contradiction_ids
    elif status is EvidenceSupportStatus.CONTRADICTED:
        valid = not support_ids and bool(contradiction_ids)
    elif status is EvidenceSupportStatus.CONFLICTED:
        valid = bool(support_ids) and bool(contradiction_ids)
    else:
        valid = not support_ids and not contradiction_ids

    if not valid:
        raise DatasetValidationError(
            f"{aspect_id}: evidence passage IDs are inconsistent with {status.value}."
        )


def _validate_case(
    case: dict[str, Any],
    manifest_status: str,
) -> tuple[str, set[str], set[str], str, str]:
    case_id = _require_text(case.get("case_id"), "case.case_id")
    split = _require_text(case.get("split"), f"{case_id}.split")
    if split not in VALID_SPLITS:
        raise DatasetValidationError(
            f"{case_id}: split must be one of {', '.join(VALID_SPLITS)}."
        )

    independence_group = _require_text(
        case.get("independence_group"), f"{case_id}.independence_group"
    )
    question_language = _require_text(
        case.get("question_language_group"),
        f"{case_id}.question_language_group",
    )
    transcript_language = _require_text(
        case.get("transcript_language_group"),
        f"{case_id}.transcript_language_group",
    )
    if question_language not in SUPPORTED_LANGUAGES:
        raise DatasetValidationError(
            f"{case_id}: question_language_group must be one of {', '.join(SUPPORTED_LANGUAGES)}."
        )
    if transcript_language not in SUPPORTED_LANGUAGES:
        raise DatasetValidationError(
            f"{case_id}: transcript_language_group must be one of {', '.join(SUPPORTED_LANGUAGES)}."
        )

    analysis = _require_mapping(case.get("question_analysis"), f"{case_id}.question_analysis")
    video_id = _require_text(analysis.get("video_id"), f"{case_id}.question_analysis.video_id")
    aspects_raw = analysis.get("aspects")
    if not isinstance(aspects_raw, list) or not aspects_raw:
        raise DatasetValidationError(f"{case_id}: question_analysis.aspects must be non-empty.")

    try:
        aspects = tuple(
            EvidenceAspect(
                aspect_id=item["aspect_id"],
                description=item["description"],
                required=item.get("required", True),
                weight=item.get("weight", 1.0),
                retrieval_hints=tuple(item.get("retrieval_hints", ())),
            )
            for item in aspects_raw
        )
        temporal_raw = analysis.get("temporal_constraint")
        temporal = (
            TemporalConstraint(
                start_seconds=temporal_raw["start_seconds"],
                end_seconds=temporal_raw["end_seconds"],
            )
            if temporal_raw is not None
            else None
        )
        parsed_analysis = QuestionAnalysis(
            intent=QuestionIntent(analysis.get("intent", "unknown")),
            language=analysis.get("language", question_language),
            aspects=aspects,
            video_id=video_id,
            subject=analysis.get("subject"),
            constraints=tuple(analysis.get("constraints", ())),
            temporal_constraint=temporal,
            planner_version=analysis.get("planner_version", "phase11_0C_dataset_v1"),
            analysis_source="phase11_0C_labelled_dataset",
        )
    except (KeyError, TypeError, ValueError) as error:
        raise DatasetValidationError(f"{case_id}: invalid question analysis: {error}") from error

    if parsed_analysis.language != question_language:
        raise DatasetValidationError(
            f"{case_id}: question_analysis.language must match question_language_group."
        )
    if len(aspects) > 8:
        raise DatasetValidationError(f"{case_id}: Phase 11.0B allows at most 8 aspects.")

    passages_raw = case.get("passages")
    if not isinstance(passages_raw, list) or not passages_raw:
        raise DatasetValidationError(f"{case_id}: passages must be a non-empty array.")
    if len(passages_raw) > 12:
        raise DatasetValidationError(f"{case_id}: Phase 11.0B allows at most 12 passages.")

    passages = []
    passage_ids: set[str] = set()
    source_references: set[tuple[str, str]] = set()
    for raw_passage in passages_raw:
        passage = _require_mapping(raw_passage, f"{case_id}.passage")
        passage_id = _require_text(passage.get("passage_id"), f"{case_id}.passage_id")
        if passage_id in passage_ids:
            raise DatasetValidationError(f"{case_id}: duplicate passage ID {passage_id}.")
        passage_ids.add(passage_id)

        passage_language = passage.get("transcript_language", transcript_language)
        if passage_language != transcript_language:
            raise DatasetValidationError(
                f"{case_id}/{passage_id}: transcript_language must match "
                "transcript_language_group for this case."
            )
        refs_raw = passage.get("references")
        if not isinstance(refs_raw, list) or not refs_raw:
            raise DatasetValidationError(
                f"{case_id}/{passage_id}: references must be a non-empty array."
            )
        references = []
        for raw_ref in refs_raw:
            ref = _require_mapping(raw_ref, f"{case_id}/{passage_id}.reference")
            reference = EvidenceReference(
                video_id=ref.get("video_id"),
                chunk_id=ref.get("chunk_id"),
                start_seconds=ref.get("start_seconds"),
                end_seconds=ref.get("end_seconds"),
            )
            if reference.video_id != video_id:
                raise DatasetValidationError(
                    f"{case_id}/{passage_id}: every reference must match the case video."
                )
            identity = (reference.video_id, str(reference.chunk_id))
            if identity in source_references:
                raise DatasetValidationError(
                    f"{case_id}: a source chunk may belong to only one passage."
                )
            source_references.add(identity)
            references.append(reference)

        try:
            passages.append(
                EvidencePassage(
                    passage_id=passage_id,
                    text=passage.get("text"),
                    references=tuple(references),
                    transcript_language=passage.get("transcript_language", transcript_language),
                )
            )
        except (TypeError, ValueError) as error:
            raise DatasetValidationError(
                f"{case_id}/{passage_id}: invalid evidence passage: {error}"
            ) from error

    labels_raw = case.get("labels")
    if not isinstance(labels_raw, list):
        raise DatasetValidationError(f"{case_id}: labels must be an array.")
    label_ids = []
    for label in labels_raw:
        if not isinstance(label, dict):
            raise DatasetValidationError(f"{case_id}: every label must be an object.")
        _validate_status_label(label, passage_ids, {aspect.aspect_id for aspect in aspects})
        label_ids.append(label["aspect_id"])
    if len(label_ids) != len(set(label_ids)):
        raise DatasetValidationError(f"{case_id}: duplicate aspect labels.")
    if set(label_ids) != {aspect.aspect_id for aspect in aspects}:
        raise DatasetValidationError(
            f"{case_id}: every question aspect must have exactly one reference label."
        )

    source = _require_mapping(case.get("source"), f"{case_id}.source")
    source_kind = _require_text(source.get("kind"), f"{case_id}.source.kind")
    source_group = _require_text(
        source.get("source_group", video_id), f"{case_id}.source.source_group"
    )
    annotation = _require_mapping(case.get("annotation"), f"{case_id}.annotation")
    annotation_case_status = _require_text(
        annotation.get("status"), f"{case_id}.annotation.status"
    )

    if manifest_status == "adjudicated":
        reviewer_ids = _require_string_list(
            annotation.get("reviewer_ids"), f"{case_id}.annotation.reviewer_ids"
        )
        adjudicator_id = _require_text(
            annotation.get("adjudicator_id"), f"{case_id}.annotation.adjudicator_id"
        )
        if annotation_case_status != "adjudicated":
            raise DatasetValidationError(
                f"{case_id}: every case must be marked adjudicated."
            )
        if len(reviewer_ids) < 2:
            raise DatasetValidationError(
                f"{case_id}: at least two independent reviewer IDs are required."
            )
        if adjudicator_id in reviewer_ids:
            raise DatasetValidationError(
                f"{case_id}: adjudicator must be distinct from the two reviewers."
            )
        if source_kind == "synthetic_fixture":
            raise DatasetValidationError(
                f"{case_id}: synthetic fixtures cannot be marked human-adjudicated."
            )
        _require_text(source.get("source_url"), f"{case_id}.source.source_url")
        _require_text(source.get("transcript_track"), f"{case_id}.source.transcript_track")
    elif annotation_case_status != "provisional_synthetic":
        raise DatasetValidationError(
            f"{case_id}: provisional manifests must mark every case provisional_synthetic."
        )

    return (
        split,
        {independence_group, source_group},
        {video_id},
        question_language,
        transcript_language,
    )


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate the complete dataset, provenance, annotations, and split isolation."""
    manifest = _require_mapping(manifest, "dataset")
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise DatasetValidationError(
            f"Unsupported dataset schema_version: {manifest.get('schema_version')!r}."
        )
    _require_text(manifest.get("dataset_id"), "dataset_id")
    manifest_status = _require_text(manifest.get("annotation_status"), "annotation_status")
    if manifest_status not in {"provisional_synthetic", "adjudicated"}:
        raise DatasetValidationError(
            "annotation_status must be provisional_synthetic or adjudicated."
        )
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise DatasetValidationError("dataset.cases must be a non-empty array.")

    case_ids: set[str] = set()
    group_splits: dict[str, set[str]] = defaultdict(set)
    video_splits: dict[str, set[str]] = defaultdict(set)
    status_by_split: dict[str, set[str]] = defaultdict(set)
    question_language_by_split: dict[str, set[str]] = defaultdict(set)
    transcript_language_by_split: dict[str, set[str]] = defaultdict(set)
    counts_by_split: Counter[str] = Counter()

    for case in cases:
        case = _require_mapping(case, "case")
        case_id = _require_text(case.get("case_id"), "case.case_id")
        if case_id in case_ids:
            raise DatasetValidationError(f"Duplicate case_id: {case_id}.")
        case_ids.add(case_id)
        (
            split,
            groups,
            videos,
            question_language,
            transcript_language,
        ) = _validate_case(case, manifest_status)
        counts_by_split[split] += 1
        question_language_by_split[split].add(question_language)
        transcript_language_by_split[split].add(transcript_language)
        for group in groups:
            group_splits[group].add(split)
        for video in videos:
            video_splits[video].add(split)
        for label in case["labels"]:
            status_by_split[split].add(label["expected_status"])

    for group, splits in group_splits.items():
        if len(splits) > 1:
            raise DatasetValidationError(
                f"Data leakage: independence/source group {group!r} crosses splits."
            )
    for video, splits in video_splits.items():
        if len(splits) > 1:
            raise DatasetValidationError(
                f"Data leakage: video {video!r} occurs in multiple splits."
            )

    for split in VALID_SPLITS:
        if not counts_by_split[split]:
            raise DatasetValidationError(f"Dataset has no {split} cases.")
        if status_by_split[split] != set(STATUS_ORDER):
            missing = sorted(set(STATUS_ORDER) - status_by_split[split])
            raise DatasetValidationError(
                f"{split}: every evidence status must be represented; missing {missing}."
            )
        if question_language_by_split[split] != set(SUPPORTED_LANGUAGES):
            missing = sorted(
                set(SUPPORTED_LANGUAGES) - question_language_by_split[split]
            )
            raise DatasetValidationError(
                f"{split}: English, Hindi, and Arabic question languages must all "
                f"be represented; missing {missing}."
            )
        if transcript_language_by_split[split] != set(SUPPORTED_LANGUAGES):
            missing = sorted(
                set(SUPPORTED_LANGUAGES) - transcript_language_by_split[split]
            )
            raise DatasetValidationError(
                f"{split}: English, Hindi, and Arabic transcript languages must all "
                f"be represented; missing {missing}."
            )

    return manifest


def load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    """Load and validate a UTF-8 JSON manifest; return its exact SHA-256."""
    try:
        raw_bytes = path.read_bytes()
    except OSError as error:
        raise DatasetValidationError(f"Cannot read dataset file: {path}") from error
    digest = hashlib.sha256(raw_bytes).hexdigest()
    try:
        manifest = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DatasetValidationError(f"Dataset is not valid UTF-8 JSON: {path}") from error
    return validate_manifest(manifest), digest


def _case_inputs(case: dict[str, Any]) -> tuple[QuestionAnalysis, tuple[EvidencePassage, ...]]:
    analysis_raw = case["question_analysis"]
    aspects = tuple(
        EvidenceAspect(
            aspect_id=item["aspect_id"],
            description=item["description"],
            required=item.get("required", True),
            weight=item.get("weight", 1.0),
            retrieval_hints=tuple(item.get("retrieval_hints", ())),
        )
        for item in analysis_raw["aspects"]
    )
    temporal_raw = analysis_raw.get("temporal_constraint")
    temporal = (
        TemporalConstraint(
            start_seconds=temporal_raw["start_seconds"],
            end_seconds=temporal_raw["end_seconds"],
        )
        if temporal_raw is not None
        else None
    )
    analysis = QuestionAnalysis(
        intent=QuestionIntent(analysis_raw.get("intent", "unknown")),
        language=analysis_raw.get("language", case["question_language_group"]),
        aspects=aspects,
        video_id=analysis_raw["video_id"],
        subject=analysis_raw.get("subject"),
        constraints=tuple(analysis_raw.get("constraints", ())),
        temporal_constraint=temporal,
        planner_version=analysis_raw.get("planner_version", "phase11_0C_dataset_v1"),
        analysis_source="phase11_0C_labelled_dataset",
    )
    passages = tuple(
        EvidencePassage(
            passage_id=item["passage_id"],
            text=item["text"],
            transcript_language=item.get("transcript_language", case["transcript_language_group"]),
            references=tuple(
                EvidenceReference(
                    video_id=ref["video_id"],
                    chunk_id=ref["chunk_id"],
                    start_seconds=ref.get("start_seconds"),
                    end_seconds=ref.get("end_seconds"),
                )
                for ref in item["references"]
            ),
        )
        for item in case["passages"]
    )
    return analysis, passages


def _enum_value(value: Any) -> str:
    return value.value if hasattr(value, "value") else str(value)


def _case_prediction_record(
    case: dict[str, Any],
    result: Any,
) -> dict[str, Any]:
    diagnostics = result.diagnostics
    diag_status = _enum_value(diagnostics.status)
    analysis_raw = case["question_analysis"]
    video_id = analysis_raw["video_id"]
    ref_to_passage: dict[tuple[str, str], str] = {}
    for passage in case["passages"]:
        for reference in passage["references"]:
            ref_to_passage[(reference["video_id"], str(reference["chunk_id"]))] = (
                passage["passage_id"]
            )

    expected_by_id = {item["aspect_id"]: item for item in case["labels"]}
    predicted_by_id = {item.aspect_id: item for item in result.assessments}
    assessments = []
    for aspect_id in sorted(expected_by_id):
        expected = expected_by_id[aspect_id]
        prediction = predicted_by_id[aspect_id]
        is_semantic_result = diag_status == AssessmentRunStatus.COMPLETED.value
        supporting_ids = sorted({
            ref_to_passage[(ref.video_id, str(ref.chunk_id))]
            for ref in prediction.supporting_evidence
            if (ref.video_id, str(ref.chunk_id)) in ref_to_passage
        })
        contradicting_ids = sorted({
            ref_to_passage[(ref.video_id, str(ref.chunk_id))]
            for ref in prediction.contradicting_evidence
            if (ref.video_id, str(ref.chunk_id)) in ref_to_passage
        })
        assessments.append({
            "aspect_id": aspect_id,
            "expected_status": expected["expected_status"],
            "predicted_status": (
                _enum_value(prediction.status) if is_semantic_result else None
            ),
            "support_score": (
                float(prediction.support_score) if is_semantic_result else None
            ),
            "expected_supporting_passage_ids": expected.get("supporting_passage_ids", []),
            "predicted_supporting_passage_ids": supporting_ids if is_semantic_result else [],
            "expected_contradicting_passage_ids": expected.get("contradicting_passage_ids", []),
            "predicted_contradicting_passage_ids": contradicting_ids if is_semantic_result else [],
        })

    return {
        "case_id": case["case_id"],
        "split": case["split"],
        "question_language_group": case["question_language_group"],
        "transcript_language_group": case["transcript_language_group"],
        "challenge_tags": list(case.get("challenge_tags", [])),
        "assessment_run_status": diag_status,
        "error_code": diagnostics.error_code,
        "request_count": diagnostics.request_count,
        "elapsed_ms": round(float(diagnostics.elapsed_ms), 3),
        "input_chars": diagnostics.input_chars,
        "input_limit_chars": diagnostics.input_limit_chars,
        "input_tokens": diagnostics.input_tokens,
        "output_tokens": diagnostics.output_tokens,
        "passage_count": diagnostics.passage_count,
        "aspect_count": diagnostics.aspect_count,
        "assessments": assessments,
        # Deliberately no transcript text, prompt, raw provider response, or summary.
    }


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> dict[str, float] | None:
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return {
        "lower_95": max(0.0, center - margin),
        "upper_95": min(1.0, center + margin),
    }


def _class_metrics(
    rows: list[dict[str, Any]],
    statuses: tuple[str, ...] = STATUS_ORDER,
) -> dict[str, Any]:
    matrix = {
        expected: {predicted: 0 for predicted in statuses}
        for expected in statuses
    }
    for row in rows:
        matrix[row["expected_status"]][row["predicted_status"]] += 1

    per_class = {}
    f1_values = []
    for status in statuses:
        true_positive = matrix[status][status]
        false_positive = sum(
            matrix[other][status] for other in statuses if other != status
        )
        false_negative = sum(
            matrix[status][other] for other in statuses if other != status
        )
        support = sum(matrix[status].values())
        predicted = sum(matrix[other][status] for other in statuses)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        if support:
            f1_values.append(f1)
        per_class[status] = {
            "support": support,
            "predicted": predicted,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    total = len(rows)
    correct = sum(matrix[status][status] for status in statuses)
    return {
        "observations": total,
        "accuracy": correct / total if total else None,
        "macro_f1": statistics.mean(f1_values) if f1_values else None,
        "confusion_matrix": matrix,
        "per_status": per_class,
    }


def _binary_supported_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for row in rows:
        actual = row["expected_status"] == EvidenceSupportStatus.SUPPORTED.value
        predicted = row["predicted_status"] == EvidenceSupportStatus.SUPPORTED.value
        if actual and predicted:
            tp += 1
        elif not actual and predicted:
            fp += 1
        elif actual and not predicted:
            fn += 1
        else:
            tn += 1

    precision_denominator = tp + fp
    recall_denominator = tp + fn
    negative_denominator = fp + tn
    precision = tp / precision_denominator if precision_denominator else None
    recall = tp / recall_denominator if recall_denominator else None
    specificity = tn / negative_denominator if negative_denominator else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "supported_precision": precision,
        "supported_precision_wilson_95": _wilson_interval(tp, precision_denominator),
        "supported_recall": recall,
        "supported_recall_wilson_95": _wilson_interval(tp, recall_denominator),
        "unsupported_specificity": specificity,
        "false_supported_rate_on_unsupported": (
            fp / negative_denominator if negative_denominator else None
        ),
        "supported_f1": f1,
    }


def _reference_set_metrics(rows: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    expected_key = f"expected_{kind}_passage_ids"
    predicted_key = f"predicted_{kind}_passage_ids"
    expected_all: set[tuple[str, str, str]] = set()
    predicted_all: set[tuple[str, str, str]] = set()
    # Passage IDs are unique only within a case, and one case may have several
    # distinct aspects. Include both case and aspect IDs to prevent cross-aspect
    # matches from being counted as correct provenance.
    for row in rows:
        identity = (row["case_id"], row["aspect_id"])
        expected_all.update((*identity, item) for item in row[expected_key])
        predicted_all.update((*identity, item) for item in row[predicted_key])
    true_positive = len(expected_all & predicted_all)
    precision = true_positive / len(predicted_all) if predicted_all else None
    recall = true_positive / len(expected_all) if expected_all else None
    return {
        "expected_references": len(expected_all),
        "predicted_references": len(predicted_all),
        "matched_references": true_positive,
        "precision": precision,
        "recall": recall,
    }


def _nearest_rank_percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def build_metrics(case_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build status, supported-vs-unsupported, language, provenance and runtime metrics."""
    aspect_rows = [
        {
            **aspect,
            "case_id": record["case_id"],
            "question_language_group": record["question_language_group"],
            "transcript_language_group": record["transcript_language_group"],
        }
        for record in case_records
        for aspect in record["assessments"]
        if aspect["predicted_status"] is not None
    ]
    metrics = _class_metrics(aspect_rows)
    metrics["supported_binary"] = _binary_supported_metrics(aspect_rows)
    metrics["supporting_reference_alignment"] = _reference_set_metrics(
        aspect_rows, "supporting"
    )
    metrics["contradicting_reference_alignment"] = _reference_set_metrics(
        aspect_rows, "contradicting"
    )

    per_question_language = {}
    per_transcript_language = {}
    for language in SUPPORTED_LANGUAGES:
        question_rows = [
            row for row in aspect_rows
            if row["question_language_group"] == language
        ]
        transcript_rows = [
            row for row in aspect_rows
            if row["transcript_language_group"] == language
        ]
        per_question_language[language] = _class_metrics(question_rows)
        per_transcript_language[language] = _class_metrics(transcript_rows)

    diagnostics = [record for record in case_records]
    status_counts = dict(Counter(
        record["assessment_run_status"] for record in diagnostics
    ))
    attempted_calls = sum(record["request_count"] for record in diagnostics)
    input_token_values = [
        record["input_tokens"] for record in diagnostics
        if record["input_tokens"] is not None
    ]
    output_token_values = [
        record["output_tokens"] for record in diagnostics
        if record["output_tokens"] is not None
    ]
    elapsed_values = [
        record["elapsed_ms"] for record in diagnostics
        if record["request_count"] > 0
    ]
    completed = status_counts.get(AssessmentRunStatus.COMPLETED.value, 0)
    return {
        "semantic": metrics,
        "per_question_language": per_question_language,
        "per_transcript_language": per_transcript_language,
        "operational": {
            "attempted_cases": len(diagnostics),
            "completed_assessments": completed,
            "assessment_run_status_counts": status_counts,
            "application_level_provider_calls": attempted_calls,
            "provider_error_count": status_counts.get(
                AssessmentRunStatus.PROVIDER_ERROR.value, 0
            ),
            "invalid_response_count": status_counts.get(
                AssessmentRunStatus.INVALID_RESPONSE.value, 0
            ),
            "budget_exceeded_count": status_counts.get(
                AssessmentRunStatus.BUDGET_EXCEEDED.value, 0
            ),
            "rate_limit_count": sum(
                1 for record in diagnostics
                if record["error_code"] == "provider_rate_limited"
            ),
            "input_token_usage_observed_count": len(input_token_values),
            "output_token_usage_observed_count": len(output_token_values),
            "input_tokens_total": sum(input_token_values) if input_token_values else None,
            "output_tokens_total": sum(output_token_values) if output_token_values else None,
            "mean_input_tokens": (
                statistics.mean(input_token_values) if input_token_values else None
            ),
            "mean_output_tokens": (
                statistics.mean(output_token_values) if output_token_values else None
            ),
            "latency_ms_p50": _nearest_rank_percentile(elapsed_values, 0.50),
            "latency_ms_p95": _nearest_rank_percentile(elapsed_values, 0.95),
            "latency_ms_mean": (
                statistics.mean(elapsed_values) if elapsed_values else None
            ),
        },
    }


def _quality_gate_eligibility(
    manifest: dict[str, Any],
    split: str,
    run_status: str,
    case_records: list[dict[str, Any]],
) -> dict[str, Any]:
    reasons = []
    if split != "held_out":
        reasons.append("final_quality_gate_requires_held_out_split")
    if manifest.get("annotation_status") != "adjudicated":
        reasons.append("labels_are_not_independently_adjudicated")
    if run_status != "completed":
        reasons.append("held_out_run_is_incomplete_or_interrupted")

    expected_case_ids = {
        case["case_id"] for case in manifest.get("cases", [])
        if case.get("split") == "held_out"
    }
    actual_case_ids = {record.get("case_id") for record in case_records}
    if actual_case_ids != expected_case_ids:
        reasons.append("held_out_case_coverage_is_incomplete_or_inconsistent")
    if any(
        record.get("assessment_run_status") != AssessmentRunStatus.COMPLETED.value
        for record in case_records
    ):
        reasons.append("one_or_more_held_out_assessment_calls_failed_or_returned_invalid_output")
    if any(
        aspect.get("predicted_status") not in STATUS_ORDER
        for record in case_records
        for aspect in record.get("assessments", [])
    ):
        reasons.append("one_or_more_held_out_aspects_has_no_valid_prediction")

    aspect_rows = [
        aspect
        for record in case_records
        for aspect in record["assessments"]
    ]
    if len(aspect_rows) < MIN_HELD_OUT_ASPECTS:
        reasons.append(
            f"held_out_aspects_below_minimum_{MIN_HELD_OUT_ASPECTS}"
        )
    status_counts = Counter(row["expected_status"] for row in aspect_rows)
    for status in STATUS_ORDER:
        if status_counts[status] < MIN_HELD_OUT_PER_STATUS:
            reasons.append(f"held_out_{status}_below_minimum_{MIN_HELD_OUT_PER_STATUS}")
    question_language_counts = Counter(
        record["question_language_group"] for record in case_records
    )
    transcript_language_counts = Counter(
        record["transcript_language_group"] for record in case_records
    )
    for language in SUPPORTED_LANGUAGES:
        if question_language_counts[language] < MIN_HELD_OUT_PER_LANGUAGE:
            reasons.append(
                f"held_out_{language}_question_language_cases_below_minimum_"
                f"{MIN_HELD_OUT_PER_LANGUAGE}"
            )
        if transcript_language_counts[language] < MIN_HELD_OUT_PER_LANGUAGE:
            reasons.append(
                f"held_out_{language}_transcript_language_cases_below_minimum_"
                f"{MIN_HELD_OUT_PER_LANGUAGE}"
            )
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "minimums": {
            "total_aspect_observations": MIN_HELD_OUT_ASPECTS,
            "per_status": MIN_HELD_OUT_PER_STATUS,
            "per_question_language_cases": MIN_HELD_OUT_PER_LANGUAGE,
            "per_transcript_language_cases": MIN_HELD_OUT_PER_LANGUAGE,
        },
        "note": (
            "Eligibility is not a semantic pass. Human review of metrics and failure "
            "examples remains mandatory; no automatic quality_passed flag is emitted."
        ),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _effective_assessor_protocol(provider: str, model: str) -> dict[str, Any]:
    """Capture non-secret settings and code fingerprints that affect calibration."""
    from Backend.rag_system import config

    if provider == "groq":
        reasoning_effort = config.GROQ_REASONING_EFFORT
        configured_max_tokens = config.GROQ_MAX_TOKENS
    elif provider == "huggingface":
        reasoning_effort = config.HF_REASONING_EFFORT
        configured_max_tokens = config.HF_MAX_TOKENS
    else:
        raise ValueError("provider must be 'groq' or 'huggingface'.")
    return {
        "provider": provider,
        "model": model,
        "temperature": 0.0,
        "reasoning_effort": reasoning_effort,
        "max_input_chars": DEFAULT_MAX_ASSESSMENT_INPUT_CHARS,
        "max_passages": DEFAULT_MAX_ASSESSMENT_PASSAGES,
        "max_aspects": DEFAULT_MAX_ASSESSMENT_ASPECTS,
        "configured_max_output_tokens": configured_max_tokens,
        "effective_max_output_tokens": min(
            DEFAULT_MAX_ASSESSMENT_OUTPUT_TOKENS,
            configured_max_tokens,
        ),
        "assessor_module_sha256": _sha256_file(
            Path(evidence_assessment_module.__file__)
        ),
        "contracts_module_sha256": _sha256_file(
            Path(evidence_contracts_module.__file__)
        ),
        "config_module_sha256": _sha256_file(Path(config.__file__)),
        "client_factory_module_sha256": _sha256_file(
            PROJECT_ROOT / "Backend" / "rag_system" / "chain.py"
        ),
        "evaluation_runner_sha256": _sha256_file(Path(__file__)),
    }


def load_threshold_lock(
    lock_path: Path,
    *,
    dataset_id: str,
    dataset_sha256: str,
    annotation_status: str,
    expected_development_case_ids: set[str],
    provider: str,
    model: str,
) -> tuple[float, dict[str, Any]]:
    """Require a frozen threshold backed by a completed development report."""
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ThresholdLockError(f"Cannot read threshold lock: {lock_path}") from error
    if not isinstance(lock, dict) or lock.get("schema_version") != LOCK_SCHEMA_VERSION:
        raise ThresholdLockError("Threshold lock has an unsupported schema.")
    if lock.get("dataset_id") != dataset_id or lock.get("dataset_sha256") != dataset_sha256:
        raise ThresholdLockError("Threshold lock does not match this exact dataset revision.")
    if annotation_status != "adjudicated":
        raise ThresholdLockError(
            "Threshold calibration requires independently adjudicated human labels."
        )

    current_assessor_protocol = _effective_assessor_protocol(provider, model)
    if lock.get("assessor_protocol") != current_assessor_protocol:
        raise ThresholdLockError(
            "Threshold lock assessor protocol does not match the current provider, model, "
            "configuration, or code fingerprints."
        )

    threshold = lock.get("minimum_support_score")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not 0.0 <= float(threshold) <= 1.0
    ):
        raise ThresholdLockError("Threshold lock minimum_support_score is invalid.")

    report_rel = _require_text(
        lock.get("development_report_path"), "threshold_lock.development_report_path"
    )
    report_path = Path(report_rel)
    if not report_path.is_absolute():
        report_path = lock_path.parent / report_path
    expected_report_sha = _require_text(
        lock.get("development_report_sha256"),
        "threshold_lock.development_report_sha256",
    )
    try:
        actual_report_sha = _sha256_file(report_path)
        dev_report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ThresholdLockError("Referenced development report cannot be read.") from error
    if actual_report_sha != expected_report_sha:
        raise ThresholdLockError("Development report hash does not match the threshold lock.")
    if not isinstance(dev_report, dict):
        raise ThresholdLockError("Development report must be a JSON object.")
    dev_cases = dev_report.get("case_results")
    split_case_count = dev_report.get("case_count_in_split")
    if (
        not isinstance(dev_cases, list)
        or isinstance(split_case_count, bool)
        or not isinstance(split_case_count, int)
        or split_case_count <= 0
    ):
        raise ThresholdLockError("Development report is missing complete case-level results.")
    dev_case_ids = [
        case.get("case_id") if isinstance(case, dict) else None
        for case in dev_cases
    ]
    if (
        dev_report.get("split") != "development"
        or dev_report.get("run_status") != "completed"
        or dev_report.get("annotation_status") != "adjudicated"
        or lock.get("annotation_status") != "adjudicated"
        or dev_report.get("dataset_id") != dataset_id
        or dev_report.get("dataset_sha256") != dataset_sha256
        or dev_report.get("minimum_support_score") != float(threshold)
        or dev_report.get("assessor_protocol") != current_assessor_protocol
        or dev_report.get("provider") != provider
        or dev_report.get("model") != model
        or dev_report.get("selected_case_count") != split_case_count
        or len(dev_cases) != split_case_count
        or len(dev_case_ids) != len(set(dev_case_ids))
        or set(dev_case_ids) != set(expected_development_case_ids)
        or split_case_count != len(expected_development_case_ids)
        or any(
            not isinstance(case, dict)
            or case.get("assessment_run_status") != AssessmentRunStatus.COMPLETED.value
            or not isinstance(case.get("assessments"), list)
            or not case["assessments"]
            or any(
                not isinstance(aspect, dict)
                or aspect.get("predicted_status") not in STATUS_ORDER
                for aspect in case["assessments"]
            )
            for case in dev_cases
        )
    ):
        raise ThresholdLockError(
            "Threshold lock must reference a complete, successful development run on "
            "this exact human-adjudicated dataset and selected threshold."
        )
    _require_text(lock.get("selected_by"), "threshold_lock.selected_by")
    _require_text(lock.get("selection_rationale"), "threshold_lock.selection_rationale")
    return float(threshold), {
        "lock_path": str(lock_path),
        "lock_sha256": _sha256_file(lock_path),
        "development_report_path": str(report_path),
        "development_report_sha256": actual_report_sha,
        "selected_by": lock["selected_by"],
    }


def _write_report_atomic(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_evaluation(
    *,
    manifest: dict[str, Any],
    dataset_sha256: str,
    split: str,
    client: Any,
    provider: str,
    model: str,
    minimum_support_score: float,
    output_path: Path,
    max_cases: int | None = None,
    sleep_seconds: float = 1.0,
    threshold_lock_metadata: dict[str, Any] | None = None,
    assessor: Callable[..., Any] = assess_evidence_support,
) -> dict[str, Any]:
    """Run one case at a time, checkpoint reports, and stop after provider 429."""
    split_cases = [case for case in manifest["cases"] if case["split"] == split]
    split_case_count = len(split_cases)
    if max_cases is not None:
        if isinstance(max_cases, bool) or not isinstance(max_cases, int) or max_cases <= 0:
            raise ValueError("max_cases must be a positive integer or None.")
        selected = split_cases[:max_cases]
    else:
        selected = split_cases
    if not selected:
        raise ValueError(f"No cases selected for split {split!r}.")
    selection_was_truncated = len(selected) < split_case_count
    if (
        isinstance(sleep_seconds, bool)
        or not isinstance(sleep_seconds, (int, float))
        or not math.isfinite(float(sleep_seconds))
        or sleep_seconds < 0
    ):
        raise ValueError("sleep_seconds must be finite and non-negative.")

    report = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "report_id": "phase11_0C_semantic_evidence_assessment_evaluation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_status": "running",
        "dataset_id": manifest["dataset_id"],
        "dataset_sha256": dataset_sha256,
        "annotation_status": manifest["annotation_status"],
        "split": split,
        "provider": provider,
        "model": model,
        "assessor_protocol": _effective_assessor_protocol(provider, model),
        "minimum_support_score": float(minimum_support_score),
        "threshold_lock": threshold_lock_metadata,
        "case_count_in_split": split_case_count,
        "selected_case_count": len(selected),
        "case_results": [],
        "metrics": None,
        "quality_gate_eligibility": None,
        "stop_reason": None,
    }
    _write_report_atomic(output_path, report)

    stop_reason = None
    for index, case in enumerate(selected):
        analysis, passages = _case_inputs(case)
        result = assessor(
            analysis,
            passages,
            client,
            provider=provider,
            model=model,
            minimum_support_score=float(minimum_support_score),
        )
        record = _case_prediction_record(case, result)
        report["case_results"].append(record)
        _write_report_atomic(output_path, report)

        if record["error_code"] == "provider_rate_limited":
            stop_reason = "provider_rate_limited_no_retry_or_further_calls"
            break
        if index < len(selected) - 1 and sleep_seconds:
            time.sleep(float(sleep_seconds))

    processed_ids = {record["case_id"] for record in report["case_results"]}
    skipped = [
        case["case_id"] for case in split_cases
        if case["case_id"] not in processed_ids
    ]
    complete_selection = (
        not selection_was_truncated
        and len(report["case_results"]) == len(selected)
        and len(report["case_results"]) == split_case_count
    )
    if selection_was_truncated and stop_reason is None:
        stop_reason = "max_cases_limit_reached_partial_run"
    report["run_status"] = "completed" if complete_selection else "partial"
    report["stop_reason"] = stop_reason
    report["skipped_case_ids"] = skipped
    report["metrics"] = build_metrics(report["case_results"])
    report["quality_gate_eligibility"] = _quality_gate_eligibility(
        manifest,
        split,
        report["run_status"],
        report["case_results"],
    )
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_report_atomic(output_path, report)
    return report


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _unit_interval(value: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number between 0 and 1") from error
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise argparse.ArgumentTypeError("must be a finite number between 0 and 1")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run isolated Phase 11.0C semantic evidence assessment evaluation."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=VALID_SPLITS, required=True)
    parser.add_argument(
        "--minimum-support-score",
        type=_unit_interval,
        default=None,
        help="Development threshold. Held-out runs take the threshold from --threshold-lock.",
    )
    parser.add_argument(
        "--threshold-lock",
        type=Path,
        default=None,
        help="Required for held-out runs; ties the test threshold to a completed development report.",
    )
    parser.add_argument("--max-cases", type=_positive_int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--allow-provisional-labels",
        action="store_true",
        help="Explicitly permits synthetic/provisional fixtures; reports remain ineligible for acceptance.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest, dataset_sha256 = load_manifest(args.dataset)
    selected_cases = [
        case for case in manifest["cases"] if case["split"] == args.split
    ]
    if not selected_cases:
        raise DatasetValidationError(f"No {args.split} cases exist in the manifest.")

    if manifest["annotation_status"] != "adjudicated" and not args.allow_provisional_labels:
        raise DatasetValidationError(
            "This dataset has provisional labels. Use --allow-provisional-labels only "
            "for harness smoke tests; provisional labels cannot qualify a quality gate."
        )

    if not args.validate_only:
        from Backend.rag_system import config
        provider = config.LLM_PROVIDER
        model = config.GROQ_MODEL_ID if provider == "groq" else config.HF_MODEL_ID
    else:
        provider = None
        model = None

    threshold_lock_metadata = None
    if args.split == "held_out" and not args.validate_only:
        if args.threshold_lock is None:
            raise ThresholdLockError(
                "Held-out runs require --threshold-lock from a completed development run."
            )
        threshold, threshold_lock_metadata = load_threshold_lock(
            args.threshold_lock,
            dataset_id=manifest["dataset_id"],
            dataset_sha256=dataset_sha256,
            annotation_status=manifest["annotation_status"],
            expected_development_case_ids={
                case["case_id"] for case in manifest["cases"]
                if case["split"] == "development"
            },
            provider=provider,
            model=model,
        )
        if (
            args.minimum_support_score is not None
            and args.minimum_support_score != threshold
        ):
            raise ThresholdLockError(
                "The supplied threshold differs from the frozen development threshold."
            )
    else:
        threshold = (
            0.75 if args.minimum_support_score is None
            else args.minimum_support_score
        )

    if args.validate_only:
        summary = {
            "dataset_id": manifest["dataset_id"],
            "dataset_sha256": dataset_sha256,
            "annotation_status": manifest["annotation_status"],
            "selected_split": args.split,
            "selected_cases": len(selected_cases),
            "question_language_counts": dict(Counter(
                case["question_language_group"] for case in selected_cases
            )),
            "transcript_language_counts": dict(Counter(
                case["transcript_language_group"] for case in selected_cases
            )),
            "status_counts": dict(Counter(
                label["expected_status"]
                for case in selected_cases
                for label in case["labels"]
            )),
            "validation": "passed",
            "quality_gate_eligible": False,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    from Backend.rag_system.chain import create_llm_client

    client = create_llm_client()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output or (
        DEFAULT_OUTPUT_DIR
        / f"phase11_0C_{args.split}_{timestamp}.json"
    )

    report = run_evaluation(
        manifest=manifest,
        dataset_sha256=dataset_sha256,
        split=args.split,
        client=client,
        provider=provider,
        model=model,
        minimum_support_score=threshold,
        output_path=output_path,
        max_cases=args.max_cases,
        sleep_seconds=args.sleep_seconds,
        threshold_lock_metadata=threshold_lock_metadata,
    )
    print(json.dumps({
        "report_path": str(output_path),
        "run_status": report["run_status"],
        "dataset_id": report["dataset_id"],
        "split": report["split"],
        "minimum_support_score": report["minimum_support_score"],
        "case_count": report["selected_case_count"],
        "metrics": report["metrics"],
        "quality_gate_eligibility": report["quality_gate_eligibility"],
        "stop_reason": report["stop_reason"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["run_status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
