"""Versioned, provider-neutral contracts for question and evidence evaluation.

These data structures deliberately do not perform retrieval, semantic matching, or
LLM verification. They define the boundary between those future components so
each can be implemented and tested independently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
import math
import re
from typing import Sequence


CONTRACT_SCHEMA_VERSION = "1.0"
_LANGUAGE_TAG_PATTERN = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")


class QuestionIntent(StrEnum):
    """Broad answer intent; not a claim that the transcript contains an answer."""

    DIRECT_FACT = "direct_fact"
    EXPLANATION = "explanation"
    COMPARISON = "comparison"
    LIST = "list"
    SEQUENCE = "sequence"
    TEMPORAL = "temporal"
    OVERVIEW = "overview"
    UNKNOWN = "unknown"


class EvidenceSupportStatus(StrEnum):
    """Assessment of one requested evidence aspect against cited passages."""

    SUPPORTED = "supported"
    PARTIAL = "partial"
    MISSING = "missing"
    CONTRADICTED = "contradicted"
    CONFLICTED = "conflicted"
    UNKNOWN = "unknown"


class CoverageStatus(StrEnum):
    """Aggregate required-evidence state; not a model-confidence label."""

    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    INDETERMINATE = "indeterminate"


class RetrievalMethod(StrEnum):
    """Retrieval method label for diagnostics, independent of implementation."""

    SEMANTIC = "semantic"
    LEXICAL = "lexical"
    HYBRID = "hybrid"
    CROSS_LINGUAL = "cross_lingual"
    TEMPORAL = "temporal"
    OTHER = "other"


def _clean_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    cleaned = " ".join(value.split())
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty.")
    return cleaned


def _optional_text(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return _clean_text(value, field_name)


def _string_tuple(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{field_name} must be a sequence of strings.")
    cleaned = tuple(_clean_text(value, field_name) for value in values)
    if len(cleaned) != len(set(cleaned)):
        raise ValueError(f"{field_name} must not contain duplicates.")
    return cleaned


def _reference_tuple(values: Sequence["EvidenceReference"], field_name: str):
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{field_name} must be a sequence of EvidenceReference values.")
    references = tuple(values)
    if any(not isinstance(item, EvidenceReference) for item in references):
        raise TypeError(f"{field_name} must contain only EvidenceReference values.")
    identities = [(item.video_id, item.chunk_id) for item in references]
    if len(identities) != len(set(identities)):
        raise ValueError(f"{field_name} must not contain duplicate video/chunk references.")
    return references


def _finite_number(value: float, field_name: str, *, minimum: float | None = None,
                   maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite.")
    if minimum is not None and result < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}.")
    if maximum is not None and result > maximum:
        raise ValueError(f"{field_name} must be at most {maximum}.")
    return result


@dataclass(frozen=True, slots=True)
class TemporalConstraint:
    """Requested interval on the transcript timeline, in seconds."""

    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        start = _finite_number(self.start_seconds, "start_seconds", minimum=0.0)
        end = _finite_number(self.end_seconds, "end_seconds", minimum=0.0)
        if end <= start:
            raise ValueError("end_seconds must be greater than start_seconds.")
        object.__setattr__(self, "start_seconds", start)
        object.__setattr__(self, "end_seconds", end)


@dataclass(frozen=True, slots=True)
class EvidenceAspect:
    """One distinct information need implied by the user's question.

    retrieval_hints are optional planner hints, not gold phrases and not proof
    that a passage supports the aspect.
    """

    aspect_id: str
    description: str
    required: bool = True
    weight: float = 1.0
    retrieval_hints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "aspect_id", _clean_text(self.aspect_id, "aspect_id"))
        object.__setattr__(self, "description", _clean_text(self.description, "description"))
        if not isinstance(self.required, bool):
            raise TypeError("required must be a boolean.")
        object.__setattr__(
            self,
            "weight",
            _finite_number(self.weight, "weight", minimum=0.000001),
        )
        object.__setattr__(
            self,
            "retrieval_hints",
            _string_tuple(self.retrieval_hints, "retrieval_hints"),
        )


@dataclass(frozen=True, slots=True)
class QuestionAnalysis:
    """Provider-neutral interpretation of a user question.

    The analysis describes what is being asked. It does not determine
    answerability; that requires transcript evidence.
    """

    intent: QuestionIntent
    language: str | None
    aspects: tuple[EvidenceAspect, ...]
    video_id: str | None = None
    subject: str | None = None
    constraints: tuple[str, ...] = ()
    temporal_constraint: TemporalConstraint | None = None
    planner_version: str = CONTRACT_SCHEMA_VERSION
    analysis_source: str = "unspecified"
    confidence: float | None = None

    def __post_init__(self) -> None:
        try:
            intent = self.intent if isinstance(self.intent, QuestionIntent) else QuestionIntent(self.intent)
        except (TypeError, ValueError) as error:
            raise ValueError("intent must be a supported QuestionIntent.") from error
        object.__setattr__(self, "intent", intent)

        language = self.language
        if language is not None:
            language = _clean_text(language, "language")
            if not _LANGUAGE_TAG_PATTERN.fullmatch(language):
                raise ValueError("language must be a BCP-47-style language tag, such as 'en', 'hi', or 'ar'.")
        object.__setattr__(self, "language", language)

        if isinstance(self.aspects, (str, bytes)) or not isinstance(self.aspects, Sequence):
            raise TypeError("aspects must be a sequence of EvidenceAspect values.")
        aspects = tuple(self.aspects)
        if any(not isinstance(item, EvidenceAspect) for item in aspects):
            raise TypeError("aspects must contain only EvidenceAspect values.")
        aspect_ids = [item.aspect_id for item in aspects]
        if len(aspect_ids) != len(set(aspect_ids)):
            raise ValueError("aspect IDs must be unique within a QuestionAnalysis.")
        object.__setattr__(self, "aspects", aspects)
        object.__setattr__(self, "video_id", _optional_text(self.video_id, "video_id"))

        object.__setattr__(self, "subject", _optional_text(self.subject, "subject"))
        object.__setattr__(self, "constraints", _string_tuple(self.constraints, "constraints"))
        if self.temporal_constraint is not None and not isinstance(
            self.temporal_constraint, TemporalConstraint
        ):
            raise TypeError("temporal_constraint must be a TemporalConstraint or None.")
        object.__setattr__(
            self,
            "planner_version",
            _clean_text(self.planner_version, "planner_version"),
        )
        object.__setattr__(
            self,
            "analysis_source",
            _clean_text(self.analysis_source, "analysis_source"),
        )
        if self.confidence is not None:
            object.__setattr__(
                self,
                "confidence",
                _finite_number(self.confidence, "confidence", minimum=0.0, maximum=1.0),
            )


@dataclass(frozen=True, slots=True)
class EvidenceReference:
    """Video-scoped transcript chunk provenance; transcript text is not stored."""

    video_id: str
    chunk_id: str | int
    start_seconds: float | None = None
    end_seconds: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "video_id", _clean_text(self.video_id, "video_id"))
        if isinstance(self.chunk_id, bool) or self.chunk_id is None:
            raise TypeError("chunk_id must be a non-empty string or integer.")
        if not isinstance(self.chunk_id, (str, int)):
            raise TypeError("chunk_id must be a non-empty string or integer.")
        object.__setattr__(self, "chunk_id", _clean_text(str(self.chunk_id), "chunk_id"))
        if (self.start_seconds is None) != (self.end_seconds is None):
            raise ValueError("start_seconds and end_seconds must both be set or both be None.")
        if self.start_seconds is not None:
            start = _finite_number(self.start_seconds, "start_seconds", minimum=0.0)
            end = _finite_number(self.end_seconds, "end_seconds", minimum=0.0)
            if end <= start:
                raise ValueError("end_seconds must be greater than start_seconds.")
            object.__setattr__(self, "start_seconds", start)
            object.__setattr__(self, "end_seconds", end)


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    """Evidence evaluator's assessment of one aspect and its source references."""

    aspect_id: str
    status: EvidenceSupportStatus
    support_score: float
    supporting_evidence: tuple[EvidenceReference, ...] = ()
    contradicting_evidence: tuple[EvidenceReference, ...] = ()
    method: str = "unspecified"

    def __post_init__(self) -> None:
        object.__setattr__(self, "aspect_id", _clean_text(self.aspect_id, "aspect_id"))
        try:
            status = (
                self.status
                if isinstance(self.status, EvidenceSupportStatus)
                else EvidenceSupportStatus(self.status)
            )
        except (TypeError, ValueError) as error:
            raise ValueError("status must be a supported EvidenceSupportStatus.") from error
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "support_score",
            _finite_number(self.support_score, "support_score", minimum=0.0, maximum=1.0),
        )
        supporting = _reference_tuple(self.supporting_evidence, "supporting_evidence")
        contradicting = _reference_tuple(self.contradicting_evidence, "contradicting_evidence")
        object.__setattr__(self, "supporting_evidence", supporting)
        object.__setattr__(self, "contradicting_evidence", contradicting)
        object.__setattr__(self, "method", _clean_text(self.method, "method"))

        if status in {EvidenceSupportStatus.SUPPORTED, EvidenceSupportStatus.PARTIAL}:
            if not supporting:
                raise ValueError(f"{status.value} assessments require supporting_evidence.")
            if contradicting:
                raise ValueError(
                    "Use CONFLICTED when an aspect has both supporting and contradicting evidence."
                )
            if status is EvidenceSupportStatus.SUPPORTED and self.support_score <= 0.0:
                raise ValueError("SUPPORTED assessments require a positive support_score.")
            if status is EvidenceSupportStatus.PARTIAL and self.support_score >= 1.0:
                raise ValueError("PARTIAL assessments require support_score below 1.0.")
        elif status is EvidenceSupportStatus.MISSING:
            if supporting or contradicting:
                raise ValueError("MISSING assessments must not include evidence references.")
            if self.support_score != 0.0:
                raise ValueError("MISSING assessments must have a zero support_score.")
        elif status is EvidenceSupportStatus.CONTRADICTED:
            if not contradicting or supporting:
                raise ValueError(
                    "CONTRADICTED assessments require contradicting_evidence only."
                )
            if self.support_score != 0.0:
                raise ValueError("CONTRADICTED assessments must have a zero support_score.")
        elif status is EvidenceSupportStatus.CONFLICTED:
            if not supporting or not contradicting:
                raise ValueError(
                    "CONFLICTED assessments require both supporting and contradicting evidence."
                )
        elif status is EvidenceSupportStatus.UNKNOWN:
            if self.support_score != 0.0:
                raise ValueError("UNKNOWN assessments must have a zero support_score.")


@dataclass(frozen=True, slots=True)
class EvidenceCoverageSummary:
    """Deterministic aggregation of aspect assessments; not semantic verification."""

    status: CoverageStatus
    required_aspect_count: int
    supported_required_aspect_count: int
    coverage_ratio: float | None
    missing_aspect_ids: tuple[str, ...]
    contradicted_aspect_ids: tuple[str, ...]
    unknown_aspect_ids: tuple[str, ...]
    assessments: tuple[EvidenceAssessment, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.status, CoverageStatus):
            object.__setattr__(self, "status", CoverageStatus(self.status))
        for name in ("required_aspect_count", "supported_required_aspect_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if self.supported_required_aspect_count > self.required_aspect_count:
            raise ValueError("supported_required_aspect_count cannot exceed required_aspect_count.")
        if self.coverage_ratio is not None:
            object.__setattr__(
                self,
                "coverage_ratio",
                _finite_number(self.coverage_ratio, "coverage_ratio", minimum=0.0, maximum=1.0),
            )
        for name in (
            "missing_aspect_ids",
            "contradicted_aspect_ids",
            "unknown_aspect_ids",
        ):
            object.__setattr__(self, name, _string_tuple(getattr(self, name), name))
        if isinstance(self.assessments, (str, bytes)) or not isinstance(self.assessments, Sequence):
            raise TypeError("assessments must be a sequence of EvidenceAssessment values.")
        assessments = tuple(self.assessments)
        if any(not isinstance(item, EvidenceAssessment) for item in assessments):
            raise TypeError("assessments must contain only EvidenceAssessment values.")
        assessment_ids = [item.aspect_id for item in assessments]
        if len(assessment_ids) != len(set(assessment_ids)):
            raise ValueError("summary assessments must not contain duplicate aspect IDs.")
        object.__setattr__(self, "assessments", assessments)
        if self.status is CoverageStatus.INDETERMINATE:
            if self.required_aspect_count != 0 or self.coverage_ratio is not None:
                raise ValueError("INDETERMINATE summaries require zero required aspects and no ratio.")
        elif self.required_aspect_count == 0 or self.coverage_ratio is None:
            raise ValueError("SUFFICIENT and INSUFFICIENT summaries require assessed required aspects.")
        if self.status is CoverageStatus.SUFFICIENT:
            if self.supported_required_aspect_count != self.required_aspect_count:
                raise ValueError("SUFFICIENT summaries must support every required aspect.")
            if self.missing_aspect_ids or self.contradicted_aspect_ids or self.unknown_aspect_ids:
                raise ValueError("SUFFICIENT summaries must not contain missing or unresolved aspects.")
        elif self.status is CoverageStatus.INSUFFICIENT:
            if self.supported_required_aspect_count >= self.required_aspect_count:
                raise ValueError("INSUFFICIENT summaries must have at least one unsupported required aspect.")
            expected_missing_count = (
                self.required_aspect_count - self.supported_required_aspect_count
            )
            if len(self.missing_aspect_ids) != expected_missing_count:
                raise ValueError("INSUFFICIENT summaries must list every unsupported required aspect.")
            missing_ids = set(self.missing_aspect_ids)
            if not set(self.contradicted_aspect_ids).issubset(missing_ids):
                raise ValueError("contradicted_aspect_ids must be included in missing_aspect_ids.")
            if not set(self.unknown_aspect_ids).issubset(missing_ids):
                raise ValueError("unknown_aspect_ids must be included in missing_aspect_ids.")


def evaluate_evidence_coverage(
    analysis: QuestionAnalysis,
    assessments: Sequence[EvidenceAssessment],
    *,
    minimum_support_score: float = 0.75,
) -> EvidenceCoverageSummary:
    """Aggregate assessments for required aspects without performing semantic matching.

    An aspect counts as covered only if its assessment is SUPPORTED, its score
    meets the configured threshold, and it has at least one supporting transcript
    reference. Every required aspect must meet that condition for the aggregate
    state to be SUFFICIENT. No required aspects yields INDETERMINATE, never
    automatic success.
    """

    if not isinstance(analysis, QuestionAnalysis):
        raise TypeError("analysis must be a QuestionAnalysis.")
    threshold = _finite_number(
        minimum_support_score,
        "minimum_support_score",
        minimum=0.0,
        maximum=1.0,
    )
    if isinstance(assessments, (str, bytes)) or not isinstance(assessments, Sequence):
        raise TypeError("assessments must be a sequence of EvidenceAssessment values.")
    assessment_values = tuple(assessments)
    if any(not isinstance(item, EvidenceAssessment) for item in assessment_values):
        raise TypeError("assessments must contain only EvidenceAssessment values.")

    aspect_by_id = {aspect.aspect_id: aspect for aspect in analysis.aspects}
    assessment_by_id = {}
    for assessment in assessment_values:
        evidence_references = (
            *assessment.supporting_evidence,
            *assessment.contradicting_evidence,
        )
        if analysis.video_id is not None and any(
            reference.video_id != analysis.video_id
            for reference in evidence_references
        ):
            raise ValueError(
                "Evidence references must match QuestionAnalysis.video_id."
            )
        if assessment.aspect_id not in aspect_by_id:
            raise ValueError(
                f"Assessment references unknown aspect ID: {assessment.aspect_id!r}."
            )
        if assessment.aspect_id in assessment_by_id:
            raise ValueError(
                f"Duplicate assessment for aspect ID: {assessment.aspect_id!r}."
            )
        assessment_by_id[assessment.aspect_id] = assessment

    required_aspects = [aspect for aspect in analysis.aspects if aspect.required]
    if not required_aspects:
        return EvidenceCoverageSummary(
            status=CoverageStatus.INDETERMINATE,
            required_aspect_count=0,
            supported_required_aspect_count=0,
            coverage_ratio=None,
            missing_aspect_ids=(),
            contradicted_aspect_ids=(),
            unknown_aspect_ids=(),
            assessments=assessment_values,
        )

    total_weight = sum(aspect.weight for aspect in required_aspects)
    covered_weight = 0.0
    supported_ids = []
    missing_ids = []
    contradicted_ids = []
    unknown_ids = []

    for aspect in required_aspects:
        assessment = assessment_by_id.get(aspect.aspect_id)
        if assessment is None:
            missing_ids.append(aspect.aspect_id)
            unknown_ids.append(aspect.aspect_id)
            continue

        if assessment.status is EvidenceSupportStatus.CONTRADICTED:
            contradicted_ids.append(aspect.aspect_id)
            missing_ids.append(aspect.aspect_id)
            continue
        if assessment.status is EvidenceSupportStatus.CONFLICTED:
            contradicted_ids.append(aspect.aspect_id)
            missing_ids.append(aspect.aspect_id)
            continue
        if assessment.status is EvidenceSupportStatus.UNKNOWN:
            unknown_ids.append(aspect.aspect_id)
            missing_ids.append(aspect.aspect_id)
            continue
        if assessment.status is EvidenceSupportStatus.MISSING:
            missing_ids.append(aspect.aspect_id)
            continue

        covered_weight += aspect.weight * assessment.support_score
        if (
            assessment.status is EvidenceSupportStatus.SUPPORTED
            and assessment.support_score >= threshold
            and assessment.supporting_evidence
        ):
            supported_ids.append(aspect.aspect_id)
        else:
            missing_ids.append(aspect.aspect_id)

    ratio = min(1.0, max(0.0, covered_weight / total_weight))
    is_sufficient = len(supported_ids) == len(required_aspects)

    return EvidenceCoverageSummary(
        status=CoverageStatus.SUFFICIENT if is_sufficient else CoverageStatus.INSUFFICIENT,
        required_aspect_count=len(required_aspects),
        supported_required_aspect_count=len(supported_ids),
        coverage_ratio=ratio,
        missing_aspect_ids=tuple(missing_ids),
        contradicted_aspect_ids=tuple(contradicted_ids),
        unknown_aspect_ids=tuple(unknown_ids),
        assessments=assessment_values,
    )


@dataclass(frozen=True, slots=True)
class RetrievalAttempt:
    """One bounded retrieval operation; query text is intentionally not logged."""

    stage: str
    query_variant_id: str
    method: RetrievalMethod
    candidate_evidence: tuple[EvidenceReference, ...]
    accepted_evidence: tuple[EvidenceReference, ...]
    elapsed_ms: float
    rejection_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", _clean_text(self.stage, "stage"))
        object.__setattr__(
            self,
            "query_variant_id",
            _clean_text(self.query_variant_id, "query_variant_id"),
        )
        try:
            method = self.method if isinstance(self.method, RetrievalMethod) else RetrievalMethod(self.method)
        except (TypeError, ValueError) as error:
            raise ValueError("method must be a supported RetrievalMethod.") from error
        object.__setattr__(self, "method", method)
        candidates = _reference_tuple(self.candidate_evidence, "candidate_evidence")
        accepted = _reference_tuple(self.accepted_evidence, "accepted_evidence")
        candidate_ids = {(item.video_id, item.chunk_id) for item in candidates}
        if any((item.video_id, item.chunk_id) not in candidate_ids for item in accepted):
            raise ValueError("accepted_evidence must be a subset of candidate_evidence.")
        object.__setattr__(self, "candidate_evidence", candidates)
        object.__setattr__(self, "accepted_evidence", accepted)
        object.__setattr__(
            self,
            "elapsed_ms",
            _finite_number(self.elapsed_ms, "elapsed_ms", minimum=0.0),
        )
        object.__setattr__(
            self,
            "rejection_codes",
            _string_tuple(self.rejection_codes, "rejection_codes"),
        )

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_evidence)

    @property
    def accepted_count(self) -> int:
        return len(self.accepted_evidence)


@dataclass(frozen=True, slots=True)
class RetrievalDiagnostics:
    """Video-scoped retrieval telemetry without transcript text or provider secrets."""

    video_id: str
    attempts: tuple[RetrievalAttempt, ...]
    final_evidence: tuple[EvidenceReference, ...]
    elapsed_ms: float
    recovery_rounds: int = 0
    budget_exhausted: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "video_id", _clean_text(self.video_id, "video_id"))
        if isinstance(self.attempts, (str, bytes)) or not isinstance(self.attempts, Sequence):
            raise TypeError("attempts must be a sequence of RetrievalAttempt values.")
        attempts = tuple(self.attempts)
        if any(not isinstance(item, RetrievalAttempt) for item in attempts):
            raise TypeError("attempts must contain only RetrievalAttempt values.")
        object.__setattr__(self, "attempts", attempts)
        final_evidence = _reference_tuple(self.final_evidence, "final_evidence")
        object.__setattr__(self, "final_evidence", final_evidence)
        all_references = [
            reference
            for attempt in attempts
            for reference in (*attempt.candidate_evidence, *attempt.accepted_evidence)
        ] + list(final_evidence)
        if any(reference.video_id != self.video_id for reference in all_references):
            raise ValueError("All retrieval evidence references must match diagnostics.video_id.")
        object.__setattr__(
            self,
            "elapsed_ms",
            _finite_number(self.elapsed_ms, "elapsed_ms", minimum=0.0),
        )
        if (
            isinstance(self.recovery_rounds, bool)
            or not isinstance(self.recovery_rounds, int)
            or self.recovery_rounds < 0
        ):
            raise ValueError("recovery_rounds must be a non-negative integer.")
        if not isinstance(self.budget_exhausted, bool):
            raise TypeError("budget_exhausted must be a boolean.")

    @property
    def candidate_count(self) -> int:
        return sum(attempt.candidate_count for attempt in self.attempts)

    @property
    def accepted_candidate_count(self) -> int:
        return sum(attempt.accepted_count for attempt in self.attempts)

    @property
    def final_evidence_count(self) -> int:
        return len(self.final_evidence)


def contract_to_dict(value) -> dict | list | str | int | float | bool | None:
    """Return a JSON-compatible representation of a contract dataclass."""

    if hasattr(value, "__dataclass_fields__"):
        return contract_to_dict(asdict(value))
    if isinstance(value, dict):
        return {str(key): contract_to_dict(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [contract_to_dict(item) for item in value]
    if isinstance(value, StrEnum):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported contract value type: {type(value).__name__}")
