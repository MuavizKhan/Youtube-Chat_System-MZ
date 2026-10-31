"""Contract tests for generic question analysis and evidence coverage."""

import json

import pytest

from Backend.rag_system.evidence_contracts import (
    CoverageStatus,
    EvidenceAspect,
    EvidenceAssessment,
    EvidenceCoverageSummary,
    EvidenceReference,
    EvidenceSupportStatus,
    QuestionAnalysis,
    QuestionIntent,
    RetrievalAttempt,
    RetrievalDiagnostics,
    RetrievalMethod,
    TemporalConstraint,
    contract_to_dict,
    evaluate_evidence_coverage,
)


def reference(chunk_id="chunk-1", video_id="video-A", start=10.0, end=20.0):
    return EvidenceReference(
        video_id=video_id,
        chunk_id=chunk_id,
        start_seconds=start,
        end_seconds=end,
    )


def analysis(*aspects, intent=QuestionIntent.EXPLANATION, language="en", video_id=None):
    return QuestionAnalysis(
        intent=intent,
        language=language,
        video_id=video_id,
        subject="airline operations",
        aspects=tuple(aspects),
        constraints=("attribute claims to the interviewee",),
        analysis_source="unit-test",
        confidence=0.9,
    )


def assessment(aspect_id, status, score, *supporting, contradicting=()):
    return EvidenceAssessment(
        aspect_id=aspect_id,
        status=status,
        support_score=score,
        supporting_evidence=tuple(supporting),
        contradicting_evidence=tuple(contradicting),
        method="test-assessor",
    )


@pytest.mark.unit
def test_question_analysis_accepts_english_hindi_and_arabic_language_tags():
    aspects = (EvidenceAspect("reason", "A stated reason for the outcome"),)

    for language in ("en", "hi", "ar", "en-IN", "ar-EG", "en-u-ca-gregory"):
        parsed = analysis(*aspects, language=language)
        assert parsed.language == language
        assert parsed.aspects == aspects



@pytest.mark.unit
def test_question_analysis_preserves_native_hindi_and_arabic_script():
    hindi = QuestionAnalysis(
        intent=QuestionIntent.EXPLANATION,
        language="hi",
        aspects=(
            EvidenceAspect("movement_cause", "किसान आंदोलन के प्रमुख कारण"),
        ),
        subject="किसान आंदोलन",
    )
    arabic = QuestionAnalysis(
        intent=QuestionIntent.EXPLANATION,
        language="ar",
        aspects=(
            EvidenceAspect("silence_effect", "تأثير الصمت على الراحة والإدراك"),
        ),
        subject="الصمت والراحة",
    )

    assert hindi.aspects[0].description == "किसान आंदोलन के प्रमुख कारण"
    assert hindi.subject == "किसान आंदोलन"
    assert arabic.aspects[0].description == "تأثير الصمت على الراحة والإدراك"
    assert arabic.subject == "الصمت والراحة"


@pytest.mark.unit
def test_evidence_coverage_rejects_cross_video_evidence_when_question_is_scoped():
    parsed = analysis(
        EvidenceAspect("answer", "Transcript-supported answer"),
        video_id="video-A",
    )
    foreign_evidence = reference("foreign-chunk", video_id="video-B")

    with pytest.raises(ValueError, match="QuestionAnalysis.video_id"):
        evaluate_evidence_coverage(
            parsed,
            (
                assessment(
                    "answer",
                    EvidenceSupportStatus.SUPPORTED,
                    0.95,
                    foreign_evidence,
                ),
            ),
        )


@pytest.mark.unit
def test_question_analysis_is_immutable_and_normalizes_text_fields():
    parsed = QuestionAnalysis(
        intent="comparison",
        language="en",
        aspects=(EvidenceAspect("first", "  First   entity  "),),
        subject="  two   companies ",
        constraints=(" compare   revenue ",),
    )

    assert parsed.intent is QuestionIntent.COMPARISON
    assert parsed.subject == "two companies"
    assert parsed.aspects[0].description == "First entity"
    assert parsed.constraints == ("compare revenue",)
    with pytest.raises((AttributeError, TypeError)):
        parsed.subject = "changed"


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs",
    [
        {"intent": "unsupported", "language": "en", "aspects": ()},
        {"intent": QuestionIntent.DIRECT_FACT, "language": "en--US", "aspects": ()},
        {"intent": QuestionIntent.DIRECT_FACT, "language": "en", "aspects": ("not-an-aspect",)},
        {"intent": QuestionIntent.DIRECT_FACT, "language": "en", "aspects": (
            EvidenceAspect("duplicate", "first"),
            EvidenceAspect("duplicate", "second"),
        )},
        {"intent": QuestionIntent.DIRECT_FACT, "language": "en", "aspects": (), "confidence": float("nan")},
    ],
)
def test_question_analysis_rejects_invalid_contract_values(kwargs):
    with pytest.raises((TypeError, ValueError)):
        QuestionAnalysis(**kwargs)


@pytest.mark.unit
def test_question_analysis_supports_temporal_constraint():
    parsed = QuestionAnalysis(
        intent=QuestionIntent.TEMPORAL,
        language="en",
        aspects=(EvidenceAspect("event", "What happened in the requested interval"),),
        temporal_constraint=TemporalConstraint(90, 120),
    )

    assert parsed.temporal_constraint.start_seconds == 90.0
    assert parsed.temporal_constraint.end_seconds == 120.0
    with pytest.raises(ValueError):
        TemporalConstraint(120, 90)


@pytest.mark.unit
def test_evidence_reference_requires_valid_video_scoped_timestamp_pair():
    assert reference().video_id == "video-A"
    with pytest.raises(ValueError):
        EvidenceReference("video-A", "chunk-1", start_seconds=10)
    with pytest.raises(ValueError):
        EvidenceReference("video-A", "chunk-1", start_seconds=10, end_seconds=10)
    with pytest.raises(ValueError):
        EvidenceReference("video-A", "chunk-1", start_seconds=float("inf"), end_seconds=20)


@pytest.mark.unit
def test_evidence_aspect_requires_positive_weight_and_unique_hints():
    with pytest.raises(ValueError):
        EvidenceAspect("a", "Aspect", weight=0)
    with pytest.raises(ValueError):
        EvidenceAspect("a", "Aspect", retrieval_hints=("cost", "cost"))


@pytest.mark.unit
def test_all_required_aspects_must_be_supported_for_sufficient_coverage():
    parsed = analysis(
        EvidenceAspect("cause", "A cause"),
        EvidenceAspect("consequence", "A consequence"),
    )
    assessed = (
        assessment("cause", EvidenceSupportStatus.SUPPORTED, 0.93, reference("c1")),
        assessment("consequence", EvidenceSupportStatus.SUPPORTED, 0.88, reference("c2")),
    )

    result = evaluate_evidence_coverage(parsed, assessed)

    assert isinstance(result, EvidenceCoverageSummary)
    assert result.status is CoverageStatus.SUFFICIENT
    assert result.required_aspect_count == 2
    assert result.supported_required_aspect_count == 2
    assert result.coverage_ratio == pytest.approx(0.905)
    assert result.missing_aspect_ids == ()


@pytest.mark.unit
def test_missing_required_aspect_blocks_sufficiency_even_with_high_weighted_ratio():
    parsed = analysis(
        EvidenceAspect("main", "Main answer aspect", weight=10),
        EvidenceAspect("detail", "Required detail", weight=1),
    )
    assessed = (
        assessment("main", EvidenceSupportStatus.SUPPORTED, 1.0, reference("main")),
        assessment("detail", EvidenceSupportStatus.MISSING, 0.0),
    )

    result = evaluate_evidence_coverage(parsed, assessed)

    assert result.status is CoverageStatus.INSUFFICIENT
    assert result.coverage_ratio == pytest.approx(10 / 11)
    assert result.missing_aspect_ids == ("detail",)


@pytest.mark.unit
def test_partial_low_confidence_and_unassessed_aspects_do_not_count_as_supported():
    parsed = analysis(
        EvidenceAspect("partial", "Partially supported aspect"),
        EvidenceAspect("low-score", "Low score aspect"),
        EvidenceAspect("unassessed", "No assessment was produced"),
    )
    assessed = (
        assessment("partial", EvidenceSupportStatus.PARTIAL, 0.7, reference("p")),
        assessment("low-score", EvidenceSupportStatus.SUPPORTED, 0.5, reference("l")),
    )

    result = evaluate_evidence_coverage(parsed, assessed)

    assert result.status is CoverageStatus.INSUFFICIENT
    assert result.missing_aspect_ids == ("partial", "low-score", "unassessed")
    assert result.unknown_aspect_ids == ("unassessed",)


@pytest.mark.unit
def test_contradictory_or_conflicting_evidence_blocks_sufficiency():
    parsed = analysis(
        EvidenceAspect("conflict", "Conflicting source accounts"),
        EvidenceAspect("contradiction", "A contradicted statement"),
    )
    assessed = (
        assessment(
            "conflict",
            EvidenceSupportStatus.CONFLICTED,
            0.8,
            reference("support"),
            contradicting=(reference("against"),),
        ),
        assessment(
            "contradiction",
            EvidenceSupportStatus.CONTRADICTED,
            0.0,
            contradicting=(reference("contradiction"),),
        ),
    )

    result = evaluate_evidence_coverage(parsed, assessed)

    assert result.status is CoverageStatus.INSUFFICIENT
    assert result.contradicted_aspect_ids == ("conflict", "contradiction")
    assert result.coverage_ratio == 0.0


@pytest.mark.unit
def test_optional_aspects_do_not_block_required_coverage():
    parsed = analysis(
        EvidenceAspect("required", "Mandatory evidence"),
        EvidenceAspect("optional", "Additional context", required=False),
    )
    assessed = (
        assessment("required", EvidenceSupportStatus.SUPPORTED, 0.9, reference("r")),
        assessment("optional", EvidenceSupportStatus.MISSING, 0.0),
    )

    result = evaluate_evidence_coverage(parsed, assessed)

    assert result.status is CoverageStatus.SUFFICIENT
    assert result.required_aspect_count == 1
    assert result.supported_required_aspect_count == 1


@pytest.mark.unit
def test_no_required_aspects_is_indeterminate_not_automatically_sufficient():
    parsed = analysis(
        EvidenceAspect("optional", "Optional background", required=False),
    )

    result = evaluate_evidence_coverage(parsed, ())

    assert result.status is CoverageStatus.INDETERMINATE
    assert result.coverage_ratio is None


@pytest.mark.unit
def test_coverage_rejects_duplicate_or_unknown_assessments():
    parsed = analysis(EvidenceAspect("known", "Known aspect"))
    valid = assessment("known", EvidenceSupportStatus.SUPPORTED, 0.9, reference())

    with pytest.raises(ValueError, match="Duplicate assessment"):
        evaluate_evidence_coverage(parsed, (valid, valid))
    with pytest.raises(ValueError, match="unknown aspect"):
        evaluate_evidence_coverage(
            parsed,
            (assessment("unknown", EvidenceSupportStatus.MISSING, 0.0),),
        )


@pytest.mark.unit
def test_assessment_requires_provenance_for_supported_and_contradicted_states():
    with pytest.raises(ValueError, match="supporting_evidence"):
        assessment("a", EvidenceSupportStatus.SUPPORTED, 0.9)
    with pytest.raises(ValueError, match="contradicting_evidence"):
        assessment("a", EvidenceSupportStatus.CONTRADICTED, 0.0)


@pytest.mark.unit
def test_retrieval_diagnostics_validate_video_scope_and_accepted_candidates():
    candidate = reference("candidate")
    accepted = reference("candidate")
    attempt = RetrievalAttempt(
        stage="initial",
        query_variant_id="original",
        method=RetrievalMethod.HYBRID,
        candidate_evidence=(candidate,),
        accepted_evidence=(accepted,),
        elapsed_ms=12.5,
    )
    diagnostic = RetrievalDiagnostics(
        video_id="video-A",
        attempts=(attempt,),
        final_evidence=(accepted,),
        elapsed_ms=15.0,
    )

    assert diagnostic.candidate_count == 1
    assert diagnostic.accepted_candidate_count == 1
    assert diagnostic.attempts[0].accepted_count == 1

    with pytest.raises(ValueError, match="subset"):
        RetrievalAttempt(
            stage="initial",
            query_variant_id="original",
            method=RetrievalMethod.HYBRID,
            candidate_evidence=(candidate,),
            accepted_evidence=(reference("not-a-candidate"),),
            elapsed_ms=1,
        )
    with pytest.raises(ValueError, match="diagnostics.video_id"):
        RetrievalDiagnostics(
            video_id="video-A",
            attempts=(attempt,),
            final_evidence=(reference("foreign", video_id="video-B"),),
            elapsed_ms=15,
        )



@pytest.mark.unit
def test_evidence_assessment_enforces_status_score_consistency():
    with pytest.raises(ValueError, match="zero support_score"):
        assessment("a", EvidenceSupportStatus.MISSING, 0.4)
    with pytest.raises(ValueError, match="below 1.0"):
        assessment("a", EvidenceSupportStatus.PARTIAL, 1.0, reference())
    with pytest.raises(ValueError, match="zero support_score"):
        assessment("a", EvidenceSupportStatus.UNKNOWN, 0.2)


@pytest.mark.unit
def test_evidence_reference_rejects_null_or_boolean_chunk_ids():
    with pytest.raises(TypeError, match="chunk_id"):
        EvidenceReference("video-A", None)
    with pytest.raises(TypeError, match="chunk_id"):
        EvidenceReference("video-A", True)


@pytest.mark.unit
def test_coverage_summary_rejects_inconsistent_status_and_counts():
    with pytest.raises(ValueError, match="support every required aspect"):
        EvidenceCoverageSummary(
            status=CoverageStatus.SUFFICIENT,
            required_aspect_count=2,
            supported_required_aspect_count=1,
            coverage_ratio=0.5,
            missing_aspect_ids=("missing",),
            contradicted_aspect_ids=(),
            unknown_aspect_ids=(),
            assessments=(),
        )
    with pytest.raises(ValueError, match="zero required aspects"):
        EvidenceCoverageSummary(
            status=CoverageStatus.INDETERMINATE,
            required_aspect_count=1,
            supported_required_aspect_count=0,
            coverage_ratio=0.0,
            missing_aspect_ids=("missing",),
            contradicted_aspect_ids=(),
            unknown_aspect_ids=(),
            assessments=(),
        )

@pytest.mark.unit
def test_contracts_serialize_to_json_without_transcript_text():
    parsed = analysis(
        EvidenceAspect(
            "policy",
            "Government policy affecting the airline",
            retrieval_hints=("fuel cost", "tax policy"),
        ),
        language="ar",
    )
    evidence = reference("chunk-99")
    retrieval = RetrievalDiagnostics(
        video_id="video-A",
        attempts=(
            RetrievalAttempt(
                stage="initial",
                query_variant_id="original",
                method=RetrievalMethod.HYBRID,
                candidate_evidence=(evidence,),
                accepted_evidence=(evidence,),
                elapsed_ms=4.2,
            ),
        ),
        final_evidence=(evidence,),
        elapsed_ms=5.0,
    )
    payload = {
        "question_analysis": contract_to_dict(parsed),
        "retrieval": contract_to_dict(retrieval),
    }

    serialized = json.dumps(payload, ensure_ascii=False)
    restored = json.loads(serialized)
    assert restored["question_analysis"]["language"] == "ar"
    assert restored["question_analysis"]["aspects"][0]["aspect_id"] == "policy"
    assert restored["retrieval"]["final_evidence"][0]["chunk_id"] == "chunk-99"
    assert "transcript_text" not in serialized
