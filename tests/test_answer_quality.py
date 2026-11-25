from types import SimpleNamespace
import json
import re

import pytest
from langchain_core.documents import Document

from Backend.rag_system import chain, claim_verification, overview_core, overview_prompts, overview_summarization
from Backend.rag_system.evidence_assessment import AssessmentRunStatus
from Backend.rag_system.evidence_contracts import EvidenceReference, EvidenceSupportStatus


def make_doc(chunk_id, start, text):
    return Document(
        page_content=text,
        metadata={
            "chunk_id": chunk_id,
            "video_id": "video-1",
            "start": start,
            "end": start + 15,
        },
    )


def test_overview_single_section_returns_only_valid_source_linked_points():
    documents = [
        make_doc(1, 0, "The speaker introduces the topic."),
        make_doc(2, 15, "The speaker explains the main idea."),
    ]

    def generate(prompt):
        return json.dumps({
            "points": [{
                "text": "The video introduces and explains its main topic.",
                "evidence_ids": ["P00001", "P00002"],
            }]
        })

    result = overview_summarization.summarize_video_overview(
        documents=documents,
        question="Summarize this video.",
        generate_text=generate,
    )

    assert result["retrieval_trace"]["route"] == "video_wide_overview"
    assert result["retrieval_trace"]["overview"]["coverage_mode"] == "full_transcript"
    assert "introduces and explains" in result["answer"]
    assert [item[0].metadata["chunk_id"] for item in result["retrieved_results"]] == [1, 2]
    assert [
        item[0].metadata["chunk_id"]
        for item in result["verification_results"]
    ] == [1, 2]
    assert result["generation_trace"]["context_budget"]["verification_chunk_ids"] == [
        "1", "2"
    ]


def test_overview_long_video_uses_chronological_map_reduce_and_reports_sampling():
    documents = [
        make_doc(index, index * 15, f"Transcript segment {index}. " + ("detail " * 110))
        for index in range(1, 102)
    ]
    calls = []

    def generate(prompt):
        calls.append(prompt)
        prompt_text = "\n".join(message.content for message in prompt.to_messages())
        ids = re.findall(r"\bP\d{5}\b", prompt_text)
        assert ids
        return json.dumps({
            "points": [{
                "text": "A supported development in this section.",
                "evidence_ids": [ids[0]],
            }]
        })

    result = overview_summarization.summarize_video_overview(
        documents=documents,
        question="Summarize the complete story in 5 pointers.",
        generate_text=generate,
    )

    stats = result["retrieval_trace"]["overview"]
    assert stats["coverage_mode"] == "timeline_stratified"
    assert stats["selected_fraction"] < 1
    assert stats["section_count"] > 1
    assert stats["map_calls"] == stats["section_count"]
    assert len(calls) == stats["map_calls"] + 1
    assert "samples sections across the full timeline" in result["answer"]
    assert len(result["retrieved_results"]) <= stats["selected_document_count"]
    assert len(result["verification_results"]) == stats["section_count"]
    assert len(result["verification_results"]) > len(result["retrieved_results"])
    assert len(
        result["generation_trace"]["context_budget"]["verification_chunk_ids"]
    ) == stats["section_count"]


@pytest.mark.regression
def test_77_chunk_video_of_about_60k_characters_keeps_full_chronological_coverage():
    documents = [
        make_doc(
            index,
            (index - 1) * 45,
            f"Transcript segment {index}. " + ("context " * 95),
        )
        for index in range(1, 78)
    ]
    calls = []

    def generate(prompt):
        calls.append(prompt)
        prompt_text = "\n".join(
            message.content for message in prompt.to_messages()
        )
        ids = re.findall(r"\bP\d{5}\b", prompt_text)
        assert ids
        return json.dumps({
            "points": [{
                "text": "A chronological development in this part of the video.",
                "evidence_ids": [ids[0]],
            }]
        })

    result = overview_summarization.summarize_video_overview(
        documents=documents,
        question="Summarize this entire video chronologically.",
        generate_text=generate,
    )

    stats = result["retrieval_trace"]["overview"]
    assert stats["coverage_mode"] == "full_transcript"
    assert stats["indexed_document_count"] == 77
    assert stats["selected_document_count"] == 77
    assert stats["selected_fraction"] == 1.0
    assert stats["section_count"] <= overview_core.MAX_SECTIONS
    assert stats["map_calls"] == stats["section_count"]
    assert len(calls) == stats["map_calls"] + 1
    assert stats["selected_source_chars"] == stats["indexed_source_chars"]


def test_overview_adaptively_samples_when_uneven_chunks_exceed_section_limit():
    # The full transcript is exactly at the 48,000-character source budget,
    # but its uneven chunk sizes require 15 chronological sections at the
    # 4,500-character per-section limit (12 large chunks + 3 small-chunk
    # sections). The overview must sample more coarsely instead of falling
    # back to sparse similarity retrieval.
    documents = [
        make_doc(
            index,
            index * 15,
            (
                "Opening development " + ("x" * (7000 - len("Opening development")))
                if index <= 12
                else "Later development " + ("y" * (3000 - len("Later development")))
            ),
        )
        for index in range(1, 25)
    ]
    calls = []

    def generate(prompt):
        calls.append(prompt)
        prompt_text = "\n".join(message.content for message in prompt.to_messages())
        evidence_ids = re.findall(r"\bP\d{5}\b", prompt_text)
        assert evidence_ids
        return json.dumps({
            "points": [{
                "text": "A transcript-supported development.",
                "evidence_ids": [evidence_ids[0]],
            }]
        })

    result = overview_summarization.summarize_video_overview(
        documents=documents,
        question="Summarize this entire video chronologically.",
        generate_text=generate,
    )

    stats = result["retrieval_trace"]["overview"]
    assert stats["coverage_mode"] == "timeline_stratified"
    assert stats["section_limit_sampling"] is True
    assert stats["selected_document_count"] < len(documents)
    assert stats["selected_fraction"] < 1
    assert 1 <= stats["section_count"] <= overview_core.MAX_SECTIONS
    assert stats["map_calls"] == stats["section_count"]
    assert len(calls) == stats["map_calls"] + 1
    assert "samples sections across the full timeline" in result["answer"]


def test_parse_points_accepts_one_valid_json_payload_with_model_wrapping_text():
    payload = {
        "points": [{
            "text": "The speaker introduces the topic.",
            "evidence_ids": ["P00001"],
        }]
    }
    raw = (
        "Here is the requested response:\n"
        "```json\n"
        + json.dumps(payload)
        + "\n```\n"
        "The response is complete."
    )

    result = overview_core.parse_points(
        raw,
        allowed_ids={"P00001"},
        limit=3,
    )

    assert result == payload["points"]


def test_parse_points_rejects_multiple_ambiguous_json_payloads():
    payload = {
        "points": [{
            "text": "The speaker introduces the topic.",
            "evidence_ids": ["P00001"],
        }]
    }
    raw = json.dumps(payload) + "\n" + json.dumps(payload)

    with pytest.raises(overview_core.OverviewError, match="multiple ambiguous"):
        overview_core.parse_points(
            raw,
            allowed_ids={"P00001"},
            limit=3,
        )


def test_overview_rejects_model_invented_evidence_ids():
    with pytest.raises(overview_summarization.OverviewError):
        overview_summarization.summarize_video_overview(
            documents=[make_doc(1, 0, "A transcript statement.")],
            question="Summarize this video.",
            generate_text=lambda _prompt: json.dumps({
                "points": [{"text": "Unsupported.", "evidence_ids": ["P99999"]}]
            }),
        )


def test_claim_splitter_removes_timestamp_footer_and_preserves_bullets():
    claims = claim_verification.split_answer_claims(
        "1. The speaker introduces the project.\n"
        "2. The project has three stages.\n"
        "Relevant timestamps: 00:10–00:30"
    )
    assert claims == [
        "The speaker introduces the project.",
        "The project has three stages.",
    ]


@pytest.mark.parametrize(
    "answer,question",
    [
        ("The event occurred in 1997.", "When did it happen?"),
        ("Grand Terror Auto is the expanded name of GTA.", "What is GTA?"),
        ("The study included 35 participants.", "What study was mentioned?"),
        ("One supported point. Another supported point.", "What happened?"),
    ],
)
def test_fact_sensitive_or_compound_answers_require_audit(answer, question):
    assert claim_verification.should_verify_answer(answer, question)


def test_simple_short_answer_does_not_add_a_verification_call():
    assert not claim_verification.should_verify_answer("It is a vibration plate.", "What is it?")


def test_completed_claim_audit_removes_unsupported_claim_and_filters_to_supporting_chunks(monkeypatch):
    documents = [
        (make_doc(1, 0, "The speaker describes a study with 35 participants."), 0.1),
        (make_doc(2, 15, "The speaker mentions a company promotion."), 0.2),
    ]

    def fake_assessment(analysis, passages, client, **kwargs):
        assert len(analysis.aspects) == 2
        assert len(passages) == 2
        assert {ref.chunk_id for passage in passages for ref in passage.references} == {"1", "2"}
        return SimpleNamespace(
            diagnostics=SimpleNamespace(status=AssessmentRunStatus.COMPLETED, error_code=None),
            assessments=[
                SimpleNamespace(
                    aspect_id="claim_01",
                    status=EvidenceSupportStatus.SUPPORTED,
                    supporting_evidence=(EvidenceReference("video-1", "1", 0, 15),),
                ),
                SimpleNamespace(
                    aspect_id="claim_02",
                    status=EvidenceSupportStatus.MISSING,
                    supporting_evidence=(),
                ),
            ],
        )

    monkeypatch.setattr(claim_verification, "assess_evidence_support", fake_assessment)
    result = claim_verification.verify_answer_claims(
        answer="1. The video describes a study with 35 participants.\n"
              "2. The company promotion is proven effective.",
        question="What does the video say about the study and promotion?",
        video_id="video-1",
        retrieved_results=documents,
        included_chunk_ids=["1", "2"],
        prompt_visible_context=(
            "The speaker describes a study with 35 participants. "
            "The speaker mentions a company promotion."
        ),
        client=object(),
        fallback_answer="I couldn't find enough information about that in the video.",
    )

    assert result["status"] == "completed"
    assert result["supported_claim_count"] == 1
    assert result["removed_claim_count"] == 1
    assert "study with 35 participants" in result["answer"]
    assert "proven effective" not in result["answer"]
    assert result["supported_chunk_ids"] == ["1"]


def test_claim_audit_does_not_call_unseen_retrieval_candidates(monkeypatch):
    documents = [
        (make_doc(1, 0, "Evidence in the generation prompt."), 0.1),
        (make_doc(2, 15, "Candidate not included in generation."), 0.2),
    ]
    seen = {}

    def fake_assessment(analysis, passages, client, **kwargs):
        seen["references"] = [
            ref.chunk_id for passage in passages for ref in passage.references
        ]
        return SimpleNamespace(
            diagnostics=SimpleNamespace(status=AssessmentRunStatus.COMPLETED, error_code=None),
            assessments=[
                SimpleNamespace(
                    aspect_id="claim_01",
                    status=EvidenceSupportStatus.SUPPORTED,
                    supporting_evidence=(EvidenceReference("video-1", "1", 0, 15),),
                ),
                SimpleNamespace(
                    aspect_id="claim_02",
                    status=EvidenceSupportStatus.MISSING,
                    supporting_evidence=(),
                ),
            ],
        )

    monkeypatch.setattr(claim_verification, "assess_evidence_support", fake_assessment)
    claim_verification.verify_answer_claims(
        answer="The first fact is supported. The second fact is not supported.",
        question="Explain the result.",
        video_id="video-1",
        retrieved_results=documents,
        included_chunk_ids=["1"],
        prompt_visible_context="Evidence in the generation prompt.",
        client=object(),
        fallback_answer="I couldn't find enough information about that in the video.",
    )
    assert seen["references"] == ["1"]


def test_claim_audit_uses_only_exact_text_visible_in_prompt(monkeypatch):
    hidden_tail = (
        "The introduction describes the subject in broad terms. "
        + ("The opening details repeat the general background. " * 5)
        + "The actual release date was 1997."
    )
    document = make_doc(3, 30, hidden_tail)
    prompt_excerpt = hidden_tail[:115]
    observed = {}

    def fake_assessment(analysis, passages, client, **kwargs):
        observed["texts"] = [passage.text for passage in passages]
        return SimpleNamespace(
            diagnostics=SimpleNamespace(status=AssessmentRunStatus.COMPLETED, error_code=None),
            assessments=[
                SimpleNamespace(
                    aspect_id="claim_01",
                    status=EvidenceSupportStatus.MISSING,
                    supporting_evidence=(),
                ),
            ],
        )

    monkeypatch.setattr(claim_verification, "assess_evidence_support", fake_assessment)
    result = claim_verification.verify_answer_claims(
        answer="The actual release date was 1997.",
        question="When did it first release?",
        video_id="video-1",
        retrieved_results=[(document, 0.1)],
        included_chunk_ids=["3"],
        prompt_visible_context=prompt_excerpt,
        client=object(),
        fallback_answer="I couldn't find enough information about that in the video.",
    )

    assert observed["texts"]
    assert "1997" not in " ".join(observed["texts"])
    assert result["removed_claim_count"] == 1
    assert result["supported_chunk_ids"] == []


def test_sampling_disclosure_is_not_a_transcript_claim_and_survives_verification(monkeypatch):
    document = make_doc(7, 60, "The speaker introduces the community project.")
    observed = {}

    def fake_assessment(analysis, passages, client, **kwargs):
        observed["claims"] = [aspect.description for aspect in analysis.aspects]
        assert len(passages) == 1
        return SimpleNamespace(
            diagnostics=SimpleNamespace(
                status=AssessmentRunStatus.COMPLETED,
                error_code=None,
            ),
            assessments=[
                SimpleNamespace(
                    aspect_id="claim_01",
                    status=EvidenceSupportStatus.SUPPORTED,
                    supporting_evidence=(
                        EvidenceReference("video-1", "7", 60, 75),
                    ),
                )
            ],
        )

    monkeypatch.setattr(
        claim_verification,
        "assess_evidence_support",
        fake_assessment,
    )
    answer = (
        overview_summarization.OVERVIEW_SAMPLING_DISCLOSURE
        + "\n\n1. The speaker introduces the community project."
    )
    result = claim_verification.verify_answer_claims(
        answer=answer,
        question="Summarize the entire video.",
        video_id="video-1",
        retrieved_results=[(document, 0.0)],
        included_chunk_ids=["7"],
        prompt_visible_context=document.page_content,
        client=object(),
        fallback_answer="I couldn't find enough information about that in the video.",
    )

    assert observed["claims"] == ["The speaker introduces the community project."]
    assert result["supported_claim_count"] == 1
    assert overview_summarization.OVERVIEW_SAMPLING_DISCLOSURE in result["answer"]


def test_overview_rejects_documents_from_multiple_videos():
    first = make_doc(1, 0, "First video's transcript.")
    second = Document(
        page_content="Second video's transcript.",
        metadata={
            "chunk_id": 2,
            "video_id": "video-2",
            "start": 15,
            "end": 30,
        },
    )
    with pytest.raises(overview_summarization.OverviewError, match="exactly one video"):
        overview_summarization.summarize_video_overview(
            documents=[first, second],
            question="Summarize this video.",
            generate_text=lambda _prompt: json.dumps({
                "points": [{"text": "Unsupported.", "evidence_ids": ["P00001"]}]
            }),
        )


def test_overview_can_skip_empty_timeline_section_without_fabricating_points():
    documents = [
        make_doc(1, 0, "Opening context. " + ("opening detail " * 170)),
        make_doc(2, 300, "Later important development. " + ("later detail " * 170)),
    ]
    calls = []

    def generate(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return json.dumps({"points": []})
        prompt_text = "\n".join(message.content for message in prompt.to_messages())
        identifiers = re.findall(r"\bP\d{5}\b", prompt_text)
        return json.dumps({
            "points": [{
                "text": "The later section contains a supported development.",
                "evidence_ids": [identifiers[0]],
            }]
        })

    result = overview_summarization.summarize_video_overview(
        documents=documents,
        question="Summarize the complete video.",
        generate_text=generate,
    )

    assert len(calls) == 3
    assert "later section" in result["answer"]
    assert [pair[0].metadata["chunk_id"] for pair in result["retrieved_results"]] == [2]


def test_claim_audit_fails_open_with_explicit_diagnostic_on_provider_error(monkeypatch):
    document = make_doc(4, 45, "The speaker describes an event with several details.")

    def fail_assessment(*args, **kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(claim_verification, "assess_evidence_support", fail_assessment)
    original = "The speaker describes an event. The event had three stages."
    result = claim_verification.verify_answer_claims(
        answer=original,
        question="Summarize what happened.",
        video_id="video-1",
        retrieved_results=[(document, 0.1)],
        included_chunk_ids=["4"],
        prompt_visible_context=document.page_content,
        client=object(),
        fallback_answer="I couldn't find enough information about that in the video.",
    )

    assert result["answer"] == original
    assert result["status"] == "provider_error"
    assert result["error_code"] == "unexpected_assessment_error"
    assert result["removed_claim_count"] == 0


def test_claim_audit_does_not_silently_merge_unchecked_overflow_claims(monkeypatch):
    document = make_doc(5, 60, "The speaker provides a long series of independent facts. " * 20)
    seen = {}

    def fake_assessment(analysis, passages, client, **kwargs):
        seen["aspect_count"] = len(analysis.aspects)
        return SimpleNamespace(
            diagnostics=SimpleNamespace(status=AssessmentRunStatus.COMPLETED, error_code=None),
            assessments=[
                SimpleNamespace(
                    aspect_id=f"claim_{index:02d}",
                    status=EvidenceSupportStatus.SUPPORTED,
                    supporting_evidence=(EvidenceReference("video-1", "5", 60, 75),),
                )
                for index in range(1, len(analysis.aspects) + 1)
            ],
        )

    monkeypatch.setattr(claim_verification, "assess_evidence_support", fake_assessment)
    answer = "\n".join(f"{index}. Supported independent statement {index}." for index in range(1, 11))
    result = claim_verification.verify_answer_claims(
        answer=answer,
        question="Summarize the key facts.",
        video_id="video-1",
        retrieved_results=[(document, 0.1)],
        included_chunk_ids=["5"],
        prompt_visible_context=document.page_content,
        client=object(),
        fallback_answer="I couldn't find enough information about that in the video.",
    )

    assert seen["aspect_count"] == 8
    assert result["truncated_claim_count"] == 2
    assert "Supported independent statement 9" not in result["answer"]
    assert "exceeded the 8-claim verification limit" in result["answer"]



def test_overview_fills_underproduced_reducer_from_chronological_map_findings():
    documents = [
        make_doc(
            index,
            index * 120,
            f"Distinct section {index} describes a separate event. " + (f"detail-{index} " * 390),
        )
        for index in range(1, 7)
    ]
    map_calls = 0

    def generate(prompt):
        nonlocal map_calls
        prompt_text = "\n".join(message.content for message in prompt.to_messages())
        if "Section findings:" in prompt_text:
            return json.dumps({
                "points": [{
                    "text": "Distinct section 1 describes a separate event.",
                    "evidence_ids": ["P00001"],
                }]
            })

        map_calls += 1
        section_match = re.search(r"Chronological section (\d+) of", prompt_text)
        assert section_match
        section_number = int(section_match.group(1))
        evidence_id = re.search(r"\[(P\d{5})\]", prompt_text).group(1)
        return json.dumps({
            "points": [{
                "text": f"Distinct section {section_number} describes a separate event.",
                "evidence_ids": [evidence_id],
            }]
        })

    result = overview_summarization.summarize_video_overview(
        documents=documents,
        question="Summarize this video in 5 pointers.",
        generate_text=generate,
    )

    stats = result["retrieval_trace"]["overview"]
    assert stats["section_count"] == 6
    assert stats["reducer_returned_point_count"] == 1
    assert stats["map_fallback_points_added"] == 4
    assert stats["returned_point_count"] == 5
    assert stats["point_count_satisfied"] is True
    assert map_calls == 6
    assert [pair[0].metadata["chunk_id"] for pair in result["retrieved_results"]] == [1, 2, 3, 4, 5]


def test_overview_response_format_uses_strict_schema_for_gpt_oss():
    response_format = overview_prompts.overview_response_format_for_model(
        "openai/gpt-oss-20b"
    )

    assert response_format["type"] == "json_schema"
    schema = response_format["json_schema"]
    assert schema["strict"] is True
    assert schema["schema"]["required"] == ["points"]
    assert schema["schema"]["additionalProperties"] is False
    point_schema = schema["schema"]["properties"]["points"]["items"]
    assert point_schema["required"] == ["text", "evidence_ids"]
    assert point_schema["additionalProperties"] is False


def test_overview_response_format_uses_json_mode_for_other_groq_models():
    assert overview_prompts.overview_response_format_for_model(
        "llama-3.3-70b-versatile"
    ) == {"type": "json_object"}


def test_generate_with_groq_forwards_requested_response_format(monkeypatch):
    expected_answer = json.dumps({"points": []})
    captured = {}
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=expected_answer),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
    )

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return response

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    prompt_value = SimpleNamespace(
        messages=[
            SimpleNamespace(type="system", content="Return JSON only."),
            SimpleNamespace(type="human", content="Summarize this section."),
        ]
    )
    response_format = overview_prompts.overview_response_format_for_model(
        "openai/gpt-oss-20b"
    )
    monkeypatch.setattr(chain, "GROQ_MAX_RETRIES", 0)

    answer = chain.generate_with_groq(
        client=client,
        prompt_value=prompt_value,
        response_format=response_format,
    )

    assert answer == expected_answer
    assert captured["response_format"] == response_format
    assert captured["model"] == chain.GROQ_MODEL_ID


def test_generate_with_groq_keeps_prose_sanitization_without_response_format(monkeypatch):
    captured = {}
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="A normal prose answer."),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
    )

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return response

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    prompt_value = SimpleNamespace(
        messages=[
            SimpleNamespace(type="system", content="Answer normally."),
            SimpleNamespace(type="human", content="What happened?"),
        ]
    )
    sanitizer_calls = []

    def fake_sanitizer(value):
        sanitizer_calls.append(value)
        return "sanitized prose"

    monkeypatch.setattr(chain, "GROQ_MAX_RETRIES", 0)
    monkeypatch.setattr(chain, "sanitize_generated_answer", fake_sanitizer)

    answer = chain.generate_with_groq(
        client=client,
        prompt_value=prompt_value,
    )

    assert answer == "sanitized prose"
    assert sanitizer_calls == ["A normal prose answer."]
    assert "response_format" not in captured


def test_generate_with_llm_forwards_response_format_to_groq(monkeypatch):
    captured = {}
    response_format = overview_prompts.overview_response_format_for_model(
        "openai/gpt-oss-20b"
    )

    def fake_generate_with_groq(*, client, prompt_value, response_format=None):
        captured["client"] = client
        captured["prompt_value"] = prompt_value
        captured["response_format"] = response_format
        return "generated"

    monkeypatch.setattr(chain, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(chain, "generate_with_groq", fake_generate_with_groq)

    result = chain.generate_with_llm(
        client="test-client",
        prompt_value="test-prompt",
        response_format=response_format,
    )

    assert result == "generated"
    assert captured["client"] == "test-client"
    assert captured["prompt_value"] == "test-prompt"
    assert captured["response_format"] == response_format




def test_claim_verification_compacts_visible_excerpts_to_fit_assessment_budget(monkeypatch):
    first = make_doc(
        1,
        0,
        "The speaker discusses cobalt extraction from the northern mine. "
        + ("Cobalt mining and extraction details. " * 30),
    )
    second = make_doc(
        2,
        120,
        "The speaker reports lithium prices increased in the coastal market. "
        + ("Lithium market and price details. " * 30),
    )
    visible_context = first.page_content + "\n" + second.page_content
    answer = (
        "1. The speaker discusses cobalt extraction from the northern mine.\n"
        "2. The speaker reports lithium prices increased in the coastal market."
    )
    attempted_passage_lengths = []

    def fake_assessment(analysis, passages, client, **kwargs):
        lengths = [len(passage.text) for passage in passages]
        attempted_passage_lengths.append(lengths)
        assert all(passage.text in visible_context for passage in passages)

        if max(lengths) > 320:
            diagnostics = SimpleNamespace(
                status=AssessmentRunStatus.BUDGET_EXCEEDED,
                error_code="assessment_input_character_budget_exceeded",
                input_chars=9_000,
                input_limit_chars=8_000,
                passage_count=len(passages),
                aspect_count=len(analysis.aspects),
                request_count=0,
            )
            return SimpleNamespace(diagnostics=diagnostics, assessments=[])

        diagnostics = SimpleNamespace(
            status=AssessmentRunStatus.COMPLETED,
            error_code=None,
            input_chars=7_900,
            input_limit_chars=8_000,
            passage_count=len(passages),
            aspect_count=len(analysis.aspects),
            request_count=1,
        )
        assessments = [
            SimpleNamespace(
                aspect_id=aspect.aspect_id,
                status=EvidenceSupportStatus.SUPPORTED,
                supporting_evidence=(passages[index].references[0],),
            )
            for index, aspect in enumerate(analysis.aspects)
        ]
        return SimpleNamespace(diagnostics=diagnostics, assessments=assessments)

    monkeypatch.setattr(
        claim_verification,
        "assess_evidence_support",
        fake_assessment,
    )

    result = claim_verification.verify_answer_claims(
        answer=answer,
        question="Summarize the key facts.",
        video_id="video-1",
        retrieved_results=[(first, 0.1), (second, 0.2)],
        included_chunk_ids=["1", "2"],
        prompt_visible_context=visible_context,
        client=object(),
        fallback_answer="I couldn't find enough information about that in the video.",
    )

    assert [max(lengths) for lengths in attempted_passage_lengths] == [
        640, 560, 480, 400, 320
    ]
    assert result["status"] == "completed"
    assert result["claim_count"] == 2
    assert result["supported_claim_count"] == 2
    assert result["assessment_compaction_attempts"] == 3
    assert result["assessment_passage_excerpt_limit"] == 320
    assert result["assessment_input_chars"] == 7_900
    assert result["assessment_input_limit_chars"] == 8_000
    assert set(result["supported_chunk_ids"]) == {"1", "2"}


def test_claim_verification_compacts_eight_claim_overview_below_live_budget(monkeypatch):
    documents = [
        make_doc(
            index,
            index * 30,
            (
                f"The speaker describes timeline event {index} in detail, including "
                "the people, place, and circumstances around this development. "
                + (f"Supporting transcript context for timeline event {index}. " * 20)
            ),
        )
        for index in range(1, 9)
    ]
    visible_context = "\n".join(item.page_content for item in documents)
    claims = [
        (
            f"The speaker describes timeline event {index} and explains "
            "the main development during this part of the interview."
        )
        for index in range(1, 9)
    ]
    answer = "\n".join(
        f"{index}. {claim}" for index, claim in enumerate(claims, start=1)
    )
    attempted = []

    def fake_assessment(analysis, passages, client, **kwargs):
        lengths = [len(item.text) for item in passages]
        attempted.append((lengths, len(analysis.aspects)))
        assert len(passages) == 8
        assert len(analysis.aspects) == 8
        assert all(item.text in visible_context for item in passages)

        # Reproduce the live 8,131-character failure at eight 160-character
        # excerpts, keeping the assessor's fixed prompt/claim overhead constant.
        input_chars = 6_851 + sum(lengths)
        over_budget = input_chars > 8_000
        diagnostics = SimpleNamespace(
            status=(
                AssessmentRunStatus.BUDGET_EXCEEDED
                if over_budget
                else AssessmentRunStatus.COMPLETED
            ),
            error_code=(
                "assessment_input_character_budget_exceeded"
                if over_budget
                else None
            ),
            input_chars=input_chars,
            input_limit_chars=8_000,
            passage_count=len(passages),
            aspect_count=len(analysis.aspects),
            request_count=0 if over_budget else 1,
        )
        if over_budget:
            return SimpleNamespace(diagnostics=diagnostics, assessments=[])

        assessments = [
            SimpleNamespace(
                aspect_id=aspect.aspect_id,
                status=EvidenceSupportStatus.SUPPORTED,
                supporting_evidence=(passages[index].references[0],),
            )
            for index, aspect in enumerate(analysis.aspects)
        ]
        return SimpleNamespace(
            diagnostics=diagnostics,
            assessments=assessments,
        )

    monkeypatch.setattr(
        claim_verification,
        "assess_evidence_support",
        fake_assessment,
    )

    result = claim_verification.verify_answer_claims(
        answer=answer,
        question="Summarize this entire video chronologically.",
        video_id="video-1",
        retrieved_results=[
            (document, float(index))
            for index, document in enumerate(documents, start=1)
        ],
        included_chunk_ids=[str(index) for index in range(1, 9)],
        prompt_visible_context=visible_context,
        client=object(),
        fallback_answer="I couldn't find enough information about that in the video.",
    )

    assert [max(lengths) for lengths, _ in attempted] == [
        640, 560, 480, 400, 320, 240, 192, 160, 128
    ]
    assert all(aspect_count == 8 for _, aspect_count in attempted)
    assert result["status"] == "completed"
    assert result["claim_count"] == 8
    assert result["truncated_claim_count"] == 0
    assert result["supported_claim_count"] == 8
    assert result["partial_claim_count"] == 0
    assert result["removed_claim_count"] == 0
    assert result["assessment_input_chars"] == 7_875
    assert result["assessment_input_limit_chars"] == 8_000
    assert result["assessment_compaction_attempts"] == 9
    assert result["assessment_passage_excerpt_limit"] == 128
    assert len(result["supported_chunk_ids"]) == 8
