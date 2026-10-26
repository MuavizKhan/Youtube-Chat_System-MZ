import json

import pytest

import scripts.chat_answer_source_audit as audit_module
from Backend.rag_system import chain
from scripts.chat_answer_source_audit import (
    AUDIT_CASES,
    PHASE10_11_AUDIT_CASES,
    analyze_response,
    build_coverage_manifest,
    decode_json_response,
    select_audit_cases,
    _answer_quality_checks,
)


def _case(case_id):
    return next(case for case in AUDIT_CASES if case["id"] == case_id)


def test_select_audit_cases_preserves_requested_order_and_returns_expected_subset():
    cases = select_audit_cases(
        PHASE10_11_AUDIT_CASES,
        ["EW01", "HI09", "AR08", "Q07", "EW05"],
    )

    assert [case["id"] for case in cases] == [
        "EW01", "HI09", "AR08", "Q07", "EW05"
    ]


@pytest.mark.parametrize(
    "requested_ids, message",
    [
        (["EW01", "EW01"], "Duplicate case IDs"),
        (["EW01", "NOT_A_CASE"], "Unknown case IDs"),
        ([], "At least one case ID"),
    ],
)
def test_select_audit_cases_rejects_invalid_selection(requested_ids, message):
    with pytest.raises(ValueError, match=message):
        select_audit_cases(PHASE10_11_AUDIT_CASES, requested_ids)


def test_decode_json_response_preserves_arabic_and_unicode_punctuation():
    expected = {
        "answer": "نبض قلبك حركة الدم — Vijay\u202fMallya",
    }
    body = json.dumps(expected, ensure_ascii=False).encode("utf-8")

    assert decode_json_response(body) == expected


def test_arabic_evidence_can_be_context_only_and_absent_from_citations():
    response = {
        "answer": (
            "People may hear the pulse of the heart.\n\n"
            "Relevant timestamps: 03:18–04:21"
        ),
        "retrieval_config": {
            "embedding_model": "BAAI/bge-m3",
            "index_namespace": "experiment-bge-m3",
            "citation_policy": "selected_anchors",
        },
        "sources": [
            {
                "source_id": 1,
                "chunk_ids": [3],
                "start": 198.697,
                "end": 261.778,
                "distance": 0.84,
            }
        ],
        "observability": {
            "retrieval": {
                "final_chunk_ids": [0, 1, 2, 3, 5],
                "citation_chunk_ids": [3],
            }
        },
    }

    audit = analyze_response(_case("AR03"), response)

    assert audit["context_only_chunk_ids"] == ["0", "1", "2", "5"]
    evidence = audit["annotated_evidence_groups"][0]
    assert evidence["present_in_final_context"] is True
    assert evidence["present_in_selected_anchors"] is False
    assert evidence["present_in_returned_sources"] is False
    assert evidence["represented_in_plain_text_timestamp_summary"] is False
    assert audit["answer_text_encoding_suspect"] is False


def test_expanded_context_citation_policy_makes_neighbor_evidence_citable():
    response = {
        "answer": "الإجابة صحيحة.\n\nRelevant timestamps: 02:28–04:21",
        "retrieval_config": {
            "citation_policy": "prompt_included_chunks",
        },
        "sources": [
            {
                "source_id": 1,
                "chunk_ids": [2, 3],
                "start": 148.233,
                "end": 261.778,
                "distance": 0.84,
            }
        ],
        "observability": {
            "retrieval": {
                "final_chunk_ids": [2, 3],
                "citation_chunk_ids": [3],
            },
            "generation": {
                "context_budget": {
                    "budget_chars": 8000,
                    "final_chars": 200,
                    "included_chunk_ids": ["2", "3"],
                }
            },
        },
    }

    audit = analyze_response(_case("AR08"), response)
    evidence = audit["annotated_evidence_groups"][0]

    assert evidence["present_in_final_context"] is True
    assert evidence["present_in_selected_anchors"] is False
    assert evidence["present_in_returned_sources"] is True
    assert evidence["represented_in_plain_text_timestamp_summary"] is True


@pytest.mark.parametrize(
    ("case_id", "answer_text"),
    [
        (
            "AR03",
            "Your own heartbeat, the blood rushing through your veins, the sounds of your lungs and stomach, and even the subtle noise of your eyelids blinking.",
        ),
        (
            "AR08",
            "في الغرفة الهادئة قد يبدأ الناس في سماع أصوات أجسامهم: نبض قلبك، حركة الدم في عروقك، صوت الرئتين أو المعدة، وحتى صوت عينيك وهي بترمش.",
        ),
    ],
)
def test_bodily_sound_timestamp_uses_the_chunk_containing_the_eyelid_phrase(
    case_id, answer_text
):
    response = {
        "answer": answer_text + "\n\nRelevant timestamps: 02:33–03:37",
        "retrieval_config": {
            "embedding_model": "BAAI/bge-m3",
            "index_namespace": "experiment-caption-segments",
            "citation_policy": "selected_anchors",
        },
        "sources": [
            {
                "source_id": 1, "chunk_ids": [2],
                "start": 153.05, "end": 217.615, "distance": 0.84,
            },
            {
                "source_id": 2, "chunk_ids": [11],
                "start": 589.599, "end": 658.17, "distance": 0.95,
            },
        ],
        "observability": {
            "retrieval": {
                "final_chunk_ids": [2, 11],
                "citation_chunk_ids": [2, 11],
            },
            "generation": {
                "context_budget": {
                    "budget_chars": 8000, "original_chars": 14430,
                    "final_chars": 7902, "groups_available": 3,
                    "groups_included": 3, "reduced": True,
                    "included_chunk_ids": ["2", "11"],
                }
            },
        },
    }

    accepted = analyze_response(_case(case_id), response)
    assert accepted["quality"]["passed"] is True
    assert accepted["quality"]["failed_checks"] == []

    # Chunk 11 is a later sound-wave explanation, not evidence for the eyelid
    # claim. Its range alone must not satisfy this check.
    response["answer"] = answer_text + "\n\nRelevant timestamps: 09:49–10:58"
    rejected = analyze_response(_case(case_id), response)
    assert rejected["quality"]["passed"] is False
    assert "evidence_source_and_timestamp_alignment" in rejected["quality"]["failed_checks"]


def test_q07_evidence_groups_distinguish_context_sources_and_summary():
    response = {
        "answer": (
            "Vijay Mallya says he invested ₹3,000 crore in Kingfisher Airlines. "
            "The interview also mentions a DRT judgment figure of ₹4,999 crore plus interest "
            "and a separate debt-certificate figure of ₹6,203 crore; the transcript does not "
            "clearly explain how those figures relate. Separately, he says a Ministry of Finance "
            "report showed ₹14,100 crore recovered from him.\n\n"
            "Relevant timestamps: 02:01:07–02:02:31, 02:18:34–02:20:28, "
            "02:57:10–02:58:18, 04:08:59–04:10:15"
        ),
        "retrieval_config": {"citation_policy": "selected_anchors"},
        "sources": [
            {
                "source_id": 1, "chunk_ids": [108],
                "start": 7267.92, "end": 7351.84, "distance": 0.82,
            },
            {
                "source_id": 2, "chunk_ids": [124],
                "start": 8314.519, "end": 8428.16, "distance": 0.91,
            },
            {
                "source_id": 3, "chunk_ids": [159],
                "start": 10630.88, "end": 10698.72, "distance": 0.82,
            },
            {
                "source_id": 4, "chunk_ids": [221],
                "start": 14939.359, "end": 15015.239, "distance": 1.12,
            },
        ],
        "observability": {
            "retrieval": {
                "final_chunk_ids": [108, 124, 153, 159, 221],
                "citation_chunk_ids": [108, 124, 153, 159, 221],
            },
            "generation": {
                "context_budget": {
                    "budget_chars": 8000, "original_chars": 27628,
                    "final_chars": 7900, "groups_available": 12,
                    "groups_included": 8, "reduced": True,
                }
            },
        },
    }

    audit = analyze_response(_case("Q07"), response)
    groups = audit["annotated_evidence_groups"]

    assert all(group["present_in_final_context"] for group in groups)
    assert all(group["present_in_returned_sources"] for group in groups)
    assert all(group["represented_in_plain_text_timestamp_summary"] for group in groups)
    assert audit["quality"]["passed"] is True
    assert audit["quality"]["failed_checks"] == []


def test_q07_quality_gate_rejects_recovery_amount_as_company_debt():
    bad_answer = (
        "Mallya invested ₹3,000 crore in Kingfisher Airlines. "
        "The company's debt totals about ₹14,100 crore, and he owes roughly ₹6,200 crore plus interest."
    )
    checks = _answer_quality_checks("Q07", bad_answer)
    failed = {check["name"] for check in checks if not check["passed"]}

    assert "drt_4999" in failed
    assert "certificate_6203" in failed
    assert "recovery_14100" in failed
    assert "recovery_source_attribution" in failed
    assert "unclear_relationship_preserved" in failed


def test_source_bound_financial_fallback_satisfies_q07_quality_contract():
    context = """[SOURCE 1]
Transcript caption segments in chronological order:
- [02:01:07–02:02:31] I invested 3,000 crore rupees into Kingfisher Airlines.
[SOURCE 2]
Transcript caption segments in chronological order:
- [02:18:34–02:20:28] The DRT judgment debt was 4,999 crore rupees plus interest.
[SOURCE 3]
Transcript caption segments in chronological order:
- [02:49:56–02:50:25] The Ministry of Finance report says the total amount recovered was
- [02:50:25–02:51:41] 14,100 crore rupees.
[SOURCE 4]
Transcript caption segments in chronological order:
- [04:08:59–04:09:30] The debt certificate gives a figure of 6203.
- [04:09:30–04:10:15] The principal amount carries unapplied interest at 11.5% per annum."""

    answer = chain._build_financial_evidence_fallback(
        "What debt and money amounts are mentioned in the Kingfisher case?",
        context,
    )

    assert answer is not None
    failures = [
        check["name"]
        for check in _answer_quality_checks("Q07", answer)
        if not check["passed"]
    ]
    assert failures == [], f"Extractive fallback failed Q07 audit checks: {failures}"
    for raw_timestamp in (
        "[02:01:07–02:02:31]",
        "[02:18:34–02:20:28]",
        "[02:49:56–02:50:25]",
        "[02:50:25–02:51:41]",
    ):
        assert raw_timestamp not in answer
    recovery_claim = next(
        line for line in answer.splitlines() if "14,100" in line
    )
    assert "report" in recovery_claim.casefold()
    assert "recover" in recovery_claim.casefold()


def test_q07_quality_gate_rejects_financial_verification_abstention():
    answer = (
        "I couldn't reliably verify the financial figures from the retrieved "
        "transcript. I won't guess at or reconcile amounts that the available "
        "evidence does not establish."
    )

    failed = {
        check["name"]
        for check in _answer_quality_checks("Q07", answer)
        if not check["passed"]
    }

    assert "financial_answer_not_abstained" in failed


def test_q07_quality_gate_rejects_known_malformed_extra_amounts():
    answer = (
        "Mallya invested 3,000 crore in Kingfisher. The DRT judgment debt was 4,999 crore. "
        "The debt certificate gives a figure of 6,203. A Ministry of Finance report says "
        "14,100 crore was recovered. The amounts 6,23 crore, 6,2003 crore and 6,0002 crore "
        "were stated, and 9,000 crore was the official debt-recovery amount. "
        "The transcript does not clearly explain how the DRT judgment and certificate figures relate."
    )

    failed = {
        check["name"]
        for check in _answer_quality_checks("Q07", answer)
        if not check["passed"]
    }

    assert "malformed_or_unsupported_financial_amounts_absent" in failed


def test_bodily_sound_quality_gate_checks_coverage_and_question_language():
    english_answer = (
        "In the quiet room people may hear their heartbeat, blood rushing through "
        "their veins, their lungs and breathing, stomach rumbling, and eyelids blinking."
    )
    arabic_answer = (
        "في الغرفة الهادئة قد يبدأ الناس في سماع أصوات أجسامهم: نبض قلبك، "
        "حركة الدم في عروقك، صوت الرئتين أو المعدة، وحتى صوت عينيك وهي بترمش."
    )

    assert all(check["passed"] for check in _answer_quality_checks("AR03", english_answer))
    assert all(check["passed"] for check in _answer_quality_checks("AR08", arabic_answer))

    incomplete_arabic = "الأصوات التي قد يبدأ الناس في سماعها من أجسامهم هي نبضات القلب."
    failed = {
        check["name"]
        for check in _answer_quality_checks("AR08", incomplete_arabic)
        if not check["passed"]
    }
    assert "sound_blood_movement" in failed
    assert "sound_lungs_or_breathing" in failed
    assert "sound_stomach" in failed
    assert "sound_eyelids_blinking" in failed


def test_run_audit_fails_quality_gate_even_when_all_http_requests_succeed(tmp_path, monkeypatch):
    shared_config = {
        "embedding_model": "BAAI/bge-m3",
        "index_namespace": "experiment-caption-segments",
        "citation_policy": "selected_anchors",
    }

    def fake_post_chat(_base_url, case, _timeout_seconds):
        quality_passed = case["id"] != "AR08"
        return {
            "case_id": case["id"],
            "status": "success",
            "audit": {
                "effective_retrieval_config": shared_config,
                "quality": {
                    "passed": quality_passed,
                    "failed_checks": [] if quality_passed else ["sound_stomach"],
                },
            },
        }

    monkeypatch.setattr(audit_module, "_post_chat", fake_post_chat)
    output_path = tmp_path / "phase10_8_6_audit.json"
    report = audit_module.run_audit(
        base_url="http://127.0.0.1:8000",
        output_path=output_path,
        timeout_seconds=5.0,
    )

    assert report["report_id"] == "chat_answer_source_audit_v1"
    assert report["cases_attempted"] == 3
    assert report["cases_succeeded"] == 3
    assert report["cases_failed"] == 0
    assert report["effective_server_configuration_consistent_across_cases"] is True
    assert report["quality_passed"] is False
    assert report["quality_failures"] == [
        {"case_id": "AR08", "failed_checks": ["sound_stomach"]}
    ]
    saved_report = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved_report["report_id"] == "chat_answer_source_audit_v1"
    assert saved_report["quality_passed"] is False


def test_run_audit_requires_effective_config_for_every_case(tmp_path, monkeypatch):
    shared_config = {
        "embedding_model": "BAAI/bge-m3",
        "index_namespace": "experiment-caption-segments",
        "citation_policy": "selected_anchors",
    }

    def fake_post_chat(_base_url, case, _timeout_seconds):
        audit = {
            "quality": {"passed": True, "failed_checks": []},
        }
        if case["id"] != "AR08":
            audit["effective_retrieval_config"] = shared_config
        return {"case_id": case["id"], "status": "success", "audit": audit}

    monkeypatch.setattr(audit_module, "_post_chat", fake_post_chat)
    report = audit_module.run_audit(
        base_url="http://127.0.0.1:8000",
        output_path=tmp_path / "missing-config-audit.json",
        timeout_seconds=5.0,
    )

    assert report["cases_succeeded"] == 3
    assert report["cases_failed"] == 0
    assert report["effective_server_configuration_consistent_across_cases"] is False
    assert report["quality_passed"] is False
    assert report["quality_failures"] == [
        {
            "case_id": "ALL",
            "failed_checks": ["effective_server_configuration_consistency"],
        }
    ]


def test_mojibake_is_flagged_without_silently_rewriting_the_answer():
    response = {
        "answer": "\u00d8\u00a7\u00d9\u0084\u00d9\u0086\u00d8\u00a8\u00d8\u00b6",
        "sources": [],
        "observability": {"retrieval": {}},
    }

    audit = analyze_response(_case("AR08"), response)

    assert audit["answer_text_encoding_suspect"] is True
    assert audit["possible_mojibake_markers"]


def test_audit_separates_verified_context_evidence_from_retrieved_only_chunks():
    response = {
        "answer": (
            "في الغرفة قد يسمع الناس نبض القلب، وحركة الدم في العروق، "
            "وأصوات الرئتين والمعدة، ورمش الجفون.\n\n"
            "Relevant timestamps: 02:33–03:37"
        ),
        "retrieval_config": {
            "embedding_model": "BAAI/bge-m3",
            "index_namespace": "experiment-caption-segments",
            "citation_policy": "selected_anchors",
        },
        "sources": [
            {
                "source_id": 1, "chunk_ids": [2],
                "start": 153.05, "end": 217.615, "distance": 0.84,
            },
            {
                "source_id": 2, "chunk_ids": [11],
                "start": 589.599, "end": 658.17, "distance": 0.95,
            },
        ],
        "observability": {
            "retrieval": {
                "final_chunk_ids": [2, 11],
                "citation_chunk_ids": [2, 11],
            },
            "generation": {
                "context_budget": {
                    "budget_chars": 8000,
                    "original_chars": 14430,
                    "final_chars": 7902,
                    "groups_available": 3,
                    "groups_included": 2,
                    "included_chunk_ids": [2],
                    "reduced": True,
                }
            },
        },
    }

    # Production audit oracle: both the bodily-sound phrase and the eyelid
    # phrase are verified in chunk 2, which is included in the generation
    # context and represented by the 02:33–03:37 source range.
    audit = analyze_response(_case("AR08"), response)

    assert audit["final_context_chunk_ids"] == ["2"]
    assert audit["annotated_evidence_groups"][0]["present_in_final_context"] is True
    assert audit["annotated_evidence_groups"][1]["present_in_final_context"] is True
    assert audit["quality"]["passed"] is True
    assert audit["quality"]["failed_checks"] == []

    # Separate context-budget contract: chunk 11 is retrieved and returned as
    # a clickable source, but was deliberately excluded from the generation
    # context. A synthetic required group pinned to that retrieved-only chunk
    # must not be counted as evidence that the model actually saw.
    production_case = _case("AR08")
    context_budget_case = {
        **production_case,
        "evidence_groups": [
            production_case["evidence_groups"][0],
            {
                "label": "retrieved-only control chunk excluded from generation context",
                "candidate_chunk_ids": [11],
            },
        ],
    }
    truncated_audit = analyze_response(context_budget_case, response)

    assert truncated_audit["final_context_chunk_ids"] == ["2"]
    assert truncated_audit["annotated_evidence_groups"][0]["present_in_final_context"] is True
    assert truncated_audit["annotated_evidence_groups"][1]["present_in_final_context"] is False
    assert truncated_audit["annotated_evidence_groups"][1]["present_in_returned_sources"] is True
    assert truncated_audit["quality"]["passed"] is False
    assert "annotated_evidence_in_retrieved_context" in truncated_audit["quality"]["failed_checks"]


def test_q07_requires_explicit_relationship_uncertainty_not_only_separate_figures():
    answer = (
        "Vijay invested 3,000 crore in Kingfisher. The DRT judgment debt was "
        "4,999 crore plus interest. The debt certificate lists 6,203 crore. "
        "The Ministry of Finance report says 14,100 crore was recovered. "
        "The DRT and certificate figures are separate."
    )

    checks = _answer_quality_checks("Q07", answer)
    results = {item["name"]: item["passed"] for item in checks}

    assert results["investment_3000"] is True
    assert results["drt_4999"] is True
    assert results["certificate_6203"] is True
    assert results["recovery_source_attribution"] is True
    assert results["unclear_relationship_preserved"] is False


def test_q07_live_shaped_context_fallback_passes_financial_and_timestamp_contract():
    context = """[SOURCE 1]
- [02:01:07–02:02:31] I invested 3,000 crores of UB Group money into Kingfisher Airlines.
[SOURCE 2]
- [02:18:34–02:19:01] 4,999 crores was a debt
- [02:19:01–02:20:28] debt recovery tribunal's own admission
[SOURCE 3]
- [02:49:56–02:51:41] That's how you've recovered 14,100 crores from me.
[SOURCE 4]
- [04:09:55–04:10:01] I showed you the debt recovery
- [04:09:58–04:10:04] certificate amount 6203 crores I've shown you the ministry of
- [04:10:01–04:10:07] finance report 14,000
- [04:10:04–04:10:10] 100 recovered."""

    answer = chain._build_financial_evidence_fallback(
        "What does Vijay Mallya say about the amount of debt or money involved in the Kingfisher situation?",
        context,
    )
    assert answer is not None

    source_texts = [
        {
            "source_id": 1, "chunk_ids": [108],
            "start": 7267.92, "end": 7351.84, "distance": 0.82,
            "transcript_text": "I invested 3,000 crores of UB Group money into Kingfisher Airlines.",
        },
        {
            "source_id": 2, "chunk_ids": [124],
            "start": 8314.519, "end": 8428.16, "distance": 0.91,
            "transcript_text": "4,999 crores was a debt. The debt recovery tribunal's own admission.",
        },
        {
            "source_id": 3, "chunk_ids": [153],
            "start": 10196.399, "end": 10301.84, "distance": 0.87,
            "transcript_text": "That's how you've recovered 14,100 crores from me.",
        },
        {
            "source_id": 4, "chunk_ids": [221],
            "start": 14939.359, "end": 15015.239, "distance": 0.81,
            "transcript_text": (
                "I showed you the debt recovery certificate amount 6203 crores. "
                "I've shown you the Ministry of Finance report 14,000 100 recovered."
            ),
        },
    ]
    answer = chain.append_relevant_source_timestamps(
        answer=answer,
        sources=source_texts,
        has_evidence=True,
    )

    response = {
        "answer": answer,
        "retrieval_config": {"citation_policy": "selected_anchors"},
        "sources": source_texts,
        "observability": {
            "retrieval": {
                "final_chunk_ids": [108, 124, 153, 221],
                "citation_chunk_ids": [108, 124, 153, 221],
            },
            "generation": {
                "context_budget": {
                    "budget_chars": 8000,
                    "original_chars": len(context),
                    "final_chars": min(len(context), 7900),
                    "groups_available": 4,
                    "groups_included": 4,
                    "included_chunk_ids": ["108", "124", "153", "221"],
                    "reduced": True,
                }
            },
        },
    }

    audit = analyze_response(_case("Q07"), response)
    assert audit["quality"]["passed"] is True, audit["quality"]["failed_checks"]
    assert audit["quality"]["failed_checks"] == []
    assert len(audit["plain_text_timestamp_ranges"]) >= 4
    assert all(
        group["represented_in_plain_text_timestamp_summary"]
        for group in audit["annotated_evidence_groups"]
    )


def test_timestamp_mojibake_using_windows_punctuation_is_detected():
    response = {
        "answer": "A grounded answer.\n\nRelevant timestamps: 00:10â€“00:20",
        "retrieval_config": {"citation_policy": "selected_anchors"},
        "sources": [{"source_id": 1, "chunk_ids": [2], "start": 10, "end": 20}],
        "observability": {
            "retrieval": {"final_chunk_ids": [2], "citation_chunk_ids": [2]},
            "generation": {
                "context_budget": {
                    "budget_chars": 8000, "original_chars": 1000, "final_chars": 1000,
                    "groups_available": 1, "groups_included": 1, "included_chunk_ids": ["2"],
                    "reduced": False,
                }
            },
        },
    }

    audit = analyze_response(_case("AR03"), response)

    assert audit["answer_text_encoding_suspect"] is True
    assert audit["possible_mojibake_markers"] == ["â€“"]
    assert "encoding_integrity" in audit["quality"]["failed_checks"]


def test_q07_audit_rejects_unstructured_deterministic_financial_fallback():
    response = {
        "answer": (
            "The transcript supports these distinct financial statements:\n\n"
            "- put in 3,000 crores of UB group's money\n"
            "- debt recovery tribunal's own admission 4,999 crores was a debt right rest is all\n"
            "- right I showed you the debt recovery certificate amount 6,203 crores I've shown you the ministry of\n"
            "- The transcript cites the Ministry of Finance report with a recovery figure of 14,100 crore.\n\n"
            "The transcript does not clearly explain how the DRT judgment and debt-certificate figures relate, so they should not be treated as one total."
        ),
        "retrieval_config": {"citation_policy": "selected_anchors"},
        "sources": [
            {"source_id": 1, "chunk_ids": [108], "start": 7267.92, "end": 7351.84},
            {"source_id": 2, "chunk_ids": [124], "start": 8314.519, "end": 8428.16},
            {"source_id": 3, "chunk_ids": [153], "start": 10196.399, "end": 10301.84},
            {"source_id": 4, "chunk_ids": [221], "start": 14939.359, "end": 15015.239},
        ],
        "observability": {
            "retrieval": {"final_chunk_ids": [108, 124, 153, 221], "citation_chunk_ids": [108, 124, 153, 221]},
            "generation": {
                "grounded_financial_fallback_used": True,
                "context_budget": {
                    "budget_chars": 8000, "original_chars": 5000, "final_chars": 5000,
                    "groups_available": 4, "groups_included": 4,
                    "included_chunk_ids": ["108", "124", "153", "221"], "reduced": False,
                },
            },
        },
    }

    audit = analyze_response(_case("Q07"), response)

    assert "financial_fallback_statements_are_structured" in audit["quality"]["failed_checks"]


def test_prompt_included_citations_reject_source_chunks_omitted_from_prompt():
    response = {
        "answer": (
            "A heartbeat, blood rushing through veins, lung and stomach sounds, "
            "and eyelids blinking can become noticeable."
            "\n\nRelevant timestamps: 02:28–03:34"
        ),
        "retrieval_config": {
            "citation_policy": "prompt_included_chunks",
        },
        "sources": [
            {"source_id": 1, "chunk_ids": [2], "start": 148.233, "end": 214.294},
            {"source_id": 2, "chunk_ids": [3], "start": 214.294, "end": 261.778},
        ],
        "observability": {
            "retrieval": {
                "final_chunk_ids": [2, 3],
                "citation_chunk_ids": [2, 3],
            },
            "generation": {
                "context_budget": {
                    "budget_chars": 8000,
                    "original_chars": 1000,
                    "final_chars": 900,
                    "groups_available": 2,
                    "groups_included": 1,
                    "included_chunk_ids": ["2"],
                }
            },
        },
    }

    audit = analyze_response(_case("AR03"), response)

    assert audit["citation_provenance"]["mode"] == "generation_context"
    assert audit["citation_provenance"]["source_chunk_ids_outside_allowed_set"] == ["3"]
    assert audit["citation_provenance"]["passed"] is False
    assert "citation_prompt_provenance" in audit["quality"]["failed_checks"]


def test_prompt_included_citation_policy_uses_anchor_fallback_without_prompt_trace():
    anchor_only = audit_module._citation_provenance_diagnostics(
        citation_policy="prompt_included_chunks",
        retrieval={"citation_chunk_ids": [2]},
        generation={"context_budget": {"budget_chars": 8000, "final_chars": 100}},
        returned_source_ids=["2"],
    )
    neighbor_included = audit_module._citation_provenance_diagnostics(
        citation_policy="prompt_included_chunks",
        retrieval={"citation_chunk_ids": [2]},
        generation={"context_budget": {"budget_chars": 8000, "final_chars": 100}},
        returned_source_ids=["2", "3"],
    )

    assert anchor_only["mode"] == "selected_anchor_fallback"
    assert anchor_only["used_selected_anchor_fallback"] is True
    assert anchor_only["passed"] is True
    assert neighbor_included["passed"] is False
    assert neighbor_included["source_chunk_ids_outside_allowed_set"] == ["3"]


def test_unanswerable_phase10_11_case_requires_abstention_and_no_sources():
    case = next(case for case in PHASE10_11_AUDIT_CASES if case["id"] == "HI10")
    response = {
        "answer": (
            "I couldn't find enough information about that in the video. "
            "It may not be covered by this video."
        ),
        "retrieval_config": {"citation_policy": "prompt_included_chunks"},
        "sources": [],
        "observability": {
            "retrieval": {"final_chunk_ids": [], "citation_chunk_ids": []},
            "generation": {
                "context_budget": {
                    "budget_chars": 8000,
                    "original_chars": 50,
                    "final_chars": 50,
                    "included_chunk_ids": [],
                    "groups_available": 0,
                    "groups_included": 0,
                }
            },
        },
    }

    audit = analyze_response(case, response)

    assert audit["quality"]["passed"] is True, audit["quality"]["failed_checks"]
    assert audit["citation_provenance"]["passed"] is True


def test_phase10_11_registered_answer_requirements_and_language_are_checked():
    case = next(case for case in PHASE10_11_AUDIT_CASES if case["id"] == "HI09")
    checks = _answer_quality_checks(case, "नया टैक्स चवरी कर के नाम से था।")

    assert all(check["passed"] for check in checks), checks


def test_ar04_accepts_perceive_and_navigate_as_supported_perception_synonyms():
    case = next(case for case in PHASE10_11_AUDIT_CASES if case["id"] == "AR04")
    answer = (
        "The episode explains that sound plays a key role in how we perceive "
        "the world and helps us navigate our surroundings."
    )
    checks = _answer_quality_checks(case, answer)
    by_name = {check["name"]: check["passed"] for check in checks}
    assert by_name["answer_requirement_sound_role"] is True
    assert by_name["answer_requirement_perception_or_compass"] is True

    unsupported = _answer_quality_checks(case, "Sound plays a key role in the episode.")
    unsupported_by_name = {
        check["name"]: check["passed"] for check in unsupported
    }
    assert unsupported_by_name["answer_requirement_perception_or_compass"] is False


def test_ew01_accepts_concise_direct_count_answer_but_requires_count_and_unit():
    case = next(case for case in PHASE10_11_AUDIT_CASES if case["id"] == "EW01")

    checks = _answer_quality_checks(case, "Three ways.")
    assert all(check["passed"] for check in checks), checks
    assert {
        "substantive_answer",
        "answer_requirement_three_ways",
        "question_language",
    } == {check["name"] for check in checks}

    formatted_answer = _answer_quality_checks(
        case,
        "The speaker says he will give **three** ways to express yourself in meetings.",
    )
    assert all(check["passed"] for check in formatted_answer), formatted_answer

    incomplete = _answer_quality_checks(case, "Three.")
    by_name = {check["name"]: check["passed"] for check in incomplete}
    assert by_name["answer_requirement_three_ways"] is False


def test_phase10_11_coverage_manifest_accounts_for_pilot_and_blocked_urdu():
    manifest = build_coverage_manifest(PHASE10_11_AUDIT_CASES)

    assert manifest["registered_video_count"] == 5
    assert manifest["live_audit_video_count"] == 5
    assert manifest["live_audit_case_count"] == len(PHASE10_11_AUDIT_CASES)
    assert manifest["all_registered_videos_have_answerable_and_unanswerable_live_cases"] is True
    assert manifest["unregistered_live_cases"] == []
    assert manifest["duplicate_live_cases"] == []
    assert manifest["question_case_accounting"]["accounting_complete"] is True
    assert manifest["question_case_accounting"]["not_live_audited_count"] > 0
    assert manifest["blocked_languages"][0]["language"] == "ur"
    assert all(row["representative_live_coverage_passed"] for row in manifest["videos"])


def test_run_audit_fails_when_effective_configuration_does_not_match_expected(
    tmp_path, monkeypatch
):
    shared_config = {
        "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "index_namespace": "phase10-10-acceptance-minilm-20261023",
        "citation_policy": "selected_anchors",
    }

    def fake_post_chat(_base_url, case, _timeout_seconds):
        return {
            "case_id": case["id"],
            "status": "success",
            "audit": {
                "effective_retrieval_config": shared_config,
                "quality": {"passed": True, "failed_checks": []},
            },
        }

    monkeypatch.setattr(audit_module, "_post_chat", fake_post_chat)
    report = audit_module.run_audit(
        base_url="http://127.0.0.1:8000",
        output_path=tmp_path / "config-mismatch.json",
        timeout_seconds=5.0,
        expected_configuration={"citation_policy": "prompt_included_chunks"},
    )

    assert report["effective_server_configuration_consistent_across_cases"] is True
    assert report["effective_server_configuration_matches_expected"] is False
    assert report["quality_passed"] is False
    assert {
        "case_id": "ALL",
        "failed_checks": ["effective_server_configuration_matches_expected"],
    } in report["quality_failures"]


def test_unanswerable_control_rejects_a_guessed_answer_even_without_sources():
    case = next(case for case in PHASE10_11_AUDIT_CASES if case["id"] == "Q13")
    response = {
        "answer": "Canberra is the capital of Australia.",
        "retrieval_config": {"citation_policy": "prompt_included_chunks"},
        "sources": [],
        "observability": {
            "retrieval": {"final_chunk_ids": [], "citation_chunk_ids": []},
            "generation": {
                "context_budget": {
                    "budget_chars": 8000,
                    "original_chars": 100,
                    "final_chars": 50,
                    "included_chunk_ids": [],
                    "groups_available": 0,
                    "groups_included": 0,
                }
            },
        },
    }

    audit = analyze_response(case, response)

    assert audit["quality"]["passed"] is False
    assert "unanswerable_case_abstained" in audit["quality"]["failed_checks"]


def test_full_pilot_coverage_gate_fails_for_three_case_legacy_suite(tmp_path, monkeypatch):
    shared_config = {
        "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "index_namespace": "phase10-10-acceptance-minilm-20261023",
        "citation_policy": "prompt_included_chunks",
    }

    def fake_post_chat(_base_url, case, _timeout_seconds):
        return {
            "case_id": case["id"],
            "status": "success",
            "audit": {
                "effective_retrieval_config": shared_config,
                "quality": {"passed": True, "failed_checks": []},
            },
        }

    monkeypatch.setattr(audit_module, "_post_chat", fake_post_chat)
    report = audit_module.run_audit(
        base_url="http://127.0.0.1:8000",
        output_path=tmp_path / "incomplete-pilot-coverage.json",
        timeout_seconds=5.0,
        require_full_pilot_coverage=True,
    )

    assert report["coverage_manifest"]["all_registered_videos_have_answerable_and_unanswerable_live_cases"] is False
    assert report["quality_passed"] is False
    assert {
        "case_id": "ALL",
        "failed_checks": ["representative_five_video_live_coverage"],
    } in report["quality_failures"]
