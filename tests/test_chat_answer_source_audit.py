import json
from io import BytesIO
from urllib.error import HTTPError

import pytest

import scripts.chat_answer_source_audit as audit_module
from Backend.rag_system import chain
from scripts.chat_answer_source_audit import (
    AUDIT_CASES,
    PHASE10_11_AUDIT_CASES,
    PHASE11_0_ADDITIONAL_CASES,
    PHASE11_0_AUDIT_CASES,
    analyze_response,
    build_coverage_manifest,
    _answer_timestamp_ranges,
    _timestamp_range_covered_by_sources,
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
        "answer": (
            "In the quiet room, people may hear their heartbeat, blood moving "
            "through their veins, sounds from their lungs and stomach, and even "
            "their eyelids blinking. Arabic phrase: “نبض قلبك.”\n\n"
            "Relevant timestamps: 02:28–04:21"
        ),
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
            "In the quiet room, people may hear their heartbeat, blood moving "
            "through their veins, sounds from their lungs or stomach, and even "
            "their eyelids blinking. Arabic phrase: “نبض قلبك.”",
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


def test_bodily_sound_quality_gate_requires_english_for_arabic_transcript():
    english_answer = (
        "In the quiet room people may hear their heartbeat, blood rushing through "
        "their veins, their lungs and breathing, stomach rumbling, and eyelids blinking. "
        "Arabic transcript phrase: “نبض قلبك.”"
    )
    arabic_answer = (
        "في الغرفة الهادئة قد يبدأ الناس في سماع أصوات أجسامهم: نبض قلبك، "
        "حركة الدم في عروقك، صوت الرئتين أو المعدة، وحتى صوت عينيك وهي بترمش."
    )

    assert all(check["passed"] for check in _answer_quality_checks("AR03", english_answer))
    assert all(check["passed"] for check in _answer_quality_checks("AR08", english_answer))

    arabic_checks = {
        check["name"]: check["passed"]
        for check in _answer_quality_checks("AR08", arabic_answer)
    }
    assert arabic_checks["response_language_english"] is False

    incomplete_english = "In the quiet room people may hear their heartbeat."
    failed = {
        check["name"]
        for check in _answer_quality_checks("AR08", incomplete_english)
        if not check["passed"]
    }
    assert "sound_blood_movement" in failed
    assert "sound_lungs_or_breathing" in failed
    assert "sound_stomach" in failed
    assert "sound_eyelids_blinking" in failed


def test_english_response_language_allows_one_short_quoted_arabic_excerpt():
    answer = (
        "“...حسـدك في أوضة ممكن يا عزيزي تدفع فيها ما يصل إلي 600 دولار.” "
        "In English, this means a one-hour stay in the Orfield room can cost up to $600."
    )
    case = next(
        case for case in PHASE11_0_AUDIT_CASES if case["id"] == "AR07"
    )
    checks = {
        check["name"]: check["passed"]
        for check in _answer_quality_checks(case, answer)
    }

    assert checks["response_language_english"] is True
    assert checks["answer_requirement_room_cost"] is True


def test_english_response_language_rejects_unquoted_source_language_text():
    answer = (
        "A one-hour stay can cost up to $600. "
        "هذا هو السعر المذكور في الفيديو. هذا هو السعر المذكور في الفيديو. "
        "هذا هو السعر المذكور في الفيديو."
    )

    assert audit_module._answer_language_matches(answer, "en") is False


def test_english_response_language_rejects_multiple_source_language_quotes():
    answer = "“نبض قلبك” is followed by “حركة الدم في عروقك” in the passage."

    assert audit_module._answer_language_matches(answer, "en") is False


def test_english_response_language_rejects_overlong_source_language_quotes():
    long_excerpt = " ".join(["نبض قلبك"] * 40)
    answer = f'The passage describes internal sounds: “{long_excerpt}”.'

    assert audit_module._answer_language_matches(answer, "en") is False


def test_english_response_language_rejects_quote_without_english_main_answer():
    assert audit_module._answer_language_matches("“نبض قلبك”", "en") is False


def test_system_instruction_places_source_excerpt_before_english_meaning():
    language_rules = chain.SYSTEM_INSTRUCTIONS.split("EVIDENCE AND ATTRIBUTION", 1)[0].lower()

    assert "put the original-language excerpt first" in language_rules
    assert "follow it immediately with its english meaning" in language_rules


def test_ar08_accepts_pulse_as_a_heartbeat_synonym():
    pulse_answer = (
        "In the quiet room, people may become aware of their pulse, "
        "blood moving through their veins, their lungs, stomach, and blinking."
    )
    checks = {
        check["name"]: check["passed"]
        for check in _answer_quality_checks("AR08", pulse_answer)
    }

    assert checks["sound_heartbeat"] is True

    unrelated_answer = (
        "In the room, people discuss heart health, breathing, and the quiet."
    )
    unrelated_checks = {
        check["name"]: check["passed"]
        for check in _answer_quality_checks("AR08", unrelated_answer)
    }
    assert unrelated_checks["sound_heartbeat"] is False


def test_ar09_accepts_english_sound_and_perception_role_requirements():
    case = next(
        case for case in PHASE11_0_AUDIT_CASES if case["id"] == "AR09"
    )
    answer = (
        "Sound carries signals about the world and can shape what we see. "
        "The sound-induced flash illusion shows that auditory input can bias "
        "visual perception."
    )
    checks = {
        check["name"]: check["passed"]
        for check in _answer_quality_checks(case, answer)
    }

    assert checks["answer_requirement_sound_subject"] is True
    assert checks["answer_requirement_perception_role"] is True


def test_ar09_does_not_pass_perception_requirement_from_sound_alone():
    case = next(
        case for case in PHASE11_0_AUDIT_CASES if case["id"] == "AR09"
    )
    answer = (
        "Sound is present in the room, and the speaker discusses the "
        "experience of sitting in silence."
    )
    checks = {
        check["name"]: check["passed"]
        for check in _answer_quality_checks(case, answer)
    }

    assert checks["answer_requirement_sound_subject"] is True
    assert checks["answer_requirement_perception_role"] is False


def test_post_chat_retries_rate_limit_after_retry_after_and_records_wait(monkeypatch):
    case = {
        "id": "Q13",
        "video_id": "MdeQMVBuGgY",
        "question": "What is the capital of Australia?",
        "expected_behavior": "unanswerable",
    }
    response_payload = {
        "answer": "I couldn't find enough information about that in the video. "
        "It may not be covered by this video.",
        "video_id": case["video_id"],
        "sources": [],
        "retrieval_config": {
            "embedding_model": "BAAI/bge-m3",
            "index_namespace": "test-namespace",
            "citation_policy": "prompt_included_chunks",
        },
        "observability": {"retrieval": {}, "generation": {}},
    }
    successful_body = json.dumps(response_payload, ensure_ascii=False).encode("utf-8")
    rate_limited_body = json.dumps({
        "error": "ai_provider_rate_limited",
        "message": "The AI provider is temporarily rate-limited.",
    }).encode("utf-8")

    class FakeResponse:
        status = 200
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return successful_body

    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        if len(calls) == 1:
            raise HTTPError(
                request.full_url,
                429,
                "Too Many Requests",
                {"Retry-After": "2", "Content-Type": "application/json"},
                BytesIO(rate_limited_body),
            )
        return FakeResponse()

    sleeps = []
    monkeypatch.setattr(audit_module, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        audit_module.time,
        "sleep",
        lambda seconds: sleeps.append(seconds),
    )

    result = audit_module._post_chat(
        "http://127.0.0.1:8000",
        case,
        timeout_seconds=5.0,
    )

    assert result["status"] == "success"
    assert result["http_status"] == 200
    assert result["request_attempts"] == 2
    assert result["rate_limit_retry_count"] == 1
    assert result["rate_limit_retry_wait_seconds"] == 3.0
    assert sleeps == [3.0]
    assert len(calls) == 2


def test_q03_quality_gate_normalizes_unicode_narrow_no_break_spaces():
    case = next(
        case for case in PHASE11_0_AUDIT_CASES if case["id"] == "Q03"
    )
    answer = (
        "Vijay Mallya discusses United\u202fBreweries and "
        "United\u202fSpirits, alongside Kingfisher Airlines."
    )

    checks = _answer_quality_checks(case, answer)
    failed = [check["name"] for check in checks if not check["passed"]]

    assert failed == []


def test_answer_requirement_normalizer_converts_smart_apostrophes():
    normalized = audit_module._normalize_answer_requirement_text(
        "The farmers’ movement was discussed."
    )

    assert normalized == "The farmers' movement was discussed."


def test_answer_requirement_normalizer_canonicalizes_unicode_hyphens():
    for dash in ("‑", "‐"):
        normalized = audit_module._normalize_answer_requirement_text(
            f"30{dash}second"
        )
        assert normalized == "30-second"


def test_ew02_live_requirement_accepts_nonbreaking_hyphen():
    case = next(
        case for case in PHASE11_0_AUDIT_CASES if case["id"] == "EW02"
    )
    checks = _answer_quality_checks(
        case,
        "The speaker offers a 30‑second structure for explaining work.",
    )

    assert all(check["passed"] for check in checks), checks


@pytest.mark.parametrize("case_id", ["HI01", "HI06"])
def test_hindi_named_movement_quality_gate_requires_bijolia(case_id):
    case = next(
        case for case in PHASE11_0_AUDIT_CASES if case["id"] == case_id
    )

    grounded_checks = _answer_quality_checks(
        case,
        "The first major farmers' movement was the Bijolia movement.",
    )
    assert all(check["passed"] for check in grounded_checks)

    generic_checks = _answer_quality_checks(
        case,
        "The first major farmers’ movement was a large-scale farmers' protest.",
    )
    failed = [check["name"] for check in generic_checks if not check["passed"]]
    assert failed == ["answer_requirement_named_movement"]


def test_answer_timestamp_parser_finds_inline_hindi_ranges_without_footer():
    answer = (
        "पहला उल्लेख 00:02–00:06 के आसपास है और दूसरा "
        "16:30–16:37 के आसपास है।"
    )

    assert _answer_timestamp_ranges(answer) == [
        "00:02–00:06",
        "16:30–16:37",
    ]


def test_answer_timestamp_parser_accepts_word_range_separators():
    answer = (
        "The first section is 00:00 to 01:08 and the next is "
        "01:10 and 01:22."
    )

    assert _answer_timestamp_ranges(answer) == [
        "00:00 to 01:08",
        "01:10 and 01:22",
    ]


@pytest.mark.parametrize(
    "timestamp_range,source_start,source_end,expected_temporal_pass",
    [
        ("00:00 to 01:08", 0, 130, False),
        ("01:10:38–01:12:04", 4238, 4324, True),
    ],
)
def test_q11_temporal_timestamps_must_match_registered_video_sections(
    timestamp_range,
    source_start,
    source_end,
    expected_temporal_pass,
):
    case = audit_module._make_audit_case(
        "Q11",
        "MdeQMVBuGgY",
        expected_behavior="answerable",
        answer_language="en",
    )
    response = {
        "answer": (
            "Kingfisher Airlines is discussed in the interview."
            f"\n\nRelevant timestamps: {timestamp_range}"
        ),
        "video_id": "MdeQMVBuGgY",
        "retrieval_config": {"citation_policy": "prompt_included_chunks"},
        "sources": [
            {
                "source_id": "source-1",
                "chunk_ids": ["1"],
                "start": source_start,
                "end": source_end,
            }
        ],
        "observability": {
            "retrieval": {
                "final_chunk_ids": ["1"],
                "citation_chunk_ids": ["1"],
            },
            "generation": {
                "context_budget": {
                    "included_chunk_ids": ["1"],
                    "budget_chars": 8000,
                    "final_chars": 250,
                }
            },
        },
    }

    audit = analyze_response(case, response)
    checks = {item["name"]: item["passed"] for item in audit["quality"]["checks"]}

    assert checks["timestamp_summary_matches_returned_sources"] is True
    assert (
        checks["timestamp_summary_matches_registered_temporal_evidence"]
        is expected_temporal_pass
    )
    assert audit["temporal_timestamp_alignment"]["passed"] is expected_temporal_pass


def test_audit_source_timestamp_formatter_rejects_non_finite_values():
    assert audit_module._format_timestamp(float("nan")) is None
    assert audit_module._format_timestamp(float("inf")) is None
    assert audit_module._format_timestamp(-1) is None


def test_temporal_alignment_honors_annotation_minimum_window_coverage():
    windows = [
        {"section_id": "one", "start_seconds": 0, "end_seconds": 10},
        {"section_id": "two", "start_seconds": 20, "end_seconds": 30},
        {"section_id": "three", "start_seconds": 40, "end_seconds": 50},
        {"section_id": "four", "start_seconds": 60, "end_seconds": 70},
    ]
    case = {
        "temporal_evidence_windows": windows,
        "temporal_evidence_min_window_coverage": 0.75,
    }

    partial = audit_module._temporal_timestamp_alignment(
        case,
        ["00:00–00:10", "00:20–00:30"],
    )
    sufficient = audit_module._temporal_timestamp_alignment(
        case,
        ["00:00–00:10", "00:20–00:30", "00:40–00:50"],
    )

    assert partial["passed"] is False
    assert partial["window_coverage"] == 0.5
    assert partial["required_window_coverage"] == 0.75
    assert sufficient["passed"] is True
    assert sufficient["window_coverage"] == 0.75


def test_audit_rejects_inline_model_timestamp_claims_outside_trusted_footer():
    case = {
        "id": "TIMESTAMP_TEST",
        "video_id": "video-test",
        "question": "What is discussed?",
        "expected_behavior": "answerable",
        "answer_requirements": [],
        "evidence_groups": [],
        "answer_language": "en",
    }
    response = {
        "answer": (
            "A grounded answer about the topic, with the discussion occurring "
            "around 01:20.\n\nRelevant timestamps: 00:10–00:20"
        ),
        "video_id": "video-test",
        "retrieval_config": {"citation_policy": "prompt_included_chunks"},
        "sources": [
            {
                "source_id": "source-1",
                "chunk_ids": ["1"],
                "start": 10,
                "end": 20,
            }
        ],
        "observability": {
            "retrieval": {
                "final_chunk_ids": ["1"],
                "citation_chunk_ids": ["1"],
            },
            "generation": {
                "context_budget": {
                    "included_chunk_ids": ["1"],
                    "budget_chars": 8000,
                    "final_chars": 200,
                }
            },
        },
    }

    audit = analyze_response(case, response)

    assert audit["unverified_model_timestamp_claims_present"] is True
    assert "unverified_model_timestamp_claims_absent" in audit["quality"]["failed_checks"]


def test_malformed_timestamp_ranges_are_not_silently_ignored():
    answer = "The model claims the segment runs from 01:75 to 02:00."

    assert _answer_timestamp_ranges(answer) == ["01:75 to 02:00"]
    assert not _timestamp_range_covered_by_sources(
        "01:75 to 02:00",
        [{"display_range": "00:00–10:00"}],
    )


def test_timestamp_range_must_be_fully_covered_by_returned_source_intervals():
    hi_sources = [
        {"display_range": "00:02–01:18"},
        {"display_range": "15:49–17:08"},
    ]
    kingfisher_sources = [
        {"display_range": "01:10:38–01:12:04"},
        {"display_range": "01:11:54–01:13:23"},
    ]

    assert _timestamp_range_covered_by_sources("00:02–00:06", hi_sources)
    assert _timestamp_range_covered_by_sources("16:30–16:37", hi_sources)
    assert _timestamp_range_covered_by_sources(
        "01:10:38–01:13:23", kingfisher_sources
    )
    adjacent_sources = [
        {"display_range": "00:00–00:10"},
        {"display_range": "00:11–00:20"},
    ]
    assert _timestamp_range_covered_by_sources(
        "00:00–00:20", adjacent_sources
    )
    assert not _timestamp_range_covered_by_sources(
        "01:10:38–01:14:34", kingfisher_sources
    )


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
            "In the quiet room, people may hear their heartbeat, blood moving "
            "through their veins, sounds from their lungs and stomach, and even "
            "their eyelids blinking. Arabic phrase: “نبض قلبك.”\n\n"
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


@pytest.mark.parametrize(
    "answer",
    [
        (
            "لا أجد معلومات كافية حول ذلك في الفيديو. "
            "قد لا يكون موضوع العاصمة أستراليا مغطى في هذا الفيديو."
        ),
        "لا أجد معلومات كافية حول ذلك في الفيديو.",
        "لم أجد في الفيديو معلومات كافية للإجابة عن هذا السؤال.",
        "قد لا يكون موضوع العاصمة أستراليا مغطى في هذا الفيديو.",
    ],
)
def test_ar10_recognizes_but_rejects_non_english_abstention_variants(answer):
    case = next(case for case in PHASE10_11_AUDIT_CASES if case["id"] == "AR10")

    checks = _answer_quality_checks(case, answer)

    by_name = {check["name"]: check["passed"] for check in checks}
    assert by_name["unanswerable_case_abstained"] is True, checks
    assert by_name["response_language_english"] is False, checks


def test_ar10_does_not_mistake_a_guessed_capital_for_arabic_abstention():
    case = next(case for case in PHASE10_11_AUDIT_CASES if case["id"] == "AR10")

    checks = _answer_quality_checks(case, "عاصمة أستراليا هي كانبيرا.")

    by_name = {check["name"]: check["passed"] for check in checks}
    assert by_name["unanswerable_case_abstained"] is False
    assert by_name["response_language_english"] is False


def test_phase10_11_registered_answer_requirements_and_english_response_are_checked():
    case = next(case for case in PHASE10_11_AUDIT_CASES if case["id"] == "HI09")
    checks = _answer_quality_checks(
        case,
        'The speaker identifies the new tax as “चवरी कर” (the original Hindi term).',
    )

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
        "response_language_english",
    } == {check["name"] for check in checks}

    formatted_answer = _answer_quality_checks(
        case,
        "The speaker says he will give **three** ways to express yourself in meetings.",
    )
    assert all(check["passed"] for check in formatted_answer), formatted_answer

    incomplete = _answer_quality_checks(case, "Three.")
    by_name = {check["name"]: check["passed"] for check in incomplete}
    assert by_name["answer_requirement_three_ways"] is False


@pytest.mark.parametrize(
    "answer",
    [
        "I couldn't find enough information about that in the video. "
        "It may not be covered by this video.",
        "वीडियो में इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं मिली।",
        "لم أجد في الفيديو معلومات كافية للإجابة عن هذا السؤال.",
    ],
)
def test_answerable_case_rejects_localized_abstention(answer):
    case = next(case for case in PHASE10_11_AUDIT_CASES if case["id"] == "EW01")

    checks = _answer_quality_checks(case, answer)
    by_name = {check["name"]: check["passed"] for check in checks}

    assert by_name["substantive_answer"] is False


def test_answerable_case_keeps_substantive_caveat_as_an_answer():
    case = next(case for case in PHASE10_11_AUDIT_CASES if case["id"] == "EW01")
    answer = (
        "The video does not specify every detail, but the speaker says there "
        "are three ways to express yourself in meetings."
    )

    checks = _answer_quality_checks(case, answer)
    by_name = {check["name"]: check["passed"] for check in checks}

    assert by_name["substantive_answer"] is True


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


def test_phase11_0_adds_exactly_the_27_uncovered_registered_questions():
    expected_additional_ids = {
        "Q01", "Q02", "Q03", "Q04", "Q05", "Q06",
        "Q08", "Q09", "Q10", "Q11", "Q12",
        "EW02", "EW03", "EW04",
        "AI02", "AI03", "AI04",
        "HI01", "HI02", "HI03", "HI06", "HI07", "HI08",
        "AR01", "AR02", "AR06", "AR07",
    }

    assert len(PHASE10_11_AUDIT_CASES) == 16
    assert len(PHASE11_0_ADDITIONAL_CASES) == 27
    assert len(PHASE11_0_AUDIT_CASES) == 43
    assert PHASE11_0_AUDIT_CASES[:16] == PHASE10_11_AUDIT_CASES
    assert {
        case["id"] for case in PHASE11_0_ADDITIONAL_CASES
    } == expected_additional_ids
    assert len({
        (case["video_id"], case["id"])
        for case in PHASE11_0_AUDIT_CASES
    }) == 43


def test_phase11_0_full_registry_manifest_has_no_unevaluated_questions():
    manifest = build_coverage_manifest(PHASE11_0_AUDIT_CASES)

    assert manifest["registered_video_count"] == 5
    assert manifest["registered_question_case_count"] == 43
    assert manifest["live_audit_case_count"] == 43
    assert manifest["live_audit_video_count"] == 5
    assert manifest["unregistered_live_cases"] == []
    assert manifest["duplicate_live_cases"] == []
    assert manifest["question_case_accounting"]["live_audited_count"] == 43
    assert manifest["question_case_accounting"]["not_live_audited_count"] == 0
    assert manifest["question_case_accounting"]["not_live_audited_cases"] == []
    assert manifest["question_case_accounting"]["accounting_complete"] is True
    assert manifest[
        "all_registered_videos_have_answerable_and_unanswerable_live_cases"
    ] is True


def test_phase11_0_keeps_registry_unanswerable_q12_as_abstention_case():
    case = next(item for item in PHASE11_0_AUDIT_CASES if item["id"] == "Q12")

    assert case["expected_behavior"] == "unanswerable"
    assert case.get("answer_language") is None


def test_phase11_0_hindi_and_arabic_transcript_cases_use_english_questions_and_answers():
    cases = {
        item["id"]: item
        for item in PHASE11_0_AUDIT_CASES
    }

    for case_id in ("HI06", "HI07", "HI08", "AR06", "AR07"):
        assert cases[case_id]["answer_language"] == "en"
        assert cases[case_id]["question"].isascii()

def test_phase11_0_full_registry_gate_requires_all_43_cases(monkeypatch, tmp_path):
    def successful_fake_chat(_base_url, case, _timeout_seconds):
        return {
            "case_id": case["id"],
            "status": "success",
            "audit": {
                "effective_retrieval_config": {"top_k": 4},
                "quality": {"passed": True, "failed_checks": []},
            },
        }

    monkeypatch.setattr(audit_module, "_post_chat", successful_fake_chat)

    full_report = audit_module.run_audit(
        base_url="http://unused.invalid",
        output_path=tmp_path / "phase11_0_full.json",
        timeout_seconds=1.0,
        cases=PHASE11_0_AUDIT_CASES,
        report_id="phase11_0_test_full",
        expected_configuration={"top_k": 4},
        require_full_pilot_coverage=True,
        require_full_registry_coverage=True,
    )

    assert full_report["cases_attempted"] == 43
    assert full_report["cases_succeeded"] == 43
    assert full_report["full_registry_coverage_passed"] is True
    assert full_report["quality_passed"] is True

    subset_report = audit_module.run_audit(
        base_url="http://unused.invalid",
        output_path=tmp_path / "phase11_0_subset.json",
        timeout_seconds=1.0,
        cases=PHASE11_0_ADDITIONAL_CASES,
        report_id="phase11_0_test_incomplete",
        expected_configuration={"top_k": 4},
        require_full_pilot_coverage=True,
        require_full_registry_coverage=True,
    )

    assert subset_report["cases_succeeded"] == 27
    assert subset_report["full_registry_coverage_passed"] is False
    assert subset_report["quality_passed"] is False
    assert {
        check
        for failure in subset_report["quality_failures"]
        for check in failure["failed_checks"]
    } >= {
        "representative_five_video_live_coverage",
        "all_registered_question_cases_live_audited",
    }


def test_temporal_location_audit_fails_when_incomplete_first_pass_is_not_repaired():
    case = {
        "id": "Q11",
        "question": (
            "Where in the video does Vijay Mallya discuss Kingfisher Airlines, "
            "and what is discussed there?"
        ),
    }
    generation = {
        "temporal_location_coverage_before": {
            "is_temporal_location_question": True,
            "candidate_source_group_numbers": [1, 2, 3, 4, 5],
            "covered_source_group_numbers": [1],
            "required_covered_source_group_count": 4,
            "coverage_ratio": 0.2,
            "needs_review": True,
        },
        "temporal_location_review_attempted": False,
        "temporal_location_review_passed": None,
        "temporal_location_coverage_after": None,
    }

    check = audit_module._temporal_location_review_quality_check(
        case, generation, "The final answer covers all relevant sections."
    )

    assert check is not None
    assert check["name"] == "temporal_location_answer_covers_substantive_sections"
    assert check["passed"] is False


def test_temporal_location_audit_passes_when_review_covers_distinct_sections():
    case = {
        "id": "Q11",
        "question": (
            "Where in the video does Vijay Mallya discuss Kingfisher Airlines, "
            "and what is discussed there?"
        ),
    }
    context = (
        "[SOURCE 1]\nKingfisher Airlines is briefly introduced.\n"
        "[SOURCE 2]\nKingfisher Airlines was launched in 2002 after an initial planning period.\n"
        "[SOURCE 3]\nKingfisher Airlines focused on premium passenger comfort and safety.\n"
        "[SOURCE 4]\nThe Deccan acquisition and one-rupee tickets were discussed.\n"
        "[SOURCE 5]\nKingfisher's financial turmoil and bank support were discussed."
    )
    answer = (
        "The opening briefly introduces Kingfisher Airlines. The interviewer asks "
        "when Mallya decided to start the airline; he gives 2002. Mallya describes "
        "a premium experience focused on passenger comfort and safety. The later "
        "discussion covers the Deccan acquisition and one-rupee tickets, followed "
        "by Kingfisher's financial turmoil and bank support."
    )
    generation = {
        "temporal_location_coverage_before": {
            "is_temporal_location_question": True,
            "candidate_source_group_numbers": [1, 2, 3, 4, 5],
            "covered_source_group_numbers": [1],
            "required_covered_source_group_count": 4,
            "coverage_ratio": 0.2,
            "needs_review": True,
        },
        "temporal_location_review_attempted": True,
        "temporal_location_review_passed": True,
        "temporal_location_coverage_after": {
            "is_temporal_location_question": True,
            "candidate_source_group_numbers": [1, 2, 3, 4, 5],
            "covered_source_group_numbers": [2, 3, 4, 5],
            "required_covered_source_group_count": 4,
            "coverage_ratio": 0.8,
            "needs_review": False,
        },
        "temporal_location_context": context,
    }

    check = audit_module._temporal_location_review_quality_check(
        case, generation, answer
    )

    assert check is not None
    assert check["passed"] is True


def test_temporal_location_audit_passes_when_no_repair_was_needed():
    case = {
        "id": "Q11",
        "question": (
            "Where in the video does Vijay Mallya discuss Kingfisher Airlines, "
            "and what is discussed there?"
        ),
    }
    context = (
        "[SOURCE 1]\nKingfisher Airlines was briefly introduced.\n"
        "[SOURCE 2]\nKingfisher Airlines launched in 2002 after planning.\n"
        "[SOURCE 3]\nKingfisher Airlines emphasized premium passenger comfort and safety."
    )
    answer = (
        "Kingfisher Airlines was briefly introduced. It launched in 2002 after "
        "planning. Its premium focus included passenger comfort and safety."
    )
    generation = {
        "temporal_location_coverage_before": {
            "is_temporal_location_question": True,
            "candidate_source_group_numbers": [1, 2, 3],
            "covered_source_group_numbers": [1, 2, 3],
            "required_covered_source_group_count": 2,
            "coverage_ratio": 1.0,
            "needs_review": False,
        },
        "temporal_location_review_attempted": False,
        "temporal_location_review_passed": None,
        "temporal_location_coverage_after": None,
        "temporal_location_context": context,
    }

    check = audit_module._temporal_location_review_quality_check(
        case, generation, answer
    )

    assert check is not None
    assert check["passed"] is True


def test_temporal_location_audit_does_not_apply_to_ordinary_topic_questions():
    case = {
        "id": "Q01",
        "question": "What does the speaker say about the airline's passenger experience?",
    }

    assert audit_module._temporal_location_review_quality_check(case, {}, "") is None


def test_temporal_location_audit_fails_when_no_topic_evidence_reached_context():
    case = {
        "id": "Q11",
        "question": (
            "Where in the video does Vijay Mallya discuss Kingfisher Airlines, "
            "and what is discussed there?"
        ),
    }
    generation = {
        "temporal_location_coverage_before": {
            "is_temporal_location_question": True,
            "candidate_source_group_numbers": [],
            "covered_source_group_numbers": [],
            "required_covered_source_group_count": 0,
            "coverage_ratio": 0.0,
            "needs_review": False,
        },
        "temporal_location_review_attempted": False,
        "temporal_location_review_passed": None,
        "temporal_location_coverage_after": None,
    }

    check = audit_module._temporal_location_review_quality_check(
        case, generation, "The final answer covers all relevant sections."
    )

    assert check is not None
    assert check["passed"] is False
    assert "no topic-matching transcript source blocks" in check["detail"].lower()



def test_temporal_location_audit_rejects_final_answer_replaced_after_successful_repair():
    case = {
        "id": "Q11",
        "question": (
            "Where in the video does Vijay Mallya discuss Kingfisher Airlines, "
            "and what is discussed there?"
        ),
    }
    context = (
        "[SOURCE 1]\nKingfisher Airlines is briefly introduced.\n"
        "[SOURCE 2]\nKingfisher Airlines was launched in 2002 after an initial planning period.\n"
        "[SOURCE 3]\nKingfisher Airlines focused on premium passenger comfort and safety.\n"
        "[SOURCE 4]\nThe Deccan acquisition and one-rupee tickets were discussed.\n"
        "[SOURCE 5]\nKingfisher's financial turmoil and bank support were discussed."
    )
    generation = {
        "temporal_location_coverage_before": {
            "is_temporal_location_question": True,
            "candidate_source_group_numbers": [1, 2, 3, 4, 5],
            "covered_source_group_numbers": [1],
            "required_covered_source_group_count": 4,
            "coverage_ratio": 0.2,
            "needs_review": True,
        },
        "temporal_location_review_attempted": True,
        "temporal_location_review_passed": True,
        "temporal_location_coverage_after": {
            "is_temporal_location_question": True,
            "candidate_source_group_numbers": [1, 2, 3, 4, 5],
            "covered_source_group_numbers": [2, 3, 4, 5],
            "required_covered_source_group_count": 4,
            "coverage_ratio": 0.8,
            "needs_review": False,
        },
        "temporal_location_context": context,
    }

    check = audit_module._temporal_location_review_quality_check(
        case,
        generation,
        "I couldn't find enough information about that in the video. It may not be covered by this video.",
    )

    assert check is not None
    assert check["passed"] is False
    assert "final returned answer" in check["detail"].lower()


def test_temporal_location_audit_passes_only_when_final_answer_matches_source_coverage():
    case = {
        "id": "Q11",
        "question": (
            "Where in the video does Vijay Mallya discuss Kingfisher Airlines, "
            "and what is discussed there?"
        ),
    }
    context = (
        "[SOURCE 1]\\nKingfisher Airlines is briefly introduced.\\n"
        "[SOURCE 2]\\nKingfisher Airlines was launched in 2002 after an initial planning period.\\n"
        "[SOURCE 3]\\nKingfisher Airlines focused on premium passenger comfort and safety.\\n"
        "[SOURCE 4]\\nThe Deccan acquisition and one-rupee tickets were discussed.\\n"
        "[SOURCE 5]\\nKingfisher's financial turmoil and bank support were discussed."
    )
    answer = (
        "The interviewer asks when Mallya decided to start Kingfisher Airlines; "
        "he gives 2002. He describes premium passenger comfort and safety. "
        "The discussion later covers the Deccan acquisition, one-rupee tickets, "
        "and Kingfisher's financial turmoil and bank support."
    )
    generation = {
        "temporal_location_coverage_before": {
            "is_temporal_location_question": True,
            "candidate_source_group_numbers": [1, 2, 3, 4, 5],
            "covered_source_group_numbers": [1],
            "required_covered_source_group_count": 4,
            "coverage_ratio": 0.2,
            "needs_review": True,
        },
        "temporal_location_review_attempted": True,
        "temporal_location_review_passed": True,
        "temporal_location_coverage_after": {
            "is_temporal_location_question": True,
            "candidate_source_group_numbers": [1, 2, 3, 4, 5],
            "covered_source_group_numbers": [2, 3, 4, 5],
            "required_covered_source_group_count": 4,
            "coverage_ratio": 0.8,
            "needs_review": False,
        },
        "temporal_location_context": context,
    }

    check = audit_module._temporal_location_review_quality_check(
        case,
        generation,
        answer,
    )

    assert check is not None
    assert check["passed"] is True


def test_temporal_location_audit_rejects_repair_diagnostics_with_changed_candidates():
    case = {
        "id": "Q11",
        "question": (
            "Where in the video does Vijay Mallya discuss Kingfisher Airlines, "
            "and what is discussed there?"
        ),
    }
    generation = {
        "temporal_location_coverage_before": {
            "is_temporal_location_question": True,
            "candidate_source_group_numbers": [1, 2, 3, 4, 5],
            "covered_source_group_numbers": [1],
            "required_covered_source_group_count": 4,
            "coverage_ratio": 0.2,
            "needs_review": True,
        },
        "temporal_location_review_attempted": True,
        "temporal_location_review_passed": True,
        "temporal_location_coverage_after": {
            "is_temporal_location_question": True,
            "candidate_source_group_numbers": [2, 3, 4, 5],
            "covered_source_group_numbers": [2, 3, 4, 5],
            "required_covered_source_group_count": 3,
            "coverage_ratio": 1.0,
            "needs_review": False,
        },
    }

    check = audit_module._temporal_location_review_quality_check(
        case, generation, "The final answer covers all relevant sections."
    )

    assert check is not None
    assert check["passed"] is False

