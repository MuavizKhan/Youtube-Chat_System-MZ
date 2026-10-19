import json

import scripts.chat_answer_source_audit as audit_module
from scripts.chat_answer_source_audit import (
    AUDIT_CASES,
    analyze_response,
    decode_json_response,
    _answer_quality_checks,
)


def _case(case_id):
    return next(case for case in AUDIT_CASES if case["id"] == case_id)


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
            "citation_policy": "expanded_context_experiment",
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
            }
        },
    }

    audit = analyze_response(_case("AR08"), response)
    evidence = audit["annotated_evidence_groups"][0]

    assert evidence["present_in_final_context"] is True
    assert evidence["present_in_selected_anchors"] is False
    assert evidence["present_in_returned_sources"] is True
    assert evidence["represented_in_plain_text_timestamp_summary"] is True


def test_ar08_requires_both_bodily_sound_source_intervals_in_timestamp_summary():
    answer_text = (
        "في الغرفة الهادئة قد يبدأ الناس في سماع أصوات أجسامهم: نبض قلبك، "
        "حركة الدم في عروقك، صوت الرئتين أو المعدة، وحتى صوت عينيك وهي بترمش."
    )
    response = {
        "answer": answer_text + "\n\nRelevant timestamps: 02:33–03:37, 09:49–10:58",
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
                }
            },
        },
    }

    accepted = analyze_response(_case("AR08"), response)
    assert accepted["quality"]["passed"] is True
    assert accepted["quality"]["failed_checks"] == []

    response["answer"] = answer_text + "\n\nRelevant timestamps: 04:15–05:18, 09:49–10:58"
    rejected = analyze_response(_case("AR08"), response)
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

    assert report["cases_attempted"] == 3
    assert report["cases_succeeded"] == 3
    assert report["cases_failed"] == 0
    assert report["effective_server_configuration_consistent_across_cases"] is True
    assert report["quality_passed"] is False
    assert report["quality_failures"] == [
        {"case_id": "AR08", "failed_checks": ["sound_stomach"]}
    ]
    saved_report = json.loads(output_path.read_text(encoding="utf-8"))
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


def test_audit_uses_chunks_included_in_generation_context_not_all_retrieved_chunks():
    response = {
        "answer": (
            "في الغرفة قد يسمع الناس نبض القلب، وحركة الدم في العروق، "
            "وأصوات الرئتين والمعدة، ورمش الجفون.\n\n"
            "Relevant timestamps: 02:33–03:37, 09:49–10:58"
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

    audit = analyze_response(_case("AR08"), response)

    assert audit["final_context_chunk_ids"] == ["2"]
    assert audit["annotated_evidence_groups"][0]["present_in_final_context"] is True
    assert audit["annotated_evidence_groups"][1]["present_in_final_context"] is False
    assert audit["quality"]["passed"] is False
    assert "annotated_evidence_in_retrieved_context" in audit["quality"]["failed_checks"]
