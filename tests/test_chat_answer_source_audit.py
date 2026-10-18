import json

from scripts.chat_answer_source_audit import (
    AUDIT_CASES,
    analyze_response,
    decode_json_response,
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


def test_q07_evidence_groups_distinguish_context_sources_and_summary():
    response = {
        "answer": (
            "Several amounts were discussed.\n\n"
            "Relevant timestamps: 02:08:48–02:10:10, "
            "03:31:45–03:33:40, 03:56:46–03:58:20"
        ),
        "retrieval_config": {"citation_policy": "selected_anchors"},
        "sources": [
            {
                "source_id": 1,
                "chunk_ids": [115],
                "start": 7728.079,
                "end": 7810.0,
                "distance": 0.758,
            },
            {
                "source_id": 2,
                "chunk_ids": [124],
                "start": 8314.519,
                "end": 8428.16,
                "distance": 0.908,
            },
            {
                "source_id": 3,
                "chunk_ids": [159],
                "start": 10630.88,
                "end": 10698.72,
                "distance": 0.823,
            },
            {
                "source_id": 4,
                "chunk_ids": [189],
                "start": 12705.08,
                "end": 12820.76,
                "distance": 0.769,
            },
            {
                "source_id": 5,
                "chunk_ids": [221],
                "start": 14939.359,
                "end": 15015.239,
                "distance": 1.12,
            },
        ],
        "observability": {
            "retrieval": {
                "final_chunk_ids": [115, 124, 159, 189, 221],
                "citation_chunk_ids": [115, 124, 159, 189, 221],
            }
        },
    }

    audit = analyze_response(_case("Q07"), response)
    groups = audit["annotated_evidence_groups"]

    assert all(group["present_in_returned_sources"] for group in groups)
    assert groups[0]["represented_in_plain_text_timestamp_summary"] is False
    assert groups[1]["represented_in_plain_text_timestamp_summary"] is False
    assert groups[2]["represented_in_plain_text_timestamp_summary"] is False


def test_mojibake_is_flagged_without_silently_rewriting_the_answer():
    response = {
        "answer": "\\u00d8\\u00a7\\u00d9\\u0084\\u00d9\\u0086\\u00d8\\u00a8\\u00d8\\u00b6",
        "sources": [],
        "observability": {"retrieval": {}},
    }

    audit = analyze_response(_case("AR08"), response)

    assert audit["answer_text_encoding_suspect"] is True
    assert audit["possible_mojibake_markers"]
