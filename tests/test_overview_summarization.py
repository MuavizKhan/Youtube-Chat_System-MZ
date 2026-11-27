import json
import re

from langchain_core.documents import Document

from Backend.rag_system import overview_summarization
from Backend.rag_system.overview_summarization import (
    _balance_points_over_timeline,
    _timeline_coverage_stats,
)


def _document_for_minute(index, minute):
    start = minute * 60.0
    return Document(
        page_content=f"Transcript passage at minute {minute}.",
        metadata={
            "chunk_id": index,
            "video_id": "VKrtiTgBeng",
            "start": start,
            "end": start + 45.0,
        },
    )


def _point(text, evidence_id):
    return {"text": text, "evidence_ids": [evidence_id]}


def test_overview_replaces_clustered_reducer_points_with_late_map_findings():
    # Model an 87-minute interview whose map phase found material across the
    # timeline, but whose reducer selected most of its points from the opening.
    minutes = [
        1, 2, 5, 15, 23, 30, 42, 46, 55, 60, 70, 78, 84,
    ]
    lookup = {
        f"P{index:05d}": _document_for_minute(index, minute)
        for index, minute in enumerate(minutes, start=1)
    }

    reducer_points = [
        _point("Opening context is introduced.", "P00001"),
        _point("A second early detail is discussed.", "P00002"),
        _point("A third opening event occurs.", "P00003"),
        _point("The first major family event is described.", "P00004"),
        _point("The family story develops further.", "P00005"),
        _point("A later middle discussion begins.", "P00008"),
        _point("Another middle discussion follows.", "P00010"),
    ]
    map_findings = [
        _point("A distinct development occurs around minute thirty.", "P00006"),
        _point("The discussion changes direction around minute forty-two.", "P00007"),
        _point("A later event is explained around minute fifty-five.", "P00009"),
        _point("A significant development occurs around minute seventy.", "P00011"),
        _point("The concluding discussion occurs near the end.", "P00013"),
    ]

    selected, balance = _balance_points_over_timeline(
        reducer_points=reducer_points,
        map_findings=map_findings,
        lookup=lookup,
        previous_points=[],
        requested=7,
    )
    coverage = _timeline_coverage_stats(selected, lookup, 7)

    assert len(selected) == 7
    assert balance["balancing_applied"] is True
    assert balance["map_points_selected"] >= 2
    assert "A significant development occurs around minute seventy." in [
        point["text"] for point in selected
    ]
    assert "The concluding discussion occurs near the end." in [
        point["text"] for point in selected
    ]
    assert coverage["covered_buckets"] == [1, 2, 3, 4, 5, 6, 7]
    assert coverage["missing_buckets"] == []
    assert coverage["coverage_fraction"] == 1.0

    selected_midpoints = [
        lookup[point["evidence_ids"][0]].metadata["start"]
        for point in selected
    ]
    assert selected_midpoints == sorted(selected_midpoints)


def test_overview_timeline_coverage_reports_missing_late_buckets_without_inventing():
    lookup = {
        "P00001": _document_for_minute(1, 1),
        "P00002": _document_for_minute(2, 15),
        "P00003": _document_for_minute(3, 30),
        "P00004": _document_for_minute(4, 42),
    }
    points = [
        _point("Opening fact.", "P00001"),
        _point("Early development.", "P00002"),
        _point("Middle development.", "P00003"),
    ]

    coverage = _timeline_coverage_stats(points, lookup, 4)

    assert coverage["covered_buckets"] == [1, 2, 3]
    assert coverage["missing_buckets"] == [4]
    assert coverage["coverage_fraction"] == 0.75


def test_overview_continuation_does_not_rebalance_already_returned_points():
    lookup = {
        "P00001": _document_for_minute(1, 1),
        "P00002": _document_for_minute(2, 15),
        "P00003": _document_for_minute(3, 30),
        "P00004": _document_for_minute(4, 42),
    }
    reducer_points = [_point("A new continuation point.", "P00003")]
    map_findings = [_point("Another continuation point.", "P00004")]

    selected, balance = _balance_points_over_timeline(
        reducer_points=reducer_points,
        map_findings=map_findings,
        lookup=lookup,
        previous_points=[{"text": "1. Opening point", "evidence_ids": []}],
        requested=2,
    )

    assert selected == reducer_points
    assert balance["balancing_applied"] is False
    assert balance["map_points_selected"] == 0


def test_verified_timeline_coverage_uses_only_supported_claim_sources():
    documents = [
        _document_for_minute(1, 1),
        _document_for_minute(2, 15),
        _document_for_minute(3, 30),
        _document_for_minute(4, 42),
    ]
    result = overview_summarization.timeline_coverage_from_claim_evidence(
        [
            {"claim_index": 1, "status": "supported", "supporting_chunk_ids": ["1"]},
            {"claim_index": 2, "status": "missing", "supporting_chunk_ids": []},
            {"claim_index": 3, "status": "partial", "supporting_chunk_ids": ["4"]},
        ],
        documents,
        timeline_start=60.0,
        timeline_end=2565.0,
        bucket_count=4,
    )

    assert result["verified_point_count"] == 1
    assert result["covered_buckets"] == [1]
    assert result["missing_buckets"] == [2, 3, 4]


def test_verified_overview_point_selector_replaces_missing_primary_from_timeline_backup():
    documents = [
        _document_for_minute(1, 0),
        _document_for_minute(2, 10),
        _document_for_minute(3, 20),
        _document_for_minute(4, 30),
    ]
    verification_claims = [
        {"text": "Unsupported opening point.", "evidence_chunk_ids": ["1"], "is_backup": False},
        {"text": "Supported closing point.", "evidence_chunk_ids": ["4"], "is_backup": False},
        {"text": "Verified opening alternative.", "evidence_chunk_ids": ["1"], "is_backup": True},
        {"text": "Verified middle alternative.", "evidence_chunk_ids": ["3"], "is_backup": True},
    ]
    claim_evidence = [
        {"claim_index": 1, "status": "missing", "supporting_chunk_ids": []},
        {"claim_index": 2, "status": "supported", "supporting_chunk_ids": ["4"]},
        {"claim_index": 3, "status": "supported", "supporting_chunk_ids": ["1"]},
        {"claim_index": 4, "status": "supported", "supporting_chunk_ids": ["3"]},
    ]

    selected = overview_summarization.select_verified_overview_points(
        verification_claims,
        claim_evidence,
        documents,
        requested_count=2,
        timeline_start=0.0,
        timeline_end=2075.0,
        bucket_count=4,
    )

    assert len(selected) == 2
    assert {item["text"] for item in selected} == {
        "Verified opening alternative.",
        "Supported closing point.",
    }
    assert sum(item["is_backup"] for item in selected) == 1


def test_verified_overview_selector_replaces_partial_primary_points_with_supported_backups():
    documents = [
        _document_for_minute(index, (index - 1) * 10)
        for index in range(1, 8)
    ]
    verification_claims = [
        {
            "text": f"Primary point {index}.",
            "evidence_chunk_ids": [str(index)],
            "is_backup": False,
        }
        for index in range(1, 8)
    ] + [
        {
            "text": f"Fully supported map alternative {index}.",
            "evidence_chunk_ids": [str(index)],
            "is_backup": True,
        }
        for index in range(1, 8)
    ]
    claim_evidence = [
        {
            "claim_index": 1,
            "status": "supported",
            "supporting_chunk_ids": ["1"],
        }
    ] + [
        {
            "claim_index": index,
            "status": "partial",
            "supporting_chunk_ids": [str(index)],
        }
        for index in range(2, 8)
    ] + [
        {
            "claim_index": index + 7,
            "status": "supported",
            "supporting_chunk_ids": [str(index)],
        }
        for index in range(1, 8)
    ]

    selected = overview_summarization.select_verified_overview_points(
        verification_claims,
        claim_evidence,
        documents,
        requested_count=7,
        timeline_start=0.0,
        timeline_end=3645.0,
        bucket_count=7,
    )

    assert len(selected) == 7
    assert all(item["status"] == "supported" for item in selected)
    assert sum(item["is_backup"] for item in selected) == 6
    assert [item["text"] for item in selected] == [
        "Primary point 1.",
        "Fully supported map alternative 2.",
        "Fully supported map alternative 3.",
        "Fully supported map alternative 4.",
        "Fully supported map alternative 5.",
        "Fully supported map alternative 6.",
        "Fully supported map alternative 7.",
    ]


def test_verified_overview_selector_can_replace_all_seven_partial_primary_points():
    documents = [
        _document_for_minute(index, (index - 1) * 10)
        for index in range(1, 8)
    ]
    verification_claims = [
        {
            "text": f"Partial primary point {index}.",
            "evidence_chunk_ids": [str(index)],
            "is_backup": False,
        }
        for index in range(1, 8)
    ] + [
        {
            "text": f"Verified reserve point {index}.",
            "evidence_chunk_ids": [str(index)],
            "is_backup": True,
        }
        for index in range(1, 8)
    ]
    claim_evidence = [
        {
            "claim_index": index,
            "status": "partial",
            "supporting_chunk_ids": [str(index)],
        }
        for index in range(1, 8)
    ] + [
        {
            "claim_index": index + 7,
            "status": "supported",
            "supporting_chunk_ids": [str(index)],
        }
        for index in range(1, 8)
    ]

    selected = overview_summarization.select_verified_overview_points(
        verification_claims,
        claim_evidence,
        documents,
        requested_count=7,
        timeline_start=0.0,
        timeline_end=3645.0,
        bucket_count=7,
    )

    assert len(selected) == 7
    assert all(item["status"] == "supported" for item in selected)
    assert all(item["is_backup"] for item in selected)
    assert [item["text"] for item in selected] == [
        f"Verified reserve point {index}." for index in range(1, 8)
    ]


def test_verified_overview_selector_deduplicates_same_source_paraphrases_and_covers_all_buckets():
    documents = [
        _document_for_minute(index, 1 + (index - 1) * 10)
        for index in range(1, 8)
    ]
    claims = [
        {
            "text": (
                "She recalls a beautiful childhood that changed abruptly when "
                "the Gulf War began in 1990, marking a turning point in her life."
            ),
            "evidence_chunk_ids": ["1"],
            "is_backup": False,
        },
        {
            "text": (
                "She reflects on her childhood, describing it as beautiful and "
                "happy until the Gulf War in 1990, which abruptly changed her life."
            ),
            "evidence_chunk_ids": ["1"],
            "is_backup": False,
        },
    ] + [
        {
            "text": f"Distinct development in timeline bucket {index}.",
            "evidence_chunk_ids": [str(index + 1)],
            "is_backup": False,
        }
        for index in range(1, 7)
    ]
    evidence = [
        {
            "claim_index": index,
            "status": "supported",
            "supporting_chunk_ids": ["1"] if index in {1, 2} else [str(index - 1)],
        }
        for index in range(1, 9)
    ]

    selected = overview_summarization.select_verified_overview_points(
        claims,
        evidence,
        documents,
        requested_count=7,
        timeline_start=60.0,
        timeline_end=3705.0,
        bucket_count=7,
    )

    assert len(selected) == 7
    assert all(item["status"] == "supported" for item in selected)
    assert [item["claim_index"] for item in selected] == [1, 3, 4, 5, 6, 7, 8]
    assert len({item["bucket"] for item in selected}) == 7


def test_verified_overview_selector_does_not_pad_missing_bucket_with_second_point():
    documents = [
        _document_for_minute(index, minute)
        for index, minute in enumerate([1, 11, 31, 41, 51, 61], start=1)
    ]
    claims = [
        {
            "text": f"Distinct supported event {index}.",
            "evidence_chunk_ids": [str(index)],
            "is_backup": index == 6,
        }
        for index in range(1, 7)
    ]
    evidence = [
        {
            "claim_index": index,
            "status": "supported",
            "supporting_chunk_ids": [str(index)],
        }
        for index in range(1, 7)
    ]

    selected = overview_summarization.select_verified_overview_points(
        claims,
        evidence,
        documents,
        requested_count=7,
        timeline_start=60.0,
        timeline_end=3705.0,
        bucket_count=7,
    )

    assert len(selected) == 6
    assert len({item["bucket"] for item in selected}) == 6


def test_overview_backup_reserve_selects_two_distinct_candidates_per_timeline_bucket():
    minutes = [
        bucket * 10 + offset
        for bucket in range(7)
        for offset in (1, 3, 5)
    ]
    documents = [
        _document_for_minute(index, minute)
        for index, minute in enumerate(minutes, start=1)
    ]
    lookup = {
        f"P{index:05d}": document
        for index, document in enumerate(documents, start=1)
    }
    map_findings = [
        _point(f"Independent reserve development {index}.", f"P{index:05d}")
        for index in range(1, len(documents) + 1)
    ]

    selected = overview_summarization._select_backup_points(
        primary_points=[],
        map_findings=map_findings,
        lookup=lookup,
        requested=7,
        maximum_backups=14,
    )

    selected_buckets = [
        overview_summarization._point_timeline_buckets(
            point,
            lookup,
            min(
                overview_summarization.sort_key(document)[0]
                for document in lookup.values()
            ),
            max(
                overview_summarization.sort_key(document)[1]
                for document in lookup.values()
            ),
            7,
        )[0].pop()
        for point in selected
    ]
    assert len(selected) == 14
    assert sorted(selected_buckets) == [
        bucket for bucket in range(7) for _ in range(2)
    ]


def test_verified_overview_selector_omits_partial_points_without_supported_replacements():
    documents = [_document_for_minute(1, 0)]
    selected = overview_summarization.select_verified_overview_points(
        [
            {
                "text": "Partially supported compound point.",
                "evidence_chunk_ids": ["1"],
                "is_backup": False,
            }
        ],
        [
            {
                "claim_index": 1,
                "status": "partial",
                "supporting_chunk_ids": ["1"],
            }
        ],
        documents,
        requested_count=1,
    )

    assert selected == []


def test_verified_overview_selector_caps_output_when_timestamps_are_degenerate():
    documents = [
        _document_for_minute(1, 0),
        _document_for_minute(2, 0),
        _document_for_minute(3, 0),
    ]
    for document in documents:
        document.metadata["start"] = 0.0
        document.metadata["end"] = 0.0
    claims = [
        {"text": "Primary opening.", "evidence_chunk_ids": ["1"], "is_backup": False},
        {"text": "Unsupported primary middle.", "evidence_chunk_ids": ["2"], "is_backup": False},
        {"text": "Verified reserve middle.", "evidence_chunk_ids": ["3"], "is_backup": True},
    ]
    evidence = [
        {"claim_index": 1, "status": "supported", "supporting_chunk_ids": ["1"]},
        {"claim_index": 2, "status": "missing", "supporting_chunk_ids": []},
        {"claim_index": 3, "status": "supported", "supporting_chunk_ids": ["3"]},
    ]

    selected = overview_summarization.select_verified_overview_points(
        claims,
        evidence,
        documents,
        requested_count=2,
    )

    assert len(selected) == 2
    assert {item["text"] for item in selected} == {
        "Primary opening.",
        "Verified reserve middle.",
    }


def test_overview_backup_candidates_remain_available_without_usable_timestamps():
    lookup = {
        f"P{index:05d}": _document_for_minute(index, 0)
        for index in range(1, 5)
    }
    for document in lookup.values():
        document.metadata["start"] = 0.0
        document.metadata["end"] = 0.0
    primary = [_point("Primary point.", "P00001")]
    alternatives = [
        _point(f"Alternative {index}.", f"P{index:05d}")
        for index in range(2, 5)
    ]

    selected = overview_summarization._select_backup_points(
        primary_points=primary,
        map_findings=alternatives,
        lookup=lookup,
        requested=1,
        maximum_backups=3,
    )

    assert len(selected) == 3
    assert [item["text"] for item in selected] == [
        "Alternative 2.",
        "Alternative 3.",
        "Alternative 4.",
    ]


def test_overview_backup_reserve_spans_timeline_buckets_even_when_primary_points_cover_them():
    minutes = [0, 10, 20, 30, 40, 50, 60]
    documents = [
        _document_for_minute(index, minute)
        for index, minute in enumerate(minutes, start=1)
    ]
    lookup = {
        f"P{index:05d}": document
        for index, document in enumerate(documents, start=1)
    }
    primary_points = [
        _point(f"Primary point {index}.", f"P{index:05d}")
        for index in range(1, 8)
    ]
    map_findings = [
        _point(f"Distinct alternative development {index}.", f"P{index:05d}")
        for index in range(1, 8)
    ]

    selected = overview_summarization._select_backup_points(
        primary_points=primary_points,
        map_findings=map_findings,
        lookup=lookup,
        requested=7,
        maximum_backups=7,
    )

    assert len(selected) == 7
    selected_buckets = [
        overview_summarization._point_timeline_buckets(
            point,
            lookup,
            0.0,
            3645.0,
            7,
        )[0].pop()
        for point in selected
    ]
    assert selected_buckets == list(range(7))


def test_verified_overview_selector_prefers_latest_terminal_tail_point_in_final_bucket():
    minutes = [1, 11, 21, 31, 41, 51, 61, 65, 66]
    documents = [
        _document_for_minute(index, minute)
        for index, minute in enumerate(minutes, start=1)
    ]
    claims = [
        {
            "text": f"Distinct development in timeline interval {index}.",
            "evidence_chunk_ids": [str(index)],
            "is_backup": False,
        }
        for index in range(1, 7)
    ]
    claims.extend([
        {
            "text": "Earlier point from the broad final interval.",
            "evidence_chunk_ids": ["7"],
            "is_backup": False,
        },
        {
            "text": "A closing reflection from the earlier part of the terminal window.",
            "evidence_chunk_ids": ["8"],
            "is_backup": False,
        },
        {
            "text": "The final exchange explains how the host keeps interviews spontaneous.",
            "evidence_chunk_ids": ["9"],
            "is_backup": True,
        },
    ])
    evidence = [
        {
            "claim_index": index,
            "status": "supported",
            "supporting_chunk_ids": [str(index)],
        }
        for index in range(1, 10)
    ]

    selected = overview_summarization.select_verified_overview_points(
        claims,
        evidence,
        documents,
        requested_count=7,
        timeline_start=60.0,
        timeline_end=66 * 60.0 + 45.0,
        bucket_count=7,
    )

    assert len(selected) == 7
    assert selected[-1]["text"] == (
        "The final exchange explains how the host keeps interviews spontaneous."
    )
    assert selected[-1]["supporting_chunk_ids"] == ["9"]
    # A later supported terminal finding outranks an earlier primary point
    # from the same broad final bucket.
    assert selected[-1]["is_backup"] is True

def test_verified_terminal_tail_coverage_is_independent_of_broad_last_bucket():
    early_final_bucket = Document(
        page_content="A development from the earlier part of the final broad bucket.",
        metadata={
            "chunk_id": "70",
            "video_id": "VKrtiTgBeng",
            "start": 4841.0,
            "end": 4930.0,
        },
    )
    actual_tail = Document(
        page_content="A substantive conclusion from the final minutes.",
        metadata={
            "chunk_id": "75",
            "video_id": "VKrtiTgBeng",
            "start": 5174.0,
            "end": 5260.0,
        },
    )
    actual_closing_exchange = Document(
        page_content="A meaningful exchange from the closing-window.",
        metadata={
            "chunk_id": "72",
            "video_id": "VKrtiTgBeng",
            "start": 4965.0,
            "end": 5080.0,
        },
    )
    partial_boundary_chunk = Document(
        page_content="A chunk that crosses into the terminal window.",
        metadata={
            "chunk_id": "71",
            "video_id": "VKrtiTgBeng",
            "start": 4850.0,
            "end": 4950.0,
        },
    )
    documents = [
        early_final_bucket,
        partial_boundary_chunk,
        actual_closing_exchange,
        actual_tail,
    ]

    # The old evidence belongs to broad bucket seven, but it is outside the
    # separate seven-percent terminal window of this 88-minute transcript.
    broad_bucket = overview_summarization._timeline_bucket_index(
        (4841.0 + 4930.0) / 2.0,
        4.0,
        5287.0,
        7,
    )
    assert broad_bucket == 6

    early_coverage = overview_summarization.terminal_tail_coverage_from_claim_evidence(
        [
            {
                "claim_index": 1,
                "status": "supported",
                "supporting_chunk_ids": ["70"],
            }
        ],
        documents,
        timeline_start=4.0,
        timeline_end=5287.0,
    )
    assert early_coverage["eligible"] is True
    assert early_coverage["covered"] is False
    assert early_coverage["supporting_chunk_ids"] == []

    tail_coverage = overview_summarization.terminal_tail_coverage_from_claim_evidence(
        [
            {
                "claim_index": 1,
                "status": "supported",
                "supporting_chunk_ids": ["75"],
            }
        ],
        documents,
        timeline_start=4.0,
        timeline_end=5287.0,
    )
    assert tail_coverage["eligible"] is True
    assert tail_coverage["covered"] is True
    assert tail_coverage["supporting_chunk_ids"] == ["75"]

    # Chunk 72 starts just before the five-percent tail boundary but overlaps
    # it for more than a minute. Midpoint-only checks would incorrectly drop it.
    boundary_coverage = overview_summarization.terminal_tail_coverage_from_claim_evidence(
        [
            {
                "claim_index": 1,
                "status": "supported",
                "supporting_chunk_ids": ["72"],
            }
        ],
        documents,
        timeline_start=4.0,
        timeline_end=5287.0,
    )
    assert boundary_coverage["covered"] is True
    assert boundary_coverage["supporting_chunk_ids"] == ["72"]

    # A passage that merely straddles the beginning of the tail cannot be
    # treated as terminal evidence: its claim could come from pre-tail text.
    partial_boundary_coverage = overview_summarization.terminal_tail_coverage_from_claim_evidence(
        [
            {
                "claim_index": 1,
                "status": "supported",
                "supporting_chunk_ids": ["71"],
            }
        ],
        documents,
        timeline_start=4.0,
        timeline_end=5287.0,
    )
    assert partial_boundary_coverage["covered"] is False
    assert partial_boundary_coverage["supporting_chunk_ids"] == []


def test_backup_reserve_prioritizes_terminal_tail_alternative_in_final_bucket():
    minutes = [1, 11, 21, 31, 41, 51, 61, 66]
    documents = [
        _document_for_minute(index, minute)
        for index, minute in enumerate(minutes, start=1)
    ]
    lookup = {
        f"P{index:05d}": document
        for index, document in enumerate(documents, start=1)
    }
    findings = [
        _point(f"Distinct reserve event {index}.", f"P{index:05d}")
        for index in range(1, 9)
    ]

    selected = overview_summarization._select_backup_points(
        primary_points=[],
        map_findings=findings,
        lookup=lookup,
        requested=7,
        maximum_backups=7,
    )

    selected_last_bucket = [
        point
        for point in selected
        if overview_summarization._point_timeline_buckets(
            point,
            lookup,
            60.0,
            66 * 60.0 + 45.0,
            7,
        )[0] == {6}
    ]
    assert len(selected_last_bucket) == 1
    assert selected_last_bucket[0]["evidence_ids"] == ["P00008"]


def test_overview_runs_dedicated_terminal_scan_and_adds_tail_findings_to_audit():
    starts = [
        0, 600, 1200, 1800, 2400, 3000,
        3600, 4200, 4800, 5100, 5160, 5220,
    ]
    documents = [
        Document(
            page_content=f"Transcript development for segment {index}.",
            metadata={
                "chunk_id": str(index),
                "video_id": "generic-video",
                "start": float(start),
                "end": float(start + 45),
            },
        )
        for index, start in enumerate(starts, start=1)
    ]
    terminal_ids = []
    calls = {"map": 0, "terminal": 0, "reduce": 0}

    def generate(prompt):
        prompt_text = "\n".join(
            message.content for message in prompt.to_messages()
        )
        passage_ids = re.findall(r"\[(P\d{5})\]", prompt_text)

        if "Inspect only the closing portion" in prompt_text:
            calls["terminal"] += 1
            terminal_ids.extend(passage_ids)
            return json.dumps({
                "points": [{
                    "text": (
                        "The closing exchange introduces a distinct final "
                        "reflection from the guest."
                    ),
                    "evidence_ids": [passage_ids[-1]],
                }]
            })

        if "Synthesize the chronological section findings" in prompt_text:
            calls["reduce"] += 1
            note_ids = re.findall(
                r"\[evidence_ids: (P\d{5})\]",
                prompt_text,
            )
            assert len(note_ids) >= 3
            return json.dumps({
                "points": [
                    {
                        "text": "An opening development is discussed.",
                        "evidence_ids": [note_ids[0]],
                    },
                    {
                        "text": "A middle development is explained.",
                        "evidence_ids": [note_ids[len(note_ids) // 2]],
                    },
                    {
                        "text": (
                            "The closing exchange introduces a distinct final "
                            "reflection from the guest."
                        ),
                        "evidence_ids": [note_ids[-1]],
                    },
                ]
            })

        calls["map"] += 1
        assert passage_ids
        return json.dumps({
            "points": [{
                "text": f"Distinct development in map section {calls['map']}.",
                "evidence_ids": [passage_ids[0]],
            }]
        })

    result = overview_summarization.summarize_video_overview(
        documents=documents,
        question="Summarize the entire video chronologically in 3 points.",
        requested_points=3,
        generate_text=generate,
    )

    scan = result["retrieval_trace"]["overview"]["terminal_tail_scan"]
    assert calls["terminal"] == 1
    assert calls["reduce"] == 1
    assert scan["status"] == "completed"
    assert scan["finding_count"] == 1
    assert scan["source_chunk_ids"][-3:] == ["10", "11", "12"]
    assert set(scan["passage_ids"]).issubset(set(terminal_ids))
    assert any(
        "The closing exchange introduces a distinct final reflection"
        in claim["text"]
        and "12" in claim["evidence_chunk_ids"]
        for claim in result["verification_claims"]
    )

def test_overview_survives_unavailable_supplementary_terminal_scan():
    starts = [0, 600, 1200, 1800, 2400, 3000, 3600, 4200]
    documents = [
        Document(
            page_content=f"Supported transcript section {index}.",
            metadata={
                "chunk_id": str(index),
                "video_id": "video-terminal-fallback",
                "start": float(start),
                "end": float(start + 45),
            },
        )
        for index, start in enumerate(starts, start=1)
    ]

    def generate(prompt):
        prompt_text = "\n".join(
            message.content for message in prompt.to_messages()
        )
        if "Inspect only the closing portion" in prompt_text:
            raise RuntimeError("simulated provider throttle")
        if "Synthesize the chronological section findings" in prompt_text:
            note_ids = re.findall(
                r"\[evidence_ids: (P\d{5})\]",
                prompt_text,
            )
            return json.dumps({
                "points": [{
                    "text": "The overview retains a supported chronological development.",
                    "evidence_ids": [note_ids[0]],
                }]
            })
        ids = re.findall(r"\[(P\d{5})\]", prompt_text)
        return json.dumps({
            "points": [{
                "text": "A supported development is discussed in this section.",
                "evidence_ids": [ids[0]],
            }]
        })

    result = overview_summarization.summarize_video_overview(
        documents=documents,
        question="Summarize the entire video in 3 points.",
        requested_points=3,
        generate_text=generate,
    )

    scan = result["retrieval_trace"]["overview"]["terminal_tail_scan"]
    assert scan["status"] == "unavailable"
    assert scan["failure_type"] == "RuntimeError"
    assert result["retrieval_trace"]["route"] == "video_wide_overview"
    assert result["answer"]



def test_terminal_end_cap_scan_recovers_final_passages_when_broad_scan_stops_early():
    starts = [
        0, 600, 1200, 1800, 2400, 3000,
        3600, 4200, 4800, 5100, 5160, 5220,
    ]
    documents = [
        Document(
            page_content=f"Transcript development for segment {index}.",
            metadata={
                "chunk_id": str(index),
                "video_id": "generic-video-end-cap",
                "start": float(start),
                "end": float(start + 45),
            },
        )
        for index, start in enumerate(starts, start=1)
    ]
    calls = {"broad_tail": 0, "end_cap": 0, "reduce": 0}

    def generate(prompt):
        prompt_text = "\n".join(
            message.content for message in prompt.to_messages()
        )
        passage_ids = re.findall(r"\[(P\d{5})\]", prompt_text)

        if "Inspect only the closing portion" in prompt_text:
            calls["broad_tail"] += 1
            # Reproduce the production miss: the wide scan returns an early
            # terminal finding despite later substantive passages.
            return json.dumps({
                "points": [{
                    "text": "An earlier closing reflection is discussed.",
                    "evidence_ids": [passage_ids[0]],
                }]
            })

        if "Inspect only the final two transcript passages" in prompt_text:
            calls["end_cap"] += 1
            return json.dumps({
                "points": [{
                    "text": "The host explains that he avoids researching guests deeply so the interview remains fresh and spontaneous.",
                    "evidence_ids": [passage_ids[-1]],
                }]
            })

        if "Synthesize the chronological section findings" in prompt_text:
            calls["reduce"] += 1
            note_ids = re.findall(r"\[evidence_ids: (P\d{5})\]", prompt_text)
            return json.dumps({
                "points": [
                    {"text": "An opening development is discussed.", "evidence_ids": [note_ids[0]]},
                    {"text": "A middle development is explained.", "evidence_ids": [note_ids[len(note_ids) // 2]]},
                    {"text": "A later development is discussed.", "evidence_ids": [note_ids[-1]]},
                ]
            })

        return json.dumps({
            "points": [{
                "text": "A distinct development in this section is discussed.",
                "evidence_ids": [passage_ids[0]],
            }]
        })

    result = overview_summarization.summarize_video_overview(
        documents=documents,
        question="Summarize the entire video chronologically in 3 points.",
        requested_points=3,
        generate_text=generate,
    )

    scan = result["retrieval_trace"]["overview"]["terminal_tail_scan"]
    assert calls["broad_tail"] == 1
    assert calls["end_cap"] == 1
    assert calls["reduce"] == 1
    assert scan["status"] == "completed"
    assert scan["end_cap_scan"]["status"] == "completed"
    assert scan["end_cap_scan"]["source"] == "dedicated_end_cap_scan"
    assert scan["end_cap_scan"]["finding_count"] == 1
    assert scan["end_cap_scan"]["source_chunk_ids"] == ["11", "12"]
    assert any(
        claim["text"].startswith("The host explains that he avoids researching guests")
        and "12" in claim["evidence_chunk_ids"]
        for claim in result["verification_claims"]
    )


def test_terminal_end_cap_coverage_is_stricter_than_terminal_tail_coverage():
    documents = [
        Document(
            page_content="Earlier tail passage.",
            metadata={
                "chunk_id": "74",
                "video_id": "generic-video-end-cap",
                "start": 5089.0,
                "end": 5182.0,
            },
        ),
        Document(
            page_content="Final substantive passage.",
            metadata={
                "chunk_id": "76",
                "video_id": "generic-video-end-cap",
                "start": 5238.0,
                "end": 5287.0,
            },
        ),
    ]
    early = overview_summarization.terminal_tail_coverage_from_claim_evidence(
        [{"claim_index": 1, "status": "supported", "supporting_chunk_ids": ["74"]}],
        documents,
        timeline_start=4.0,
        timeline_end=5287.0,
        end_cap_start=5174.0,
    )
    assert early["covered"] is True
    assert early["end_cap_eligible"] is True
    assert early["end_cap_covered"] is False
    assert early["end_cap_supporting_chunk_ids"] == []

    actual_ending = overview_summarization.terminal_tail_coverage_from_claim_evidence(
        [{"claim_index": 1, "status": "supported", "supporting_chunk_ids": ["76"]}],
        documents,
        timeline_start=4.0,
        timeline_end=5287.0,
        end_cap_start=5174.0,
    )
    assert actual_ending["covered"] is True
    assert actual_ending["end_cap_covered"] is True
    assert actual_ending["end_cap_supporting_chunk_ids"] == ["76"]