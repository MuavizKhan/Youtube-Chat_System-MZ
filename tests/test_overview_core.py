from langchain_core.documents import Document

from Backend.rag_system.overview_core import build_sections, timeline_sample


def _doc(index, start, text="x" * 120):
    return Document(
        page_content=text,
        metadata={
            "chunk_id": index,
            "video_id": "video-1",
            "start": float(start),
            "end": float(start + 5),
        },
    )


def test_timeline_sample_preserves_late_video_content_when_chunk_density_is_skewed():
    documents = [
        _doc(index, second)
        for index, second in enumerate(
            [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 90, 300, 600, 900, 1200],
            start=1,
        )
    ]

    sampled = timeline_sample(documents, 7)

    assert len(sampled) == 7
    assert sampled[0].metadata["start"] == 0.0
    assert sampled[-1].metadata["start"] == 1200.0
    assert any(item.metadata["start"] >= 600 for item in sampled)
    assert [item.metadata["start"] for item in sampled] == sorted(
        item.metadata["start"] for item in sampled
    )


def test_build_sections_aligns_nonempty_map_sections_to_video_time_buckets():
    starts = [0, 70, 140, 210, 280, 350, 420, 490, 560, 630, 700, 770, 840]
    documents = [_doc(index, start) for index, start in enumerate(starts, start=1)]

    sections, lookup, stats = build_sections(
        documents,
        preferred_sections=7,
    )

    assert len(sections) == 7
    assert len(lookup) == len(documents)
    assert stats["sectioning_strategy"] == "timeline_binned"
    assert stats["preferred_timeline_sections"] == 7

    timeline_start = min(item.metadata["start"] for item in documents)
    timeline_end = max(item.metadata["end"] for item in documents)
    duration = timeline_end - timeline_start
    observed_buckets = []
    for section in sections:
        starts_in_section = [
            item["document"].metadata["start"]
            for item in section
        ]
        buckets = {
            min(6, int(
                (((start + 2.5) - timeline_start) / duration) * 7
            ))
            for start in starts_in_section
        }
        assert len(buckets) == 1
        observed_buckets.append(next(iter(buckets)))

    assert observed_buckets == sorted(observed_buckets)
    assert observed_buckets == list(range(7))


def test_build_sections_retains_character_limit_when_a_time_bucket_is_dense():
    documents = [
        _doc(index, index, text=("Transcript content " * 450))
        for index in range(1, 15)
    ]

    sections, _lookup, stats = build_sections(
        documents,
        preferred_sections=4,
    )

    assert len(sections) >= 4
    assert all(
        sum(len(item["line"]) for item in section) <= 9_000
        for section in sections
    )
    assert stats["sectioning_strategy"] == "timeline_binned"


def test_build_sections_reports_character_packing_for_degenerate_timestamps():
    documents = [
        Document(
            page_content=f"Transcript content {index}.",
            metadata={
                "chunk_id": index,
                "video_id": "video-1",
                "start": 0.0,
                "end": 0.0,
            },
        )
        for index in range(1, 5)
    ]

    sections, _lookup, stats = build_sections(
        documents,
        preferred_sections=4,
    )

    assert sections
    assert stats["sectioning_strategy"] == "character_packed"


def test_build_sections_uses_one_section_when_transcript_is_sparser_than_requested_buckets():
    documents = [
        _doc(1, 0),
        _doc(2, 20),
    ]
    sections, lookup, stats = build_sections(documents, preferred_sections=7)

    assert len(sections) == 1
    assert len(lookup) == 2
    assert stats["sectioning_strategy"] == "character_packed"


def test_build_sections_always_preserves_latest_transcript_chunks_when_sampling():
    documents = [
        _doc(index, index * 60)
        for index in range(1, 151)
    ]

    sections, lookup, stats = build_sections(
        documents,
        preferred_sections=7,
    )

    selected_chunk_ids = {
        str(document.metadata["chunk_id"])
        for document in lookup.values()
    }
    assert stats["coverage_mode"] == "timeline_stratified"
    assert stats["selected_document_count"] == 100
    assert {str(index) for index in range(145, 151)}.issubset(
        selected_chunk_ids
    )
    assert min(
        float(item["document"].metadata["start"])
        for section in sections
        for item in section
    ) == 60.0

