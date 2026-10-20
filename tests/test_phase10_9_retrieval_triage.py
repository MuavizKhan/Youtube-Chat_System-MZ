import hashlib
import json

import pytest

import scripts.phase10_9_retrieval_triage as triage


def _snapshot(video_id, language=None, state="valid"):
    benchmark = triage.VIDEO_BENCHMARKS[video_id]
    actual_language = language or benchmark.transcript_language
    return {
        "state": state,
        "directory_label": f"Backend/rag_system/vector_stores/{video_id}",
        "files": {},
        "metadata_sha256": "example-hash",
        "metadata": {
            "video_id": video_id,
            "transcript_language_code": actual_language,
            "embedding_model": "test-embedding",
            "chunk_count": 12,
        },
        "metadata_error": None,
    }


def _trace(_video_url, question_ids, *, display):
    assert display is False
    diagnostics = []
    for question_id in question_ids:
        diagnostics.append(
            {
                "id": question_id,
                "question": f"Question for {question_id}",
                "groups": [
                    {
                        "group": 0,
                        "matched_index_phrase": "curated evidence",
                        "candidate_count": 1,
                        "candidate_chunk_ids": ["chunk-4"],
                        "raw_anchor_top_k_status": "evidence_in_top_k",
                        "diagnosis": "evidence_in_top_k",
                        "candidates": [
                            {
                                "chunk_id": "chunk-4",
                                "raw_faiss_rank": 1,
                                "semantic_mmr_rank": 1,
                                "semantic_fused_rank": 1,
                                "lexical_rank": None,
                                "hybrid_fused_rank": 1,
                                "facet_soft_reranked_rank": 1,
                                "anchor_rank": 1,
                                "final_context_rank": 1,
                            }
                        ],
                    }
                ],
            }
        )
    return {
        "video_id": _video_url.rsplit("v=", 1)[-1],
        "questions": list(question_ids),
        "diagnostics": diagnostics,
    }


def test_default_targets_cover_only_the_three_known_failure_videos():
    assert set(triage.DEFAULT_DIAGNOSTIC_TARGETS) == {
        "MdeQMVBuGgY",
        "A-JBaNvv3Tk",
        "fAd2nKfrDB8",
    }
    assert triage.DEFAULT_DIAGNOSTIC_TARGETS["MdeQMVBuGgY"] == ("Q03",)
    assert triage.DEFAULT_DIAGNOSTIC_TARGETS["A-JBaNvv3Tk"] == ("AI01",)
    assert triage.DEFAULT_DIAGNOSTIC_TARGETS["fAd2nKfrDB8"] == (
        "AR01", "AR02", "AR03", "AR04"
    )


def test_run_triage_records_provenance_index_and_per_stage_group_summary():
    report = triage.run_phase10_9_triage(
        diagnostic_runner=_trace,
        provenance_builder=lambda: {"git_commit": "test-commit"},
        index_snapshot_builder=_snapshot,
    )

    assert report["report_id"] == triage.REPORT_ID
    assert report["report_schema_version"] == triage.REPORT_SCHEMA_VERSION
    assert report["experiment_metadata"] == {"git_commit": "test-commit"}
    assert report["summary"] == {
        "selected_video_count": 3,
        "completed_video_count": 3,
        "blocked_video_count": 0,
        "error_video_count": 0,
        "diagnosed_question_count": 6,
        "diagnosed_group_count": 8,
        "diagnostics_completed": True,
    }

    arabic = next(item for item in report["videos"] if item["video_id"] == "fAd2nKfrDB8")
    assert arabic["status"] == "completed"
    assert arabic["index_snapshot"]["metadata"]["transcript_language_code"] == "ar"
    assert len(arabic["diagnostics"]) == 4
    group = arabic["group_summaries"][0]
    assert group["question_id"] == "AR01"
    assert group["existing_group_diagnosis"] == "evidence_in_top_k"
    assert group["returned_gold_candidates_by_stage"] == {
        "raw_faiss": 1,
        "semantic_mmr": 1,
        "semantic_fusion": 1,
        "lexical": 0,
        "hybrid_fusion": 1,
        "facet_rerank": 1,
        "anchor_selection": 1,
        "production_context": 1,
    }
    assert "answer_quality_passed" not in report["summary"]


def test_triage_blocks_index_with_wrong_transcript_language_before_tracing():
    calls = []

    def runner(*args, **kwargs):
        calls.append((args, kwargs))
        return _trace(*args, **kwargs)

    report = triage.run_phase10_9_triage(
        video_ids=["fAd2nKfrDB8"],
        diagnostic_runner=runner,
        provenance_builder=dict,
        index_snapshot_builder=lambda video_id: _snapshot(video_id, language="en"),
    )

    assert calls == []
    assert report["summary"]["blocked_video_count"] == 1
    assert report["summary"]["diagnostics_completed"] is False
    assert report["videos"][0]["failure_reasons"] == [
        "index_transcript_language_mismatch"
    ]


def test_triage_blocks_missing_index_and_keeps_the_reason_explicit():
    report = triage.run_phase10_9_triage(
        video_ids=["MdeQMVBuGgY"],
        diagnostic_runner=_trace,
        provenance_builder=dict,
        index_snapshot_builder=lambda video_id: _snapshot(
            video_id, state="missing"
        ),
    )

    assert report["videos"][0]["status"] == "blocked"
    assert report["videos"][0]["failure_reasons"] == ["index_state_not_valid"]
    assert report["videos"][0]["diagnostics"] == []
    assert report["summary"]["blocked_video_count"] == 1


def test_triage_rejects_duplicate_video_ids_before_touching_indexes():
    snapshot_calls = []
    with pytest.raises(ValueError, match="must not contain duplicates"):
        triage.run_phase10_9_triage(
            video_ids=["MdeQMVBuGgY", "MdeQMVBuGgY"],
            index_snapshot_builder=lambda video_id: snapshot_calls.append(video_id),
        )
    assert snapshot_calls == []


def test_triage_rejects_unregistered_targets_before_touching_indexes():
    snapshot_calls = []
    with pytest.raises(ValueError, match="not a configured Phase 10.9 triage target"):
        triage.run_phase10_9_triage(
            video_ids=["not-a-benchmark"],
            index_snapshot_builder=lambda video_id: snapshot_calls.append(video_id),
        )
    assert snapshot_calls == []


def test_index_snapshot_records_metadata_hash_without_returning_transcript_text(
    tmp_path, monkeypatch
):
    video_id = "MdeQMVBuGgY"
    vector_root = tmp_path / "vector_stores"
    index_dir = vector_root / video_id
    index_dir.mkdir(parents=True)
    (index_dir / "index.faiss").write_bytes(b"test-faiss")
    (index_dir / "index.pkl").write_bytes(b"test-docstore")
    metadata = {
        "metadata_version": 1,
        "transcript_segment_schema_version": 1,
        "video_id": video_id,
        "embedding_model": "test-embedding",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "chunk_count": 1,
        "transcript_language": "English",
        "transcript_language_code": "en",
        "transcript_generated": True,
        "requested_languages": ["en", "hi", "ur", "ar"],
        "private_transcript_text": "must never be included in the snapshot",
    }
    raw_metadata = json.dumps(metadata).encode("utf-8")
    (index_dir / "metadata.json").write_bytes(raw_metadata)

    monkeypatch.setattr(triage.rag_config, "VECTOR_STORE_ROOT", vector_root)
    monkeypatch.setattr(triage.rag_config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(triage, "get_index_state", lambda _video_id: "valid")

    snapshot = triage.build_index_snapshot(video_id)

    assert snapshot["state"] == "valid"
    assert snapshot["metadata_sha256"] == hashlib.sha256(raw_metadata).hexdigest()
    assert snapshot["files"]["index.faiss"]["size_bytes"] == len(b"test-faiss")
    assert snapshot["metadata"]["transcript_language_code"] == "en"
    assert "private_transcript_text" not in snapshot["metadata"]
    assert "must never be included" not in json.dumps(snapshot)


def test_cli_saves_report_and_returns_nonzero_for_blocked_targets(tmp_path, monkeypatch):
    output = tmp_path / "nested" / "triage.json"
    report = {
        "report_id": triage.REPORT_ID,
        "report_schema_version": triage.REPORT_SCHEMA_VERSION,
        "generated_at": "2026-10-20T00:00:00Z",
        "experiment_metadata": {},
        "summary": {
            "selected_video_count": 1,
            "completed_video_count": 0,
            "blocked_video_count": 1,
            "error_video_count": 0,
            "diagnosed_question_count": 0,
            "diagnosed_group_count": 0,
            "diagnostics_completed": False,
        },
        "videos": [],
    }
    monkeypatch.setattr(triage, "run_phase10_9_triage", lambda **_kwargs: report)

    exit_code = triage.main(
        ["--videos", "fAd2nKfrDB8", "--output", str(output)]
    )

    assert exit_code == 1
    assert json.loads(output.read_text(encoding="utf-8")) == report

def test_triage_does_not_mark_incomplete_gold_group_trace_as_completed():
    def incomplete_trace(_video_url, question_ids, *, display):
        return {
            "questions": list(question_ids),
            "diagnostics": [
                {"id": question_id, "groups": []}
                for question_id in question_ids
            ],
        }

    report = triage.run_phase10_9_triage(
        video_ids=["MdeQMVBuGgY"],
        diagnostic_runner=incomplete_trace,
        provenance_builder=dict,
        index_snapshot_builder=_snapshot,
    )

    assert report["videos"][0]["status"] == "error"
    assert report["videos"][0]["failure_reasons"] == [
        "diagnostic_execution_failed"
    ]
    assert "did not include all gold groups" in report["videos"][0]["error"]["message"]
    assert report["summary"]["diagnostics_completed"] is False

