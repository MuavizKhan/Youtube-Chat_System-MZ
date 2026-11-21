"""Local summary and private export commands for the Phase 11.0E feedback queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from Backend.rag_system.feedback import FeedbackStore, default_feedback_database_path


def _private_root() -> Path:
    return Path(__file__).resolve().parents[1] / "evaluation_data" / "private"


def build_evidence_snapshot(
    record: dict,
    documents: list,
    *,
    index_error: str | None = None,
) -> dict:
    """Resolve explicitly shared source chunk IDs to exact local transcript text."""
    video_id = record.get("video_id")
    if not record.get("context_shared"):
        return {"status": "not_shared", "video_id": video_id, "passages": []}

    requested_ids = sorted({
        chunk_id
        for source in record.get("sources", [])
        if isinstance(source, dict)
        for chunk_id in source.get("chunk_ids", [])
        if type(chunk_id) is int and chunk_id >= 0
    })
    if not requested_ids:
        return {
            "status": "no_chunk_ids",
            "video_id": video_id,
            "requested_chunk_ids": [],
            "missing_chunk_ids": [],
            "passages": [],
        }
    if index_error:
        return {
            "status": "index_unavailable",
            "video_id": video_id,
            "requested_chunk_ids": requested_ids,
            "missing_chunk_ids": requested_ids,
            "passages": [],
            "diagnostic": index_error,
        }

    documents_by_id = {}
    duplicate_ids = set()
    for document in documents:
        metadata = getattr(document, "metadata", None)
        if not isinstance(metadata, dict) or metadata.get("video_id") != video_id:
            continue
        chunk_id = metadata.get("chunk_id")
        if type(chunk_id) is not int or chunk_id < 0:
            continue
        if chunk_id in documents_by_id:
            duplicate_ids.add(chunk_id)
        documents_by_id[chunk_id] = document

    passages = []
    for chunk_id in requested_ids:
        document = documents_by_id.get(chunk_id)
        if document is None or chunk_id in duplicate_ids:
            continue
        text = getattr(document, "page_content", None)
        metadata = getattr(document, "metadata", {})
        if not isinstance(text, str) or not text.strip():
            continue
        segments = metadata.get("transcript_segments", [])
        if not isinstance(segments, list):
            segments = []
        passages.append({
            "passage_id": f"{video_id}:{chunk_id}",
            "video_id": video_id,
            "chunk_id": chunk_id,
            "start": metadata.get("start"),
            "end": metadata.get("end"),
            "language_code": metadata.get("language_code"),
            "is_generated": metadata.get("is_generated"),
            "text": text,
            "transcript_segments": segments,
        })

    resolved_ids = {passage["chunk_id"] for passage in passages}
    missing_ids = [chunk_id for chunk_id in requested_ids if chunk_id not in resolved_ids]
    return {
        "status": "complete" if not missing_ids else "partial",
        "video_id": video_id,
        "requested_chunk_ids": requested_ids,
        "missing_chunk_ids": missing_ids,
        "passages": passages,
    }


def attach_evidence_snapshots(records: list[dict]) -> list[dict]:
    """Hydrate opted-in feedback records from existing local indexes; no provider calls."""
    video_ids = sorted({
        record.get("video_id")
        for record in records
        if record.get("context_shared")
        and any(
            isinstance(source, dict) and source.get("chunk_ids")
            for source in record.get("sources", [])
        )
        and isinstance(record.get("video_id"), str)
    })
    documents_by_video = {}
    errors_by_video = {}
    if video_ids:
        try:
            from Backend.rag_system.retrieval import get_all_documents, load_vector_store
        except Exception:
            for video_id in video_ids:
                errors_by_video[video_id] = "retrieval_module_unavailable"
        else:
            for video_id in video_ids:
                try:
                    documents_by_video[video_id] = get_all_documents(
                        load_vector_store(video_id)
                    )
                except FileNotFoundError:
                    errors_by_video[video_id] = "video_index_not_found"
                except Exception:
                    errors_by_video[video_id] = "video_index_load_failed"

    hydrated = []
    for record in records:
        item = dict(record)
        video_id = item.get("video_id")
        if item.get("context_shared"):
            item["evidence_snapshot"] = build_evidence_snapshot(
                item,
                documents_by_video.get(video_id, []),
                index_error=errors_by_video.get(video_id),
            )
        else:
            item["evidence_snapshot"] = {
                "status": "not_shared",
                "video_id": video_id,
                "passages": [],
            }
        hydrated.append(item)
    return hydrated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize or privately export user feedback records."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary = subparsers.add_parser(
        "summary",
        help="Print aggregate counts only; no question or answer text is emitted.",
    )
    summary.add_argument("--db", type=Path, default=default_feedback_database_path())

    export = subparsers.add_parser(
        "export",
        help="Export the review queue as JSONL under evaluation_data/private/.",
    )
    export.add_argument("--db", type=Path, default=default_feedback_database_path())
    export.add_argument("--output", type=Path, required=True)
    export.add_argument(
        "--include-evidence",
        action="store_true",
        help="Resolve opted-in cited chunk IDs from existing local indexes into the private export.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = FeedbackStore(args.db)

    if args.command == "summary":
        print(json.dumps(store.summary(), ensure_ascii=False, indent=2))
        return 0

    private_root = _private_root().resolve()
    output = args.output.expanduser().resolve()
    try:
        output.relative_to(private_root)
    except ValueError:
        print(
            "Refusing to export feedback outside evaluation_data/private/.",
            file=sys.stderr,
        )
        return 2

    if output.exists():
        print(f"Refusing to overwrite existing feedback export: {output}", file=sys.stderr)
        return 2

    records = store.list_records()
    if args.include_evidence:
        records = attach_evidence_snapshots(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    print(
        json.dumps(
            {"exported_records": len(records), "output": str(output)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
