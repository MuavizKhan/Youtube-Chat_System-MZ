"""Local summary and private export commands for the Phase 11.0E feedback queue."""

from __future__ import annotations

import argparse
import json
import re
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


def split_answer_claims(answer: str) -> list[str]:
    """Split rendered answer bullets/sentences into bounded assessment aspects."""
    if not isinstance(answer, str):
        return []
    claims = []
    in_code_block = False
    for raw_line in answer.splitlines() or [answer]:
        line = raw_line.strip()
        if line.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block or not line:
            continue
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", line)
        line = re.sub(r"^#{1,6}\s+", "", line)
        parts = re.split(r"(?<=[.!?؟।])\s+|;\s+", line)
        for part in parts:
            cleaned = part.strip().strip("*_ 	")
            if cleaned and cleaned not in claims:
                claims.append(cleaned)
    return claims


def _reference_to_dict(reference) -> dict:
    return {
        "video_id": reference.video_id,
        "chunk_id": reference.chunk_id,
        "start_seconds": reference.start_seconds,
        "end_seconds": reference.end_seconds,
    }


def _serialize_ai_proposal(result) -> dict:
    notes = {note.aspect_id: note.summary for note in result.notes}
    return {
        "status": result.diagnostics.status.value,
        "scope": "cited_answer_claim_support_only",
        "scope_limitations": [
            "Does not determine whether the answer covers every part of the user's question.",
            "Does not determine whether a negative user rating is justified.",
            "Does not replace review of the original video, caption track, or conversational intent.",
        ],
        "ground_truth": False,
        "human_review_required": True,
        "coverage": {
            "status": result.coverage.status.value,
            "coverage_ratio": result.coverage.coverage_ratio,
            "required_aspect_count": result.coverage.required_aspect_count,
            "supported_required_aspect_count": result.coverage.supported_required_aspect_count,
            "missing_aspect_ids": list(result.coverage.missing_aspect_ids),
            "contradicted_aspect_ids": list(result.coverage.contradicted_aspect_ids),
            "unknown_aspect_ids": list(result.coverage.unknown_aspect_ids),
        },
        "assessments": [
            {
                "aspect_id": item.aspect_id,
                "status": item.status.value,
                "support_score": item.support_score,
                "supporting_evidence": [
                    _reference_to_dict(reference)
                    for reference in item.supporting_evidence
                ],
                "contradicting_evidence": [
                    _reference_to_dict(reference)
                    for reference in item.contradicting_evidence
                ],
                "summary": notes.get(item.aspect_id),
            }
            for item in result.assessments
        ],
        "diagnostics": {
            "provider": result.diagnostics.provider,
            "model": result.diagnostics.model,
            "status": result.diagnostics.status.value,
            "request_count": result.diagnostics.request_count,
            "elapsed_ms": result.diagnostics.elapsed_ms,
            "input_chars": result.diagnostics.input_chars,
            "input_limit_chars": result.diagnostics.input_limit_chars,
            "passage_count": result.diagnostics.passage_count,
            "aspect_count": result.diagnostics.aspect_count,
            "error_code": result.diagnostics.error_code,
            "input_tokens": result.diagnostics.input_tokens,
            "output_tokens": result.diagnostics.output_tokens,
        },
    }


def attach_ai_assessments(
    records: list[dict],
    *,
    max_cases: int = 10,
) -> list[dict]:
    """Run Phase 11.0B only on negative reports with explicit AI-provider consent."""
    if isinstance(max_cases, bool) or not isinstance(max_cases, int) or not 1 <= max_cases <= 20:
        raise ValueError("max_cases must be between 1 and 20.")

    eligible = []
    output = []
    for record in records:
        item = dict(record)
        if item.get("helpful"):
            item["ai_proposed_assessment"] = {"status": "not_a_negative_report"}
        elif not item.get("context_shared"):
            item["ai_proposed_assessment"] = {"status": "context_not_shared"}
        elif not item.get("allow_ai_assessment"):
            item["ai_proposed_assessment"] = {"status": "not_consented"}
        elif (item.get("evidence_snapshot") or {}).get("status") != "complete":
            item["ai_proposed_assessment"] = {"status": "evidence_incomplete"}
        else:
            eligible.append(len(output))
        output.append(item)

    if not eligible:
        return output

    selected = set(eligible[:max_cases])
    for index in eligible[max_cases:]:
        output[index]["ai_proposed_assessment"] = {
            "status": "deferred_batch_limit",
            "human_review_required": True,
            "ground_truth": False,
        }

    from Backend.rag_system import config
    from Backend.rag_system.evidence_assessment import (
        EvidencePassage,
        assess_evidence_support,
    )
    from Backend.rag_system.evidence_contracts import (
        EvidenceAspect,
        EvidenceReference,
        QuestionAnalysis,
        QuestionIntent,
    )

    provider = config.LLM_PROVIDER
    model = (
        config.GROQ_MODEL_ID if provider == "groq"
        else config.HF_MODEL_ID
    )
    try:
        from Backend.rag_system.chain import create_llm_client
        client = create_llm_client()
    except Exception:
        client = None

    for index in sorted(selected):
        item = output[index]
        question = item.get("question")
        answer = item.get("answer")
        snapshot = item.get("evidence_snapshot") or {}
        claims = split_answer_claims(answer or "")

        if not isinstance(question, str) or not question.strip() or not claims:
            item["ai_proposed_assessment"] = {
                "status": "insufficient_feedback_context",
                "ground_truth": False,
                "human_review_required": True,
            }
            continue
        if len(claims) > 8:
            item["ai_proposed_assessment"] = {
                "status": "answer_claim_budget_exceeded",
                "claim_count": len(claims),
                "maximum_claims": 8,
                "ground_truth": False,
                "human_review_required": True,
            }
            continue

        passage_records = snapshot.get("passages", [])
        if not isinstance(passage_records, list) or not passage_records or len(passage_records) > 12:
            item["ai_proposed_assessment"] = {
                "status": "passage_budget_exceeded_or_empty",
                "passage_count": len(passage_records) if isinstance(passage_records, list) else None,
                "maximum_passages": 12,
                "ground_truth": False,
                "human_review_required": True,
            }
            continue

        passages = []
        try:
            for passage_record in passage_records:
                video_id = passage_record["video_id"]
                chunk_id = passage_record["chunk_id"]
                start = passage_record.get("start")
                end = passage_record.get("end")
                reference_kwargs = {}
                if (
                    isinstance(start, (int, float))
                    and not isinstance(start, bool)
                    and isinstance(end, (int, float))
                    and not isinstance(end, bool)
                    and float(start) >= 0
                    and float(end) > float(start)
                ):
                    reference_kwargs = {
                        "start_seconds": float(start),
                        "end_seconds": float(end),
                    }
                reference = EvidenceReference(
                    video_id=video_id,
                    chunk_id=chunk_id,
                    **reference_kwargs,
                )
                language = passage_record.get("language_code")
                if isinstance(language, str):
                    language = {
                        "english": "en",
                        "hindi": "hi",
                        "arabic": "ar",
                    }.get(language.strip().casefold(), language.strip().replace("_", "-"))
                else:
                    language = None
                passages.append(
                    EvidencePassage(
                        passage_id=passage_record["passage_id"],
                        text=passage_record["text"],
                        references=(reference,),
                        transcript_language=language,
                    )
                )
            analysis = QuestionAnalysis(
                intent=QuestionIntent.UNKNOWN,
                language=None,
                aspects=tuple(
                    EvidenceAspect(
                        aspect_id=f"answer_claim_{claim_index + 1}",
                        description=claim,
                        required=True,
                    )
                    for claim_index, claim in enumerate(claims)
                ),
                video_id=item["video_id"],
                subject=(
                    "Prior conversation context (if supplied): "
                    + " | ".join(
                        f"{turn.get('role', 'user')}: {turn.get('content', '')}"
                        for turn in item.get("conversation_history", [])
                        if isinstance(turn, dict)
                        and isinstance(turn.get("content"), str)
                    )
                    + f" | Current user question: {question.strip()}"
                ),
                analysis_source="feedback_answer_sentence_splitter_v1",
            )
            result = assess_evidence_support(
                analysis,
                passages,
                client,
                provider=provider,
                model=model,
            )
            item["ai_proposed_assessment"] = _serialize_ai_proposal(result)
        except Exception:
            item["ai_proposed_assessment"] = {
                "status": "assessment_preparation_failed",
                "ground_truth": False,
                "human_review_required": True,
            }

    return output


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
    export.add_argument(
        "--assess",
        action="store_true",
        help="Run Phase 11.0B only for reports with separate AI-provider consent.",
    )
    export.add_argument(
        "--max-ai-cases",
        type=int,
        default=10,
        help="Maximum consented reports to assess in this command (1-20; default 10).",
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
    if args.assess:
        if not 1 <= args.max_ai_cases <= 20:
            print("--max-ai-cases must be between 1 and 20.", file=sys.stderr)
            return 2
        records = attach_evidence_snapshots(records)
        records = attach_ai_assessments(records, max_cases=args.max_ai_cases)
    elif args.include_evidence:
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
