"""Capture reproducible /chat answers and compare transcript evidence with citations.

Run from the repository root while the local FastAPI backend is running:
    python -m scripts.chat_answer_source_audit

The response body is decoded from UTF-8 bytes explicitly. Effective retrieval
settings are copied from the running API response, not from this process's
environment (which may differ from the backend process).
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation_reports" / "phase10_8_answer_source_audit.json"

# Chunk IDs refer to the annotated evidence from the Phase 10.7 local audit.
# For Q07, each group includes alternatives: one candidate may be missing from
# retrieval while another supplies equivalent evidence in context or citations.
AUDIT_CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "Q07",
        "video_id": "MdeQMVBuGgY",
        "question": (
            "What does Vijay Mallya say about the amount of debt or money "
            "involved in the Kingfisher situation?"
        ),
        "evidence_groups": [
            {"label": "DRT judgment debt: 4,999 crore", "candidate_chunk_ids": [9, 124]},
            {"label": "principal plus unapplied interest: 6,203 crore", "candidate_chunk_ids": [10, 221]},
            {
                "label": "recovery figure: 14,100 crore",
                "candidate_chunk_ids": [9, 10, 106, 111, 112, 129, 130, 153, 159],
            },
        ],
    },
    {
        "id": "AR03",
        "video_id": "fAd2nKfrDB8",
        "question": "What bodily sounds might people begin to hear in the room?",
        "evidence_groups": [
            {"label": "heartbeat and blood movement phrase", "candidate_chunk_ids": [2]},
        ],
    },
    {
        "id": "AR08",
        "video_id": "fAd2nKfrDB8",
        "question": "ما الأصوات التي قد يبدأ الناس بسماعها من أجسامهم داخل الغرفة؟",
        "evidence_groups": [
            {"label": "heartbeat and blood movement phrase", "candidate_chunk_ids": [2]},
        ],
    },
)

# These marker sequences are strong indicators that non-ASCII answer text has
# already been decoded with the wrong character encoding. They are warnings,
# not an automatic repair, because the client must not rewrite model output.
_MOJIBAKE_MARKERS = ("Ø", "Ù", "â", "Ã", "\ufffd")
_TIMESTAMP_RANGE_PATTERN = re.compile(
    r"(?<!\d)(?:\d{1,2}:)?\d{1,2}:\d{2}\s*[–—-]\s*"
    r"(?:\d{1,2}:)?\d{1,2}:\d{2}(?!\d)"
)


def decode_json_response(body: bytes) -> dict[str, Any]:
    """Decode a JSON HTTP body as UTF-8 and require an object payload."""
    text = body.decode("utf-8", errors="strict")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Expected the API JSON response to be an object.")
    return payload


def _chunk_ids(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return list(dict.fromkeys(str(value) for value in values if value is not None))


def _format_timestamp(seconds: Any) -> str | None:
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return None
    if value < 0:
        return None

    total = int(value)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _normalize_range(value: str) -> tuple[str, str] | None:
    parts = re.split(r"\s*[–—-]\s*", value.strip(), maxsplit=1)
    if len(parts) != 2:
        return None
    return parts[0].strip(), parts[1].strip()


def _answer_timestamp_ranges(answer: str) -> list[str]:
    match = re.search(
        r"Relevant timestamps?:\s*([^\r\n]+)",
        answer,
        flags=re.IGNORECASE,
    )
    if not match:
        return []
    return [item.group(0).strip() for item in _TIMESTAMP_RANGE_PATTERN.finditer(match.group(1))]


def analyze_response(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    """Compare annotated candidate evidence with context, anchors, sources, and timestamp line."""
    observability = response.get("observability")
    if not isinstance(observability, dict):
        observability = {}
    retrieval = observability.get("retrieval")
    if not isinstance(retrieval, dict):
        retrieval = {}

    final_context_ids = _chunk_ids(retrieval.get("final_chunk_ids"))
    anchor_ids = _chunk_ids(retrieval.get("citation_chunk_ids"))
    source_objects = response.get("sources")
    if not isinstance(source_objects, list):
        source_objects = []

    returned_source_ids: list[str] = []
    source_ranges: list[dict[str, Any]] = []
    for source in source_objects:
        if not isinstance(source, dict):
            continue
        ids = _chunk_ids(source.get("chunk_ids"))
        returned_source_ids.extend(ids)
        start = source.get("start")
        end = source.get("end")
        start_label = _format_timestamp(start)
        end_label = _format_timestamp(end)
        source_ranges.append(
            {
                "source_id": source.get("source_id"),
                "chunk_ids": ids,
                "start": start,
                "end": end,
                "display_range": (
                    f"{start_label}–{end_label}"
                    if start_label is not None and end_label is not None
                    else None
                ),
            }
        )

    returned_source_ids = list(dict.fromkeys(returned_source_ids))
    answer = response.get("answer")
    if not isinstance(answer, str):
        answer = ""

    timestamp_ranges = _answer_timestamp_ranges(answer)
    normalized_summary_ranges = {
        normalized
        for item in timestamp_ranges
        if (normalized := _normalize_range(item)) is not None
    }
    summary_source_ids: list[str] = []
    for source in source_ranges:
        normalized = (
            _normalize_range(source["display_range"])
            if isinstance(source.get("display_range"), str)
            else None
        )
        if normalized in normalized_summary_ranges:
            summary_source_ids.extend(source["chunk_ids"])
    summary_source_ids = list(dict.fromkeys(summary_source_ids))

    context_set = set(final_context_ids)
    anchor_set = set(anchor_ids)
    source_set = set(returned_source_ids)
    summary_set = set(summary_source_ids)

    evidence_group_results = []
    for group in case.get("evidence_groups", []):
        candidates = _chunk_ids(group.get("candidate_chunk_ids"))
        candidate_set = set(candidates)
        evidence_group_results.append(
            {
                "label": group.get("label"),
                "candidate_chunk_ids": candidates,
                "present_in_final_context": bool(candidate_set & context_set),
                "present_in_selected_anchors": bool(candidate_set & anchor_set),
                "present_in_returned_sources": bool(candidate_set & source_set),
                "represented_in_plain_text_timestamp_summary": bool(
                    candidate_set & summary_set
                ),
                "context_chunk_ids": [
                    chunk_id for chunk_id in candidates if chunk_id in context_set
                ],
                "anchor_chunk_ids": [
                    chunk_id for chunk_id in candidates if chunk_id in anchor_set
                ],
                "source_chunk_ids": [
                    chunk_id for chunk_id in candidates if chunk_id in source_set
                ],
                "timestamp_summary_chunk_ids": [
                    chunk_id for chunk_id in candidates if chunk_id in summary_set
                ],
            }
        )

    mojibake_markers = [marker for marker in _MOJIBAKE_MARKERS if marker in answer]
    return {
        "effective_retrieval_config": response.get("retrieval_config"),
        "retrieval_method": response.get("retrieval_method"),
        "final_context_chunk_ids": final_context_ids,
        "selected_anchor_chunk_ids": anchor_ids,
        "context_only_chunk_ids": [
            chunk_id for chunk_id in final_context_ids if chunk_id not in anchor_set
        ],
        "returned_source_chunk_ids": returned_source_ids,
        "plain_text_timestamp_ranges": timestamp_ranges,
        "source_ranges": source_ranges,
        "answer_text_encoding_suspect": bool(mojibake_markers),
        "possible_mojibake_markers": mojibake_markers,
        "annotated_evidence_groups": evidence_group_results,
    }


def _read_git_state() -> dict[str, Any]:
    def run_git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        except OSError:
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    status = run_git("status", "--porcelain")
    return {
        "runner_git_commit": run_git("rev-parse", "HEAD"),
        "runner_working_tree_clean": status == "",
    }


def _post_chat(
    base_url: str,
    case: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    request_body = json.dumps(
        {"video_id": case["video_id"], "question": case["question"]},
        ensure_ascii=False,
    ).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}/chat",
        data=request_body,
        headers={"Content-Type": "application/json; charset=utf-8", "Accept": "application/json"},
        method="POST",
    )

    status_code: int | None = None
    content_type: str | None = None
    raw_body = b""
    transport_error = None
    try:
        with urlopen(request, timeout=timeout_seconds) as result:
            status_code = result.status
            content_type = result.headers.get("Content-Type")
            raw_body = result.read()
    except HTTPError as error:
        status_code = error.code
        content_type = error.headers.get("Content-Type") if error.headers else None
        raw_body = error.read()
    except (URLError, TimeoutError, OSError) as error:
        transport_error = str(error)

    payload: dict[str, Any] | None = None
    parse_error = None
    if raw_body:
        try:
            payload = decode_json_response(raw_body)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            parse_error = str(error)

    success = (
        status_code is not None
        and 200 <= status_code < 300
        and isinstance(payload, dict)
        and isinstance(payload.get("answer"), str)
    )

    result: dict[str, Any] = {
        "case_id": case["id"],
        "video_id": case["video_id"],
        "question": case["question"],
        "status": "success" if success else "error",
        "http_status": status_code,
        "content_type": content_type,
        "response_body_sha256": hashlib.sha256(raw_body).hexdigest() if raw_body else None,
        "transport_error": transport_error,
        "response_parse_error": parse_error,
        "response": payload,
    }
    if isinstance(payload, dict) and isinstance(payload.get("answer"), str):
        result["audit"] = analyze_response(case, payload)
    elif raw_body and payload is None:
        result["unparsed_response_body"] = raw_body.decode("utf-8", errors="replace")
    return result


def run_audit(
    *,
    base_url: str,
    output_path: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    results = [
        _post_chat(base_url, case, timeout_seconds)
        for case in AUDIT_CASES
    ]

    successful_audits = [
        item.get("audit")
        for item in results
        if item.get("status") == "success" and isinstance(item.get("audit"), dict)
    ]
    configs = [
        audit.get("effective_retrieval_config")
        for audit in successful_audits
        if isinstance(audit.get("effective_retrieval_config"), dict)
    ]
    effective_config = configs[0] if configs else None
    configs_consistent = bool(configs) and all(config == configs[0] for config in configs)

    report = {
        "report_schema_version": "1.0.0",
        "report_id": "phase10_8_answer_source_audit",
        "generated_at": datetime.now().astimezone().isoformat(),
        "backend_base_url": base_url.rstrip("/"),
        **_read_git_state(),
        "effective_server_configuration": effective_config,
        "effective_server_configuration_consistent_across_cases": configs_consistent,
        "cases_attempted": len(AUDIT_CASES),
        "cases_succeeded": sum(item.get("status") == "success" for item in results),
        "cases_failed": sum(item.get("status") != "success" for item in results),
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture UTF-8 /chat answers and audit evidence-to-source coverage."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL of the running local FastAPI service.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"JSON report path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=90.0,
        help="Per-request timeout in seconds.",
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero.")

    report = run_audit(
        base_url=args.base_url,
        output_path=args.output,
        timeout_seconds=args.timeout,
    )
    print(
        json.dumps(
            {
                "report_path": str(args.output),
                "cases_attempted": report["cases_attempted"],
                "cases_succeeded": report["cases_succeeded"],
                "cases_failed": report["cases_failed"],
                "effective_server_configuration": report["effective_server_configuration"],
                "effective_server_configuration_consistent_across_cases": (
                    report["effective_server_configuration_consistent_across_cases"]
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["cases_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
