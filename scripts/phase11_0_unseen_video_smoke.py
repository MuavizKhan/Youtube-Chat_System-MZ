"""Live smoke checks for indexing videos outside the five-video benchmark pilot.

Run the backend and this script with the same isolated
RAG_INDEX_NAMESPACE=phase11-0-unseen-video-smoke. The namespace guard prevents
the smoke run from overwriting the frozen Phase 10.10/10.11 indexes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from Backend.rag_system.config import PROJECT_ROOT, RAG_INDEX_NAMESPACE, VECTOR_STORE_ROOT


REQUIRED_NAMESPACE = "phase11-0-unseen-video-smoke"
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "evaluation_data" / "phase11_0_unseen_video_smoke.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "evaluation_reports" / "phase11_0_unseen_video_smoke.json"
)
VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
TERMINAL_STATES = {"ready", "failed"}


def classify_transcript_timeline(
    span_seconds: float,
    *,
    short_max_seconds: float = 600.0,
    long_min_seconds: float = 1800.0,
) -> str:
    """Classify caption timeline span, not claimed full media duration."""
    if not math.isfinite(float(span_seconds)) or span_seconds < 0:
        raise ValueError("Transcript timeline span must be finite and non-negative.")
    if short_max_seconds <= 0 or long_min_seconds <= short_max_seconds:
        raise ValueError("Timeline class thresholds are invalid.")
    if span_seconds <= short_max_seconds:
        return "short"
    if span_seconds >= long_min_seconds:
        return "long"
    return "medium"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_response(request: Request, timeout_seconds: float) -> dict[str, Any]:
    status_code: int | None = None
    raw_body = b""
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            status_code = response.status
            raw_body = response.read()
    except HTTPError as error:
        status_code = error.code
        raw_body = error.read()
    except (URLError, TimeoutError, OSError) as error:
        return {
            "http_status": None,
            "payload": None,
            "transport_error": str(error),
        }

    try:
        payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return {
            "http_status": status_code,
            "payload": None,
            "transport_error": None,
            "parse_error": str(error),
        }

    if not isinstance(payload, dict):
        return {
            "http_status": status_code,
            "payload": None,
            "transport_error": None,
            "parse_error": "Expected an object JSON response.",
        }

    return {
        "http_status": status_code,
        "payload": payload,
        "transport_error": None,
    }


def _api_request(
    base_url: str,
    path: str,
    *,
    method: str,
    payload: dict[str, Any] | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        headers=headers,
        method=method,
    )
    return _read_json_response(request, timeout_seconds)


def _poll_index_status(
    base_url: str,
    video_id: str,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
    max_wait_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + max_wait_seconds
    last_result: dict[str, Any] | None = None

    while time.monotonic() < deadline:
        last_result = _api_request(
            base_url,
            f"/index/{video_id}",
            method="GET",
            payload=None,
            timeout_seconds=timeout_seconds,
        )
        if last_result.get("transport_error") or last_result.get("parse_error"):
            return last_result
        status = last_result.get("payload") or {}
        if status.get("state") in TERMINAL_STATES:
            return last_result
        time.sleep(poll_interval_seconds)

    return {
        "http_status": None if last_result is None else last_result.get("http_status"),
        "payload": None if last_result is None else last_result.get("payload"),
        "transport_error": f"Index job did not reach a terminal state within {max_wait_seconds:.1f}s.",
    }


def _validate_case(case: dict[str, Any]) -> None:
    case_id = case.get("case_id")
    video_id = case.get("video_id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("Every smoke case requires a non-empty case_id.")
    if not isinstance(video_id, str) or not VIDEO_ID_PATTERN.fullmatch(video_id):
        raise ValueError(f"{case_id}: video_id must be an 11-character YouTube ID.")
    languages = case.get("languages")
    if (
        not isinstance(languages, list)
        or not languages
        or any(language not in {"en", "hi", "ur", "ar"} for language in languages)
        or len(languages) != len(set(languages))
    ):
        raise ValueError(f"{case_id}: languages must be unique supported language codes.")
    if case.get("expected_outcome", "ready") not in {"ready", "failed"}:
        raise ValueError(f"{case_id}: expected_outcome must be 'ready' or 'failed'.")
    if case.get("expected_outcome", "ready") == "failed":
        if not isinstance(case.get("expected_error_code"), str):
            raise ValueError(f"{case_id}: failed cases require expected_error_code.")
    if case.get("expected_outcome", "ready") == "ready":
        if not isinstance(case.get("expected_language_code"), str):
            raise ValueError(f"{case_id}: ready cases require expected_language_code.")


def _case_result(
    case: dict[str, Any],
    *,
    base_url: str,
    timeout_seconds: float,
    poll_interval_seconds: float,
    max_wait_seconds: float,
    verify_repeat_reuse: bool,
) -> dict[str, Any]:
    case_id = case["case_id"]
    video_id = case["video_id"]
    languages = case["languages"]
    expected_outcome = case.get("expected_outcome", "ready")

    initial = _api_request(
        base_url,
        "/index",
        method="POST",
        payload={"video_id": video_id, "languages": languages},
        timeout_seconds=timeout_seconds,
    )
    if initial.get("transport_error") or initial.get("parse_error"):
        return {
            "case_id": case_id,
            "video_id": video_id,
            "status": "error",
            "failure": initial.get("transport_error") or initial.get("parse_error"),
        }

    current = initial
    payload = current.get("payload") or {}
    if payload.get("state") not in TERMINAL_STATES:
        current = _poll_index_status(
            base_url,
            video_id,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            max_wait_seconds=max_wait_seconds,
        )
        if current.get("transport_error") or current.get("parse_error"):
            return {
                "case_id": case_id,
                "video_id": video_id,
                "status": "error",
                "http_status": current.get("http_status"),
                "failure": current.get("transport_error") or current.get("parse_error"),
            }
        payload = current.get("payload") or {}

    actual_state = payload.get("state")
    result: dict[str, Any] = {
        "case_id": case_id,
        "video_id": video_id,
        "cohort": case.get("cohort", "unseen_new_video"),
        "expected_outcome": expected_outcome,
        "http_status": current.get("http_status"),
        "state": actual_state,
        "action": payload.get("action"),
        "ready": payload.get("ready"),
        "error_code": payload.get("error_code"),
        "error_message": payload.get("error_message"),
        "status": "success",
        "checks": [],
    }

    if expected_outcome == "failed":
        passed = (
            actual_state == "failed"
            and payload.get("error_code") == case["expected_error_code"]
            and bool(payload.get("error_message"))
        )
        result["checks"].append({
            "name": "stable_failure_code_and_message",
            "passed": passed,
            "detail": (
                "The expected stable failure code and safe message were returned."
                if passed
                else f"Expected {case['expected_error_code']}, got "
                f"{payload.get('error_code')} in state {actual_state}."
            ),
        })
        result["status"] = "success" if passed else "quality_failure"
        return result

    if actual_state != "ready" or payload.get("ready") is not True:
        result["status"] = "quality_failure"
        result["checks"].append({
            "name": "index_ready",
            "passed": False,
            "detail": f"Expected ready, got state={actual_state!r}.",
        })
        return result

    video_store = VECTOR_STORE_ROOT / video_id
    required_files = ("index.faiss", "index.pkl", "metadata.json")
    missing_files = [name for name in required_files if not (video_store / name).is_file()]
    result["checks"].append({
        "name": "persisted_index_files",
        "passed": not missing_files,
        "detail": "All three persisted index files exist."
        if not missing_files else "Missing: " + ", ".join(missing_files),
    })
    if missing_files:
        result["status"] = "quality_failure"
        return result

    try:
        metadata = json.loads(
            (video_store / "metadata.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        result["status"] = "quality_failure"
        result["checks"].append({
            "name": "metadata_readable",
            "passed": False,
            "detail": str(error),
        })
        return result

    stats_present = all(
        field in metadata
        for field in (
            "transcript_segment_count",
            "transcript_character_count",
            "transcript_timeline_start_seconds",
            "transcript_timeline_end_seconds",
            "transcript_timeline_span_seconds",
        )
    )
    result["transcript_metadata"] = {
        key: metadata.get(key)
        for key in (
            "transcript_language",
            "transcript_language_code",
            "transcript_generated",
            "requested_languages",
            "chunk_count",
            "transcript_segment_count",
            "transcript_character_count",
            "transcript_timeline_start_seconds",
            "transcript_timeline_end_seconds",
            "transcript_timeline_span_seconds",
        )
    }
    result["checks"].append({
        "name": "transcript_statistics_present",
        "passed": stats_present,
        "detail": "Unique transcript counts and timeline statistics are present."
        if stats_present else "Transcript statistics are absent or incomplete; rebuild in the Phase 11.0 namespace.",
    })
    if not stats_present:
        result["status"] = "quality_failure"
        return result

    expected_language = case["expected_language_code"]
    language_ok = metadata.get("transcript_language_code") == expected_language
    result["checks"].append({
        "name": "selected_transcript_language",
        "passed": language_ok,
        "detail": f"Expected {expected_language}, got {metadata.get('transcript_language_code')}.",
    })

    expected_generated = case.get("expected_generated")
    if expected_generated is not None:
        generated_ok = metadata.get("transcript_generated") is expected_generated
        result["checks"].append({
            "name": "caption_track_variant",
            "passed": generated_ok,
            "detail": f"Expected transcript_generated={expected_generated}, got {metadata.get('transcript_generated')}.",
        })
    else:
        generated_ok = True

    requested_languages_ok = metadata.get("requested_languages") == languages
    result["checks"].append({
        "name": "requested_language_preferences_persisted",
        "passed": requested_languages_ok,
        "detail": f"Expected {languages}, got {metadata.get('requested_languages')}.",
    })

    segment_count = metadata.get("transcript_segment_count")
    character_count = metadata.get("transcript_character_count")
    timeline_span = metadata.get("transcript_timeline_span_seconds")
    statistics_ok = (
        type(segment_count) is int and segment_count > 0
        and type(character_count) is int and character_count > 0
        and isinstance(timeline_span, (int, float))
        and not isinstance(timeline_span, bool)
        and timeline_span > 0
    )
    result["checks"].append({
        "name": "nonempty_transcript_statistics",
        "passed": statistics_ok,
        "detail": "Unique captions, characters, and timeline span are positive."
        if statistics_ok else "One or more transcript statistics are missing or invalid.",
    })

    expected_class = case.get("expected_timeline_class")
    if expected_class is not None:
        try:
            actual_class = classify_transcript_timeline(
                float(timeline_span),
                short_max_seconds=float(case.get("short_max_seconds", 600)),
                long_min_seconds=float(case.get("long_min_seconds", 1800)),
            )
            class_ok = actual_class == expected_class
        except (TypeError, ValueError) as error:
            actual_class = None
            class_ok = False
            result["checks"].append({
                "name": "timeline_classification",
                "passed": False,
                "detail": str(error),
            })
        else:
            result["checks"].append({
                "name": "timeline_classification",
                "passed": class_ok,
                "detail": f"Expected {expected_class}, observed {actual_class} from caption timeline span.",
            })
    else:
        actual_class = classify_transcript_timeline(float(timeline_span))
    result["observed_timeline_class"] = actual_class

    all_metadata_checks = (
        language_ok and generated_ok and requested_languages_ok and statistics_ok
    )
    if not all_metadata_checks:
        result["status"] = "quality_failure"
        return result

    if verify_repeat_reuse and case.get("verify_repeat_reuse", True):
        before_hashes = {
            name: _sha256(video_store / name)
            for name in required_files
        }
        repeated = _api_request(
            base_url,
            "/index",
            method="POST",
            payload={"video_id": video_id, "languages": languages},
            timeout_seconds=timeout_seconds,
        )
        repeated_payload = repeated.get("payload") or {}
        reuse_ok = (
            repeated.get("transport_error") is None
            and repeated.get("parse_error") is None
            and repeated_payload.get("state") == "ready"
            and repeated_payload.get("ready") is True
            and repeated_payload.get("action") == "reuse"
        )
        after_hashes = {
            name: _sha256(video_store / name)
            for name in required_files
        }
        files_unchanged = before_hashes == after_hashes
        result["checks"].append({
            "name": "repeated_index_reuses_existing_index",
            "passed": reuse_ok and files_unchanged,
            "detail": (
                "Second POST /index returned ready/reuse and persisted files were unchanged."
                if reuse_ok and files_unchanged
                else "Repeated indexing did not prove reuse without rewriting persisted index files."
            ),
        })
        result["repeat_index_response"] = {
            "http_status": repeated.get("http_status"),
            "state": repeated_payload.get("state"),
            "action": repeated_payload.get("action"),
            "ready": repeated_payload.get("ready"),
            "transport_error": repeated.get("transport_error"),
            "parse_error": repeated.get("parse_error"),
            "persisted_files_unchanged": files_unchanged,
        }

    if any(not check["passed"] for check in result["checks"]):
        result["status"] = "quality_failure"
    return result


def run_smoke(
    *,
    manifest_path: Path,
    output_path: Path,
    base_url: str,
    timeout_seconds: float,
    poll_interval_seconds: float,
    max_wait_seconds: float,
    verify_repeat_reuse: bool,
) -> dict[str, Any]:
    if RAG_INDEX_NAMESPACE != REQUIRED_NAMESPACE:
        raise RuntimeError(
            "Phase 11.0 smoke checks must run with "
            f"RAG_INDEX_NAMESPACE={REQUIRED_NAMESPACE!r}; got {RAG_INDEX_NAMESPACE!r}. "
            "This guard protects the frozen pilot indexes."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = manifest.get("cases") if isinstance(manifest, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError("Smoke manifest must contain a non-empty cases array.")
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Every smoke case must be a JSON object.")
        _validate_case(case)
    case_ids = [case["case_id"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Smoke case IDs must be unique.")

    results = [
        _case_result(
            case,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            max_wait_seconds=max_wait_seconds,
            verify_repeat_reuse=verify_repeat_reuse,
        )
        for case in cases
    ]
    failed = [result for result in results if result["status"] != "success"]
    unseen_videos = {
        result["video_id"]
        for result in results
        if result.get("cohort", "unseen_new_video") == "unseen_new_video"
        and result["status"] == "success"
    }
    track_kinds = set()
    timeline_classes = set()
    failure_codes = set()
    for result in results:
        metadata = result.get("transcript_metadata") or {}
        if metadata.get("transcript_generated") is True:
            track_kinds.add("youtube_auto_generated")
        elif metadata.get("transcript_generated") is False:
            track_kinds.add("youtube_publisher_uploaded")
        if result.get("observed_timeline_class"):
            timeline_classes.add(result["observed_timeline_class"])
        if result.get("status") == "success" and result.get("expected_outcome") == "failed":
            failure_codes.add(result.get("error_code"))

    required = manifest.get("required_coverage", {})
    missing_coverage = {
        "caption_track_kinds": sorted(
            set(required.get("caption_track_kinds", [])) - track_kinds
        ),
        "transcript_timeline_classes": sorted(
            set(required.get("transcript_timeline_classes", [])) - timeline_classes
        ),
        "failure_codes": sorted(
            set(required.get("failure_codes", [])) - failure_codes
        ),
        "minimum_unseen_video_count": (
            len(unseen_videos) < int(required.get("minimum_unseen_video_count", 0))
        ),
    }
    coverage_passed = (
        not missing_coverage["caption_track_kinds"]
        and not missing_coverage["transcript_timeline_classes"]
        and not missing_coverage["failure_codes"]
        and not missing_coverage["minimum_unseen_video_count"]
    )
    report = {
        "report_schema_version": "1.0.0",
        "report_id": "phase11_0_unseen_video_index_smoke_v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "backend_base_url": base_url.rstrip("/"),
        "rag_index_namespace": RAG_INDEX_NAMESPACE,
        "vector_store_root": str(VECTOR_STORE_ROOT),
        "manifest_path": str(manifest_path),
        "case_count": len(cases),
        "cases_succeeded": len(results) - len(failed),
        "cases_failed": len(failed),
        "unseen_video_count": len(unseen_videos),
        "caption_track_kinds_observed": sorted(track_kinds),
        "transcript_timeline_classes_observed": sorted(timeline_classes),
        "failure_codes_verified": sorted(failure_codes),
        "required_coverage": required,
        "missing_required_coverage": missing_coverage,
        "required_coverage_passed": coverage_passed,
        "quality_passed": not failed and coverage_passed,
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
        description="Run isolated Phase 11.0 smoke checks against unseen YouTube videos."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--max-wait-seconds", type=float, default=600.0)
    parser.add_argument(
        "--skip-repeat-reuse",
        action="store_true",
        help="Skip the second POST /index reuse check.",
    )
    args = parser.parse_args()
    if args.timeout <= 0 or args.poll_interval <= 0 or args.max_wait_seconds <= 0:
        parser.error("Timeout and polling values must be greater than zero.")
    report = run_smoke(
        manifest_path=args.manifest,
        output_path=args.output,
        base_url=args.base_url,
        timeout_seconds=args.timeout,
        poll_interval_seconds=args.poll_interval,
        max_wait_seconds=args.max_wait_seconds,
        verify_repeat_reuse=not args.skip_repeat_reuse,
    )
    print(json.dumps(
        {
            "report_path": str(args.output),
            "case_count": report["case_count"],
            "cases_succeeded": report["cases_succeeded"],
            "cases_failed": report["cases_failed"],
            "unseen_video_count": report["unseen_video_count"],
            "caption_track_kinds_observed": report["caption_track_kinds_observed"],
            "transcript_timeline_classes_observed": report["transcript_timeline_classes_observed"],
            "failure_codes_verified": report["failure_codes_verified"],
            "missing_required_coverage": report["missing_required_coverage"],
            "required_coverage_passed": report["required_coverage_passed"],
            "quality_passed": report["quality_passed"],
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0 if report["quality_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
