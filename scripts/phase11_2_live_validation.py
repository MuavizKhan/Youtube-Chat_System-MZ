from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
REPORT_ID = "phase11_2_live_validation_v1"

OVERVIEW_STAGES = {
    "overview",
    "overview_empty",
    "weak_initial_overview",
    "best_effort_overview",
}
ABSTENTION_MARKERS = (
    "i couldn't find enough information",
    "i could not find enough information",
    "it may not be covered by this video",
)
TIMESTAMP_SUMMARY_PATTERN = re.compile(
    r"(?im)^\s*relevant\s+timestamps\s*:"
)

# Keep this deliberately small. Each case can trigger query planning and
# generation against the configured live provider. Run sequentially.
VALIDATION_CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "HI_ANALYTICAL_01",
        "video_id": "INpnuL7qnzY",
        "question": (
            "Explain the major historical developments mentioned in the video "
            "and their consequences."
        ),
        "kind": "multi_event_analytical",
        "expected_answerable": True,
        "expected_route": "targeted",
        "purpose": (
            "Must remain on targeted retrieval even if the planner proposes overview."
        ),
    },
    {
        "id": "HI_OVERVIEW_01",
        "video_id": "INpnuL7qnzY",
        "question": "What is the main subject discussed in this video?",
        "kind": "explicit_overview",
        "expected_answerable": True,
        "expected_route": "overview_allowed",
        "purpose": "Confirms that genuine broad overview requests remain supported.",
    },
    {
        "id": "HI_UNRELATED_01",
        "video_id": "INpnuL7qnzY",
        "question": "What is the capital of Australia?",
        "kind": "unanswerable_control",
        "expected_answerable": False,
        "expected_route": "targeted",
        "purpose": (
            "Must abstain without replacing an unrelated question with overview samples."
        ),
    },
    {
        "id": "AR_MULTIPART_01",
        "video_id": "fAd2nKfrDB8",
        "question": (
            "What does the episode say sound contributes to perception, "
            "and how does it describe sound's role?"
        ),
        "kind": "multi_part_explanation",
        "expected_answerable": True,
        "expected_route": "targeted",
        "purpose": (
            "Checks targeted retrieval and answer/source behaviour on Arabic captions."
        ),
    },
    {
        "id": "AI_MULTIFACET_01",
        "video_id": "A-JBaNvv3Tk",
        "question": (
            "What concern does the guest raise about some future open-source models, "
            "and what has the guest unified into one system?"
        ),
        "kind": "multi_facet_explanation",
        "expected_answerable": True,
        "expected_route": "targeted",
        "purpose": (
            "Checks that a multi-facet question is not downgraded to representative sampling."
        ),
    },
)


def _git_state() -> dict[str, Any]:
    def run_git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    return {
        "runner_git_commit": run_git("rev-parse", "HEAD"),
        "runner_working_tree_clean": (
            run_git("status", "--porcelain") == ""
        ),
    }


def _decode_json(raw_body: bytes) -> tuple[Any, str | None]:
    if not raw_body:
        return None, None
    try:
        return json.loads(raw_body.decode("utf-8")), None
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return None, f"{type(error).__name__}: {error}"


def _request_json(
    url: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    timeout_seconds: float,
) -> tuple[int | None, Any, dict[str, str], str | None]:
    data = (
        json.dumps(body, ensure_ascii=False).encode("utf-8")
        if body is not None
        else None
    )
    request = Request(
        url,
        data=data,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json; charset=utf-8",
        },
        method=method,
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw_body = response.read()
            status = response.status
            headers = dict(response.headers.items())
    except HTTPError as error:
        raw_body = error.read()
        status = error.code
        headers = dict(error.headers.items()) if error.headers else {}
    except (URLError, TimeoutError, OSError) as error:
        return None, None, {}, f"{type(error).__name__}: {error}"

    payload, parse_error = _decode_json(raw_body)
    if parse_error:
        return status, None, headers, parse_error
    return status, payload, headers, None


def _retry_after_seconds(headers: dict[str, str], fallback: float) -> float:
    raw_value = next(
        (value for key, value in headers.items() if key.casefold() == "retry-after"),
        None,
    )
    if raw_value is None:
        return fallback
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return fallback
    return value if value >= 0 else fallback


def _get_index_status(
    base_url: str,
    video_id: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    status, payload, _headers, error = _request_json(
        f"{base_url.rstrip('/')}/index/{video_id}",
        timeout_seconds=timeout_seconds,
    )
    ready = (
        status == 200
        and isinstance(payload, dict)
        and payload.get("ready") is True
    )
    return {
        "http_status": status,
        "ready": ready,
        "state": payload.get("state") if isinstance(payload, dict) else None,
        "action": payload.get("action") if isinstance(payload, dict) else None,
        "error": error,
    }


def _post_chat(
    base_url: str,
    case: dict[str, Any],
    *,
    timeout_seconds: float,
    max_rate_limit_retries: int,
    max_rate_limit_wait_seconds: float,
) -> dict[str, Any]:
    attempts = 0
    cumulative_wait = 0.0
    last_status = None
    last_payload = None
    last_error = None

    for attempt_index in range(max_rate_limit_retries + 1):
        attempts += 1
        status, payload, headers, error = _request_json(
            f"{base_url.rstrip('/')}/chat",
            method="POST",
            body={
                "video_id": case["video_id"],
                "question": case["question"],
            },
            timeout_seconds=timeout_seconds,
        )
        last_status = status
        last_payload = payload
        last_error = error

        if status != 429 or attempt_index >= max_rate_limit_retries:
            break

        retry_after = _retry_after_seconds(headers, fallback=5.0)
        wait_seconds = retry_after + 1.0
        if wait_seconds > max_rate_limit_wait_seconds:
            last_error = (
                "Rate-limit retry deferred because Retry-After plus safety "
                f"buffer ({wait_seconds:.1f}s) exceeds the configured wait cap "
                f"({max_rate_limit_wait_seconds:.1f}s)."
            )
            break

        time.sleep(wait_seconds)
        cumulative_wait += wait_seconds

    return {
        "http_status": last_status,
        "request_attempts": attempts,
        "rate_limit_retry_wait_seconds": round(cumulative_wait, 3),
        "transport_error": last_error,
        "response": last_payload,
    }


def _is_abstention(answer: str) -> bool:
    normalized = answer.casefold()
    return any(marker in normalized for marker in ABSTENTION_MARKERS)


def _valid_source_intervals(sources: list[Any]) -> bool:
    if not sources:
        return False
    for source in sources:
        if not isinstance(source, dict):
            return False
        start = source.get("start")
        end = source.get("end")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or start < 0
            or end <= start
        ):
            return False
    return True


def evaluate_case_response(
    case: dict[str, Any],
    response: dict[str, Any],
) -> dict[str, Any]:
    answer = response.get("answer")
    answer = answer if isinstance(answer, str) else ""
    sources = response.get("sources")
    sources = sources if isinstance(sources, list) else []
    observability = response.get("observability")
    observability = observability if isinstance(observability, dict) else {}
    retrieval = observability.get("retrieval")
    retrieval = retrieval if isinstance(retrieval, dict) else {}
    generation = observability.get("generation")
    generation = generation if isinstance(generation, dict) else {}

    final_stage = retrieval.get("final_stage")
    attempts = retrieval.get("attempts")
    attempts = attempts if isinstance(attempts, list) else []
    attempted_stages = [
        str(item.get("stage"))
        for item in attempts
        if isinstance(item, dict) and item.get("stage") is not None
    ]

    checks: list[dict[str, Any]] = []

    def add_check(name: str, passed: bool, details: str) -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "details": details,
            }
        )

    add_check(
        "retrieval_trace_present",
        isinstance(final_stage, str) and bool(final_stage),
        (
            f"Retrieval final stage: {final_stage}."
            if final_stage
            else "The API response did not include observability.retrieval.final_stage."
        ),
    )

    if case.get("expected_route") == "targeted":
        overview_used = (
            final_stage in OVERVIEW_STAGES
            or any(stage in OVERVIEW_STAGES for stage in attempted_stages)
        )
        add_check(
            "specific_question_avoids_overview_sampling",
            not overview_used,
            (
                "No overview sampling stage was used for this specific question."
                if not overview_used
                else (
                    "A specific question used an overview stage: "
                    f"final_stage={final_stage}, attempts={attempted_stages}."
                )
            ),
        )

    if case.get("expected_answerable") is True:
        add_check(
            "answerable_case_has_answer_text",
            bool(answer.strip()) and not _is_abstention(answer),
            "A non-abstention answer was returned."
            if answer.strip() and not _is_abstention(answer)
            else "The answer is empty or is an abstention for an answerable case.",
        )
        add_check(
            "answerable_case_has_sources",
            bool(sources),
            f"Returned clickable source count: {len(sources)}.",
        )
        add_check(
            "source_intervals_are_valid",
            _valid_source_intervals(sources),
            "All returned source intervals have numeric start/end values with end > start."
            if _valid_source_intervals(sources)
            else "At least one source interval is missing or malformed.",
        )
    else:
        abstained = _is_abstention(answer)
        sources_suppressed = len(sources) == 0
        timestamps_suppressed = TIMESTAMP_SUMMARY_PATTERN.search(answer) is None
        add_check(
            "unanswerable_case_abstains",
            abstained,
            "The answer uses the application's explicit insufficient-evidence fallback."
            if abstained
            else "The answer did not use a recognised insufficient-evidence abstention.",
        )
        add_check(
            "unanswerable_case_suppresses_sources",
            sources_suppressed,
            f"Returned clickable source count: {len(sources)}.",
        )
        add_check(
            "unanswerable_case_suppresses_timestamps",
            timestamps_suppressed,
            "No source-derived timestamp summary was appended."
            if timestamps_suppressed
            else "A source-derived timestamp summary was appended to an abstention.",
        )

    context_budget = generation.get("context_budget")
    if isinstance(context_budget, dict):
        budget_chars = context_budget.get("budget_chars")
        final_chars = context_budget.get("final_chars")
        budget_valid = (
            isinstance(budget_chars, int)
            and isinstance(final_chars, int)
            and 0 <= final_chars <= budget_chars
        )
        add_check(
            "generation_context_within_budget",
            budget_valid,
            (
                f"Generation context used {final_chars} / {budget_chars} characters."
                if budget_valid
                else "Generation context diagnostics exceed the configured budget."
            ),
        )

    return {
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "failed_checks": [
            item["name"] for item in checks if not item["passed"]
        ],
        "retrieval": {
            "final_stage": final_stage,
            "planner_used": retrieval.get("planner_used"),
            "planner_intent": retrieval.get("planner_intent"),
            "planner_overview_rejected": retrieval.get("planner_overview_rejected"),
            "planner_failure_reason": retrieval.get("planner_failure_reason"),
            "planner_failure_fallback_attempted": retrieval.get(
                "planner_failure_fallback_attempted"
            ),
            "planner_failure_fallback_accepted": retrieval.get(
                "planner_failure_fallback_accepted"
            ),
            "best_effort_overview_suppressed": retrieval.get(
                "best_effort_overview_suppressed"
            ),
            "attempt_stages": attempted_stages,
            "retrieved_chunks": response.get("retrieved_chunks"),
            "source_segments": response.get("source_segments"),
        },
        "generation": {
            "generated": generation.get("generated"),
            "fallback_used": generation.get("fallback_used"),
            "latency_ms": generation.get("latency_ms"),
            "context_budget": context_budget,
        },
        "sources": [
            {
                "source_id": source.get("source_id"),
                "start": source.get("start"),
                "end": source.get("end"),
                "chunk_ids": source.get("chunk_ids"),
                "url": source.get("url"),
            }
            for source in sources
            if isinstance(source, dict)
        ],
    }


def select_cases(case_ids: str | None) -> tuple[dict[str, Any], ...]:
    if case_ids is None:
        return VALIDATION_CASES

    requested = [item.strip().upper() for item in case_ids.split(",") if item.strip()]
    if not requested:
        raise ValueError("--case-ids must contain at least one case ID.")
    if len(requested) != len(set(requested)):
        raise ValueError("--case-ids contains duplicate IDs.")

    by_id = {str(case["id"]).upper(): case for case in VALIDATION_CASES}
    unknown = [item for item in requested if item not in by_id]
    if unknown:
        raise ValueError("Unknown Phase 11.2 case IDs: " + ", ".join(unknown))
    return tuple(by_id[item] for item in requested)


def run_validation(
    *,
    base_url: str,
    cases: tuple[dict[str, Any], ...],
    timeout_seconds: float,
    pause_seconds: float,
    max_rate_limit_retries: int,
    max_rate_limit_wait_seconds: float,
) -> dict[str, Any]:
    index_statuses: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []

    for index, case in enumerate(cases):
        video_id = str(case["video_id"])
        if video_id not in index_statuses:
            index_statuses[video_id] = _get_index_status(
                base_url,
                video_id,
                timeout_seconds,
            )
        index_status = index_statuses[video_id]

        result: dict[str, Any] = {
            "case_id": case["id"],
            "video_id": video_id,
            "question": case["question"],
            "kind": case["kind"],
            "purpose": case["purpose"],
            "expected_answerable": case["expected_answerable"],
            "expected_route": case["expected_route"],
            "index_status": index_status,
            "manual_evidence_review": (
                "pending" if case["expected_answerable"] else "not_applicable"
            ),
        }

        if not index_status["ready"]:
            result.update(
                {
                    "status": "index_not_ready",
                    "automated_contracts_passed": False,
                    "error": (
                        "The existing video index is not ready. The runner only "
                        "checks index status; it never triggers index creation."
                    ),
                }
            )
            results.append(result)
            continue

        request_result = _post_chat(
            base_url,
            case,
            timeout_seconds=timeout_seconds,
            max_rate_limit_retries=max_rate_limit_retries,
            max_rate_limit_wait_seconds=max_rate_limit_wait_seconds,
        )
        response = request_result.get("response")
        successful = (
            request_result.get("http_status") == 200
            and isinstance(response, dict)
            and isinstance(response.get("answer"), str)
        )

        result.update(
            {
                "status": "success" if successful else "request_failed",
                "http_status": request_result.get("http_status"),
                "request_attempts": request_result.get("request_attempts"),
                "rate_limit_retry_wait_seconds": request_result.get(
                    "rate_limit_retry_wait_seconds"
                ),
                "transport_error": request_result.get("transport_error"),
            }
        )

        if successful:
            result["response"] = response
            result["audit"] = evaluate_case_response(case, response)
            result["automated_contracts_passed"] = result["audit"]["passed"]
        else:
            result["response"] = response
            result["automated_contracts_passed"] = False

        results.append(result)

        if index < len(cases) - 1 and pause_seconds > 0:
            time.sleep(pause_seconds)

    successful_results = [
        item for item in results if item.get("status") == "success"
    ]
    failed_cases = [
        {
            "case_id": item.get("case_id"),
            "status": item.get("status"),
            "failed_checks": item.get("audit", {}).get("failed_checks", []),
            "error": item.get("error") or item.get("transport_error"),
        }
        for item in results
        if item.get("automated_contracts_passed") is not True
    ]

    configurations = [
        item["response"].get("retrieval_config")
        for item in successful_results
        if isinstance(item.get("response"), dict)
        and isinstance(item["response"].get("retrieval_config"), dict)
    ]
    configuration_consistent = (
        len(configurations) == len(successful_results)
        and bool(configurations)
        and all(config == configurations[0] for config in configurations)
    )

    return {
        "report_schema_version": "1.0.0",
        "report_id": REPORT_ID,
        "generated_at": datetime.now().astimezone().isoformat(),
        "backend_base_url": base_url.rstrip("/"),
        **_git_state(),
        "suite_case_ids": [str(case["id"]) for case in cases],
        "cases_attempted": len(cases),
        "cases_succeeded": len(successful_results),
        "cases_failed": len(cases) - len(successful_results),
        "automated_contracts_passed": (
            len(results) == len(cases)
            and not failed_cases
            and configuration_consistent
        ),
        "effective_server_configuration_consistent_across_cases": (
            configuration_consistent
        ),
        "effective_server_configuration": (
            configurations[0] if configurations else None
        ),
        "live_answer_quality_status": "manual_transcript_review_required",
        "manual_review_required_case_ids": [
            item["case_id"]
            for item in results
            if item.get("manual_evidence_review") == "pending"
        ],
        "automated_contract_failures": failed_cases,
        "index_status_by_video": index_statuses,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a small sequential Phase 11.2 live validation against an "
            "already-running FastAPI backend."
        )
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL of the running FastAPI backend.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Timeout for each API request in seconds.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON report path. Defaults to a new timestamped report.",
    )
    parser.add_argument(
        "--case-ids",
        default=None,
        help="Optional comma-separated subset of Phase 11.2 case IDs.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=20.0,
        help="Pause between chat requests to reduce provider TPM pressure.",
    )
    parser.add_argument(
        "--max-rate-limit-retries",
        type=int,
        default=2,
        help="Maximum retries for HTTP 429 responses from the local API.",
    )
    parser.add_argument(
        "--max-rate-limit-wait-seconds",
        type=float,
        default=45.0,
        help="Maximum single Retry-After wait accepted by the runner.",
    )
    args = parser.parse_args()

    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero.")
    if args.pause_seconds < 0:
        parser.error("--pause-seconds cannot be negative.")
    if args.max_rate_limit_retries < 0:
        parser.error("--max-rate-limit-retries cannot be negative.")
    if args.max_rate_limit_wait_seconds < 0:
        parser.error("--max-rate-limit-wait-seconds cannot be negative.")

    try:
        cases = select_cases(args.case_ids)
    except ValueError as error:
        parser.error(str(error))

    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    output_path = args.output or (
        PROJECT_ROOT
        / "evaluation_reports"
        / f"phase11_2_live_validation_{timestamp}.json"
    )
    if output_path.exists():
        parser.error(f"Refusing to overwrite existing report: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = run_validation(
        base_url=args.base_url,
        cases=cases,
        timeout_seconds=args.timeout,
        pause_seconds=args.pause_seconds,
        max_rate_limit_retries=args.max_rate_limit_retries,
        max_rate_limit_wait_seconds=args.max_rate_limit_wait_seconds,
    )
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Report: {output_path}")
    print(f"Cases attempted: {report['cases_attempted']}")
    print(f"Cases succeeded: {report['cases_succeeded']}")
    print(f"Cases failed: {report['cases_failed']}")
    print(
        "Effective configuration consistent: "
        f"{report['effective_server_configuration_consistent_across_cases']}"
    )
    print(f"Automated contracts passed: {report['automated_contracts_passed']}")
    print("Live answer quality: manual transcript review required")

    for item in report["results"]:
        failed_checks = item.get("audit", {}).get("failed_checks", [])
        print(
            f"{item['case_id']}: status={item['status']} "
            f"automated_contracts_passed={item.get('automated_contracts_passed')} "
            f"failed_checks={','.join(failed_checks)}"
        )

    return 0 if report["automated_contracts_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
