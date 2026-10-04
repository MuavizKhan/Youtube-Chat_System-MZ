"""
End-to-end evaluation runner.

This module is deliberately separate from the retrieval regression suite.
It executes the same public RAG function used by the API and applies the
deterministic response contract to each evaluation case.

Live model execution is opt-in because it consumes inference quota and is
nondeterministic.
"""

import argparse
import json

from .chain import answer_question
from .rag_evaluation import EVALUATION_CASES
from .metrics import evaluate_response_contract


def evaluate_video(video_reference: str) -> dict:
    results = []

    for case in EVALUATION_CASES:
        result = answer_question(
            video_reference=video_reference,
            question=case.question,
        )

        quality = evaluate_response_contract(
            answer=result["answer"],
            answerable=case.answerable,
            retrieved_chunks=result["retrieved_chunks"],
            source_segments=result["source_segments"],
            sources=result["sources"],
        )

        results.append(
            {
                "id": case.id,
                "retrieval_case_id": case.retrieval_case_id,
                "question": case.question,
                "answerable": case.answerable,
                "passed": quality.passed,
                "checks": quality.checks,
                "failures": list(quality.failures),
                "answer": result["answer"],
                "retrieved_chunks": result["retrieved_chunks"],
                "source_segments": result["source_segments"],
                "sources": result["sources"],
            }
        )

    passed = sum(item["passed"] for item in results)

    return {
        "video_reference": video_reference,
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": passed / len(results) if results else 1.0,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run live end-to-end RAG quality evaluation."
    )
    parser.add_argument(
        "video",
        help="YouTube video ID or YouTube URL.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON.",
    )
    args = parser.parse_args()

    summary = evaluate_video(args.video)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print("\nRAG END-TO-END EVALUATION")
        print("=" * 80)
        for item in summary["results"]:
            status = "PASS" if item["passed"] else "FAIL"
            print(f"{item['id']} | {status} | {item['question']}")
            if item["failures"]:
                print(f"  Failures: {', '.join(item['failures'])}")

        print("-" * 80)
        print(
            f"Overall: {summary['passed']}/{summary['total']} "
            f"passed ({summary['pass_rate']:.1%})"
        )

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
