"""
evaluation.py

Retrieval regression evaluation for the YouTube RAG system.

This module evaluates retrieval behavior only. It never calls the
generation model.

The regression contract intentionally separates strict retrieval
requirements from cases where an answerability decision belongs to
the generation/grounding layer.
"""

import argparse
import json
from typing import Any

from .config import (
    MAX_DISTANCE,
    MMR_FETCH_K,
    MMR_LAMBDA,
    TOP_K,
)

from .retrieval import (
    extract_video_id,
    load_vector_store,
    retrieve_question_context,
)


DEFAULT_VIDEO = "https://www.youtube.com/watch?v=MdeQMVBuGgY"


TEST_QUESTIONS = [
    {
        "id": "Q01",
        "type": "direct_fact",
        "question": "What is Kingfisher Airlines?",
    },
    {
        "id": "Q02",
        "type": "direct_fact",
        "question": (
            "What connection does Vijay Mallya discuss "
            "with Formula One?"
        ),
    },
    {
        "id": "Q03",
        "type": "direct_fact",
        "question": (
            "What businesses or companies does Vijay Mallya "
            "discuss in the interview?"
        ),
    },
    {
        "id": "Q04",
        "type": "explanation",
        "question": (
            "Why does Vijay Mallya say Kingfisher Airlines "
            "struggled or failed?"
        ),
    },
    {
        "id": "Q05",
        "type": "explanation",
        "question": (
            "How does Vijay Mallya describe the challenges "
            "he faced while running Kingfisher Airlines?"
        ),
    },
    {
        "id": "Q06",
        "type": "explanation",
        "question": (
            "What does Vijay Mallya say about building "
            "the Kingfisher brand?"
        ),
    },
    {
        "id": "Q07",
        "type": "specific_detail",
        "question": (
            "What does Vijay Mallya say about the amount "
            "of debt or money involved in the Kingfisher "
            "situation?"
        ),
    },
    {
        "id": "Q08",
        "type": "specific_detail",
        "question": (
            "What does Vijay Mallya say about banks and "
            "the recovery of money?"
        ),
    },
    {
        "id": "Q09",
        "type": "specific_detail",
        "question": (
            "What does Vijay Mallya say about the role of "
            "Indian government policy in the problems "
            "faced by Kingfisher Airlines?"
        ),
    },
    {
        "id": "Q10",
        "type": "multi_part",
        "question": (
            "What does Vijay Mallya say about the failure "
            "of Kingfisher Airlines, and what reasons does "
            "he give for it?"
        ),
    },
    {
        "id": "Q11",
        "type": "temporal",
        "question": (
            "Where in the video does Vijay Mallya discuss "
            "Kingfisher Airlines, and what is discussed "
            "there?"
        ),
    },
    {
        "id": "Q12",
        "type": "unanswerable",
        "question": (
            "What programming language does Vijay Mallya "
            "say is his favorite?"
        ),
    },
    {
        "id": "Q13",
        "type": "unrelated",
        "question": "What is the capital of Australia?",
    },
]


EXPECTED_RETRIEVAL = {
    **{f"Q{i:02d}": "must_retrieve" for i in range(1, 12)},
    "Q12": "may_retrieve",
    "Q13": "must_not_retrieve",
}


def _expected_for(question_id: str) -> str:
    try:
        return EXPECTED_RETRIEVAL[question_id]
    except KeyError as error:
        raise ValueError(
            f"Unknown evaluation question ID: {question_id}"
        ) from error


def evaluate_retrieval_result(
    question_id: str,
    retrieved_count: int,
) -> tuple[bool, str]:
    """Evaluate one retrieval result against the regression contract."""

    if retrieved_count < 0:
        raise ValueError("retrieved_count cannot be negative.")

    expected = _expected_for(question_id)

    if expected == "must_retrieve":
        passed = retrieved_count > 0
        message = (
            "Relevant retrieval found."
            if passed
            else "Expected retrieval, but nothing was retrieved."
        )
        return passed, message

    if expected == "may_retrieve":
        return True, (
            "Retrieval may return context; answerability is "
            "validated by generation."
        )

    if expected == "must_not_retrieve":
        passed = retrieved_count == 0
        message = (
            "Correctly rejected as unrelated."
            if passed
            else "Unexpected context retrieved."
        )
        return passed, message

    raise ValueError(f"Unsupported retrieval expectation: {expected}")


def _build_result_record(
    test_case: dict[str, str],
    results: list[tuple[Any, float]],
) -> dict[str, Any]:
    retrieved_count = len(results)
    passed, evaluation_message = evaluate_retrieval_result(
        question_id=test_case["id"],
        retrieved_count=retrieved_count,
    )

    distances = [float(distance) for _, distance in results]

    chunk_ids = []
    for document, _ in results:
        chunk_id = document.metadata.get("chunk_id")
        if chunk_id is not None and chunk_id not in chunk_ids:
            chunk_ids.append(chunk_id)

    return {
        "id": test_case["id"],
        "type": test_case["type"],
        "question": test_case["question"],
        "expected": EXPECTED_RETRIEVAL[test_case["id"]],
        "retrieved": retrieved_count,
        "passed": passed,
        "message": evaluation_message,
        "best_distance": min(distances) if distances else None,
        "retrieved_chunk_ids": chunk_ids,
    }


def build_evaluation_summary(
    video_id: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a stable machine-readable evaluation summary."""

    total = len(results)
    passed = sum(1 for result in results if result["passed"])
    failed = total - passed

    strict_results = [
        result
        for result in results
        if result["expected"] != "may_retrieve"
    ]
    strict_total = len(strict_results)
    strict_passed = sum(
        1 for result in strict_results if result["passed"]
    )
    strict_failed = strict_total - strict_passed

    return {
        "video_id": video_id,
        "retrieval_strategy": "question_aware_mmr_lexical",
        "retrieval_config": {
            "top_k": TOP_K,
            "fetch_k": MMR_FETCH_K,
            "lambda_mult": MMR_LAMBDA,
            "max_distance": MAX_DISTANCE,
        },
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / total if total else 1.0,
        "strict_total": strict_total,
        "strict_passed": strict_passed,
        "strict_failed": strict_failed,
        "strict_pass_rate": (
            strict_passed / strict_total if strict_total else 1.0
        ),
        "results": results,
    }


def display_results(results: list[tuple[Any, float]]) -> None:
    """Print retrieved evidence for human inspection."""

    if not results:
        print("\nNO RELEVANT DOCUMENTS RETRIEVED.")
        return

    for rank, (document, distance) in enumerate(results, start=1):
        metadata = document.metadata
        print(f"\n--- Result {rank} ---")
        print(f"Distance: {float(distance):.6f}")
        print(f"Chunk ID: {metadata.get('chunk_id')}")
        print(
            f"Timestamp: {metadata.get('start', 0):.2f}s"
            f" -> {metadata.get('end', 0):.2f}s"
        )
        print("\nTEXT:")
        print(document.page_content)


def evaluate_video(
    video_reference: str,
    *,
    display: bool = True,
) -> dict[str, Any]:
    """Run the retrieval regression suite and return a summary."""

    video_id = extract_video_id(video_reference)
    vector_store = load_vector_store(video_id)

    results = []

    if display:
        print("\n" + "=" * 80)
        print("RAG RETRIEVAL REGRESSION EVALUATION")
        print("=" * 80)
        print(f"\nVideo ID: {video_id}")
        print("\nRetrieval Configuration:")
        print("  strategy     = question-aware retrieval")
        print(f"  top_k        = {TOP_K}")
        print(f"  fetch_k      = {MMR_FETCH_K}")
        print(f"  lambda       = {MMR_LAMBDA}")
        print(f"  max_distance = {MAX_DISTANCE}")
        print("\nLoading vector store...")

    for test_case in TEST_QUESTIONS:
        question_id = test_case["id"]
        question_type = test_case["type"]
        question = test_case["question"]

        if display:
            print("\n" + "#" * 80)
            print(f"{question_id} | {question_type}")
            print(f"QUESTION: {question}")
            print("#" * 80)

        retrieved = retrieve_question_context(
            vector_store=vector_store,
            query=question,
        )

        record = _build_result_record(
            test_case=test_case,
            results=retrieved,
        )
        results.append(record)

        if display:
            display_results(retrieved)
            print(f"\nExpected: {record['expected']}")
            print(f"Result: {'PASS' if record['passed'] else 'FAIL'}")
            print(f"Evaluation: {record['message']}")

    summary = build_evaluation_summary(
        video_id=video_id,
        results=results,
    )

    if display:
        print("\n" + "=" * 80)
        print("RETRIEVAL SUMMARY")
        print("=" * 80)

        for result in results:
            print(
                f"\n{result['id']} | {result['type']} | "
                f"Retrieved: {result['retrieved']} | "
                f"{'PASS' if result['passed'] else 'FAIL'}"
            )

        print("\n" + "-" * 80)
        print(
            f"Overall: {summary['passed']}/{summary['total']} passed "
            f"({summary['pass_rate']:.1%})"
        )
        print(
            f"Strict:  {summary['strict_passed']}/"
            f"{summary['strict_total']} passed "
            f"({summary['strict_pass_rate']:.1%})"
        )
        print("-" * 80)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the YouTube RAG retrieval regression suite."
    )
    parser.add_argument(
        "video",
        nargs="?",
        default=DEFAULT_VIDEO,
        help="YouTube video ID or URL.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print only machine-readable JSON.",
    )

    args = parser.parse_args()

    summary = evaluate_video(
        args.video,
        display=not args.json,
    )

    if args.json:
        print(json.dumps(summary, indent=2))

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
