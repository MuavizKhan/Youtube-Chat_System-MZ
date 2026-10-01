"""
evaluation.py

Retrieval evaluation for the YouTube RAG system.

This script evaluates the retrieval stage only.

It does NOT call the generation model.

For each test question it reports:

- retrieved chunk count
- expected retrieval behavior
- retrieval pass/fail status
- FAISS distance
- chunk ID
- timestamp
- retrieved transcript text
"""

import argparse


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


# ============================================================
# DEFAULT TEST VIDEO
# ============================================================

DEFAULT_VIDEO = (
    "https://www.youtube.com/watch?v=MdeQMVBuGgY"
)


# ============================================================
# TEST QUESTIONS
# ============================================================

TEST_QUESTIONS = [

    {
        "id": "Q01",
        "type": "direct_fact",
        "question":
            "What is Kingfisher Airlines?",
    },

    {
        "id": "Q02",
        "type": "direct_fact",
        "question":
            "What connection does Vijay Mallya discuss "
            "with Formula One?",
    },

    {
        "id": "Q03",
        "type": "direct_fact",
        "question":
            "What businesses or companies does Vijay Mallya "
            "discuss in the interview?",
    },

    {
        "id": "Q04",
        "type": "explanation",
        "question":
            "Why does Vijay Mallya say Kingfisher Airlines "
            "struggled or failed?",
    },

    {
        "id": "Q05",
        "type": "explanation",
        "question":
            "How does Vijay Mallya describe the challenges "
            "he faced while running Kingfisher Airlines?",
    },

    {
        "id": "Q06",
        "type": "explanation",
        "question":
            "What does Vijay Mallya say about building "
            "the Kingfisher brand?",
    },

    {
        "id": "Q07",
        "type": "specific_detail",
        "question":
            "What does Vijay Mallya say about the amount "
            "of debt or money involved in the Kingfisher "
            "situation?",
    },

    {
        "id": "Q08",
        "type": "specific_detail",
        "question":
            "What does Vijay Mallya say about banks and "
            "the recovery of money?",
    },

    {
        "id": "Q09",
        "type": "specific_detail",
        "question":
            "What does Vijay Mallya say about the role of "
            "Indian government policy in the problems "
            "faced by Kingfisher Airlines?",
    },

    {
        "id": "Q10",
        "type": "multi_part",
        "question":
            "What does Vijay Mallya say about the failure "
            "of Kingfisher Airlines, and what reasons does "
            "he give for it?",
    },

    {
        "id": "Q11",
        "type": "temporal",
        "question":
            "Where in the video does Vijay Mallya discuss "
            "Kingfisher Airlines, and what is discussed "
            "there?",
    },

    {
        "id": "Q12",
        "type": "unanswerable",
        "question":
            "What programming language does Vijay Mallya "
            "say is his favorite?",
    },

    {
        "id": "Q13",
        "type": "unrelated",
        "question":
            "What is the capital of Australia?",
    },
]

# ============================================================
# EXPECTED RETRIEVAL BEHAVIOR
# ============================================================

EXPECTED_RETRIEVAL = {
    **{
        f"Q{i:02d}": "must_retrieve"
        for i in range(1, 12)
    },

    "Q12": "may_retrieve",
    "Q13": "must_not_retrieve",
}

# ============================================================
# DISPLAY HELPERS
# ============================================================

def display_results(
    results,
):
    """
    Print retrieved chunks in a readable format.
    """

    if not results:

        print(
            "\nNO RELEVANT DOCUMENTS RETRIEVED."
        )

        return


    for rank, (
        document,
        distance,
    ) in enumerate(
        results,
        start=1,
    ):

        metadata = document.metadata


        print(
            f"\n--- Result {rank} ---"
        )


        print(
            f"Distance: "
            f"{float(distance):.6f}"
        )


        print(
            f"Chunk ID: "
            f"{metadata.get('chunk_id')}"
        )


        print(
            f"Timestamp: "
            f"{metadata.get('start', 0):.2f}s"
            f" → "
            f"{metadata.get('end', 0):.2f}s"
        )


        print(
            "\nTEXT:"
        )


        print(
            document.page_content
        )



# ============================================================
# RETRIEVAL RESULT CHECK
# ============================================================

def evaluate_retrieval_result(
    question_id: str,
    retrieved_count: int,
) -> tuple[bool, str]:
    """
    Determine whether the retrieval result matches
    the expected behavior for the test question.
    """

    expected = EXPECTED_RETRIEVAL[question_id]

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
            "Retrieval may return context; "
            "answerability is validated by generation."
        )

    if expected == "must_not_retrieve":

        passed = retrieved_count == 0

        message = (
            "Correctly rejected as unrelated."
            if passed
            else "Unexpected context retrieved."
        )

        return passed, message

    raise ValueError(
        f"Unknown retrieval expectation: {expected}"
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_video(
    video_reference: str,
):
    """
    Run the retrieval inspection suite.
    """

    video_id = extract_video_id(
        video_reference
    )


    print(
        "\n" + "=" * 80
    )

    print(
        "RAG RETRIEVAL REGRESSION EVALUATION"
    )

    print(
        "=" * 80
    )


    print(
        f"\nVideo ID: {video_id}"
    )


    print(
        f"\nRetrieval Configuration:"
    )
    
    print(
        "  strategy    = question-aware retrieval"
    )

    print(
        f"  top_k       = {TOP_K}"
    )

    print(
        f"  fetch_k     = {MMR_FETCH_K}"
    )

    print(
        f"  lambda      = {MMR_LAMBDA}"
    )

    print(
        f"  max_distance= {MAX_DISTANCE}"
    )


    print(
        "\nLoading vector store..."
    )


    vector_store = load_vector_store(
        video_id
    )


    evaluation_results = []


    for test_case in TEST_QUESTIONS:

        question_id = test_case["id"]
        question_type = test_case["type"]
        question = test_case["question"]

        print(
            "\n" + "#" * 80
        )

        print(
            f"{question_id} | {question_type}"
        )

        print(
            f"QUESTION: {question}"
        )

        print(
            "#" * 80
        )

        results = retrieve_question_context(
            vector_store=vector_store,
            query=question,
        )
        
        display_results(
            results
        )
        retrieved_count = len(results)

        passed, evaluation_message = (
            evaluate_retrieval_result(
                question_id=question_id,
                retrieved_count=retrieved_count,
            )
        )

        print(
            f"\nExpected: "
            f"{EXPECTED_RETRIEVAL[question_id]}"
        )

        print(
            f"Result: "
            f"{'PASS' if passed else 'FAIL'}"
        )

        print(
            f"Evaluation: "
            f"{evaluation_message}"
        )

        evaluation_results.append(
            {
                "id": question_id,
                "type": question_type,
                "question": question,
                "retrieved": retrieved_count,
                "passed": passed,
            }
        )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "RETRIEVAL SUMMARY"
    )

    print(
        "=" * 80
    )


    passed_count = sum(
        result["passed"]
        for result in evaluation_results
    )

    total_count = len(
        evaluation_results
    )

    for result in evaluation_results:

        print(
            f"\n{result['id']} | "
            f"{result['type']} | "
            f"Retrieved: {result['retrieved']} | "
            f"{'PASS' if result['passed'] else 'FAIL'}"
        )

    print(
        "\n" + "-" * 80
    )

    print(
        f"Evaluation Result: "
        f"{passed_count}/{total_count} tests passed"
    )

    print(
        "-" * 80
    )


# ============================================================
# COMMAND-LINE ENTRY POINT
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Inspect RAG retrieval quality for a YouTube video."
        )
    )


    parser.add_argument(
        "video",
        nargs="?",
        default=DEFAULT_VIDEO,
        help=(
            "YouTube video ID or URL."
        ),
    )


    args = parser.parse_args()


    evaluate_video(
        args.video
    )


if __name__ == "__main__":

    main()