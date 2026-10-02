"""
End-to-end RAG evaluation dataset.

This file contains the evaluation contract only. It deliberately does not
contain generated answers because model output is nondeterministic.

Each case defines:
- the user question
- whether the video should answer it
- what quality dimensions must be checked
- the retrieval regression case it corresponds to
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RAGEvaluationCase:
    id: str
    question: str
    answerable: bool
    retrieval_case_id: str
    expected_behavior: str


EVALUATION_CASES = (
    RAGEvaluationCase(
        id="E01",
        retrieval_case_id="Q01",
        question="What is Kingfisher Airlines?",
        answerable=True,
        expected_behavior="Answer using only information stated in the transcript.",
    ),
    RAGEvaluationCase(
        id="E02",
        retrieval_case_id="Q04",
        question=(
            "Why does Vijay Mallya say Kingfisher Airlines "
            "struggled or failed?"
        ),
        answerable=True,
        expected_behavior="Address the reasons given by the speaker without adding outside explanations.",
    ),
    RAGEvaluationCase(
        id="E03",
        retrieval_case_id="Q07",
        question=(
            "What does Vijay Mallya say about the amount of "
            "debt or money involved in the Kingfisher situation?"
        ),
        answerable=True,
        expected_behavior="Report only amounts or financial claims supported by the retrieved transcript.",
    ),
    RAGEvaluationCase(
        id="E04",
        retrieval_case_id="Q10",
        question=(
            "What does Vijay Mallya say about the failure of "
            "Kingfisher Airlines, and what reasons does he give for it?"
        ),
        answerable=True,
        expected_behavior="Cover both parts of the question when the retrieved context supports both.",
    ),
    RAGEvaluationCase(
        id="E05",
        retrieval_case_id="Q12",
        question=(
            "What programming language does Vijay Mallya "
            "say is his favorite?"
        ),
        answerable=False,
        expected_behavior="Return the exact fallback because the evaluation contract marks this as unanswerable.",
    ),
    RAGEvaluationCase(
        id="E06",
        retrieval_case_id="Q13",
        question="What is the capital of Australia?",
        answerable=False,
        expected_behavior="Return the exact fallback because the question is unrelated to the video.",
    ),
)


def get_evaluation_cases() -> tuple[RAGEvaluationCase, ...]:
    """Return the immutable end-to-end evaluation set."""
    return EVALUATION_CASES
