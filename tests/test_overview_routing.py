import pytest

from Backend.rag_system import chain, retrieval


@pytest.mark.parametrize(
    "question",
    [
        "Summarize the complete story.",
        "Give me a full summary of this video.",
        "Summarize everything in the video.",
        "What are the key takeaways from this video?",
        "Summarize the video in 5 pointers.",
        "Give me the main events from this video.",
        "Recap the entire video.",
    ],
)
def test_generic_full_video_requests_use_overview_intent(question):
    assert retrieval.is_overview_question(question)
    assert chain._is_explicit_overview_request(question)


@pytest.mark.parametrize(
    "question",
    [
        "What were the main causes of the event?",
        "Compare the two approaches described by the speaker.",
        "What is the second method?",
        "Why did the final experiment fail?",
    ],
)
def test_targeted_and_analytical_questions_are_not_overview_requests(question):
    assert not chain._is_explicit_overview_request(question)
