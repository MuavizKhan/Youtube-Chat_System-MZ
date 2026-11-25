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
        "Summarize this entire video chronologically.",
        "What are the most important claims and findings discussed?",
        "Give me the timestamps for the important sections.",
        "Give me the main events from this video.",
        "Recap the entire video.",
        "What is the context of this video?",
        "Explain the context of this video.",
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

@pytest.mark.regression
def test_full_video_summary_with_multiple_content_requirements_overrides_dense_classifier():
    question = (
        "Summarize this entire video chronologically. "
        "Include the important arguments, explanations, "
        "events and conclusions. Use only information "
        "supported by the transcript, provide relevant "
        "timestamps, and clearly qualify anything that "
        "the transcript does not establish."
    )

    # The broad multi-facet wording is expected to trigger the generic
    # evidence-dense heuristic. Explicit whole-video summary scope must win.
    assert retrieval.is_overview_question(question)
    assert retrieval.is_evidence_dense_question(question)
    assert chain._is_explicit_overview_request(question)
    assert chain._should_use_overview_route(question, "overview")


@pytest.mark.regression
def test_targeted_multi_facet_analysis_remains_outside_overview_route():
    question = (
        "Explain the major historical developments mentioned in the video "
        "and their consequences."
    )

    assert not chain._is_explicit_overview_request(question)
    assert not chain._should_use_overview_route(question, "overview")

