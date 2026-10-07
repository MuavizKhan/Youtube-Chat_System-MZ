from types import SimpleNamespace

import pytest

from Backend.rag_system import query_understanding


class FakeCompletions:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.kwargs = None
        self.call_count = 0

    def create(self, **kwargs):
        self.kwargs = kwargs
        self.call_count += 1

        if self.error is not None:
            raise self.error

        return self.response


def fake_client(content: str):
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                )
            )
        ]
    )
    completions = FakeCompletions(response=response)
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=completions,
        )
    )
    return client, completions


@pytest.mark.unit
def test_understand_query_parses_structured_plan():
    fake, completions = fake_client(
        '{"intent":"causal","standalone_question":"Why are people protesting?",'
        '"search_queries":["reasons for protest","protest demands"]}'
    )

    plan = query_understanding.understand_query(
        fake,
        "Why are they protesting?",
    )

    assert plan.intent == "causal"
    assert plan.standalone_question == "Why are people protesting?"
    assert plan.search_queries == [
        "Why are people protesting?",
        "reasons for protest",
        "protest demands",
    ]
    assert completions.kwargs["max_tokens"] == 256
    assert completions.kwargs["temperature"] == 0.0


@pytest.mark.unit
def test_understand_query_handles_markdown_json():
    fake, _ = fake_client(
        "\x60\x60\x60json\n"
        '{"intent":"overview","standalone_question":"What is this video about?",'
        '"search_queries":["main topic","video overview"]}'
        "\n\x60\x60\x60"
    )

    plan = query_understanding.understand_query(
        fake,
        "What is the main subject here?",
    )

    assert plan.intent == "overview"
    assert plan.standalone_question == "What is this video about?"


@pytest.mark.unit
def test_understand_query_normalizes_unknown_intent():
    fake, _ = fake_client(
        '{"intent":"something_new","standalone_question":"What happened?",'
        '"search_queries":["what happened"]}'
    )

    plan = query_understanding.understand_query(
        fake,
        "What happened?",
    )

    assert plan.intent == "general"


@pytest.mark.unit
def test_understand_query_deduplicates_search_queries():
    fake, _ = fake_client(
        '{"intent":"factual","standalone_question":"Who is the speaker?",'
        '"search_queries":["who is the speaker?","Who is the speaker?","speaker identity"]}'
    )

    plan = query_understanding.understand_query(
        fake,
        "Who is the speaker?",
    )

    assert plan.search_queries == [
        "Who is the speaker?",
        "speaker identity",
    ]


@pytest.mark.unit
@pytest.mark.parametrize(
    "content,error_match",
    [
        ("not json", "did not return a JSON object"),
        ("", "empty response"),
        ('{"intent":"general"}', "invalid query plan"),
    ],
)
def test_understand_query_rejects_unusable_output(content, error_match):
    fake, _ = fake_client(content)

    with pytest.raises(
        query_understanding.QueryUnderstandingError,
        match=error_match,
    ):
        query_understanding.understand_query(
            fake,
            "What happened?",
        )


@pytest.mark.unit
def test_understand_query_rejects_empty_question():
    fake, _ = fake_client("{}")

    with pytest.raises(ValueError, match="question cannot be empty"):
        query_understanding.understand_query(fake, "   ")


@pytest.mark.unit
def test_understand_query_uses_conversation_history_for_follow_up():
    fake, completions = fake_client(
        '{"intent":"follow_up","standalone_question":"Who are the people who joined the protest?",'
        '"search_queries":["people who joined the protest"]}'
    )

    plan = query_understanding.understand_query(
        fake,
        "Who are they people?",
        conversation_history=[
            {
                "role": "user",
                "content": "Why are people protesting?",
            },
            {
                "role": "assistant",
                "content": "People are protesting to demand election commission reforms.",
            },
        ],
    )

    assert plan.intent == "follow_up"
    assert plan.standalone_question == "Who are the people who joined the protest?"
    user_message = completions.kwargs["messages"][1]["content"]
    assert "Why are people protesting?" in user_message
    assert "Who are they people?" in user_message


@pytest.mark.unit
@pytest.mark.parametrize(
    "question,expected",
    [
        ("Who are they?", True),
        ("What happened here?", True),
        ("What is the main topic?", False),
        ("Why are people protesting?", False),
    ],
)
def test_is_likely_follow_up(question, expected):
    assert query_understanding.is_likely_follow_up(question) is expected
