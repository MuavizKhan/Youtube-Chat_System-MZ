import pytest

from Backend.rag_system.indexing import IndexAction, IndexState, get_index_action


@pytest.mark.parametrize(
    ("state", "expected_action"),
    [
        (IndexState.MISSING, IndexAction.CREATE),
        (IndexState.VALID, IndexAction.REUSE),
        (IndexState.INVALID, IndexAction.REBUILD),
        (IndexState.STALE, IndexAction.REBUILD),
    ],
)
def test_get_index_action(state, expected_action):
    assert get_index_action(state) is expected_action


def test_get_index_action_rejects_unsupported_state():
    with pytest.raises(ValueError, match="Unsupported index state"):
        get_index_action("unsupported")