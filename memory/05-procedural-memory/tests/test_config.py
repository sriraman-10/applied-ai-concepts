import pytest
from config import Settings


def test_settings_from_environment():
    s = Settings.from_env(
        {
            "DATABASE_URL": "postgresql://u:p@db/test",
            "OPENAI_EMBEDDING_DIMENSIONS": "24",
            "PROCEDURE_TOP_K": "2",
            "PROCEDURE_MAX_TOP_K": "8",
            "PROCEDURE_MIN_SIMILARITY": "0.4",
        }
    )
    assert (s.embedding_dimensions, s.top_k, s.max_top_k, s.min_similarity) == (
        24,
        2,
        8,
        0.4,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"database_url": "sqlite:///x"},
        {"embedding_dimensions": 0},
        {"top_k": 9, "max_top_k": 3},
        {"min_similarity": 2},
        {"max_agent_turns": 0},
    ],
)
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)
