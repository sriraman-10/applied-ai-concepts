from pathlib import Path

import pytest

from config import Settings


def test_settings_from_environment():
    settings = Settings.from_env(
        {
            "DATABASE_URL": "postgresql://u:p@db/test",
            "OPENAI_MODEL": "chat",
            "OPENAI_EMBEDDING_MODEL": "embed",
            "OPENAI_EMBEDDING_DIMENSIONS": "24",
            "EPISODIC_TOP_K": "3",
            "EPISODIC_MAX_TOP_K": "9",
            "EPISODIC_MIN_SIMILARITY": "0.3",
            "EPISODIC_MAX_AGENT_TURNS": "6",
            "MIGRATIONS_PATH": "schema",
        }
    )
    assert settings.database_url == "postgresql://u:p@db/test"
    assert settings.embedding_dimensions == 24
    assert settings.top_k == 3 and settings.max_top_k == 9
    assert settings.min_similarity == 0.3
    assert settings.migrations_path == Path("schema")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"database_url": "sqlite:///bad"},
        {"embedding_dimensions": 0},
        {"embedding_dimensions": 2001},
        {"top_k": 10, "max_top_k": 5},
        {"min_similarity": 2.0},
        {"max_agent_turns": 0},
    ],
)
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)
