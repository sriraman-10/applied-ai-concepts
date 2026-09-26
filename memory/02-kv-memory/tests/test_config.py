"""Configuration tests."""

from pathlib import Path

import pytest

from config import Settings


def test_settings_from_environment() -> None:
    settings = Settings.from_env(
        {
            "KV_MEMORY_DB_PATH": "custom/memory.db",
            "OPENAI_MODEL": "test-model",
            "KV_MAX_TOOL_TURNS": "4",
            "KV_MAX_TTL_SECONDS": "3600",
        }
    )
    assert settings.database_path == Path("custom/memory.db")
    assert settings.model == "test-model"
    assert settings.max_tool_turns == 4
    assert settings.max_ttl_seconds == 3600


def test_invalid_integer_is_clear() -> None:
    with pytest.raises(ValueError, match="KV_MAX_TOOL_TURNS must be an integer"):
        Settings.from_env({"KV_MAX_TOOL_TURNS": "many"})
