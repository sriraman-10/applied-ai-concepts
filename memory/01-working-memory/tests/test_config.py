"""Offline tests for environment configuration."""

import pytest

from config import Settings


def test_settings_load_small_demo_limits() -> None:
    settings = Settings.from_env(
        {
            "OPENAI_MODEL": "test-model",
            "MEMORY_HARD_LIMIT": "1500",
            "MEMORY_TARGET_LIMIT": "900",
            "MEMORY_KEEP_RECENT": "4",
        }
    )
    assert settings.model == "test-model"
    assert settings.summary_model == "test-model"
    assert settings.hard_token_limit == 1500
    assert settings.target_token_limit == 900
    assert settings.keep_recent_items == 4


def test_invalid_integer_has_clear_error() -> None:
    with pytest.raises(ValueError, match="MEMORY_HARD_LIMIT must be an integer"):
        Settings.from_env({"MEMORY_HARD_LIMIT": "many"})
