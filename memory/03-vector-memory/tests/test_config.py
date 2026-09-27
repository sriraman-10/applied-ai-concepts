from pathlib import Path

import pytest

from config import Settings


def test_settings_from_environment():
    settings = Settings.from_env(
        {
            "CHROMA_PATH": "custom",
            "CHROMA_COLLECTION": "custom_memory",
            "OPENAI_MODEL": "chat",
            "OPENAI_EMBEDDING_MODEL": "embed",
            "VECTOR_TOP_K": "3",
            "VECTOR_MAX_TOP_K": "8",
            "VECTOR_MIN_SIMILARITY": "0.35",
        }
    )
    assert settings.chroma_path == Path("custom")
    assert (
        settings.collection_name,
        settings.model,
        settings.embedding_model,
        settings.top_k,
        settings.max_top_k,
        settings.min_similarity,
    ) == ("custom_memory", "chat", "embed", 3, 8, 0.35)


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError):
        Settings(top_k=0)
    with pytest.raises(ValueError):
        Settings(top_k=10, max_top_k=5)
    with pytest.raises(ValueError):
        Settings(min_similarity=1.1)
