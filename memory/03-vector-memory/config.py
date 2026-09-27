"""Environment-backed settings for the vector-memory demo."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True, slots=True)
class Settings:
    chroma_path: Path = Path(".data/chroma")
    collection_name: str = "agent_memory"
    model: str = "gpt-5-mini"
    embedding_model: str = "text-embedding-3-small"
    top_k: int = 5
    max_top_k: int = 25
    min_similarity: float = 0.20

    def __post_init__(self) -> None:
        if not 3 <= len(self.collection_name) <= 512:
            raise ValueError("CHROMA_COLLECTION must contain 3 to 512 characters")
        if self.top_k <= 0:
            raise ValueError("VECTOR_TOP_K must be greater than zero")
        if self.max_top_k <= 0 or self.top_k > self.max_top_k:
            raise ValueError(
                "VECTOR_MAX_TOP_K must be positive and at least VECTOR_TOP_K"
            )
        if not -1.0 <= self.min_similarity <= 1.0:
            raise ValueError("VECTOR_MIN_SIMILARITY must be between -1 and 1")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        values = os.environ if environ is None else environ
        defaults = cls()

        def integer(name: str, default: int) -> int:
            raw = values.get(name)
            if raw is None:
                return default
            try:
                return int(raw)
            except ValueError as error:
                raise ValueError(f"{name} must be an integer, got {raw!r}") from error

        def number(name: str, default: float) -> float:
            raw = values.get(name)
            if raw is None:
                return default
            try:
                return float(raw)
            except ValueError as error:
                raise ValueError(f"{name} must be a number, got {raw!r}") from error

        return cls(
            chroma_path=Path(values.get("CHROMA_PATH", str(defaults.chroma_path))),
            collection_name=values.get("CHROMA_COLLECTION", defaults.collection_name),
            model=values.get("OPENAI_MODEL", defaults.model),
            embedding_model=values.get(
                "OPENAI_EMBEDDING_MODEL", defaults.embedding_model
            ),
            top_k=integer("VECTOR_TOP_K", defaults.top_k),
            max_top_k=integer("VECTOR_MAX_TOP_K", defaults.max_top_k),
            min_similarity=number("VECTOR_MIN_SIMILARITY", defaults.min_similarity),
        )
