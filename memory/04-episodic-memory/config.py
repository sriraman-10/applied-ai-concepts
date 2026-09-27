"""Environment-backed configuration for episodic memory."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True, slots=True)
class Settings:
    database_url: str = (
        "postgresql://episodic:episodic_dev@localhost:5432/episodic_memory"
    )
    model: str = "gpt-5-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    top_k: int = 5
    max_top_k: int = 25
    min_similarity: float = 0.20
    max_agent_turns: int = 10
    migrations_path: Path = Path("migrations")

    def __post_init__(self) -> None:
        if not self.database_url.startswith(("postgresql://", "postgresql+psycopg://")):
            raise ValueError("DATABASE_URL must be a PostgreSQL URL")
        if self.embedding_dimensions <= 0 or self.embedding_dimensions > 2000:
            raise ValueError("OPENAI_EMBEDDING_DIMENSIONS must be from 1 to 2000")
        if self.top_k <= 0 or self.max_top_k < self.top_k:
            raise ValueError("EPISODIC_MAX_TOP_K must be at least EPISODIC_TOP_K")
        if not -1.0 <= self.min_similarity <= 1.0:
            raise ValueError("EPISODIC_MIN_SIMILARITY must be between -1 and 1")
        if self.max_agent_turns <= 0:
            raise ValueError("EPISODIC_MAX_AGENT_TURNS must be positive")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        values = os.environ if environ is None else environ
        defaults = cls()

        def integer(name: str, default: int) -> int:
            raw = values.get(name)
            try:
                return default if raw is None else int(raw)
            except ValueError as error:
                raise ValueError(f"{name} must be an integer") from error

        def number(name: str, default: float) -> float:
            raw = values.get(name)
            try:
                return default if raw is None else float(raw)
            except ValueError as error:
                raise ValueError(f"{name} must be a number") from error

        return cls(
            database_url=values.get("DATABASE_URL", defaults.database_url),
            model=values.get("OPENAI_MODEL", defaults.model),
            embedding_model=values.get(
                "OPENAI_EMBEDDING_MODEL", defaults.embedding_model
            ),
            embedding_dimensions=integer(
                "OPENAI_EMBEDDING_DIMENSIONS", defaults.embedding_dimensions
            ),
            top_k=integer("EPISODIC_TOP_K", defaults.top_k),
            max_top_k=integer("EPISODIC_MAX_TOP_K", defaults.max_top_k),
            min_similarity=number("EPISODIC_MIN_SIMILARITY", defaults.min_similarity),
            max_agent_turns=integer(
                "EPISODIC_MAX_AGENT_TURNS", defaults.max_agent_turns
            ),
            migrations_path=Path(
                values.get("MIGRATIONS_PATH", str(defaults.migrations_path))
            ),
        )
