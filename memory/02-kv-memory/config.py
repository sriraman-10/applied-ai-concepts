"""Configuration for the SQLite KV-memory demonstration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True, slots=True)
class Settings:
    database_path: Path = Path(".data/kv_memory.db")
    model: str = "gpt-5-mini"
    max_tool_turns: int = 8
    max_ttl_seconds: int = 31_536_000

    def __post_init__(self) -> None:
        if self.max_tool_turns <= 0:
            raise ValueError("max_tool_turns must be greater than zero")
        if self.max_ttl_seconds <= 0:
            raise ValueError("max_ttl_seconds must be greater than zero")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
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

        return cls(
            database_path=Path(values.get("KV_MEMORY_DB_PATH", str(defaults.database_path))),
            model=values.get("OPENAI_MODEL", defaults.model),
            max_tool_turns=integer("KV_MAX_TOOL_TURNS", defaults.max_tool_turns),
            max_ttl_seconds=integer("KV_MAX_TTL_SECONDS", defaults.max_ttl_seconds),
        )
