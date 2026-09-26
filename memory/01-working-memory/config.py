"""Environment-backed configuration for the working-memory demo."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated runtime settings."""

    model: str = "gpt-5-mini"
    summary_model: str = "gpt-5-mini"
    hard_token_limit: int = 8_000
    target_token_limit: int = 5_000
    keep_recent_items: int = 6

    def __post_init__(self) -> None:
        if self.hard_token_limit <= 0:
            raise ValueError("hard_token_limit must be greater than zero")
        if self.target_token_limit <= 0:
            raise ValueError("target_token_limit must be greater than zero")
        if self.target_token_limit >= self.hard_token_limit:
            raise ValueError("target_token_limit must be lower than hard_token_limit")
        if self.keep_recent_items < 0:
            raise ValueError("keep_recent_items cannot be negative")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Load settings from environment variables with clear validation errors."""
        values = os.environ if environ is None else environ

        def integer(name: str, default: int) -> int:
            raw = values.get(name)
            if raw is None:
                return default
            try:
                return int(raw)
            except ValueError as error:
                raise ValueError(f"{name} must be an integer, got {raw!r}") from error

        defaults = cls()
        model = values.get("OPENAI_MODEL", defaults.model)
        return cls(
            model=model,
            summary_model=values.get("OPENAI_SUMMARY_MODEL", model),
            hard_token_limit=integer("MEMORY_HARD_LIMIT", defaults.hard_token_limit),
            target_token_limit=integer("MEMORY_TARGET_LIMIT", defaults.target_token_limit),
            keep_recent_items=integer("MEMORY_KEEP_RECENT", defaults.keep_recent_items),
        )
