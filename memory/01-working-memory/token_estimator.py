"""Approximate token counting kept separate from memory policy."""

from __future__ import annotations

import logging
from typing import Protocol

import tiktoken

logger = logging.getLogger(__name__)


class TokenEstimator(Protocol):
    """Small interface that makes token accounting deterministic in tests."""

    def estimate(self, text: str) -> int:
        """Estimate the token count for text."""

    def truncate(self, text: str, max_tokens: int) -> str:
        """Return at most max_tokens of text."""


class TiktokenEstimator:
    """Use tiktoken when available, with an offline-safe byte approximation."""

    def __init__(self, model: str) -> None:
        self._encoding: tiktoken.Encoding | None = None
        try:
            self._encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            try:
                self._encoding = tiktoken.get_encoding("o200k_base")
            except Exception as error:
                logger.warning(
                    "Could not load tiktoken encoding; using byte estimate: %s",
                    type(error).__name__,
                )
        except Exception as error:
            logger.warning(
                "Could not load tiktoken encoding; using byte estimate: %s",
                type(error).__name__,
            )

    def estimate(self, text: str) -> int:
        if self._encoding is not None:
            return len(self._encoding.encode(text))
        if not text:
            return 0
        return len(text.encode("utf-8"))

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        if self._encoding is not None:
            tokens = self._encoding.encode(text)
            if len(tokens) <= max_tokens:
                return text
            return self._encoding.decode(tokens[:max_tokens]).rstrip()
        encoded = text.encode("utf-8")
        if len(encoded) <= max_tokens:
            return text
        return encoded[:max_tokens].decode("utf-8", errors="ignore").rstrip()
