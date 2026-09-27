"""Embedding boundary used by vector memory."""

from __future__ import annotations

import os
from typing import Protocol, Sequence


class EmbeddingError(RuntimeError):
    """Raised when an embedding cannot be generated safely."""


class EmbeddingProvider(Protocol):
    async def embed(self, text: str) -> Sequence[float]:
        """Return one finite embedding vector for text."""


class OpenAIEmbeddingProvider:
    def __init__(self, model: str, *, client: object | None = None) -> None:
        self.model = model
        self._client = client

    def _client_or_raise(self) -> object:
        if self._client is not None:
            return self._client
        if not os.getenv("OPENAI_API_KEY"):
            raise EmbeddingError(
                "OPENAI_API_KEY is required for /remember, /search, and /chat. "
                "Local /list, /memory, /delete, and /stats commands still work."
            )
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI()
        return self._client

    async def embed(self, text: str) -> Sequence[float]:
        try:
            response = await self._client_or_raise().embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float",
            )
            vector = response.data[0].embedding
        except EmbeddingError:
            raise
        except Exception as error:
            raise EmbeddingError(f"Embedding request failed: {error}") from error
        if not vector:
            raise EmbeddingError("Embedding provider returned an empty vector")
        return vector
