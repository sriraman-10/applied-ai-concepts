"""Embedding provider boundary; tests use a deterministic fake."""

from __future__ import annotations

import math
import os
from typing import Protocol, Sequence


class EmbeddingError(RuntimeError):
    pass


class EmbeddingProvider(Protocol):
    async def embed(self, text: str) -> Sequence[float]: ...


class OpenAIEmbeddingProvider:
    def __init__(
        self, model: str, dimensions: int, *, client: object | None = None
    ) -> None:
        self.model, self.dimensions, self._client = model, dimensions, client

    def _client_or_raise(self) -> object:
        if self._client is not None:
            return self._client
        if not os.getenv("OPENAI_API_KEY"):
            raise EmbeddingError("OPENAI_API_KEY is required for embedding operations")
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI()
        return self._client

    async def embed(self, text: str) -> list[float]:
        try:
            response = await self._client_or_raise().embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimensions,
                encoding_format="float",
            )
            vector = [float(value) for value in response.data[0].embedding]
        except EmbeddingError:
            raise
        except Exception as error:
            raise EmbeddingError(f"Embedding request failed: {error}") from error
        if len(vector) != self.dimensions or not all(
            math.isfinite(value) for value in vector
        ):
            raise EmbeddingError(
                f"Expected {self.dimensions} finite values, received {len(vector)}"
            )
        return vector
