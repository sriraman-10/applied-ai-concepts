from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import pytest

from memory import VectorMemoryManager


class FakeEmbeddingProvider:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        vector = [0.0] * 24
        for token in re.findall(r"[a-z0-9]+", text.casefold()):
            digest = hashlib.sha256(token.encode()).digest()
            vector[digest[0] % len(vector)] += 1.0
            vector[digest[1] % len(vector)] += 0.35
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


@pytest.fixture
def provider() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


@pytest.fixture
def manager(tmp_path: Path, provider: FakeEmbeddingProvider) -> VectorMemoryManager:
    return VectorMemoryManager(
        tmp_path / "chroma", "test_memory", provider, max_top_k=10
    )
