from __future__ import annotations

from types import SimpleNamespace

import pytest

from embeddings import EmbeddingError, OpenAIEmbeddingProvider

pytestmark = pytest.mark.asyncio


class FakeEmbeddings:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])


async def test_openai_provider_uses_configured_model():
    embeddings = FakeEmbeddings()
    client = SimpleNamespace(embeddings=embeddings)
    provider = OpenAIEmbeddingProvider("text-embedding-test", client=client)
    assert await provider.embed("hello") == [0.1, 0.2, 0.3]
    assert embeddings.calls == [
        {"model": "text-embedding-test", "input": "hello", "encoding_format": "float"}
    ]


async def test_openai_provider_rejects_empty_vector():
    class Empty:
        async def create(self, **kwargs):
            return SimpleNamespace(data=[SimpleNamespace(embedding=[])])

    provider = OpenAIEmbeddingProvider(
        "model", client=SimpleNamespace(embeddings=Empty())
    )
    with pytest.raises(EmbeddingError, match="empty vector"):
        await provider.embed("hello")
