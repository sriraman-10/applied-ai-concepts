from types import SimpleNamespace

import pytest

from embeddings import EmbeddingError, OpenAIEmbeddingProvider

pytestmark = pytest.mark.asyncio


class FakeEmbeddings:
    def __init__(self, vector):
        self.vector = vector
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(embedding=self.vector)])


async def test_openai_embedding_provider_sends_configured_dimensions():
    embeddings = FakeEmbeddings([0.1, 0.2, 0.3])
    provider = OpenAIEmbeddingProvider(
        "text-embedding-test", 3, client=SimpleNamespace(embeddings=embeddings)
    )
    assert await provider.embed("episode summary") == [0.1, 0.2, 0.3]
    assert embeddings.calls == [
        {
            "model": "text-embedding-test",
            "input": "episode summary",
            "dimensions": 3,
            "encoding_format": "float",
        }
    ]


async def test_openai_embedding_provider_rejects_wrong_dimension():
    provider = OpenAIEmbeddingProvider(
        "model", 3, client=SimpleNamespace(embeddings=FakeEmbeddings([0.1]))
    )
    with pytest.raises(EmbeddingError, match="Expected 3"):
        await provider.embed("episode summary")
