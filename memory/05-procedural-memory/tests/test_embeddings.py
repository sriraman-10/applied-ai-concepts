from types import SimpleNamespace
import pytest
from embeddings import EmbeddingError, OpenAIEmbeddingProvider

pytestmark = pytest.mark.asyncio


class Fake:
    async def create(self, **kwargs):
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])


async def test_openai_embedding_provider():
    p = OpenAIEmbeddingProvider("embed", 3, client=SimpleNamespace(embeddings=Fake()))
    assert await p.embed("procedure") == [0.1, 0.2, 0.3]


async def test_embedding_dimension_checked():
    p = OpenAIEmbeddingProvider("embed", 2, client=SimpleNamespace(embeddings=Fake()))
    with pytest.raises(EmbeddingError):
        await p.embed("procedure")
