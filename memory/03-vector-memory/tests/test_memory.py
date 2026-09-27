from __future__ import annotations

from pathlib import Path

import pytest

from memory import MemoryValidationError, VectorMemoryManager
from models import MemoryType
from conftest import FakeEmbeddingProvider

pytestmark = pytest.mark.asyncio


async def test_add_and_get_round_trip(manager):
    result = await manager.add_memory(
        "Payment calls fraud service",
        memory_type="fact",
        source="incident-42",
        metadata={"service": "payment"},
        importance=4,
    )
    loaded = manager.get_memory(result.memory.id)
    assert result.created is True
    assert loaded == result.memory
    assert loaded.memory_type is MemoryType.FACT


async def test_exact_normalized_content_is_deduplicated(manager, provider):
    first = await manager.add_memory("Use Python 3.12")
    second = await manager.add_memory("  use   PYTHON 3.12  ")
    assert first.created is True and second.created is False
    assert second.memory.id == first.memory.id
    assert manager.count() == 1
    assert len(provider.calls) == 1


async def test_persists_across_manager_instances(tmp_path: Path, provider):
    path = tmp_path / "persistent"
    first = VectorMemoryManager(path, "persistent_test", provider)
    added = await first.add_memory("Durable incident note", memory_type="incident")
    second = VectorMemoryManager(path, "persistent_test", provider)
    assert second.get_memory(added.memory.id).content == "Durable incident note"


async def test_semantic_search_ranks_shared_terms(manager):
    await manager.add_memory(
        "checkout payment fraud latency incident", memory_type="incident"
    )
    await manager.add_memory("python formatting preference", memory_type="preference")
    results = await manager.search("checkout fraud latency", top_k=2)
    assert results[0].memory.memory_type is MemoryType.INCIDENT
    assert results[0].similarity_score > results[1].similarity_score


async def test_search_empty_collection_skips_embedding(manager, provider):
    assert await manager.search("anything") == []
    assert provider.calls == []


async def test_filter_by_memory_type(manager):
    await manager.add_memory("checkout latency", memory_type="incident")
    await manager.add_memory("checkout theme", memory_type="preference")
    results = await manager.search("checkout", filters={"memory_type": "preference"})
    assert [item.memory.memory_type for item in results] == [MemoryType.PREFERENCE]


async def test_filter_by_source_and_importance(manager):
    await manager.add_memory("important payment incident", source="pager", importance=5)
    await manager.add_memory("routine payment note", source="cli", importance=2)
    results = await manager.search(
        "payment", filters={"source": "pager", "importance": 5}
    )
    assert len(results) == 1 and results[0].memory.source == "pager"


async def test_filter_by_scalar_user_metadata(manager):
    await manager.add_memory(
        "fraud CPU saturation",
        metadata={"service": "fraud-service", "region": "ap-south-1"},
    )
    await manager.add_memory("payment timeout", metadata={"service": "payment-service"})
    results = await manager.search(
        "service problem", filters={"service": "fraud-service"}
    )
    assert (
        len(results) == 1 and results[0].memory.metadata["service"] == "fraud-service"
    )


async def test_min_similarity_threshold(manager):
    await manager.add_memory("checkout fraud latency")
    assert await manager.search("checkout fraud latency", min_similarity=0.99)
    assert await manager.search("unrelated banana", min_similarity=0.99) == []


async def test_update_content_reembeds_and_preserves_created_at(manager, provider):
    added = await manager.add_memory("old checkout note")
    before_calls = len(provider.calls)
    updated = await manager.update_memory(
        added.memory.id, content="new fraud note", importance=5
    )
    assert updated.created_at == added.memory.created_at
    assert updated.updated_at >= added.memory.updated_at
    assert updated.importance == 5
    assert len(provider.calls) == before_calls + 1


async def test_update_metadata_does_not_reembed(manager, provider):
    added = await manager.add_memory("stable content")
    before_calls = len(provider.calls)
    updated = await manager.update_memory(
        added.memory.id, metadata={"team": "payments"}
    )
    assert updated.metadata == {"team": "payments"}
    assert len(provider.calls) == before_calls


async def test_update_rejects_duplicate_content(manager):
    first = await manager.add_memory("first memory")
    second = await manager.add_memory("second memory")
    with pytest.raises(MemoryValidationError, match="same normalized content"):
        await manager.update_memory(second.memory.id, content=" FIRST  MEMORY ")


async def test_update_missing_returns_none(manager):
    assert await manager.update_memory("missing", content="valid") is None


async def test_delete_lifecycle(manager):
    added = await manager.add_memory("delete me")
    assert manager.delete_memory(added.memory.id) is True
    assert manager.delete_memory(added.memory.id) is False
    assert manager.count() == 0


async def test_clear_returns_deleted_count(manager):
    await manager.add_memory("one")
    await manager.add_memory("two")
    assert manager.clear() == 2
    assert manager.count() == 0


async def test_list_is_newest_first_and_filterable(manager):
    await manager.add_memory("fact one", memory_type="fact")
    latest = await manager.add_memory("incident two", memory_type="incident")
    assert manager.list_memories()[0].id == latest.memory.id
    assert len(manager.list_memories(memory_type="fact")) == 1


@pytest.mark.parametrize("content", ["", "   "])
async def test_empty_content_is_rejected(manager, content):
    with pytest.raises(MemoryValidationError, match="cannot be empty"):
        await manager.add_memory(content)


async def test_invalid_importance_and_type_are_rejected(manager):
    with pytest.raises(MemoryValidationError, match="importance"):
        await manager.add_memory("valid", importance=8)
    with pytest.raises(MemoryValidationError, match="memory_type"):
        await manager.add_memory("valid", memory_type="secret")


async def test_invalid_search_controls_are_rejected(manager):
    await manager.add_memory("valid")
    with pytest.raises(MemoryValidationError, match="top_k"):
        await manager.search("query", top_k=99)
    with pytest.raises(MemoryValidationError, match="min_similarity"):
        await manager.search("query", min_similarity=2.0)


async def test_nested_metadata_round_trips(manager):
    added = await manager.add_memory(
        "nested metadata",
        metadata={"labels": ["a", "b"], "owner": {"team": "platform"}},
    )
    loaded = manager.get_memory(added.memory.id)
    assert loaded.metadata == {"labels": ["a", "b"], "owner": {"team": "platform"}}
