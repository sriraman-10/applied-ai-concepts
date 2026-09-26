"""SQLite KV-memory lifecycle and isolation tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from memory import KVMemoryManager
from models import ValueType


class MutableClock:
    def __init__(self) -> None:
        self.current = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.current

    def advance(self, *, seconds: int) -> None:
        self.current += timedelta(seconds=seconds)


@pytest.fixture
def clock() -> MutableClock:
    return MutableClock()


@pytest.fixture
def memory(tmp_path, clock) -> KVMemoryManager:
    manager = KVMemoryManager(tmp_path / "memory.db", clock=clock)
    yield manager
    manager.close()


def test_set_and_get(memory: KVMemoryManager) -> None:
    stored = memory.set("project", "memory-series", "python_version", "3.12")
    loaded = memory.get("project", "memory-series", "python_version")
    assert loaded == stored
    assert loaded.value_type is ValueType.STRING


def test_persistence_across_manager_restart(tmp_path, clock: MutableClock) -> None:
    path = tmp_path / "persistent.db"
    first = KVMemoryManager(path, clock=clock)
    first.set("user", "123", "preferred_language", "Java")
    first.close()
    second = KVMemoryManager(path, clock=clock)
    try:
        assert second.get("user", "123", "preferred_language").value == "Java"
    finally:
        second.close()


def test_update_preserves_creation_and_changes_update_time(
    memory: KVMemoryManager, clock: MutableClock
) -> None:
    original = memory.set("project", "memory-series", "test_framework", "unittest")
    clock.advance(seconds=5)
    updated = memory.set("project", "memory-series", "test_framework", "pytest")
    assert updated.id == original.id
    assert updated.created_at == original.created_at
    assert updated.updated_at > original.updated_at
    assert updated.value == "pytest"


def test_delete(memory: KVMemoryManager) -> None:
    memory.set("user", "123", "timezone", "Asia/Kolkata")
    assert memory.delete("user", "123", "timezone") is True
    assert memory.delete("user", "123", "timezone") is False
    assert memory.get("user", "123", "timezone") is None


def test_exists(memory: KVMemoryManager) -> None:
    assert memory.exists("service", "checkout", "region") is False
    memory.set("service", "checkout", "region", "ap-south-1")
    assert memory.exists("service", "checkout", "region") is True


def test_list_is_limited_to_namespace_and_entity(memory: KVMemoryManager) -> None:
    memory.set("project", "alpha", "language", "Python")
    memory.set("project", "alpha", "tests", "pytest")
    memory.set("project", "beta", "language", "Java")
    memory.set("user", "alpha", "language", "Go")
    assert [item.key for item in memory.list("project", "alpha")] == ["language", "tests"]


def test_namespace_and_entity_isolation(memory: KVMemoryManager) -> None:
    memory.set("user", "123", "language", "Java")
    memory.set("project", "123", "language", "Python")
    memory.set("user", "456", "language", "Go")
    assert memory.get("user", "123", "language").value == "Java"
    assert memory.get("project", "123", "language").value == "Python"
    assert memory.get("user", "456", "language").value == "Go"


def test_ttl_expiration_behaves_as_not_found(
    memory: KVMemoryManager, clock: MutableClock
) -> None:
    memory.set("deployment", "release-1", "region", "temporary", ttl_seconds=60)
    clock.advance(seconds=60)
    assert memory.get("deployment", "release-1", "region") is None
    assert memory.exists("deployment", "release-1", "region") is False
    assert memory.list("deployment", "release-1") == []


def test_cleanup_expired(memory: KVMemoryManager, clock: MutableClock) -> None:
    memory.set("deployment", "one", "temporary", True, ttl_seconds=10)
    memory.set("deployment", "one", "permanent", True)
    clock.advance(seconds=11)
    assert memory.cleanup_expired() == 1
    assert memory.cleanup_expired() == 0
    assert memory.get("deployment", "one", "permanent") is not None


@pytest.mark.parametrize(
    ("value", "expected_type"),
    [
        ("Python", ValueType.STRING),
        (3.12, ValueType.NUMBER),
        (True, ValueType.BOOLEAN),
        (["pytest", "ruff"], ValueType.LIST),
        ({"version": 3, "strict": True}, ValueType.OBJECT),
        (None, ValueType.NULL),
    ],
)
def test_json_values_round_trip(memory, value, expected_type) -> None:
    loaded = memory.set("project", "json", expected_type.value, value)
    assert loaded.value == value
    assert loaded.value_type is expected_type


def test_duplicate_logical_key_does_not_create_duplicate(memory: KVMemoryManager) -> None:
    memory.set("project", "one", "language", "Python")
    memory.set("project", "one", "language", "Java")
    assert len(memory.list("project", "one")) == 1
    assert memory.stats().active_entries == 1


def test_sql_injection_like_strings_are_data(memory: KVMemoryManager) -> None:
    namespace = "project'; DROP TABLE kv_memories; --"
    entity = "entity' OR 1=1 --"
    key = "key'); DELETE FROM kv_memories; --"
    value = "value'; DROP TABLE kv_memories; --"
    memory.set(namespace, entity, key, value)
    assert memory.get(namespace, entity, key).value == value
    memory.set("safe", "entity", "key", "still works")
    assert memory.get("safe", "entity", "key").value == "still works"


def test_unknown_key_returns_none(memory: KVMemoryManager) -> None:
    assert memory.get("project", "missing", "unknown") is None


def test_clear_one_entity_then_whole_namespace(memory: KVMemoryManager) -> None:
    memory.set("project", "alpha", "language", "Python")
    memory.set("project", "alpha", "tests", "pytest")
    memory.set("project", "beta", "language", "Java")
    memory.set("user", "alpha", "language", "Go")
    assert memory.clear_namespace("project", "alpha") == 2
    assert len(memory.list("project", "beta")) == 1
    assert memory.clear_namespace("project") == 1
    assert len(memory.list("user", "alpha")) == 1


def test_stats_separate_active_and_expired(
    memory: KVMemoryManager, clock: MutableClock
) -> None:
    memory.set("project", "alpha", "active", True)
    memory.set("project", "alpha", "expired", True, ttl_seconds=1)
    clock.advance(seconds=2)
    stats = memory.stats()
    assert stats.active_entries == 1
    assert stats.expired_entries == 1
    assert stats.namespaces == 1
    assert stats.entities == 1
