"""Offline tests for working-memory policy and model input construction."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from memory import MemoryItem, MemoryKind, MemoryManager, MemoryRole, Priority


class WordEstimator:
    def estimate(self, text: str) -> int:
        return len(text.split())

    def truncate(self, text: str, max_tokens: int) -> str:
        return " ".join(text.split()[:max_tokens])


class FakeSummarizer:
    def __init__(self, outputs: list[str] | None = None) -> None:
        self.outputs = list(outputs or ["compacted useful state"])
        self.calls: list[tuple[str | None, tuple[MemoryItem, ...]]] = []

    async def __call__(
        self, existing_summary: str | None, items: Sequence[MemoryItem]
    ) -> str:
        self.calls.append((existing_summary, tuple(items)))
        if len(self.outputs) > 1:
            return self.outputs.pop(0)
        return self.outputs[0]


def manager(
    *,
    hard: int = 100,
    target: int = 60,
    keep: int = 2,
    summarizer=None,
) -> MemoryManager:
    return MemoryManager(
        estimator=WordEstimator(),
        hard_token_limit=hard,
        target_token_limit=target,
        keep_recent_items=keep,
        summarizer=summarizer,
    )


async def add_user(memory: MemoryManager, text: str) -> MemoryItem:
    return await memory.add(
        role=MemoryRole.USER,
        content=text,
        kind=MemoryKind.USER_MESSAGE,
    )


@pytest.mark.asyncio
async def test_adding_memory_records_metadata_and_tokens() -> None:
    memory = manager()
    item = await add_user(memory, "remember this fact")
    assert item in memory.items
    assert item.kind is MemoryKind.USER_MESSAGE
    assert item.priority is Priority.NORMAL
    assert item.estimated_tokens == 3
    assert item.timestamp.tzinfo is not None
    assert memory.stats.item_count == 1


@pytest.mark.asyncio
async def test_token_accounting_includes_item_and_summary_overhead() -> None:
    memory = manager()
    await add_user(memory, "three useful words")
    assert memory.estimated_tokens == 7  # three content tokens plus message overhead
    assert memory.stats.summary_tokens == 0


@pytest.mark.asyncio
async def test_low_priority_tool_observation_is_evicted_first() -> None:
    memory = manager(hard=40, target=33, keep=2)
    noise = await memory.add(
        role=MemoryRole.TOOL,
        content="health check returned two hundred",
        kind=MemoryKind.TOOL_OBSERVATION,
        priority=Priority.LOW,
    )
    for index in range(4):
        await add_user(memory, f"useful item number {index}")
    assert noise not in memory.items
    assert memory.stats.compaction_count == 1
    assert memory.last_report is not None
    assert memory.last_report.items_dropped == 1


@pytest.mark.asyncio
async def test_recent_items_are_protected_during_summary_compaction() -> None:
    summarizer = FakeSummarizer(["state"])
    memory = manager(hard=30, target=20, keep=2, summarizer=summarizer)
    added = [await add_user(memory, f"message number {index}") for index in range(5)]
    remaining_ids = {item.id for item in memory.items}
    assert added[-1].id in remaining_ids
    assert added[-2].id in remaining_ids
    assert all(item.id not in remaining_ids for item in added[:-2])


@pytest.mark.asyncio
async def test_compaction_triggers_only_at_hard_limit() -> None:
    summarizer = FakeSummarizer()
    memory = manager(hard=30, target=20, keep=2, summarizer=summarizer)
    for index in range(4):
        await add_user(memory, f"message number {index}")
    assert memory.stats.compaction_count == 0
    await add_user(memory, "message number four")
    assert memory.stats.compaction_count == 1


@pytest.mark.asyncio
async def test_repeated_compaction_merges_existing_summary() -> None:
    summarizer = FakeSummarizer(["first compact summary", "merged compact summary"])
    memory = manager(hard=50, target=35, keep=2, summarizer=summarizer)
    for index in range(6):
        await add_user(memory, f"useful message number {index} here")
    assert memory.summary == "first compact summary"
    for index in range(6, 8):
        await add_user(memory, f"useful message number {index} here")
    assert len(summarizer.calls) == 2
    assert summarizer.calls[1][0] == "first compact summary"
    assert memory.summary == "merged compact summary"


def test_target_limit_must_be_lower_than_hard_limit() -> None:
    with pytest.raises(ValueError, match="lower than"):
        manager(hard=20, target=20)
    with pytest.raises(ValueError, match="lower than"):
        manager(hard=20, target=21)


@pytest.mark.asyncio
async def test_model_input_maps_observable_memory_to_supported_roles() -> None:
    memory = manager(hard=200, target=100)
    await add_user(memory, "What is the incident status?")
    await memory.add(
        role=MemoryRole.TOOL,
        content="checkout p95 is high",
        kind=MemoryKind.TOOL_OBSERVATION,
        priority=Priority.HIGH,
    )
    await memory.add(
        role=MemoryRole.SYSTEM,
        content="Investigate payment next",
        kind=MemoryKind.DECISION,
        priority=Priority.HIGH,
    )
    await memory.add(
        role=MemoryRole.ASSISTANT,
        content="I will inspect payment.",
        kind=MemoryKind.ASSISTANT_MESSAGE,
    )
    model_input = memory.build_model_input()
    assert [message["role"] for message in model_input] == [
        "user",
        "developer",
        "developer",
        "assistant",
    ]
    assert model_input[1]["content"].startswith("[Tool observation]")
    assert model_input[2]["content"].startswith("[Recorded decision]")


@pytest.mark.asyncio
async def test_hard_eviction_bounds_one_oversized_recent_message() -> None:
    memory = manager(hard=10, target=5, keep=1)
    await add_user(memory, "one two three four five six seven eight nine ten")
    assert memory.estimated_tokens <= 5
    assert memory.items == ()
    assert memory.last_report is not None
    assert memory.last_report.items_dropped == 1


@pytest.mark.asyncio
async def test_oversized_summary_is_removed_before_recent_items() -> None:
    summarizer = FakeSummarizer([" ".join(["summary"] * 100)])
    memory = manager(hard=30, target=14, keep=2, summarizer=summarizer)
    added = [await add_user(memory, f"message number {index}") for index in range(5)]
    assert [item.id for item in memory.items] == [added[-2].id, added[-1].id]
    assert memory.summary is None
    assert memory.estimated_tokens == 14


@pytest.mark.asyncio
async def test_compaction_observer_receives_report() -> None:
    reports = []
    memory = MemoryManager(
        estimator=WordEstimator(),
        hard_token_limit=10,
        target_token_limit=5,
        keep_recent_items=1,
        on_compaction=reports.append,
    )
    await add_user(memory, "one two three four five six seven eight nine ten")
    assert reports == [memory.last_report]
    assert reports[0].after_tokens <= 5
