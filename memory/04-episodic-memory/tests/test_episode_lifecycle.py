from __future__ import annotations

import pytest

from memory import EpisodeValidationError
from models import EpisodeStatus
from conftest import completed_episode

pytestmark = pytest.mark.asyncio


async def test_start_episode(manager):
    episode = await manager.start_episode("Card charged", issue_type="payment_failure")
    assert episode.episode_key == "CASE-001"
    assert episode.status is EpisodeStatus.OPEN
    assert episode.episode_summary is None


async def test_duplicate_case_id_rejected(manager):
    await manager.start_episode(
        "First", issue_type="general", episode_key="CASE-CUSTOM"
    )
    with pytest.raises(EpisodeValidationError, match="already exists"):
        await manager.start_episode(
            "Second", issue_type="general", episode_key="CASE-CUSTOM"
        )


async def test_actions_and_observations_preserve_order(manager):
    episode = await manager.start_episode("Issue", issue_type="general")
    for index in range(3):
        await manager.record_action(episode.episode_key, action_type=f"action-{index}")
        await manager.record_observation(
            episode.episode_key, source="tool", content=f"result-{index}"
        )
    loaded = await manager.get_episode(episode.episode_key)
    assert [item.sequence for item in loaded.actions] == [1, 2, 3]
    assert [item.action_type for item in loaded.actions] == [
        "action-0",
        "action-1",
        "action-2",
    ]
    assert [item.sequence for item in loaded.observations] == [1, 2, 3]


async def test_completion_builds_summary_and_embedding(manager):
    episode = await completed_episode(manager)
    assert episode.status is EpisodeStatus.COMPLETED
    assert "Issue: Payment failed" in episode.episode_summary
    assert "transaction failed" in episode.episode_summary
    assert "Resolution: Confirmed automatic refund" in episode.episode_summary
    assert episode.completed_at is not None
    assert len(manager.embedding_provider.calls) == 1
    assert manager.embedding_provider.calls[0] == episode.episode_summary


async def test_completion_requires_action(manager):
    episode = await manager.start_episode("Issue", issue_type="general")
    await manager.record_observation(
        episode.episode_key, source="user", content="observed"
    )
    with pytest.raises(EpisodeValidationError, match="action"):
        await manager.complete_episode(
            episode.episode_key, resolution="Done", outcome="resolved"
        )


async def test_completion_requires_observation(manager):
    episode = await manager.start_episode("Issue", issue_type="general")
    await manager.record_action(episode.episode_key, action_type="check")
    with pytest.raises(EpisodeValidationError, match="observation"):
        await manager.complete_episode(
            episode.episode_key, resolution="Done", outcome="resolved"
        )


async def test_completion_requires_resolution(manager):
    episode = await manager.start_episode("Issue", issue_type="general")
    await manager.record_action(episode.episode_key, action_type="check")
    await manager.record_observation(episode.episode_key, source="tool", content="seen")
    with pytest.raises(EpisodeValidationError, match="resolution"):
        await manager.complete_episode(
            episode.episode_key, resolution=" ", outcome="resolved"
        )


async def test_completed_episode_cannot_be_changed_or_completed_twice(manager):
    episode = await completed_episode(manager)
    with pytest.raises(EpisodeValidationError, match="cannot be changed"):
        await manager.record_action(episode.episode_key, action_type="late")
    with pytest.raises(EpisodeValidationError, match="cannot be changed"):
        await manager.complete_episode(
            episode.episode_key, resolution="Again", outcome="resolved"
        )


async def test_get_by_case_id_and_delete(manager):
    episode = await completed_episode(manager)
    assert (await manager.get_episode(episode.episode_key)).id == episode.id
    assert await manager.delete_episode(episode.episode_key) is True
    assert await manager.delete_episode(episode.episode_key) is False


async def test_completion_rolls_back_when_vector_dimension_is_wrong(manager):
    episode = await manager.start_episode("Issue", issue_type="general")
    await manager.record_action(episode.episode_key, action_type="check")
    await manager.record_observation(episode.episode_key, source="tool", content="seen")
    manager.embedding_provider.dimensions = 3
    with pytest.raises(Exception):
        await manager.complete_episode(
            episode.episode_key, resolution="Done", outcome="resolved"
        )
    loaded = await manager.get_episode(episode.episode_key)
    assert loaded.status is EpisodeStatus.OPEN
    assert loaded.episode_summary is None
