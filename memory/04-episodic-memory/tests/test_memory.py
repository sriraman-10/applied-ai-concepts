from __future__ import annotations

import pytest

from models import EpisodeStatus
from store import PostgresEpisodeStore
from conftest import completed_episode

pytestmark = pytest.mark.asyncio


async def test_episode_persists_for_new_store_instance(manager, database):
    episode = await completed_episode(manager)
    new_store = PostgresEpisodeStore(database)
    loaded = await new_store.get(episode.episode_key)
    assert loaded == episode


async def test_list_and_stats(manager):
    await manager.start_episode("open issue", issue_type="general")
    await completed_episode(manager, "resolved issue")
    await completed_episode(manager, "escalated issue", outcome="escalated")
    episodes = await manager.list_episodes()
    stats = await manager.stats()
    assert len(episodes) == 3
    assert stats.total == 3
    assert stats.open == 1
    assert stats.completed == 2
    assert stats.resolved == 1
    assert stats.escalated == 1


async def test_list_can_filter_status(manager):
    await manager.start_episode("open issue", issue_type="general")
    completed = await completed_episode(manager)
    episodes = await manager.list_episodes(status=EpisodeStatus.COMPLETED)
    assert [item.id for item in episodes] == [completed.id]
