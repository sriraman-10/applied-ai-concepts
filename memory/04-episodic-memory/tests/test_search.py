from __future__ import annotations

import pytest

from memory import EpisodeValidationError
from models import EpisodeOutcome
from conftest import completed_episode

pytestmark = pytest.mark.asyncio


async def test_incomplete_episode_not_retrieved(manager):
    await manager.start_episode("payment card charged", issue_type="payment_failure")
    assert await manager.search_similar("payment card charged") == []


async def test_semantic_search_orders_matching_episode_first(manager):
    matching = await completed_episode(manager, "checkout payment failed card charged")
    await completed_episode(
        manager, "account profile address update", issue_type="account"
    )
    results = await manager.search_similar("checkout card charged", top_k=2)
    assert results[0].episode.id == matching.id
    assert results[0].similarity_score > results[1].similarity_score


async def test_top_k(manager):
    for index in range(3):
        await completed_episode(manager, f"payment failed case {index}")
    assert len(await manager.search_similar("payment failed", top_k=2)) == 2


async def test_issue_type_filter(manager):
    await completed_episode(manager, "payment problem", issue_type="payment_failure")
    account = await completed_episode(manager, "account problem", issue_type="account")
    results = await manager.search_similar("problem", issue_type="account")
    assert [item.episode.id for item in results] == [account.id]


async def test_customer_filter(manager):
    target = await completed_episode(
        manager, "payment failed", customer_id="customer-a"
    )
    await completed_episode(manager, "payment failed", customer_id="customer-b")
    results = await manager.search_similar("payment failed", customer_id="customer-a")
    assert [item.episode.id for item in results] == [target.id]


async def test_default_search_only_uses_resolved(manager):
    resolved = await completed_episode(manager, "payment failed", outcome="resolved")
    await completed_episode(manager, "payment failed", outcome="failed")
    results = await manager.search_similar("payment failed")
    assert [item.episode.id for item in results] == [resolved.id]


async def test_failed_episode_can_be_requested_explicitly(manager):
    failed = await completed_episode(manager, "payment failed", outcome="failed")
    results = await manager.search_similar(
        "payment failed", outcomes=(EpisodeOutcome.FAILED,)
    )
    assert [item.episode.id for item in results] == [failed.id]


async def test_minimum_similarity_filters_results(manager):
    await completed_episode(manager, "checkout payment failed")
    unfiltered = await manager.search_similar("checkout payment failed")
    score = unfiltered[0].similarity_score

    included = await manager.search_similar(
        "checkout payment failed", min_similarity=score - 0.001
    )
    excluded = await manager.search_similar(
        "checkout payment failed", min_similarity=min(1.0, score + 0.001)
    )

    assert included
    assert excluded == []


async def test_query_embedding_and_episode_summary_embedding_are_distinct(manager):
    await completed_episode(manager, "checkout payment failed")
    completion_input = manager.embedding_provider.calls[-1]
    await manager.search_similar("my card was charged")
    query_input = manager.embedding_provider.calls[-1]
    assert completion_input.startswith("Issue:")
    assert query_input == "my card was charged"


async def test_search_controls_are_validated(manager):
    with pytest.raises(EpisodeValidationError, match="top_k"):
        await manager.search_similar("query", top_k=99)
    with pytest.raises(EpisodeValidationError, match="min_similarity"):
        await manager.search_similar("query", min_similarity=2)
