from __future__ import annotations

import hashlib
import math
import os
import re
from pathlib import Path

import pytest_asyncio

from database import PostgresDatabase
from memory import EpisodicMemoryManager
from store import PostgresEpisodeStore


class FakeEmbeddingProvider:
    def __init__(self, dimensions: int = 24) -> None:
        self.dimensions = dimensions
        self.calls: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        vector = [0.0] * self.dimensions
        for token in re.findall(r"[a-z0-9]+", text.casefold()):
            digest = hashlib.sha256(token.encode()).digest()
            vector[digest[0] % self.dimensions] += 1.0
            vector[digest[1] % self.dimensions] += 0.25
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


@pytest_asyncio.fixture(scope="session")
async def database():
    url = os.getenv(
        "TEST_DATABASE_URL",
        "postgresql://episodic:episodic_dev@localhost:5432/episodic_memory_test",
    )
    db = PostgresDatabase(url, 24, Path(__file__).parents[1] / "migrations")
    await db.open()
    yield db
    await db.close()


@pytest_asyncio.fixture
async def manager(database):
    async with database.pool.connection() as connection:
        await connection.execute("TRUNCATE support_episodes RESTART IDENTITY")
        await connection.execute(
            "ALTER SEQUENCE support_case_number_seq RESTART WITH 1"
        )
    provider = FakeEmbeddingProvider()
    return EpisodicMemoryManager(PostgresEpisodeStore(database), provider, max_top_k=10)


async def completed_episode(
    manager,
    description="Payment failed but money was deducted",
    *,
    issue_type="payment_failure",
    customer_id=None,
    outcome="resolved",
    episode_key=None,
):
    episode = await manager.start_episode(
        description,
        issue_type=issue_type,
        customer_id=customer_id,
        episode_key=episode_key,
    )
    await manager.record_action(
        episode.episode_key,
        action_type="tool_call",
        tool_name="check_transaction",
        result_summary="transaction failed",
    )
    await manager.record_observation(
        episode.episode_key,
        source="check_transaction",
        content="transaction failed; amount deducted",
    )
    return await manager.complete_episode(
        episode.episode_key,
        resolution="Confirmed automatic refund",
        outcome=outcome,
    )
