from __future__ import annotations

import hashlib
import math
import os
import re
from pathlib import Path

import pytest_asyncio

from database import PostgresDatabase
from memory import ProceduralMemoryManager
from models import ProcedureStatus
from store import PostgresProcedureStore


class FakeEmbeddingProvider:
    def __init__(self, dimensions: int = 24):
        self.dimensions = dimensions
        self.calls = []

    async def embed(self, text: str):
        self.calls.append(text)
        vector = [0.0] * self.dimensions
        for token in re.findall(r"[a-z0-9]+", text.casefold()):
            digest = hashlib.sha256(token.encode()).digest()
            vector[digest[0] % self.dimensions] += 1
            vector[digest[1] % self.dimensions] += 0.25
        norm = math.sqrt(sum(v * v for v in vector)) or 1
        return [v / norm for v in vector]


@pytest_asyncio.fixture(scope="session")
async def database():
    url = os.getenv(
        "TEST_DATABASE_URL",
        "postgresql://procedural:procedural_dev@localhost:5433/procedural_memory_test",
    )
    db = PostgresDatabase(url, 24, Path(__file__).parents[1] / "migrations")
    await db.open()
    yield db
    await db.close()


@pytest_asyncio.fixture
async def manager(database):
    async with database.pool.connection() as conn:
        await conn.execute(
            "TRUNCATE procedure_step_executions, procedure_executions, support_procedures CASCADE"
        )
    provider = FakeEmbeddingProvider()
    return ProceduralMemoryManager(
        PostgresProcedureStore(database), provider, max_top_k=10
    )


async def create_procedure(
    manager,
    key="FAILED_PAYMENT_PLAYBOOK",
    status=ProcedureStatus.ACTIVE,
    version=1,
    category="payments",
    description="failed checkout payment and charged card",
):
    return await manager.create_procedure(
        procedure_key=key,
        version=version,
        name=key.replace("_", " ").title(),
        description=description,
        category=category,
        issue_patterns=(description,),
        steps=(
            {"action": "identify_transaction"},
            {"action": "check_transaction", "tool": "check_transaction"},
            {"action": "check_gateway", "tool": "check_gateway"},
            {"action": "check_refund", "tool": "check_refund"},
            {"action": "prevent_duplicate_refund"},
            {"action": "notify_customer", "tool": "notify_customer"},
        ),
        preconditions=("transaction id supplied",),
        completion_conditions=("status known", "customer informed"),
        escalation_rules=("inconsistent transaction",),
        status=status,
        metadata={"approved_by": "test-admin"},
    )
