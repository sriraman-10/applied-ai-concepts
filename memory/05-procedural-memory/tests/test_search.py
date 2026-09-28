import pytest
from conftest import create_procedure
from models import ProcedureStatus

pytestmark = pytest.mark.asyncio


async def test_semantic_search_and_query_embedding(manager):
    target = await create_procedure(manager)
    await create_procedure(
        manager,
        "ADDRESS_PLAYBOOK",
        description="change profile shipping address",
        category="account",
    )
    results = await manager.search_procedures("checkout failed card charged", top_k=2)
    assert (
        results[0].procedure.id == target.id
        and manager.embedding_provider.calls[-1] == "checkout failed card charged"
    )


async def test_top_k(manager):
    for i in range(3):
        await create_procedure(
            manager, f"PAYMENT_{i}", description=f"payment failure variation {i}"
        )
    assert len(await manager.search_procedures("payment failure", top_k=2)) == 2


async def test_draft_and_deprecated_not_retrieved(manager):
    await create_procedure(manager, "DRAFT", ProcedureStatus.DRAFT)
    await create_procedure(manager, "OLD", ProcedureStatus.DEPRECATED)
    assert await manager.search_procedures("failed checkout payment") == []


async def test_category_filter(manager):
    await create_procedure(manager)
    refund = await create_procedure(
        manager, "REFUND", category="refunds", description="refund pending delay"
    )
    results = await manager.search_procedures("refund pending", category="refunds")
    assert [r.procedure.id for r in results] == [refund.id]


async def test_minimum_similarity_no_match(manager):
    await create_procedure(manager)
    assert (
        await manager.search_procedures("unrelated profile", min_similarity=1.0) == []
    )


async def test_search_validation(manager):
    with pytest.raises(Exception, match="top_k"):
        await manager.search_procedures("x", top_k=99)
