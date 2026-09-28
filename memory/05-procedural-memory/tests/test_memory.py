import pytest
from conftest import create_procedure
from memory import ProcedureValidationError, ProceduralMemoryManager
from store import PostgresProcedureStore

pytestmark = pytest.mark.asyncio


async def test_create_and_get_procedure(manager):
    created = await create_procedure(manager)
    assert await manager.get_procedure(created.procedure_key, 1) == created


async def test_persistence_across_manager_instances(manager, database):
    created = await create_procedure(manager)
    other = ProceduralMemoryManager(
        PostgresProcedureStore(database), manager.embedding_provider
    )
    assert (await other.get_procedure(created.procedure_key, 1)).id == created.id


async def test_ordered_jsonb_steps_preserved(manager):
    p = await create_procedure(manager)
    assert [s.sequence for s in p.steps] == list(range(1, 7))
    assert p.steps[2].tool == "check_gateway"


async def test_embedding_is_compact_summary_not_json(manager):
    p = await create_procedure(manager)
    text = manager.embedding_provider.calls[-1]
    assert (
        text == p.procedure_summary
        and "Important actions:" in text
        and '"sequence"' not in text
    )


async def test_duplicate_key_version_protected(manager):
    await create_procedure(manager)
    with pytest.raises(ProcedureValidationError, match="already exist"):
        await create_procedure(manager)


async def test_bad_steps_rejected(manager):
    with pytest.raises(ProcedureValidationError, match="consecutive"):
        await manager.create_procedure(
            procedure_key="BAD",
            name="Bad",
            description="Bad steps",
            category="test",
            steps=({"sequence": 2, "action": "x"},),
            completion_conditions=("done",),
            escalation_rules=("bad",),
            issue_patterns=("bad",),
        )


async def test_insert_rolls_back_on_wrong_vector_dimension(manager):
    manager.embedding_provider.dimensions = 3
    with pytest.raises(Exception):
        await create_procedure(manager)
    assert await manager.list_procedures() == []


async def test_delete_only_draft(manager):
    await create_procedure(manager)
    assert not await manager.delete_procedure("FAILED_PAYMENT_PLAYBOOK", 1)
    await manager.deactivate_procedure("FAILED_PAYMENT_PLAYBOOK", 1)
    assert await manager.delete_procedure("FAILED_PAYMENT_PLAYBOOK", 1)
