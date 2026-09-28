import pytest
from conftest import create_procedure
from models import ExecutionStatus, ProcedureStatus
from procedure_executor import ExecutionError, ProcedureExecutor
from store import PostgresProcedureStore
from tools import SupportTools

pytestmark = pytest.mark.asyncio


async def test_execution_starts_tracks_steps_and_completes(manager, database):
    p = await create_procedure(manager)
    store = PostgresProcedureStore(database)
    execution, observations = await ProcedureExecutor(store, SupportTools()).execute(
        p, "failed for TXN-DEMO-001"
    )
    assert execution.status is ExecutionStatus.COMPLETED and len(
        execution.steps
    ) == len(p.steps) == len(observations)
    assert [s.step_sequence for s in execution.steps] == list(range(1, 7))


async def test_execution_escalates_without_transaction_id(manager, database):
    p = await create_procedure(manager)
    execution, _ = await ProcedureExecutor(
        PostgresProcedureStore(database), SupportTools()
    ).execute(p, "payment failed")
    assert execution.status is ExecutionStatus.ESCALATED


async def test_execution_escalates_inconsistent_state(manager, database):
    p = await create_procedure(manager)
    execution, _ = await ProcedureExecutor(
        PostgresProcedureStore(database), SupportTools()
    ).execute(p, "failed TXN-INCONSISTENT-004")
    assert execution.status is ExecutionStatus.ESCALATED


async def test_definition_not_mutated_by_execution(manager, database):
    p = await create_procedure(manager)
    before = await manager.get_procedure(p.procedure_key, 1)
    await ProcedureExecutor(PostgresProcedureStore(database), SupportTools()).execute(
        p, "failed TXN-DEMO-001"
    )
    assert await manager.get_procedure(p.procedure_key, 1) == before


async def test_draft_cannot_execute(manager, database):
    p = await create_procedure(manager, status=ProcedureStatus.DRAFT)
    with pytest.raises(ExecutionError):
        await ProcedureExecutor(
            PostgresProcedureStore(database), SupportTools()
        ).execute(p, "failed TXN-DEMO-001")


async def test_execution_persists(manager, database):
    p = await create_procedure(manager)
    store = PostgresProcedureStore(database)
    e, _ = await ProcedureExecutor(store, SupportTools()).execute(
        p, "failed TXN-DEMO-001"
    )
    assert await store.get_execution(e.id) == e
