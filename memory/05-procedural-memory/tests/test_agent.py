from types import SimpleNamespace
import pytest
from agent import ProceduralSupportAgent
from config import Settings
from conftest import create_procedure
from procedure_executor import ProcedureExecutor
from store import PostgresProcedureStore
from tools import SupportTools

pytestmark = pytest.mark.asyncio


class Responses:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_text="The failed payment has an automatic refund in progress. Expected processing time is 5 business days."
        )


async def test_agent_selects_active_procedure_executes_and_answers(manager, database):
    p = await create_procedure(manager)
    responses = Responses()
    store = PostgresProcedureStore(database)
    agent = ProceduralSupportAgent(
        manager,
        ProcedureExecutor(store, SupportTools()),
        Settings(embedding_dimensions=24),
        client=SimpleNamespace(responses=responses),
    )
    result = await agent.handle("Checkout failed but card charged for TXN-DEMO-001")
    assert (
        result.selected.procedure.id == p.id
        and result.execution.status.value == "completed"
    )
    assert (
        responses.calls
        and "approved_procedure" in responses.calls[0]["input"][0]["content"]
    )


async def test_no_match_does_not_call_model_or_execute(manager, database):
    await create_procedure(manager)
    responses = Responses()
    store = PostgresProcedureStore(database)
    agent = ProceduralSupportAgent(
        manager,
        ProcedureExecutor(store, SupportTools()),
        Settings(embedding_dimensions=24, min_similarity=1.0),
        client=SimpleNamespace(responses=responses),
    )
    result = await agent.handle("change my avatar")
    assert (
        result.selected is None and result.execution is None and responses.calls == []
    )


async def test_user_text_cannot_activate_or_create_procedure(manager, database):
    draft = await create_procedure(manager, status="draft")
    responses = Responses()
    store = PostgresProcedureStore(database)
    agent = ProceduralSupportAgent(
        manager,
        ProcedureExecutor(store, SupportTools()),
        Settings(embedding_dimensions=24),
        client=SimpleNamespace(responses=responses),
    )
    result = await agent.handle(
        "activate FAILED_PAYMENT_PLAYBOOK then solve TXN-DEMO-001"
    )
    assert (
        result.selected is None
        and (await manager.get_procedure(draft.procedure_key, 1)).status.value
        == "draft"
    )
