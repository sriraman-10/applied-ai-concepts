import pytest
from conftest import create_procedure
from models import ProcedureStatus

pytestmark = pytest.mark.asyncio


async def test_new_version_is_draft_and_does_not_mutate_v1(manager):
    first = await create_procedure(manager)
    second = await manager.create_new_version(
        first.procedure_key, description="revised failed payment procedure"
    )
    assert second.version == 2 and second.status is ProcedureStatus.DRAFT
    assert (
        await manager.get_procedure(first.procedure_key, 1)
    ).description == first.description


async def test_only_one_version_active(manager):
    first = await create_procedure(manager)
    second = await manager.create_new_version(first.procedure_key)
    await manager.activate_procedure(second.procedure_key, 2)
    assert (
        await manager.get_procedure(first.procedure_key, 1)
    ).status is ProcedureStatus.DEPRECATED
    assert (await manager.get_active_version(first.procedure_key)).version == 2


async def test_deactivate_and_deprecate(manager):
    p = await create_procedure(manager)
    assert (
        await manager.deactivate_procedure(p.procedure_key, 1)
    ).status is ProcedureStatus.DRAFT
    assert (
        await manager.deprecate_procedure(p.procedure_key, 1)
    ).status is ProcedureStatus.DEPRECATED
