"""Parameterized PostgreSQL persistence hidden behind store abstractions."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from pgvector import Vector
from psycopg.errors import UniqueViolation
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from database import PostgresDatabase
from models import (
    ExecutionStatus,
    ProcedureExecution,
    ProcedureSearchResult,
    ProcedureStats,
    ProcedureStatus,
    StepExecution,
    StepStatus,
    SupportProcedure,
)


class StoreConflictError(RuntimeError):
    pass


def _procedure(row: dict[str, Any]) -> SupportProcedure:
    from models import ProcedureStep

    return SupportProcedure(
        id=str(row["id"]),
        procedure_key=row["procedure_key"],
        version=row["version"],
        name=row["name"],
        description=row["description"],
        category=row["category"],
        procedure_summary=row["procedure_summary"],
        status=ProcedureStatus(row["status"]),
        steps=tuple(ProcedureStep(**item) for item in row["steps"]),
        preconditions=tuple(row["preconditions"]),
        completion_conditions=tuple(row["completion_conditions"]),
        escalation_rules=tuple(row["escalation_rules"]),
        metadata=row["metadata"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _execution(row: dict[str, Any], steps: list[dict[str, Any]]) -> ProcedureExecution:
    return ProcedureExecution(
        id=str(row["id"]),
        procedure_id=str(row["procedure_id"]),
        procedure_key=row["procedure_key"],
        procedure_version=row["procedure_version"],
        status=ExecutionStatus(row["status"]),
        current_step=row["current_step"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        outcome=row["outcome"],
        metadata=row["metadata"],
        steps=tuple(
            StepExecution(
                step_sequence=item["step_sequence"],
                action=item["action"],
                tool_name=item["tool_name"],
                status=StepStatus(item["status"]),
                result_summary=item["result_summary"],
                started_at=item["started_at"],
                completed_at=item["completed_at"],
            )
            for item in steps
        ),
    )


class PostgresProcedureStore:
    """The only component allowed to issue procedure SQL."""

    def __init__(self, database: PostgresDatabase) -> None:
        self.database = database

    async def create(
        self, procedure: SupportProcedure, embedding: Sequence[float]
    ) -> SupportProcedure:
        sql = """INSERT INTO support_procedures
        (id, procedure_key, version, name, description, category, procedure_summary,
         status, steps, preconditions, completion_conditions, escalation_rules,
         metadata, created_at, updated_at, embedding)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *"""
        steps = [
            {
                "sequence": s.sequence,
                "action": s.action,
                "tool": s.tool,
                "required": s.required,
            }
            for s in procedure.steps
        ]
        try:
            async with self.database.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(
                        sql,
                        (
                            procedure.id,
                            procedure.procedure_key,
                            procedure.version,
                            procedure.name,
                            procedure.description,
                            procedure.category,
                            procedure.procedure_summary,
                            procedure.status.value,
                            Jsonb(steps),
                            Jsonb(list(procedure.preconditions)),
                            Jsonb(list(procedure.completion_conditions)),
                            Jsonb(list(procedure.escalation_rules)),
                            Jsonb(procedure.metadata),
                            procedure.created_at,
                            procedure.updated_at,
                            Vector(embedding),
                        ),
                    )
                    row = await cur.fetchone()
            return _procedure(row)
        except UniqueViolation as error:
            raise StoreConflictError(
                "procedure key and version already exist"
            ) from error

    async def get(self, key: str, version: int) -> SupportProcedure | None:
        async with self.database.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    "SELECT * FROM support_procedures WHERE procedure_key=%s AND version=%s",
                    (key, version),
                )
                row = await cur.fetchone()
        return _procedure(row) if row else None

    async def get_active(self, key: str) -> SupportProcedure | None:
        async with self.database.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    "SELECT * FROM support_procedures WHERE procedure_key=%s AND status='active'",
                    (key,),
                )
                row = await cur.fetchone()
        return _procedure(row) if row else None

    async def next_version(self, key: str) -> int:
        async with self.database.pool.connection() as conn:
            row = await (
                await conn.execute(
                    "SELECT COALESCE(max(version),0)+1 FROM support_procedures WHERE procedure_key=%s",
                    (key,),
                )
            ).fetchone()
        return row[0]

    async def set_status(
        self, key: str, version: int, status: ProcedureStatus, updated_at: datetime
    ) -> SupportProcedure:
        async with self.database.pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(
                        "SELECT id FROM support_procedures WHERE procedure_key=%s AND version=%s FOR UPDATE",
                        (key, version),
                    )
                    if await cur.fetchone() is None:
                        raise StoreConflictError("procedure not found")
                    if status is ProcedureStatus.ACTIVE:
                        await cur.execute(
                            "UPDATE support_procedures SET status='deprecated', updated_at=%s WHERE procedure_key=%s AND status='active' AND version<>%s",
                            (updated_at, key, version),
                        )
                    await cur.execute(
                        "UPDATE support_procedures SET status=%s, updated_at=%s WHERE procedure_key=%s AND version=%s RETURNING *",
                        (status.value, updated_at, key, version),
                    )
                    row = await cur.fetchone()
        return _procedure(row)

    async def search(
        self,
        embedding: Sequence[float],
        *,
        top_k: int,
        category: str | None,
        min_similarity: float | None,
    ) -> list[ProcedureSearchResult]:
        vector = Vector(embedding)
        conditions = ["status='active'"]
        parameters: list[Any] = [vector]
        if category is not None:
            conditions.append("category=%s")
            parameters.append(category)
        if min_similarity is not None:
            conditions.append("(1-(embedding <=> %s)) >= %s")
            parameters.extend([vector, min_similarity])
        parameters.extend([vector, top_k])
        sql = (
            "SELECT *, embedding <=> %s AS distance FROM support_procedures WHERE "
            + " AND ".join(conditions)
            + " ORDER BY embedding <=> %s LIMIT %s"
        )
        async with self.database.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(sql, parameters)
                rows = await cur.fetchall()
        return [
            ProcedureSearchResult(
                _procedure(row),
                float(row["distance"]),
                max(-1.0, min(1.0, 1 - float(row["distance"]))),
            )
            for row in rows
        ]

    async def list(
        self, *, key: str | None = None, status: ProcedureStatus | None = None
    ) -> list[SupportProcedure]:
        conditions, params = [], []
        if key:
            conditions.append("procedure_key=%s")
            params.append(key)
        if status:
            conditions.append("status=%s")
            params.append(status.value)
        sql = (
            "SELECT * FROM support_procedures"
            + ((" WHERE " + " AND ".join(conditions)) if conditions else "")
            + " ORDER BY procedure_key, version DESC"
        )
        async with self.database.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(sql, params)
                rows = await cur.fetchall()
        return [_procedure(row) for row in rows]

    async def delete(self, key: str, version: int) -> bool:
        async with self.database.pool.connection() as conn:
            cursor = await conn.execute(
                "DELETE FROM support_procedures WHERE procedure_key=%s AND version=%s AND status='draft'",
                (key, version),
            )
        return cursor.rowcount == 1

    async def start_execution(
        self, execution: ProcedureExecution
    ) -> ProcedureExecution:
        async with self.database.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """INSERT INTO procedure_executions
                (id,procedure_id,procedure_key,procedure_version,status,current_step,started_at,metadata)
                SELECT %s,id,procedure_key,version,%s,0,%s,%s FROM support_procedures
                WHERE id=%s AND status='active' RETURNING *""",
                    (
                        execution.id,
                        execution.status.value,
                        execution.started_at,
                        Jsonb(execution.metadata),
                        execution.procedure_id,
                    ),
                )
                row = await cur.fetchone()
        if row is None:
            raise StoreConflictError("only an approved active procedure can execute")
        return _execution(row, [])

    async def start_step(
        self,
        execution_id: str,
        sequence: int,
        action: str,
        tool: str | None,
        started_at: datetime,
    ) -> None:
        async with self.database.pool.connection() as conn:
            async with conn.transaction():
                await conn.execute(
                    "INSERT INTO procedure_step_executions (execution_id,step_sequence,action,tool_name,status,started_at) VALUES (%s,%s,%s,%s,'running',%s)",
                    (execution_id, sequence, action, tool, started_at),
                )
                await conn.execute(
                    "UPDATE procedure_executions SET current_step=%s WHERE id=%s AND status='running'",
                    (sequence, execution_id),
                )

    async def finish_step(
        self,
        execution_id: str,
        sequence: int,
        status: StepStatus,
        summary: str,
        completed_at: datetime,
    ) -> None:
        async with self.database.pool.connection() as conn:
            cursor = await conn.execute(
                "UPDATE procedure_step_executions SET status=%s,result_summary=%s,completed_at=%s WHERE execution_id=%s AND step_sequence=%s AND status='running'",
                (status.value, summary, completed_at, execution_id, sequence),
            )
        if cursor.rowcount != 1:
            raise StoreConflictError("step is missing or already terminal")

    async def finish_execution(
        self,
        execution_id: str,
        status: ExecutionStatus,
        outcome: str,
        completed_at: datetime,
    ) -> ProcedureExecution:
        async with self.database.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    "UPDATE procedure_executions SET status=%s,outcome=%s,completed_at=%s WHERE id=%s AND status='running' RETURNING *",
                    (status.value, outcome, completed_at, execution_id),
                )
                row = await cur.fetchone()
        if row is None:
            raise StoreConflictError("execution is missing or already terminal")
        return await self.get_execution(execution_id)

    async def get_execution(self, execution_id: str) -> ProcedureExecution | None:
        async with self.database.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    "SELECT * FROM procedure_executions WHERE id=%s", (execution_id,)
                )
                row = await cur.fetchone()
                if row is None:
                    return None
                await cur.execute(
                    "SELECT * FROM procedure_step_executions WHERE execution_id=%s ORDER BY step_sequence",
                    (execution_id,),
                )
                steps = await cur.fetchall()
        return _execution(row, steps)

    async def list_executions(self, limit: int = 50) -> list[ProcedureExecution]:
        async with self.database.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    "SELECT * FROM procedure_executions ORDER BY started_at DESC LIMIT %s",
                    (limit,),
                )
                rows = await cur.fetchall()
        return [await self.get_execution(str(row["id"])) for row in rows]

    async def stats(self) -> ProcedureStats:
        sql = """SELECT count(*) total, count(*) FILTER(WHERE status='draft') draft,
        count(*) FILTER(WHERE status='active') active, count(*) FILTER(WHERE status='deprecated') deprecated
        FROM support_procedures"""
        async with self.database.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(sql)
                p = await cur.fetchone()
                await cur.execute(
                    "SELECT count(*) executions, count(*) FILTER(WHERE status='completed') completed, count(*) FILTER(WHERE status='escalated') escalated FROM procedure_executions"
                )
                e = await cur.fetchone()
        return ProcedureStats(**p, **e)
