"""Episode persistence contract and PostgreSQL implementation."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any, Protocol

from pgvector import Vector
from psycopg.errors import UniqueViolation
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from database import PostgresDatabase
from models import (
    Episode,
    EpisodeAction,
    EpisodeObservation,
    EpisodeOutcome,
    EpisodeSearchResult,
    EpisodeStats,
    EpisodeStatus,
)


class StoreConflictError(RuntimeError):
    pass


class EpisodeStore(Protocol):
    async def next_episode_key(self) -> str: ...
    async def create(self, episode: Episode) -> Episode: ...
    async def get(self, episode_key: str) -> Episode | None: ...
    async def append_action(
        self, episode_key: str, action: EpisodeAction
    ) -> Episode: ...
    async def append_observation(
        self, episode_key: str, observation: EpisodeObservation
    ) -> Episode: ...
    async def complete(
        self,
        episode_key: str,
        *,
        summary: str,
        resolution: str,
        outcome: EpisodeOutcome,
        completed_at: datetime,
        embedding: Sequence[float],
    ) -> Episode: ...
    async def search(
        self,
        embedding: Sequence[float],
        *,
        top_k: int,
        issue_type: str | None,
        customer_id: str | None,
        outcomes: tuple[EpisodeOutcome, ...],
        min_similarity: float | None,
    ) -> list[EpisodeSearchResult]: ...
    async def list(
        self, *, status: EpisodeStatus | None = None, limit: int = 100
    ) -> list[Episode]: ...
    async def delete(self, episode_key: str) -> bool: ...
    async def stats(self) -> EpisodeStats: ...


def action_payload(action: EpisodeAction) -> dict[str, Any]:
    return {
        "sequence": action.sequence,
        "action_type": action.action_type,
        "tool_name": action.tool_name,
        "input_summary": action.input_summary,
        "result_summary": action.result_summary,
        "created_at": action.created_at.isoformat(),
    }


def observation_payload(observation: EpisodeObservation) -> dict[str, Any]:
    return {
        "sequence": observation.sequence,
        "source": observation.source,
        "content": observation.content,
        "important": observation.important,
        "created_at": observation.created_at.isoformat(),
    }


def _episode_from_row(row: dict[str, Any]) -> Episode:
    return Episode(
        id=str(row["id"]),
        episode_key=row["episode_key"],
        customer_id=row["customer_id"],
        issue_type=row["issue_type"],
        description=row["description"],
        episode_summary=row["episode_summary"],
        actions=tuple(
            EpisodeAction(
                sequence=item["sequence"],
                action_type=item["action_type"],
                tool_name=item.get("tool_name"),
                input_summary=item.get("input_summary"),
                result_summary=item.get("result_summary"),
                created_at=datetime.fromisoformat(item["created_at"]),
            )
            for item in row["actions"]
        ),
        observations=tuple(
            EpisodeObservation(
                sequence=item["sequence"],
                source=item["source"],
                content=item["content"],
                important=item["important"],
                created_at=datetime.fromisoformat(item["created_at"]),
            )
            for item in row["observations"]
        ),
        resolution=row["resolution"],
        outcome=EpisodeOutcome(row["outcome"]) if row["outcome"] else None,
        status=EpisodeStatus(row["status"]),
        metadata=row["metadata"],
        created_at=row["created_at"],
        completed_at=row["completed_at"],
    )


class PostgresEpisodeStore:
    """Parameterized SQL adapter. The LLM never receives this interface."""

    def __init__(self, database: PostgresDatabase) -> None:
        self.database = database

    async def next_episode_key(self) -> str:
        async with self.database.pool.connection() as connection:
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    "SELECT nextval('support_case_number_seq') AS number"
                )
                row = await cursor.fetchone()
        return f"CASE-{row['number']:03d}"

    async def create(self, episode: Episode) -> Episode:
        sql = """INSERT INTO support_episodes (
            id, episode_key, customer_id, issue_type, description, episode_summary,
            actions, observations, resolution, outcome, status, metadata,
            created_at, completed_at, embedding
        ) VALUES (%s, %s, %s, %s, %s, NULL, %s, %s, NULL, NULL, %s, %s, %s, NULL, NULL)
        RETURNING *"""
        try:
            async with self.database.pool.connection() as connection:
                async with connection.cursor(row_factory=dict_row) as cursor:
                    await cursor.execute(
                        sql,
                        (
                            episode.id,
                            episode.episode_key,
                            episode.customer_id,
                            episode.issue_type,
                            episode.description,
                            Jsonb([]),
                            Jsonb([]),
                            episode.status.value,
                            Jsonb(episode.metadata),
                            episode.created_at,
                        ),
                    )
                    row = await cursor.fetchone()
            return _episode_from_row(row)
        except UniqueViolation as error:
            raise StoreConflictError("episode_key already exists") from error

    async def get(self, episode_key: str) -> Episode | None:
        async with self.database.pool.connection() as connection:
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    "SELECT * FROM support_episodes WHERE episode_key = %s",
                    (episode_key,),
                )
                row = await cursor.fetchone()
        return _episode_from_row(row) if row else None

    async def append_action(self, episode_key: str, action: EpisodeAction) -> Episode:
        return await self._append_json(
            episode_key, "actions", action.sequence, action_payload(action)
        )

    async def append_observation(
        self, episode_key: str, observation: EpisodeObservation
    ) -> Episode:
        return await self._append_json(
            episode_key,
            "observations",
            observation.sequence,
            observation_payload(observation),
        )

    async def _append_json(
        self,
        episode_key: str,
        column: str,
        expected_sequence: int,
        payload: dict[str, Any],
    ) -> Episode:
        if column not in {"actions", "observations"}:
            raise ValueError("invalid JSONB column")
        async with self.database.pool.connection() as connection:
            async with connection.transaction():
                async with connection.cursor(row_factory=dict_row) as cursor:
                    await cursor.execute(
                        f"SELECT status, jsonb_array_length({column}) AS item_count "
                        "FROM support_episodes WHERE episode_key = %s FOR UPDATE",
                        (episode_key,),
                    )
                    current = await cursor.fetchone()
                    if current is None:
                        raise StoreConflictError("episode not found")
                    if current["status"] != EpisodeStatus.OPEN.value:
                        raise StoreConflictError("completed episodes cannot be changed")
                    if current["item_count"] + 1 != expected_sequence:
                        raise StoreConflictError(
                            "concurrent episode update; reload and retry"
                        )
                    await cursor.execute(
                        f"UPDATE support_episodes SET {column} = {column} || %s "
                        "WHERE episode_key = %s RETURNING *",
                        (Jsonb([payload]), episode_key),
                    )
                    row = await cursor.fetchone()
        return _episode_from_row(row)

    async def complete(
        self,
        episode_key: str,
        *,
        summary: str,
        resolution: str,
        outcome: EpisodeOutcome,
        completed_at: datetime,
        embedding: Sequence[float],
    ) -> Episode:
        sql = """UPDATE support_episodes
        SET episode_summary = %s, resolution = %s, outcome = %s,
            status = 'completed', completed_at = %s, embedding = %s
        WHERE episode_key = %s AND status = 'open'
        RETURNING *"""
        async with self.database.pool.connection() as connection:
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    sql,
                    (
                        summary,
                        resolution,
                        outcome.value,
                        completed_at,
                        Vector(embedding),
                        episode_key,
                    ),
                )
                row = await cursor.fetchone()
        if row is None:
            raise StoreConflictError("episode is missing or already completed")
        return _episode_from_row(row)

    async def search(
        self,
        embedding: Sequence[float],
        *,
        top_k: int,
        issue_type: str | None,
        customer_id: str | None,
        outcomes: tuple[EpisodeOutcome, ...],
        min_similarity: float | None,
    ) -> list[EpisodeSearchResult]:
        vector = Vector(embedding)
        conditions = ["status = 'completed'", "embedding IS NOT NULL"]
        parameters: list[Any] = []
        if outcomes == (EpisodeOutcome.RESOLVED,):
            # This exact predicate allows PostgreSQL to use the partial HNSW index.
            conditions.append("outcome = 'resolved'")
        else:
            conditions.append("outcome = ANY(%s)")
            parameters.append([item.value for item in outcomes])
        if issue_type is not None:
            conditions.append("issue_type = %s")
            parameters.append(issue_type)
        if customer_id is not None:
            conditions.append("customer_id = %s")
            parameters.append(customer_id)
        if min_similarity is not None:
            conditions.append("(1 - (embedding <=> %s)) >= %s")
            parameters.extend([vector, min_similarity])
        parameters.extend([vector, top_k])
        sql = (
            "SELECT *, embedding <=> %s AS distance FROM support_episodes WHERE "
            + " AND ".join(conditions)
            + " ORDER BY embedding <=> %s LIMIT %s"
        )
        # The first vector is for SELECT; condition parameters follow; final vector is ORDER BY.
        parameters.insert(0, vector)
        async with self.database.pool.connection() as connection:
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(sql, parameters)
                rows = await cursor.fetchall()
        return [
            EpisodeSearchResult(
                episode=_episode_from_row(row),
                distance=float(row["distance"]),
                similarity_score=max(-1.0, min(1.0, 1.0 - float(row["distance"]))),
            )
            for row in rows
        ]

    async def list(
        self, *, status: EpisodeStatus | None = None, limit: int = 100
    ) -> list[Episode]:
        sql = "SELECT * FROM support_episodes"
        parameters: list[Any] = []
        if status is not None:
            sql += " WHERE status = %s"
            parameters.append(status.value)
        sql += " ORDER BY created_at DESC LIMIT %s"
        parameters.append(limit)
        async with self.database.pool.connection() as connection:
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(sql, parameters)
                rows = await cursor.fetchall()
        return [_episode_from_row(row) for row in rows]

    async def delete(self, episode_key: str) -> bool:
        async with self.database.pool.connection() as connection:
            cursor = await connection.execute(
                "DELETE FROM support_episodes WHERE episode_key = %s", (episode_key,)
            )
        return cursor.rowcount == 1

    async def stats(self) -> EpisodeStats:
        sql = """SELECT
            count(*) AS total,
            count(*) FILTER (WHERE status = 'open') AS open,
            count(*) FILTER (WHERE status = 'completed') AS completed,
            count(*) FILTER (WHERE outcome = 'resolved') AS resolved,
            count(*) FILTER (WHERE outcome = 'escalated') AS escalated,
            count(*) FILTER (WHERE outcome = 'failed') AS failed,
            count(*) FILTER (WHERE outcome = 'abandoned') AS abandoned
        FROM support_episodes"""
        async with self.database.pool.connection() as connection:
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(sql)
                row = await cursor.fetchone()
        return EpisodeStats(**row)
