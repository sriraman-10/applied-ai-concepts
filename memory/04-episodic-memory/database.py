"""Async PostgreSQL connection pool and idempotent migration runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool


class DatabaseError(RuntimeError):
    pass


class PostgresDatabase:
    def __init__(
        self, database_url: str, embedding_dimensions: int, migrations_path: Path
    ) -> None:
        self.database_url = database_url.replace(
            "postgresql+psycopg://", "postgresql://", 1
        )
        if (
            isinstance(embedding_dimensions, bool)
            or not isinstance(embedding_dimensions, int)
            or not 1 <= embedding_dimensions <= 2_000
        ):
            raise ValueError("embedding_dimensions must be an integer from 1 to 2000")
        self.embedding_dimensions = embedding_dimensions
        self.migrations_path = migrations_path
        self._pool: AsyncConnectionPool[Any] | None = None

    async def open(self) -> None:
        try:
            connection = await AsyncConnection.connect(
                self.database_url, autocommit=True
            )
            async with connection:
                migration_files = sorted(self.migrations_path.glob("*.sql"))
                if not migration_files:
                    raise DatabaseError(
                        f"No migrations found in {self.migrations_path}"
                    )
                for path in migration_files:
                    sql_text = path.read_text(encoding="utf-8").replace(
                        "{{EMBEDDING_DIMENSIONS}}", str(self.embedding_dimensions)
                    )
                    await connection.execute(sql_text, prepare=False)

            async def configure(conn: AsyncConnection[Any]) -> None:
                await register_vector_async(conn)

            self._pool = AsyncConnectionPool(
                self.database_url,
                min_size=1,
                max_size=10,
                open=False,
                configure=configure,
            )
            await self._pool.open(wait=True)
        except DatabaseError:
            raise
        except Exception as error:
            raise DatabaseError(f"Could not initialize PostgreSQL: {error}") from error

    @property
    def pool(self) -> AsyncConnectionPool[Any]:
        if self._pool is None:
            raise DatabaseError("Database is not open")
        return self._pool

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
