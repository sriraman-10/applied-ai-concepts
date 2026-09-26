"""SQLite-backed deterministic key-value memory."""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from models import JSONValue, KVMemory, MemoryKey, MemoryStats, ValueType

logger = logging.getLogger(__name__)

Clock = Callable[[], datetime]
_MAX_NAMESPACE_LENGTH = 128
_MAX_ENTITY_LENGTH = 256
_MAX_KEY_LENGTH = 256


class MemoryValidationError(ValueError):
    """Input cannot be represented safely by the KV memory contract."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise MemoryValidationError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat()


def _parse_timestamp(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value is not None else None


def _validate_identifier(name: str, value: str, maximum: int) -> str:
    normalized = value.strip()
    if not normalized:
        raise MemoryValidationError(f"{name} cannot be empty")
    if len(normalized) > maximum:
        raise MemoryValidationError(f"{name} cannot exceed {maximum} characters")
    if any(ord(character) < 32 for character in normalized):
        raise MemoryValidationError(f"{name} cannot contain control characters")
    return normalized


def _value_type(value: JSONValue) -> ValueType:
    if value is None:
        return ValueType.NULL
    if isinstance(value, bool):
        return ValueType.BOOLEAN
    if isinstance(value, str):
        return ValueType.STRING
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise MemoryValidationError("numeric values must be finite")
        return ValueType.NUMBER
    if isinstance(value, list):
        return ValueType.LIST
    if isinstance(value, dict):
        return ValueType.OBJECT
    raise MemoryValidationError(f"unsupported value type: {type(value).__name__}")


def _serialize(value: JSONValue) -> str:
    _value_type(value)
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise MemoryValidationError("value must be JSON-serializable") from error


def _serialize_metadata(metadata: dict[str, JSONValue] | None) -> str | None:
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise MemoryValidationError("metadata must be an object")
    return _serialize(metadata)


class KVMemoryManager:
    """Persistent exact-key memory with TTL and explicit lifecycle operations."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        max_ttl_seconds: int = 31_536_000,
        clock: Clock = _utc_now,
    ) -> None:
        if max_ttl_seconds <= 0:
            raise ValueError("max_ttl_seconds must be greater than zero")
        self.database_path = Path(database_path) if str(database_path) != ":memory:" else Path(":memory:")
        self.max_ttl_seconds = max_ttl_seconds
        self._clock = clock
        if str(self.database_path) != ":memory:":
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.database_path))
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA busy_timeout = 5000")
        if str(self.database_path) != ":memory:":
            self._connection.execute("PRAGMA journal_mode = WAL")
        self._initialize_schema()

    def __enter__(self) -> KVMemoryManager:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def _initialize_schema(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS kv_memories (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    memory_key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    value_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT,
                    metadata_json TEXT,
                    UNIQUE(namespace, entity_id, memory_key)
                );
                CREATE INDEX IF NOT EXISTS idx_kv_namespace_entity
                    ON kv_memories(namespace, entity_id);
                CREATE INDEX IF NOT EXISTS idx_kv_expires_at
                    ON kv_memories(expires_at) WHERE expires_at IS NOT NULL;
                PRAGMA user_version = 1;
                """
            )

    def _validated_key(self, namespace: str, entity_id: str, key: str) -> MemoryKey:
        return MemoryKey(
            namespace=_validate_identifier("namespace", namespace, _MAX_NAMESPACE_LENGTH),
            entity_id=_validate_identifier("entity_id", entity_id, _MAX_ENTITY_LENGTH),
            key=_validate_identifier("key", key, _MAX_KEY_LENGTH),
        )

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            raise MemoryValidationError("clock must return a timezone-aware datetime")
        return value.astimezone(timezone.utc)

    def _expiry(self, now: datetime, ttl_seconds: int | None) -> datetime | None:
        if ttl_seconds is None:
            return None
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int):
            raise MemoryValidationError("ttl_seconds must be an integer or null")
        if ttl_seconds <= 0 or ttl_seconds > self.max_ttl_seconds:
            raise MemoryValidationError(
                f"ttl_seconds must be between 1 and {self.max_ttl_seconds}"
            )
        return now + timedelta(seconds=ttl_seconds)

    def _row_to_memory(self, row: sqlite3.Row) -> KVMemory:
        metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else None
        return KVMemory(
            id=row["id"],
            namespace=row["namespace"],
            entity_id=row["entity_id"],
            key=row["memory_key"],
            value=json.loads(row["value_json"]),
            value_type=ValueType(row["value_type"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            expires_at=_parse_timestamp(row["expires_at"]),
            metadata=metadata,
        )

    def set(
        self,
        namespace: str,
        entity_id: str,
        key: str,
        value: JSONValue,
        *,
        ttl_seconds: int | None = None,
        metadata: dict[str, JSONValue] | None = None,
    ) -> KVMemory:
        """Create or update one logical key while preserving its identity and creation time."""
        memory_key = self._validated_key(namespace, entity_id, key)
        value_json = _serialize(value)
        value_type = _value_type(value)
        metadata_json = _serialize_metadata(metadata)
        now = self._now()
        expires_at = self._expiry(now, ttl_seconds)

        with self._connection:
            existing = self._connection.execute(
                """SELECT id FROM kv_memories
                   WHERE namespace = ? AND entity_id = ? AND memory_key = ?""",
                (memory_key.namespace, memory_key.entity_id, memory_key.key),
            ).fetchone()
            memory_id = existing["id"] if existing else str(uuid4())
            self._connection.execute(
                """
                INSERT INTO kv_memories (
                    id, namespace, entity_id, memory_key, value_json, value_type,
                    created_at, updated_at, expires_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, entity_id, memory_key) DO UPDATE SET
                    value_json = excluded.value_json,
                    value_type = excluded.value_type,
                    updated_at = excluded.updated_at,
                    expires_at = excluded.expires_at,
                    metadata_json = excluded.metadata_json
                """,
                (
                    memory_id,
                    memory_key.namespace,
                    memory_key.entity_id,
                    memory_key.key,
                    value_json,
                    value_type.value,
                    _timestamp(now),
                    _timestamp(now),
                    _timestamp(expires_at) if expires_at else None,
                    metadata_json,
                ),
            )

        action = "memory_update" if existing else "memory_write"
        logger.info(
            "%s namespace=%s entity=%s key=%s ttl=%s",
            action,
            memory_key.namespace,
            memory_key.entity_id,
            memory_key.key,
            ttl_seconds,
        )
        result = self.get(memory_key.namespace, memory_key.entity_id, memory_key.key)
        if result is None:  # Defensive: a positive TTL cannot expire immediately.
            raise RuntimeError("newly written memory could not be read")
        return result

    def get(self, namespace: str, entity_id: str, key: str) -> KVMemory | None:
        """Return one exact non-expired value, or None."""
        memory_key = self._validated_key(namespace, entity_id, key)
        row = self._connection.execute(
            """SELECT * FROM kv_memories
               WHERE namespace = ? AND entity_id = ? AND memory_key = ?""",
            (memory_key.namespace, memory_key.entity_id, memory_key.key),
        ).fetchone()
        if row is None:
            logger.info(
                "memory_read namespace=%s entity=%s key=%s hit=false",
                memory_key.namespace,
                memory_key.entity_id,
                memory_key.key,
            )
            return None
        expires_at = _parse_timestamp(row["expires_at"])
        if expires_at is not None and expires_at <= self._now():
            logger.info(
                "memory_expired namespace=%s entity=%s key=%s",
                memory_key.namespace,
                memory_key.entity_id,
                memory_key.key,
            )
            return None
        logger.info(
            "memory_read namespace=%s entity=%s key=%s hit=true",
            memory_key.namespace,
            memory_key.entity_id,
            memory_key.key,
        )
        return self._row_to_memory(row)

    def exists(self, namespace: str, entity_id: str, key: str) -> bool:
        return self.get(namespace, entity_id, key) is not None

    def delete(self, namespace: str, entity_id: str, key: str) -> bool:
        memory_key = self._validated_key(namespace, entity_id, key)
        with self._connection:
            cursor = self._connection.execute(
                """DELETE FROM kv_memories
                   WHERE namespace = ? AND entity_id = ? AND memory_key = ?""",
                (memory_key.namespace, memory_key.entity_id, memory_key.key),
            )
        deleted = cursor.rowcount > 0
        logger.info(
            "memory_delete namespace=%s entity=%s key=%s deleted=%s",
            memory_key.namespace,
            memory_key.entity_id,
            memory_key.key,
            str(deleted).lower(),
        )
        return deleted

    def list(self, namespace: str, entity_id: str) -> list[KVMemory]:
        """List active memories for one indexed namespace/entity pair."""
        normalized_namespace = _validate_identifier(
            "namespace", namespace, _MAX_NAMESPACE_LENGTH
        )
        normalized_entity = _validate_identifier(
            "entity_id", entity_id, _MAX_ENTITY_LENGTH
        )
        now = _timestamp(self._now())
        rows = self._connection.execute(
            """SELECT * FROM kv_memories
               WHERE namespace = ? AND entity_id = ?
                 AND (expires_at IS NULL OR expires_at > ?)
               ORDER BY memory_key""",
            (normalized_namespace, normalized_entity, now),
        ).fetchall()
        logger.info(
            "memory_list namespace=%s entity=%s count=%d",
            normalized_namespace,
            normalized_entity,
            len(rows),
        )
        return [self._row_to_memory(row) for row in rows]

    def clear_namespace(self, namespace: str, entity_id: str | None = None) -> int:
        """Delete a whole namespace, optionally limited to one entity."""
        normalized_namespace = _validate_identifier(
            "namespace", namespace, _MAX_NAMESPACE_LENGTH
        )
        with self._connection:
            if entity_id is None:
                cursor = self._connection.execute(
                    "DELETE FROM kv_memories WHERE namespace = ?",
                    (normalized_namespace,),
                )
            else:
                normalized_entity = _validate_identifier(
                    "entity_id", entity_id, _MAX_ENTITY_LENGTH
                )
                cursor = self._connection.execute(
                    "DELETE FROM kv_memories WHERE namespace = ? AND entity_id = ?",
                    (normalized_namespace, normalized_entity),
                )
        logger.info(
            "memory_clear namespace=%s entity=%s deleted=%d",
            normalized_namespace,
            entity_id or "*",
            cursor.rowcount,
        )
        return cursor.rowcount

    def cleanup_expired(self) -> int:
        now = _timestamp(self._now())
        with self._connection:
            cursor = self._connection.execute(
                "DELETE FROM kv_memories WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (now,),
            )
        logger.info("memory_cleanup expired_deleted=%d", cursor.rowcount)
        return cursor.rowcount

    def stats(self) -> MemoryStats:
        now = _timestamp(self._now())
        row = self._connection.execute(
            """
            SELECT
                SUM(CASE WHEN expires_at IS NULL OR expires_at > ? THEN 1 ELSE 0 END) AS active,
                SUM(CASE WHEN expires_at IS NOT NULL AND expires_at <= ? THEN 1 ELSE 0 END) AS expired,
                COUNT(DISTINCT CASE WHEN expires_at IS NULL OR expires_at > ? THEN namespace END) AS namespaces,
                COUNT(DISTINCT CASE WHEN expires_at IS NULL OR expires_at > ?
                    THEN namespace || char(0) || entity_id END) AS entities
            FROM kv_memories
            """,
            (now, now, now, now),
        ).fetchone()
        return MemoryStats(
            active_entries=int(row["active"] or 0),
            expired_entries=int(row["expired"] or 0),
            namespaces=int(row["namespaces"] or 0),
            entities=int(row["entities"] or 0),
        )
